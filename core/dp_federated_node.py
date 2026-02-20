"""
dp_federated_node.py — Differentially Private Carbon-Aware Federated Node
══════════════════════════════════════════════════════════════════════════════
This module extends the base CarbonAwareNode with:
  1.  Differential Privacy (DP-SGD, Moments Accountant)
  2.  Live carbon intensity sourcing (CarbonAPIManager)
  3.  Privacy budget lifecycle management
  4.  Per-round DP training receipts for audit logging

Decision Flow per training round:
    ┌── Get Live Carbon Score ──────────────────────────────────────┐
    │   → Try ElectricityMaps / WattTime / UK Grid / Simulation     │
    └─────────────────────────────────────────────────────────────── ┘
                    ↓
    ┌── Carbon Gate ─────────────────────────────────────────────── ┐
    │   if renewable_score < threshold → IDLE                        │
    │   else → continue to DP training                               │
    └────────────────────────────────────────────────────────────── ┘
                    ↓
    ┌── Privacy Gate ─────────────────────────────────────────────  ┐
    │   if epsilon_spent >= target_epsilon → PAUSE (budget exhaust.) │
    │   else → DP-SGD training round                                 │
    └────────────────────────────────────────────────────────────── ┘
                    ↓
    ┌── DP-SGD Training ─────────────────────────────────────────── ┐
    │   Per-sample gradient clipping (L2 bound C)                    │
    │   + Gaussian noise (σ = noise_mult × C)                        │
    │   + Adaptive LR scaling (Oracle mode)                          │
    └────────────────────────────────────────────────────────────── ┘
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_fed_orchestrator.core.dp_sgd import (
    DPSGDOptimizer,
    DPTrainingResult,
    PrivacyLedger,
    calibrate_noise_multiplier,
    run_dp_training_round,
)
from climate_fed_orchestrator.core.live_carbon_api import (
    CarbonAPIManager,
    LiveCarbonData,
)

log = logging.getLogger("climate_fed.dp_node")


# ──────────────────────────────────────────────────────────────────────────────
# Result Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PrivateNodeRoundResult:
    """All metadata for one DP training round on one node."""

    node_id: int
    node_name: str
    is_active: bool
    skip_reason: Optional[str]

    # Model update
    weight_delta: Optional[Dict[str, torch.Tensor]]
    num_samples: int
    local_accuracy: float
    local_loss: float
    compression_ratio: float

    # Carbon metadata
    renewable_score: float
    carbon_intensity: float
    live_carbon_data: Optional[LiveCarbonData]

    # Privacy metadata
    epsilon_consumed: float
    epsilon_total: float
    dp_satisfied: bool
    noise_multiplier: float
    sigma: float


# ──────────────────────────────────────────────────────────────────────────────
# PrivateCarbonNode
# ──────────────────────────────────────────────────────────────────────────────


class PrivateCarbonNode:
    """
    Federated learning participant with ε-differential privacy and live CO₂ data.

    Integrates:
      - DPSGDOptimizer for mathematically rigorous gradient privatisation
      - CarbonAPIManager for real-time carbon intensity signals
      - PrivacyLedger for cross-round budget tracking

    Args:
        node_id:           Unique node integer ID.
        node_name:         Human-readable name (e.g. "Oslo").
        lat, lon:          Geographic coordinates.
        zone:              Grid zone code (e.g. "NO", "AU-VIC").
        data_loader:       Per-node DataLoader (non-IID partition).
        model:             Reference global model (will be deepcopied).
        base_lr:           Base learning rate for SGD.
        momentum:          SGD momentum.
        device:            Torch device.
        carbon_api:        CarbonAPIManager instance (shared across nodes).
        privacy_ledger:    PrivacyLedger shared across nodes for global tracking.
        target_epsilon:    ε budget per node (reset each experiment).
        target_delta:      δ probability bound.
        l2_norm_clip:      L2 gradient clipping bound C.
        noise_multiplier:  σ/C ratio (auto-calibrated if None).
        renewable_threshold: Minimum renewable score to participate.
        adaptive_lr:       Scale LR by renewable score (Oracle mode).
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        lat: float,
        lon: float,
        zone: str,
        data_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        device: torch.device = torch.device("cpu"),
        carbon_api: Optional[CarbonAPIManager] = None,
        privacy_ledger: Optional[PrivacyLedger] = None,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        l2_norm_clip: float = 1.0,
        noise_multiplier: Optional[float] = None,
        renewable_threshold: float = 0.45,
        adaptive_lr: bool = False,
        num_rounds: int = 1,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.lat, self.lon = lat, lon
        self.zone = zone
        self.loader = data_loader
        self.model_ref = model
        self.base_lr = base_lr
        self.momentum = momentum
        self.device = device
        self.carbon_api = carbon_api
        self.privacy_ledger = privacy_ledger
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.l2_norm_clip = l2_norm_clip
        self.renewable_threshold = renewable_threshold
        self.adaptive_lr = adaptive_lr
        self.total_steps_taken = 0

        dataset_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size or 64
        steps_per_epoch = max(1, dataset_size // batch_size)

        # Auto-calibrate noise multiplier if not provided
        if noise_multiplier is not None:
            self._noise_mult = noise_multiplier
        else:
            self._noise_mult = calibrate_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=batch_size / max(dataset_size, 1),
                steps=steps_per_epoch * num_rounds,
            )
            log.info(
                f"[Node-{node_id}] Auto-calibrated noise multiplier: σ={self._noise_mult:.4f} "
                f"for ε={target_epsilon}, δ={target_delta}"
            )

        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._criterion = nn.CrossEntropyLoss()

    # ── Live Carbon Data ──────────────────────────────────────────────────────

    def get_carbon_data(self) -> LiveCarbonData:
        """
        Fetch live carbon intensity.  Falls back to simulation if unavailable.
        """
        if self.carbon_api is not None:
            try:
                return self.carbon_api.get_sync(
                    zone=self.zone, lat=self.lat, lon=self.lon
                )
            except Exception as e:
                log.warning(
                    f"[Node-{self.node_id}] CarbonAPI failed: {e} — using simulation"
                )

        # Pure simulation
        from climate_fed_orchestrator.core.live_carbon_api import SimulationFallback

        sim = SimulationFallback(lat=self.lat)
        return sim.get(self.zone)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_private_round(
        self,
        global_weights: Dict[str, torch.Tensor],
        round_num: int,
        num_epochs: int = 1,
        carbon_data: Optional[LiveCarbonData] = None,
    ) -> PrivateNodeRoundResult:
        """
        Execute one DP-SGD training round.

        1. Optionally fetch live carbon data.
        2. Apply carbon gate.
        3. Apply privacy budget gate.
        4. Run DP-SGD training.
        5. Return PrivateNodeRoundResult (weight delta + full metadata).
        """
        # ── Carbon Score ─────────────────────────────────────────────────────
        if carbon_data is None:
            carbon_data = self.get_carbon_data()

        renewable_score = carbon_data.renewable_score
        carbon_intensity = carbon_data.carbon_intensity_g_kwh

        # ── Carbon Gate ──────────────────────────────────────────────────────
        if renewable_score < self.renewable_threshold:
            return PrivateNodeRoundResult(
                node_id=self.node_id,
                node_name=self.node_name,
                is_active=False,
                skip_reason=f"Carbon gate: score {renewable_score:.2f} < threshold {self.renewable_threshold:.2f}",
                weight_delta=None,
                num_samples=0,
                local_accuracy=0.0,
                local_loss=0.0,
                compression_ratio=1.0,
                renewable_score=renewable_score,
                carbon_intensity=carbon_intensity,
                live_carbon_data=carbon_data,
                epsilon_consumed=0.0,
                epsilon_total=(
                    self.privacy_ledger.epsilon_spent if self.privacy_ledger else 0.0
                ),
                dp_satisfied=True,
                noise_multiplier=self._noise_mult,
                sigma=self._noise_mult * self.l2_norm_clip,
            )

        # ── Privacy Gate ─────────────────────────────────────────────────────
        if self.privacy_ledger and self.privacy_ledger.budget_exhausted:
            return PrivateNodeRoundResult(
                node_id=self.node_id,
                node_name=self.node_name,
                is_active=False,
                skip_reason=f"Privacy budget exhausted (ε={self.privacy_ledger.epsilon_spent:.3f} ≥ {self.target_epsilon})",
                weight_delta=None,
                num_samples=0,
                local_accuracy=0.0,
                local_loss=0.0,
                compression_ratio=1.0,
                renewable_score=renewable_score,
                carbon_intensity=carbon_intensity,
                live_carbon_data=carbon_data,
                epsilon_consumed=0.0,
                epsilon_total=self.privacy_ledger.epsilon_spent,
                dp_satisfied=False,
                noise_multiplier=self._noise_mult,
                sigma=self._noise_mult * self.l2_norm_clip,
            )

        # ── Adaptive LR (Oracle mode) ─────────────────────────────────────────
        effective_lr = self.base_lr
        if self.adaptive_lr:
            effective_lr = self.base_lr * (0.5 + 0.5 * renewable_score)

        # ── Build fresh local model ───────────────────────────────────────────
        local_model = copy.deepcopy(self.model_ref)
        local_model.load_state_dict(
            {k: v.clone().to(self.device) for k, v in global_weights.items()},
            strict=False,
        )

        # ── DP-SGD Optimizer ─────────────────────────────────────────────────
        dp_optimizer = DPSGDOptimizer(
            model=local_model,
            lr=effective_lr,
            momentum=self.momentum,
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self._noise_mult,
            batch_size=self._batch_size,
            dataset_size=self._dataset_size,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            device=self.device,
        )

        # ── Run Training ─────────────────────────────────────────────────────
        steps = num_epochs * max(1, self._dataset_size // self._batch_size)
        dp_result: DPTrainingResult = run_dp_training_round(
            model=local_model,
            dataloader=self.loader,
            optimizer=dp_optimizer,
            criterion=self._criterion,
            steps=steps,
            privacy_ledger=self.privacy_ledger or PrivacyLedger(),
            node_id=self.node_id,
            round_num=round_num,
            device=self.device,
            total_steps_previously_taken=self.total_steps_taken,
        )
        self.total_steps_taken += steps

        # ── Build weight delta from global ───────────────────────────────────
        weight_delta: Dict[str, torch.Tensor] = {}
        for k, v in local_model.state_dict().items():
            weight_delta[k] = v - global_weights[k].to(self.device)

        log.info(
            f"[Node-{self.node_id}:{self.node_name}] R{round_num:02d} "
            f"| Acc={dp_result.local_accuracy:.4f} | Loss={dp_result.local_loss:.4f} "
            f"| ε={dp_result.epsilon_consumed:.4f} | σ={dp_result.sigma:.4f} "
            f"| CO₂={carbon_intensity:.0f}g/kWh | Score={renewable_score:.2f}"
        )

        return PrivateNodeRoundResult(
            node_id=self.node_id,
            node_name=self.node_name,
            is_active=True,
            skip_reason=None,
            weight_delta=weight_delta,
            num_samples=dp_result.num_samples,
            local_accuracy=dp_result.local_accuracy,
            local_loss=dp_result.local_loss,
            compression_ratio=1.0,
            renewable_score=renewable_score,
            carbon_intensity=carbon_intensity,
            live_carbon_data=carbon_data,
            epsilon_consumed=dp_result.epsilon_consumed,
            epsilon_total=(
                self.privacy_ledger.epsilon_spent
                if self.privacy_ledger
                else dp_result.epsilon_consumed
            ),
            dp_satisfied=dp_result.dp_satisfied,
            noise_multiplier=dp_result.noise_multiplier,
            sigma=dp_result.sigma,
        )
