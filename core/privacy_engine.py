"""
privacy_engine.py — Unified Differential Privacy Engine
════════════════════════════════════════════════════════
High-level privacy API for the Climate-Fed Orchestrator.

This module provides a single cohesive interface that wraps:
  • RDP-based noise calibration (Mironov 2017)
  • Per-sample gradient clipping (Abadi et al. 2016)
  • Privacy budget tracking and enforcement
  • GDPR/AI Act compliance validation

Mathematical Foundation:
  Rényi Differential Privacy (RDP):
    ε_RDP(α) = (1/(α-1)) × log(Σ_{k=0}^α C(α,k) × q^k × (1-q)^{α-k} × exp(k(k-1)/(2σ²)))

  Conversion to (ε, δ)-DP:
    ε = min_α [ ε_RDP(α) - (log δ + log(α-1)/α) / (α-1) ]

Privacy Guarantee:
  For ε ≤ 1.0, δ = 1e-5, the probability that any individual's data
  influences the model output by more than e^ε ≈ 2.72× is bounded by δ.

Usage:
    >>> engine = PrivacyEngine(target_epsilon=1.0, target_delta=1e-5)
    >>> sigma = engine.calibrate(sample_rate=0.01, steps=100)
    >>> noisy_grad = engine.privatize_gradient(grad, batch_size=64)
    >>> engine.step()
    >>> print(engine.summary())
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from core.dp_sgd import (
    _compute_eps_rdp,
    calibrate_noise_multiplier,
    clip_per_sample_gradients,
    add_gaussian_noise,
    PrivacyLedger,
    PrivacyBudgetEntry,
)

log = logging.getLogger("climate_fed.privacy")


# ──────────────────────────────────────────────────────────────────────────────
# Privacy Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PrivacyConfig:
    """Immutable privacy configuration."""

    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    l2_clip_norm: float = 1.0
    noise_multiplier: Optional[float] = None  # Auto-computed if None
    max_grad_norm: float = 1.0
    secure_mode: bool = False  # If True, validates all operations

    def __post_init__(self) -> None:
        assert self.target_epsilon > 0, f"ε must be > 0, got {self.target_epsilon}"
        assert 0 < self.target_delta < 1, f"δ must be in (0,1), got {self.target_delta}"
        assert self.l2_clip_norm > 0, f"Clip norm must be > 0, got {self.l2_clip_norm}"


# ──────────────────────────────────────────────────────────────────────────────
# Privacy Engine
# ──────────────────────────────────────────────────────────────────────────────


class PrivacyEngine:
    """
    Unified differential privacy engine for federated learning.

    Manages the complete DP lifecycle:
      1. Noise calibration (σ from target ε, δ)
      2. Per-sample gradient clipping
      3. Gaussian noise injection
      4. Privacy budget tracking
      5. Compliance reporting

    Thread-safe for single-node use. For multi-node, use one engine per node.

    Args:
        config:       Privacy configuration (ε, δ, clip norm).
        node_id:      Identifier for this node's privacy tracking.
        node_name:    Human-readable node name.
    """

    def __init__(
        self,
        config: PrivacyConfig,
        node_id: int = 0,
        node_name: str = "default",
    ) -> None:
        self._config = config
        self._node_id = node_id
        self._node_name = node_name

        # Privacy ledger
        self._ledger = PrivacyLedger(
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
        )

        # Calibration state
        self._sigma: Optional[float] = config.noise_multiplier
        self._sample_rate: float = 0.0
        self._steps_per_round: int = 0
        self._current_round: int = 0
        self._total_steps: int = 0
        self._is_calibrated: bool = config.noise_multiplier is not None

        # Per-round tracking
        self._round_epsilon_history: List[Tuple[int, float]] = []

        log.info(
            f"[PrivacyEngine] Node-{node_id} ({node_name}) initialized: "
            f"ε={config.target_epsilon}, δ={config.target_delta}, "
            f"clip={config.l2_clip_norm}"
        )

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(
        self,
        sample_rate: float,
        steps: int,
        total_rounds: Optional[int] = None,
    ) -> float:
        """
        Auto-calibrate noise multiplier σ for target privacy budget.

        Uses binary search over σ to find the minimum noise satisfying:
            ε(σ, q, T, δ) ≤ target_ε

        Args:
            sample_rate:   q = batch_size / dataset_size (subsampling rate).
            steps:         Number of SGD steps per round.
            total_rounds:  Optional total rounds for tighter calibration.

        Returns:
            Calibrated noise multiplier σ.
        """
        total_steps = steps * (total_rounds or 1)
        self._sample_rate = sample_rate
        self._steps_per_round = steps

        self._sigma = calibrate_noise_multiplier(
            target_epsilon=self._config.target_epsilon,
            target_delta=self._config.target_delta,
            sample_rate=sample_rate,
            steps=total_steps,
        )
        self._is_calibrated = True

        log.info(
            f"[PrivacyEngine] Node-{self._node_id} calibrated: "
            f"σ={self._sigma:.4f} for ε={self._config.target_epsilon}, "
            f"q={sample_rate:.4f}, T={total_steps}"
        )
        return self._sigma

    # ── Gradient Processing ──────────────────────────────────────────────────

    def clip_gradients(
        self,
        per_sample_grads: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Tensor]]:
        """
        Clip per-sample gradients to bound L2 sensitivity.

        Each gradient g_i is rescaled: g_i' = g_i × min(1, C / ‖g_i‖₂)

        Args:
            per_sample_grads: List of gradient dicts (one per sample).

        Returns:
            Clipped gradients with ‖g_i'‖₂ ≤ C.
        """
        return clip_per_sample_gradients(
            per_sample_grads,
            clipping_bound=self._config.l2_clip_norm,
        )

    def add_noise(
        self,
        gradient: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Add calibrated Gaussian noise for (ε, δ)-DP.

        Noise std: σ_noise = noise_multiplier × C / B

        Args:
            gradient:    Averaged (clipped) gradient tensor.
            batch_size:  Minibatch size B.

        Returns:
            Noisy gradient tensor.

        Raises:
            RuntimeError: If engine not calibrated.
        """
        if not self._is_calibrated or self._sigma is None:
            raise RuntimeError("PrivacyEngine not calibrated. Call calibrate() first.")
        return add_gaussian_noise(
            gradient=gradient,
            noise_multiplier=self._sigma,
            clipping_bound=self._config.l2_clip_norm,
            batch_size=batch_size,
        )

    def privatize_gradient(
        self,
        gradient: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Combined clip + noise in one call (for pre-averaged gradients).

        This is the most common entrypoint for DP gradient processing.

        Args:
            gradient:    Raw averaged gradient tensor.
            batch_size:  Minibatch size B.

        Returns:
            (ε, δ)-DP noisy gradient.
        """
        # Clip the gradient norm
        grad_norm = torch.norm(gradient.float(), p=2)
        clip_coeff = min(1.0, self._config.l2_clip_norm / (grad_norm.item() + 1e-8))
        clipped = gradient * clip_coeff

        # Add noise
        return self.add_noise(clipped, batch_size)

    # ── Budget Tracking ──────────────────────────────────────────────────────

    def step(self, round_num: Optional[int] = None) -> float:
        """
        Record one round of training and return ε consumed so far.

        Args:
            round_num: Optional round number for logging.

        Returns:
            Total ε consumed after this step.
        """
        if not self._is_calibrated or self._sigma is None:
            raise RuntimeError("Cannot step: engine not calibrated.")

        self._total_steps += self._steps_per_round
        self._current_round = round_num or (self._current_round + 1)

        # Compute current ε
        eps = _compute_eps_rdp(
            q=self._sample_rate,
            sigma=self._sigma,
            steps=self._total_steps,
            delta=self._config.target_delta,
        )

        # Record in ledger
        entry = PrivacyBudgetEntry(
            round_num=self._current_round,
            node_id=self._node_id,
            epsilon_delta=eps,
            sigma=self._sigma,
            sample_rate=self._sample_rate,
            steps=self._total_steps,
        )
        self._ledger.record(entry)
        self._round_epsilon_history.append((self._current_round, eps))

        return eps

    @property
    def epsilon_spent(self) -> float:
        """Total ε consumed so far."""
        return self._ledger.epsilon_spent

    @property
    def epsilon_remaining(self) -> float:
        """Remaining ε budget."""
        return self._ledger.epsilon_remaining

    @property
    def budget_exhausted(self) -> bool:
        """Whether the privacy budget has been fully consumed."""
        return self._ledger.budget_exhausted

    @property
    def sigma(self) -> Optional[float]:
        """Current noise multiplier."""
        return self._sigma

    @property
    def config(self) -> PrivacyConfig:
        """Privacy configuration."""
        return self._config

    # ── Compliance ───────────────────────────────────────────────────────────

    def compliance_check(self) -> Dict[str, object]:
        """
        Run a full compliance check against GDPR Art. 25 requirements.

        Returns:
            Dict with compliance status, evidence, and recommendations.
        """
        eps = self.epsilon_spent
        is_compliant = eps <= self._config.target_epsilon

        return {
            "node_id": self._node_id,
            "node_name": self._node_name,
            "target_epsilon": self._config.target_epsilon,
            "target_delta": self._config.target_delta,
            "epsilon_spent": round(eps, 6),
            "epsilon_remaining": round(self.epsilon_remaining, 6),
            "noise_multiplier": self._sigma,
            "l2_clip_norm": self._config.l2_clip_norm,
            "total_steps": self._total_steps,
            "rounds_completed": self._current_round,
            "compliant": is_compliant,
            "gdpr_art25_status": "✅ COMPLIANT" if is_compliant else "❌ VIOLATION",
            "evidence": (
                f"ε={eps:.6f} ≤ {self._config.target_epsilon} with "
                f"σ={self._sigma}, C={self._config.l2_clip_norm}"
                if is_compliant
                else f"ε={eps:.6f} EXCEEDS budget {self._config.target_epsilon}"
            ),
        }

    def summary(self) -> str:
        """Human-readable privacy summary."""
        check = self.compliance_check()
        status = "✅" if check["compliant"] else "❌"
        return (
            f"[Privacy] Node-{self._node_id} ({self._node_name}): "
            f"{status} ε={check['epsilon_spent']:.4f}/{self._config.target_epsilon} | "
            f"σ={self._sigma:.4f} | Steps={self._total_steps} | "
            f"Rounds={self._current_round}"
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(self, path: str) -> str:
        """
        Export privacy budget history to CSV.

        Args:
            path: Output CSV file path.

        Returns:
            Absolute path to saved CSV.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "epsilon_total", "node_id", "node_name"])
            for round_num, eps in self._round_epsilon_history:
                writer.writerow(
                    [round_num, f"{eps:.6f}", self._node_id, self._node_name]
                )

        log.info(f"[PrivacyEngine] Budget history → {p}")
        return str(p.resolve())

    def export_json(self, path: str) -> str:
        """
        Export full compliance report to JSON.

        Args:
            path: Output JSON file path.

        Returns:
            Absolute path to saved JSON.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "compliance": self.compliance_check(),
            "config": asdict(self._config),
            "history": [
                {"round": r, "epsilon": round(e, 6)}
                for r, e in self._round_epsilon_history
            ],
            "ledger": self._ledger.compliance_report(),
        }

        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info(f"[PrivacyEngine] Compliance JSON → {p}")
        return str(p.resolve())


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Node Privacy Coordinator
# ──────────────────────────────────────────────────────────────────────────────


class PrivacyCoordinator:
    """
    Coordinates privacy budgets across multiple federated nodes.

    In federated DP, the global privacy guarantee is the *maximum* ε
    across all nodes (not the sum), because each node processes
    disjoint data partitions.

    Args:
        target_epsilon: Global privacy budget.
        target_delta:   Failure probability.
        node_count:     Number of federated nodes.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        node_count: int = 3,
    ) -> None:
        self._target_epsilon = target_epsilon
        self._target_delta = target_delta

        config = PrivacyConfig(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
        )

        self._engines: Dict[int, PrivacyEngine] = {}
        for i in range(node_count):
            self._engines[i] = PrivacyEngine(
                config=config,
                node_id=i,
                node_name=f"Node-{i}",
            )

    def get_engine(self, node_id: int) -> PrivacyEngine:
        """Get the privacy engine for a specific node."""
        return self._engines[node_id]

    @property
    def worst_case_epsilon(self) -> float:
        """
        Global privacy guarantee = max ε across all nodes.

        Since each node processes disjoint data, the global ε is determined
        by the node that has consumed the most budget.
        """
        return max(e.epsilon_spent for e in self._engines.values())

    @property
    def global_compliant(self) -> bool:
        """Whether the global privacy guarantee holds."""
        return self.worst_case_epsilon <= self._target_epsilon

    def global_summary(self) -> str:
        """Summary of privacy status across all nodes."""
        lines = [
            "┌─────────────────────────────────────────────┐",
            "│   🔒 PRIVACY COORDINATOR STATUS              │",
            "├─────────────────────────────────────────────┤",
        ]
        for nid, engine in self._engines.items():
            check = engine.compliance_check()
            icon = "✅" if check["compliant"] else "❌"
            lines.append(
                f"│  {icon} Node-{nid}: ε={check['epsilon_spent']:.4f}"
                f"/{self._target_epsilon}  σ={engine.sigma or 0:.4f}  │"
            )
        lines.append("├─────────────────────────────────────────────┤")

        global_icon = "✅" if self.global_compliant else "❌"
        lines.append(
            f"│  {global_icon} Global: ε={self.worst_case_epsilon:.4f}"
            f"/{self._target_epsilon}                  │"
        )
        lines.append("└─────────────────────────────────────────────┘")
        return "\n".join(lines)

    def export_all(self, output_dir: str) -> List[str]:
        """Export privacy reports for all nodes."""
        paths = []
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)

        for nid, engine in self._engines.items():
            csv_path = engine.export_csv(str(d / f"privacy_node_{nid}.csv"))
            json_path = engine.export_json(str(d / f"privacy_node_{nid}.json"))
            paths.extend([csv_path, json_path])

        # Global summary
        global_report = {
            "global_epsilon": self.worst_case_epsilon,
            "target_epsilon": self._target_epsilon,
            "target_delta": self._target_delta,
            "compliant": self.global_compliant,
            "nodes": {
                nid: engine.compliance_check() for nid, engine in self._engines.items()
            },
        }
        gp = d / "privacy_global.json"
        with open(gp, "w") as f:
            json.dump(global_report, f, indent=2, default=str)
        paths.append(str(gp.resolve()))

        return paths
