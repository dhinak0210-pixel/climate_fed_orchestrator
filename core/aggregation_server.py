"""
aggregation_server.py — The Weight Alchemist
═════════════════════════════════════════════
Central server for energy-aware federated model aggregation.

Implements three aggregation strategies, selectable per experiment arm:
  1. FedAvg        — standard sample-count-weighted average
  2. RenewableWeighted — weight = samples × renewable_score²
     Prioritises updates from cleaner energy sources.
  3. TrimmedMean   — Byzantine-robust: discard top/bottom τ% per parameter

Additional features:
  • Server-side SGD momentum for stable global model convergence
  • Global model evaluation against held-out test set
  • Model checkpointing with round metadata

Complexity: O(N × P) for N active nodes, P model parameters.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.federated_node import NodeResult

log = logging.getLogger("climate_fed.server")


@dataclass
class AggregationResult:
    """Outcome of one aggregation step."""

    round_num: int
    global_accuracy: float
    global_loss: float
    num_participants: int
    strategy_used: str
    participants: List[str]


class CarbonAwareAggregator:
    """
    Energy-aware federated model aggregator with Byzantine-robustness.

    The key innovation of RenewableWeighted aggregation is that nodes
    contributing updates from cleaner (higher renewable score) energy
    sources receive proportionally higher weight in the global average.
    This creates a virtuous cycle: nodes with green energy are incentivised
    to participate and rewarded with greater influence over the global model.

    Server-side momentum (Polyak averaging) smooths the trajectory of the
    global model, reducing gradient variance from sparse participation.

    Args:
        model:          Initial global model (will be maintained as state).
        test_loader:    DataLoader for global test set evaluation.
        strategy:       "fedavg" | "renewable_weighted" | "trimmed_mean".
        trimmed_frac:   For trimmed mean — discard this fraction at each tail.
        server_momentum: Momentum factor for server-side update (0 = disabled).
        device:         Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        strategy: str = "renewable_weighted",
        trimmed_frac: float = 0.10,
        server_momentum: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._model = model.to(device)
        self._test_loader = test_loader
        self._strategy = strategy
        self._trimmed_frac = trimmed_frac
        self._server_momentum = server_momentum
        self._device = device
        self._criterion = nn.CrossEntropyLoss()

        # Server-side momentum buffer (Polyak averaging)
        self._momentum_buffer: Optional[Dict[str, torch.Tensor]] = None
        self._round = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def global_weights(self) -> Dict[str, torch.Tensor]:
        """Current global model state dict (cloned for safety)."""
        return {k: v.clone() for k, v in self._model.state_dict().items()}

    def aggregate(
        self,
        node_results: List[NodeResult],
        round_num: int,
    ) -> AggregationResult:
        """
        Aggregate node results into updated global model.

        Only results with `participated=True` and non-None `weight_delta`
        are included.  If no nodes participated, the global model is unchanged.

        Args:
            node_results: List of :class:`NodeResult` from this round's nodes.
            round_num:    Current round number (for logging).

        Returns:
            :class:`AggregationResult` with new global accuracy.
        """
        self._round = round_num
        active = [r for r in node_results if r.participated and r.weight_delta]

        if not active:
            log.warning(
                f"[Round {round_num}] No active nodes — global model unchanged."
            )
            acc, loss = self.evaluate()
            return AggregationResult(
                round_num=round_num,
                global_accuracy=acc,
                global_loss=loss,
                num_participants=0,
                strategy_used=self._strategy,
                participants=[],
            )

        # Dispatch to strategy
        if self._strategy == "fedavg":
            update = self._fedavg(active)
        elif self._strategy == "renewable_weighted":
            update = self._renewable_weighted(active)
        elif self._strategy == "trimmed_mean":
            update = self._trimmed_mean(active)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self._strategy!r}")

        # Apply server-side momentum
        if self._server_momentum > 0:
            update = self._apply_server_momentum(update)

        # Update global model: new_W = W_global + Δ_agg
        with torch.no_grad():
            current = self._model.state_dict()
            new_state = {k: current[k] + update[k].to(self._device) for k in current}
            self._model.load_state_dict(new_state)

        acc, loss = self.evaluate()
        log.info(
            f"[Round {round_num}] Aggregated {len(active)} node(s) "
            f"via {self._strategy} → Acc={acc:.4f} Loss={loss:.4f}"
        )

        return AggregationResult(
            round_num=round_num,
            global_accuracy=acc,
            global_loss=loss,
            num_participants=len(active),
            strategy_used=self._strategy,
            participants=[r.node_name for r in active],
        )

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate global model on the held-out test set.

        Returns:
            Tuple of (accuracy, average_cross_entropy_loss).

        Complexity: O(|test_set|)
        """
        self._model.eval()
        total_correct, total_loss, total_samples = 0, 0.0, 0

        with torch.no_grad():
            for batch_x, batch_y in self._test_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                logits = self._model(batch_x)
                loss = self._criterion(logits, batch_y)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == batch_y).sum())
                total_loss += loss.item() * batch_y.size(0)
                total_samples += batch_y.size(0)

        return (
            total_correct / max(total_samples, 1),
            total_loss / max(total_samples, 1),
        )

    # ── Aggregation Strategies ────────────────────────────────────────────────

    def _fedavg(self, active: List[NodeResult]) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg: sample-count weighted average of weight deltas.

        Formula: Δ_global = Σ_i (n_i / N) × ΔW_i
        where N = Σ n_i (total samples across active nodes).

        Complexity: O(N × P)
        """
        total_samples = sum(r.num_samples for r in active)
        agg: Dict[str, torch.Tensor] = {}

        for result in active:
            w = result.num_samples / max(total_samples, 1)
            for key, delta in result.weight_delta.items():
                d = delta.float().to(self._device)
                agg[key] = agg[key] + w * d if key in agg else w * d

        return agg

    def _renewable_weighted(self, active: List[NodeResult]) -> Dict[str, torch.Tensor]:
        """
        Energy-weighted FedAvg: weight = n_i × renewable_score_i².

        Squaring the renewable score amplifies the advantage of clean energy,
        creating a stronger selection pressure toward renewable-powered nodes.
        This is the core algorithmic innovation of the Carbon-Aware arm.

        Formula: Δ_global = Σ_i w_i × ΔW_i  / Σ_i w_i
                 where w_i = n_i × renewable_i²

        Complexity: O(N × P)
        """
        weights = [r.num_samples * (r.renewable_score**2) for r in active]
        total_w = sum(weights)

        agg: Dict[str, torch.Tensor] = {}
        for result, w in zip(active, weights):
            norm_w = w / max(total_w, 1e-9)
            for key, delta in result.weight_delta.items():
                d = delta.float().to(self._device)
                agg[key] = agg[key] + norm_w * d if key in agg else norm_w * d

        return agg

    def _trimmed_mean(self, active: List[NodeResult]) -> Dict[str, torch.Tensor]:
        """
        Byzantine-robust trimmed mean aggregation.

        For each parameter tensor, sorts the N node updates along the batch
        dimension and removes the top and bottom τ fraction before averaging.
        This rejects gradient poisoning attacks and outlier updates.

        Complexity: O(N × P × log N)
        """
        agg: Dict[str, torch.Tensor] = {}
        keys = list(active[0].weight_delta.keys())
        n = len(active)
        trim = max(1, int(n * self._trimmed_frac))

        for key in keys:
            # Stack deltas: shape (N, *param_shape)
            stacked = torch.stack(
                [r.weight_delta[key].float().to(self._device) for r in active], dim=0
            )
            # Sort along batch dim for each element position
            sorted_stack, _ = torch.sort(stacked, dim=0)
            # Trim and average
            if 2 * trim < n:
                trimmed = sorted_stack[trim : n - trim]
            else:
                trimmed = sorted_stack
            agg[key] = trimmed.mean(dim=0)

        return agg

    def _apply_server_momentum(
        self, update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Polyak momentum to the aggregated update.

        v_t = β × v_{t-1} + (1 - β) × Δ_t
        Smooths noisy aggregated updates from sparse participation.

        Complexity: O(P)
        """
        β = self._server_momentum
        if self._momentum_buffer is None:
            self._momentum_buffer = {k: v.clone() for k, v in update.items()}
            return update

        smoothed: Dict[str, torch.Tensor] = {}
        for key, delta in update.items():
            buf = self._momentum_buffer.get(key, torch.zeros_like(delta))
            new_v = β * buf + (1 - β) * delta
            self._momentum_buffer[key] = new_v
            smoothed[key] = new_v

        return smoothed

    def save_checkpoint(self, save_dir: str, round_num: int) -> str:
        """Persist global model state to disk."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"global_model_round_{round_num:03d}.pt")
        torch.save({"round": round_num, "state_dict": self._model.state_dict()}, path)
        return path
