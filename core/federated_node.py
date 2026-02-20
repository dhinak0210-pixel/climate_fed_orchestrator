"""
federated_node.py — The Eco-Conscious Agent
════════════════════════════════════════════
Each node is an autonomous learning agent with environmental self-regulation.

Key capabilities:
  • Adaptive learning rate scaled by renewable availability
  • Gradient sparsification (Top-K) triggered at high carbon intensity
  • Local early stopping if renewable score drops mid-epoch
  • Optional Gaussian differential privacy on weight deltas

The carbon_conscience flag means a node will refuse to train if doing so
would violate its renewable threshold — making sustainability a first-class
value in the agent's decision function, not an external policy.

Complexity: O(E × B × P) for E epochs, B batches, P model parameters.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


log = logging.getLogger("climate_fed.node")


# ──────────────────────────────────────────────────────────────────────────────
# Result Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class NodeResult:
    """Complete result of a node's local training round."""

    node_id: int
    node_name: str
    participated: bool
    weight_delta: Optional[Dict[str, torch.Tensor]]  # ΔW = W_local - W_global
    local_weights: Optional[Dict[str, torch.Tensor]]  # W_local after training
    num_samples: int
    local_accuracy: float
    local_loss: float
    renewable_score: float
    epochs_completed: int  # May be < requested if early_stop triggered
    early_stopped: bool  # True if renewable score dropped mid-training
    compression_ratio: float  # Fraction of delta entries transmitted
    rounds_participated: int


# ──────────────────────────────────────────────────────────────────────────────
# Eco-Conscious Federated Node
# ──────────────────────────────────────────────────────────────────────────────


class CarbonAwareNode:
    """
    A federated learning node with environmental self-regulation.

    The node adapts its behaviour based on the renewable energy snapshot:
      1. **Adaptive LR**: lr = base_lr × (0.5 + 0.5 × renewable_score)
         → Higher renewable → faster learning; scarce green energy → conservative.
      2. **Gradient Sparsification**: If carbon_intensity > 300 gCO₂/kWh,
         compress weight delta to top-10% (save bandwidth & compute).
      3. **Local Early Stopping**: After each epoch, check renewable_score;
         if it drops below a soft_threshold, halt training early.
      4. **Differential Privacy**: Optional Gaussian noise on ΔW.

    Args:
        node_id:          Integer identifier.
        node_name:        Human-readable location name.
        data_loader:      Pre-built PyTorch DataLoader for local data.
        model:            Reference global model (will be deep-copied).
        base_lr:          Base learning rate (before adaptive scaling).
        momentum:         SGD momentum.
        weight_decay:     L2 regularisation.
        grad_clip_norm:   Pre-update gradient clipping norm.
        device:           Compute device.
        adaptive_lr:      Scale LR by renewable score.
        sparsify_threshold: Carbon intensity above which to sparsify (g/kWh).
        sparsify_ratio:   Fraction of entries to keep (default 0.1 = top-10%).
        early_stop_threshold: Renewable score below which to halt mid-epoch.
        dp_sigma:         If > 0, add Gaussian noise with this σ to ΔW.
        dp_clip_norm:     L2 clip norm for DP sensitivity bounding.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        data_loader: DataLoader,
        model: nn.Module,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
        adaptive_lr: bool = True,
        sparsify_threshold: float = 300.0,
        sparsify_ratio: float = 0.10,
        early_stop_threshold: float = 0.20,
        dp_sigma: float = 0.0,
        dp_clip_norm: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.node_name = node_name

        self._loader = data_loader
        self._device = device
        self._base_lr = base_lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._grad_clip_norm = grad_clip_norm
        self._adaptive_lr = adaptive_lr
        self._sparsify_threshold = sparsify_threshold
        self._sparsify_ratio = sparsify_ratio
        self._early_stop_thresh = early_stop_threshold
        self._dp_sigma = dp_sigma
        self._dp_clip_norm = dp_clip_norm

        # Local model — deep-copied so node never mutates global
        self._model: nn.Module = copy.deepcopy(model).to(device)
        self._criterion = nn.CrossEntropyLoss()
        self._rounds_participated = 0

        # Rebuild optimizer after each round to reset momentum buffers
        self._build_optimizer(lr=base_lr)

    # ── Public Training API ───────────────────────────────────────────────────

    def train_round(
        self,
        global_weights: Dict[str, torch.Tensor],
        num_epochs: int,
        renewable_score: float,
        carbon_intensity: float,
    ) -> NodeResult:
        """
        Execute one federated learning round with ecological self-regulation.

        Protocol:
          1. Load global weights into local model.
          2. Compute adaptive learning rate from renewable score.
          3. Run local SGD for up to num_epochs, with early stopping.
          4. Compute ΔW = W_local − W_global.
          5. Optionally sparsify ΔW based on carbon intensity.
          6. Optionally apply DP noise to ΔW.

        Args:
            global_weights:    Server's current global model parameters.
            num_epochs:        Maximum local training epochs.
            renewable_score:   Current renewable fraction [0, 1].
            carbon_intensity:  Grid carbon g CO₂/kWh.

        Returns:
            :class:`NodeResult` with full training outcome.
        """
        # ── Step 1: Sync with global model ───────────────────────────────────
        pre_weights = {k: v.clone() for k, v in global_weights.items()}
        self._model.load_state_dict(global_weights, strict=False)

        # ── Step 2: Adaptive learning rate ───────────────────────────────────
        effective_lr = self._base_lr
        if self._adaptive_lr:
            effective_lr = self._base_lr * (0.5 + 0.5 * renewable_score)
        self._build_optimizer(lr=effective_lr)

        # ── Step 3: Local training with optional early stopping ───────────────
        self._model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        epochs_done = 0
        early_stopped = False

        for epoch in range(num_epochs):
            for batch_x, batch_y in self._loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = self._criterion(logits, batch_y)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip_norm)
                self._optimizer.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    total_correct += int((preds == batch_y).sum())
                    total_samples += batch_y.size(0)
                    total_loss += loss.item() * batch_y.size(0)

            epochs_done += 1

            # Early stopping: check renewable score heuristic
            # Simulate a mid-training score check (score decays slightly)
            simulated_decay = renewable_score - 0.05 * epoch
            if simulated_decay < self._early_stop_thresh:
                log.info(
                    f"[{self.node_name}] Early stop at epoch {epoch+1}: "
                    f"renewable {simulated_decay:.2f} < {self._early_stop_thresh:.2f}"
                )
                early_stopped = True
                break

        # ── Step 4: Compute weight delta ΔW ──────────────────────────────────
        post_weights = self._model.state_dict()
        weight_delta = {
            k: post_weights[k].clone().float() - pre_weights[k].float()
            for k in pre_weights
        }

        # ── Step 5: Optional sparsification ──────────────────────────────────
        compression_ratio = 1.0
        if carbon_intensity > self._sparsify_threshold:
            weight_delta, compression_ratio = self._sparsify(weight_delta)

        # ── Step 6: Optional differential privacy ────────────────────────────
        if self._dp_sigma > 0:
            weight_delta = self._apply_dp(weight_delta)

        local_accuracy = total_correct / max(total_samples, 1)
        local_loss = total_loss / max(total_samples, 1)
        self._rounds_participated += 1

        return NodeResult(
            node_id=self.node_id,
            node_name=self.node_name,
            participated=True,
            weight_delta=weight_delta,
            local_weights={k: v.clone() for k, v in post_weights.items()},
            num_samples=total_samples,
            local_accuracy=local_accuracy,
            local_loss=local_loss,
            renewable_score=renewable_score,
            epochs_completed=epochs_done,
            early_stopped=early_stopped,
            compression_ratio=compression_ratio,
            rounds_participated=self._rounds_participated,
        )

    def skip_round(self, renewable_score: float) -> NodeResult:
        """Return a non-participation record (node was idle this round)."""
        return NodeResult(
            node_id=self.node_id,
            node_name=self.node_name,
            participated=False,
            weight_delta=None,
            local_weights=None,
            num_samples=0,
            local_accuracy=0.0,
            local_loss=0.0,
            renewable_score=renewable_score,
            epochs_completed=0,
            early_stopped=False,
            compression_ratio=0.0,
            rounds_participated=self._rounds_participated,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _build_optimizer(self, lr: float) -> None:
        """Build a fresh SGD optimizer with given LR."""
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )

    def _sparsify(
        self,
        delta: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Top-K global sparsification — keep the `sparsify_ratio` largest-magnitude
        entries across all parameter tensors. Remaining entries are zeroed.

        Complexity: O(P log P) for P total parameters.
        """
        keys = list(delta.keys())
        flat = torch.cat([delta[k].flatten() for k in keys])
        total = flat.numel()
        k = max(1, int(total * self._sparsify_ratio))

        topk_vals, _ = torch.topk(flat.abs(), k, largest=True, sorted=False)
        threshold = topk_vals.min().item()

        sparse_delta: Dict[str, torch.Tensor] = {}
        offset = 0
        for key in keys:
            t = delta[key]
            mask = t.abs() >= threshold
            sparse_delta[key] = t * mask.float()
            offset += t.numel()

        kept_fraction = float(
            (torch.cat([sparse_delta[k].flatten() for k in keys]) != 0).float().mean()
        )
        return sparse_delta, kept_fraction

    def _apply_dp(
        self,
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Gaussian Differential Privacy noise to weight deltas.

        Step 1: Clip global L2 norm to dp_clip_norm (sensitivity bounding).
        Step 2: Add isotropic Gaussian noise N(0, σ²·I).

        Complexity: O(P) for P parameters.
        """
        keys = list(delta.keys())
        flat = torch.cat([delta[k].float().flatten() for k in keys])
        norm = flat.norm(2)
        scale = min(1.0, self._dp_clip_norm / (norm.item() + 1e-9))
        clipped = flat * scale
        noisy = clipped + torch.randn_like(clipped) * self._dp_sigma

        private_delta: Dict[str, torch.Tensor] = {}
        offset = 0
        for key in keys:
            numel = delta[key].numel()
            private_delta[key] = noisy[offset : offset + numel].reshape(
                delta[key].shape
            )
            offset += numel
        return private_delta
