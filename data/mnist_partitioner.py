"""
mnist_partitioner.py — Non-IID Data Engineering
════════════════════════════════════════════════
Realistic federated data heterogeneity using Dirichlet distribution.

In the real world, federated nodes accumulate data with imbalanced class
distributions reflecting their local environment.  Hospitals see different
disease distributions; phones carry different languages.

This module models three types of heterogeneity:
  1. Label Distribution Skew: Dirichlet(α) assigns class proportions
     (α → 0: extreme specialisation; α → ∞: IID uniform)
  2. Class Subset Restriction: Nodes may only hold a subset of classes
     (e.g., Melbourne nodes only see digits 0–4)
  3. Quantity Skew: Nodes have different total dataset sizes

The combination creates a realistic federated environment where the carbon-
aware scheduling must navigate not just energy constraints but also data
coverage gaps — making the aggregation problem genuinely challenging.

Reference: Hsieh et al., "Quagmire of Heterogeneous FL", NeurIPS 2020.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

log = logging.getLogger("climate_fed.data")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Partitioner
# ──────────────────────────────────────────────────────────────────────────────


class MNISTPartitioner:
    """
    Dirichlet-based non-IID MNIST partitioner with class subset support.

    Partitioning Protocol:
      1. For each allowed class c for node i, compute sample count
         proportional to Dirichlet weight.
      2. Respect `num_samples` cap per node (quantity skew).
      3. Wrap in DataLoader with appropriate batch size.

    Args:
        dataset:        Full MNIST training dataset.
        node_configs:   List of dicts with keys:
                        'node_id', 'name', 'num_samples', 'data_classes'.
        alpha:          Dirichlet concentration parameter (0.5 for moderate skew).
        batch_size:     DataLoader batch size.
        num_workers:    DataLoader worker processes.
        seed:           RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        node_configs: List[Dict],
        alpha: float = 0.5,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        self._dataset = dataset
        self._node_configs = node_configs
        self._alpha = alpha
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._rng = np.random.default_rng(seed)

        # Cache labels as numpy array for fast indexing
        if hasattr(dataset, "targets"):
            self._labels = np.array(dataset.targets)
        else:
            self._labels = np.array([y for _, y in dataset])

        self._num_classes = int(self._labels.max()) + 1

    def partition(self) -> Dict[int, DataLoader]:
        """
        Create per-node DataLoaders with heterogeneous label distributions.

        Returns:
            Dict mapping node_id → :class:`DataLoader`.

        Complexity: O(N × C × D) for N nodes, C classes, D dataset size.
        """
        loaders: Dict[int, DataLoader] = {}

        for cfg in self._node_configs:
            node_id = cfg.get("node_id", cfg.get("id"))
            name = cfg.get("name", f"Node-{node_id}")
            max_n = cfg.get("num_samples", 2000)
            allowed_classes = list(cfg.get("data_classes", range(self._num_classes)))

            indices = self._sample_indices(allowed_classes, max_n)

            if len(indices) == 0:
                log.warning(f"[{name}] No samples found for classes {allowed_classes}")
                continue

            subset = Subset(self._dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                pin_memory=False,
                drop_last=False,
            )
            loaders[node_id] = loader
            class_counts = np.bincount(
                self._labels[indices], minlength=self._num_classes
            )
            log.info(
                f"[{name}] {len(indices)} samples | "
                f"Classes: {allowed_classes} | Distribution: {class_counts.tolist()}"
            )

        return loaders

    def _sample_indices(self, allowed_classes: List[int], max_n: int) -> List[int]:
        """
        Sample indices using Dirichlet distribution over allowed classes.

        Dirichlet proportions are drawn once per node, then indices are
        sampled class-by-class to match target proportions up to max_n.

        Complexity: O(C + D) for C allowed classes, D dataset size.
        """
        # Dirichlet sample over allowed classes
        concentration = np.full(len(allowed_classes), self._alpha)
        proportions = self._rng.dirichlet(concentration)

        all_indices: List[int] = []
        for cls, prop in zip(allowed_classes, proportions):
            cls_indices = np.where(self._labels == cls)[0]
            n_cls = min(len(cls_indices), int(max_n * prop) + 1)
            if n_cls == 0:
                continue
            sampled = self._rng.choice(cls_indices, size=n_cls, replace=False)
            all_indices.extend(sampled.tolist())

        # Cap at max_n
        if len(all_indices) > max_n:
            shuffle_idx = self._rng.permutation(len(all_indices))[:max_n]
            all_indices = [all_indices[i] for i in shuffle_idx]

        return all_indices
