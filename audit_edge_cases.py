import sys
import os
from pathlib import Path
import torch
import yaml
import json

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from core.carbon_engine import NodeGeography, RenewableOracle
from core.privacy_engine import PrivacyEngine, PrivacyConfig


def test_edge_cases():
    results = {}

    # Case 1: Extreme Low Energy (Zero Capacity)
    geo = NodeGeography(
        node_id=0,
        name="Static",
        country="None",
        latitude=0,
        longitude=0,
        timezone_offset_hours=0,
        solar_capacity=0.0,
        wind_capacity=0.0,
        grid_carbon_intensity=500,
    )
    oracle = RenewableOracle([geo], threshold=0.5)
    snap = oracle.snapshot(0, 1, mode="naive")
    results["zero_capacity_train"] = snap.can_train
    print(f"Zero capacity can train: {snap.can_train}")

    # Case 2: High Threshold deferral
    oracle_high = RenewableOracle([geo], threshold=0.99)
    snap_high = oracle_high.snapshot(0, 1, mode="naive")
    results["high_threshold_train"] = snap_high.can_train
    print(f"High threshold (0.99) can train: {snap_high.can_train}")

    # Case 3: Privacy Exhaustion
    config = PrivacyConfig(target_epsilon=0.1, target_delta=1e-5)
    engine = PrivacyEngine(config)
    engine.calibrate(sample_rate=0.1, steps=100)  # This should spend a lot
    eps = engine.step()
    results["privacy_budget_exhausted"] = engine.budget_exhausted
    results["epsilon_spent"] = eps
    print(f"Privacy budget exhausted: {engine.budget_exhausted} (spent {eps:.4f})")

    # Case 4: Non-IID Skewed Partitioning
    from data.mnist_partitioner import MNISTPartitioner
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)

    node_configs = [
        {"id": 0, "name": "Skewed", "num_samples": 500, "data_classes": [0]}
    ]
    partitioner = MNISTPartitioner(ds, node_configs, alpha=0.01)
    loaders = partitioner.partition()
    loader = loaders[0]
    labels = []
    for _, y in loader:
        labels.extend(y.tolist())
    unique_labels = set(labels)
    results["skewed_partition_classes"] = list(unique_labels)
    print(f"Skewed partition classes: {unique_labels}")

    with open("edge_case_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    test_edge_cases()
