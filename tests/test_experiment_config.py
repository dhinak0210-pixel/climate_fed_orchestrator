"""Unit tests for ExperimentConfig Pydantic validation."""
import pytest
from core.experiment_config import (
    ExperimentConfig, NodeConfig, TrainingConfig,
    CarbonConfig, AggregationConfig,
)

MINIMAL_CONFIG = {
    "nodes": [
        {"id": 0, "name": "Oslo", "lat": 59.9, "lon": 10.7, "zone": "NO"},
        {"id": 1, "name": "Melbourne", "lat": -37.8, "lon": 144.9, "zone": "AU-VIC"},
    ]
}


def test_minimal_config_loads():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    assert len(cfg.nodes) == 2
    assert cfg.training.base_lr == 0.01
    assert cfg.carbon.renewable_threshold == 0.6


def test_node_id_uniqueness_enforced():
    bad = {"nodes": [
        {"id": 0, "name": "A", "lat": 0.0, "lon": 0.0},
        {"id": 0, "name": "B", "lat": 1.0, "lon": 1.0},
    ]}
    with pytest.raises(Exception, match="unique"):
        ExperimentConfig(**bad)


def test_invalid_lat_rejected():
    with pytest.raises(Exception):
        NodeConfig(id=0, name="Bad", lat=999.0, lon=0.0)


def test_invalid_strategy_rejected():
    with pytest.raises(Exception, match="strategy"):
        AggregationConfig(strategy="unknown_strategy")


def test_rounds_override():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    cfg.experiment.num_rounds = 20
    assert cfg.experiment.num_rounds == 20


def test_training_lr_bounds():
    with pytest.raises(Exception):
        TrainingConfig(base_lr=-0.01)


def test_carbon_threshold_bounds():
    with pytest.raises(Exception):
        CarbonConfig(renewable_threshold=1.5)


def test_valid_strategies():
    for s in ["fedavg", "renewable_weighted", "trimmed_mean"]:
        agg = AggregationConfig(strategy=s)
        assert agg.strategy == s
