"""
experiment_config.py — Pydantic-validated Experiment Configuration
═══════════════════════════════════════════════════════════════════
Replaces the raw YAML dict usage in main.py with typed, validated config
objects. Validation errors now fail fast with clear messages instead of
cryptic KeyError / TypeError deep inside the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────


class NodeConfig(BaseModel):
    id: int
    name: str
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    zone: str = "UNKNOWN"
    solar_capacity: float = Field(default=0.3, ge=0.0, le=1.0)
    wind_capacity: float = Field(default=0.4, ge=0.0, le=1.0)
    compute_tflops: float = Field(default=1.0, gt=0.0)


class TrainingConfig(BaseModel):
    base_lr: float = Field(default=0.01, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    local_epochs: int = Field(default=1, ge=1, le=20)
    batch_size: int = Field(default=64, ge=8, le=512)
    num_workers: int = Field(default=0, ge=0, le=8)
    adaptive_lr: bool = True


class CarbonConfig(BaseModel):
    renewable_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    pue: float = Field(default=1.4, ge=1.0, le=3.0)
    oracle_lookahead_rounds: int = Field(default=3, ge=1, le=10)
    solar_noise_std: float = Field(default=0.08, ge=0.0)
    wind_noise_std: float = Field(default=0.10, ge=0.0)


class AggregationConfig(BaseModel):
    strategy: str = Field(default="renewable_weighted")
    trimmed_fraction: float = Field(default=0.10, ge=0.0, le=0.49)
    use_server_momentum: bool = True
    server_momentum: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"fedavg", "renewable_weighted", "trimmed_mean"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{v}'")
        return v


class ModelConfig(BaseModel):
    input_channels: int = Field(default=1, ge=1)
    num_classes: int = Field(default=10, ge=2)
    conv1_filters: int = Field(default=16, ge=4)
    conv2_filters: int = Field(default=32, ge=4)
    fc_hidden: int = Field(default=64, ge=16)
    dropout: float = Field(default=0.25, ge=0.0, le=0.9)


class ExperimentMeta(BaseModel):
    num_rounds: int = Field(default=10, ge=1, le=200)
    seed: int = Field(default=42)
    name: str = "climate_fed_experiment"


class DataConfig(BaseModel):
    data_dir: str = "./data"
    alpha: float = Field(default=0.5, gt=0.0)  # Dirichlet concentration


# ─────────────────────────────────────────────────────────────────────────────
# Root config model
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    """
    Fully-typed, validated experiment configuration.

    Usage:
        cfg = ExperimentConfig.from_yaml("config/simulation_params.yaml")
        cfg.training.base_lr   # typed float, validated
    """

    experiment: ExperimentMeta = Field(default_factory=ExperimentMeta)
    nodes: List[NodeConfig]
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    carbon: CarbonConfig = Field(default_factory=CarbonConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @model_validator(mode="after")
    def validate_nodes_unique_ids(self) -> "ExperimentConfig":
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Node IDs must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        try:
            return cls(**raw)
        except Exception as e:
            raise ValueError(f"Invalid config at {path}:\n{e}") from e

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Load and validate configuration from a raw dict."""
        return cls(**d)
