"""Shared pytest fixtures for climate_fed_orchestrator tests."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from core.carbon_engine import NodeGeography, RenewableOracle
from core.energy_accountant import CarbonLedger
from models.mnist_cnn import EcoCNN


@pytest.fixture
def node_geo():
    return NodeGeography(
        node_id=0, name="Oslo", country="Norway",
        latitude=59.9, longitude=10.7, timezone_offset_hours=1,
        solar_capacity=0.6, wind_capacity=0.8, grid_carbon_intensity=28,
    )


@pytest.fixture
def three_nodes():
    return [
        NodeGeography(node_id=0, name="Oslo", country="Norway",
                      latitude=59.9, longitude=10.7, timezone_offset_hours=1,
                      solar_capacity=0.6, wind_capacity=0.8, grid_carbon_intensity=28),
        NodeGeography(node_id=1, name="Melbourne", country="Australia",
                      latitude=-37.8, longitude=144.9, timezone_offset_hours=10,
                      solar_capacity=0.9, wind_capacity=0.4, grid_carbon_intensity=510),
        NodeGeography(node_id=2, name="San Jose", country="Costa Rica",
                      latitude=9.9, longitude=-84.1, timezone_offset_hours=-6,
                      solar_capacity=0.85, wind_capacity=0.5, grid_carbon_intensity=45),
    ]


@pytest.fixture
def oracle(three_nodes):
    return RenewableOracle(three_nodes, threshold=0.6, seed=42)


@pytest.fixture
def ledger():
    return CarbonLedger(pue=1.4)


@pytest.fixture
def eco_cnn():
    return EcoCNN(input_channels=1, num_classes=10)


@pytest.fixture
def dummy_loader():
    ds = TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,)))
    return DataLoader(ds, batch_size=4)
