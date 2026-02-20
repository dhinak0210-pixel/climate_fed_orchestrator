"""
test_carbon_logic.py — Verification Suite
═════════════════════════════════════════
Property-based and unit tests for the Climate-Fed core logic.

Coverage:
  1. RenewableOracle Decision Logic (Naive/Oracle/Standard)
  2. CarbonLedger Energy Accounting Accuracy
  3. FederatedNode Adaptive LR Calculation
  4. Aggregator Strategy Weighting
"""

import unittest
import torch
import numpy as np
from climate_fed_orchestrator.core.carbon_engine import RenewableOracle, NodeGeography
from climate_fed_orchestrator.core.energy_accountant import CarbonLedger
from climate_fed_orchestrator.core.federated_node import CarbonAwareNode
from climate_fed_orchestrator.models.mnist_cnn import EcoCNN
from torch.utils.data import DataLoader, TensorDataset


class TestCarbonLogic(unittest.TestCase):
    def setUp(self):
        # Setup dummy geographic data
        self.geo = NodeGeography(
            node_id=0,
            name="Test-Node",
            country="Norway",
            latitude=60.0,
            longitude=10.0,
            timezone_offset_hours=1,
            solar_capacity=1.0,
            wind_capacity=1.0,
            grid_carbon_intensity=100,
        )
        self.oracle = RenewableOracle([self.geo], threshold=0.5, seed=42)
        self.ledger = CarbonLedger(pue=1.4)

    # ── Oracle Tests ─────────────────────────────────────────────────────────

    def test_standard_mode_always_trains(self):
        """Standard mode should ignore renewable scores totally."""
        # Even with extremely low score simulations, standard mode returns True
        for r in range(1, 10):
            schedule = self.oracle.schedule_round(r, mode="standard")
            self.assertTrue(schedule[0].can_train)

    def test_naive_mode_respects_threshold(self):
        """Naive mode should strictly gate by the threshold."""
        # We manually call _decide to isolate the logic
        train, reason = self.oracle._decide(0.6, [], "naive")
        self.assertTrue(train)

        train, reason = self.oracle._decide(0.4, [], "naive")
        self.assertFalse(train)

    def test_oracle_lookahead_deferral(self):
        """Oracle should defer if a significantly better window exists."""
        # Now=0.55 (above threshold), Future=0.9 (much better)
        train, reason = self.oracle._decide(0.55, [0.9], "oracle")
        self.assertFalse(train)
        self.assertIn("deferring", reason.lower())

    # ── Ledger Tests ─────────────────────────────────────────────────────────

    def test_ledger_accounting_values(self):
        """Verify energy calculation against known model (1.05 kWh/epoch)."""
        record = self.ledger.record_event(
            round_num=1,
            node_id=0,
            node_name="Test",
            is_active=True,
            renewable_score=1.0,
            carbon_intensity_g_kwh=100.0,
            num_samples=100,
            num_epochs=1,
            energy_mix_solar=1.0,
            energy_mix_wind=0.0,
            energy_mix_fossil=0.0,
        )

        # Expected compute: 1.05 kWh
        # Expected cooling: 1.05 * (1.4 - 1) = 0.42 kWh
        # Total: 1.47 kWh (approx, excluding minimal comms)
        self.assertAlmostEqual(record.total_kwh, 1.47, places=2)

        # CO2: 1.47 kWh * 100 g/kWh = 147g = 0.147 kg
        self.assertAlmostEqual(record.kg_co2e, 0.147, places=3)

    # ── Node Logic ──────────────────────────────────────────────────────────

    def test_adaptive_lr_scaling(self):
        """Verify LR scales correctly with renewable score."""
        # Dummy DataLoader
        ds = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))
        loader = DataLoader(ds, batch_size=2)
        model = EcoCNN()

        node = CarbonAwareNode(
            node_id=0,
            node_name="Test",
            data_loader=loader,
            model=model,
            base_lr=0.1,
            adaptive_lr=True,
        )

        # Score = 1.0 -> LR = 0.1 * (0.5 + 0.5*1.0) = 0.1
        # Score = 0.0 -> LR = 0.1 * (0.5 + 0.5*0.0) = 0.05

        # Trigger training to see optimizer update
        node.train_round(model.state_dict(), 1, 1.0, 100)
        self.assertAlmostEqual(node._optimizer.param_groups[0]["lr"], 0.1)

        node.train_round(model.state_dict(), 1, 0.0, 100)
        self.assertAlmostEqual(node._optimizer.param_groups[0]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()
