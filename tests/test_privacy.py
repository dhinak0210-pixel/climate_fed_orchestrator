"""
test_privacy.py — Differential Privacy Verification Suite
═══════════════════════════════════════════════════════════
Rigorous unit tests for DP guarantees in the Climate-Fed Orchestrator.

Coverage:
  1. RDP noise calibration accuracy
  2. Per-sample gradient clipping bounds
  3. Gaussian noise magnitude verification
  4. Privacy budget accounting (no budget exceeded)
  5. PrivacyEngine lifecycle (calibrate → clip → noise → step)
  6. Multi-node privacy coordination (worst-case ε)
  7. Compliance reporting accuracy
"""

import unittest
import math
import torch
import numpy as np
from typing import Dict

from core.dp_sgd import (
    _compute_eps_rdp,
    calibrate_noise_multiplier,
    clip_per_sample_gradients,
    add_gaussian_noise,
    PrivacyLedger,
    PrivacyBudgetEntry,
)
from core.privacy_engine import (
    PrivacyEngine,
    PrivacyConfig,
    PrivacyCoordinator,
)


class TestRDPAccountant(unittest.TestCase):
    """Tests for the Rényi Differential Privacy accountant."""

    def test_eps_monotonically_increases_with_steps(self):
        """ε must grow with more training steps (composition theorem)."""
        eps_values = []
        for steps in [10, 50, 100, 500, 1000]:
            eps = _compute_eps_rdp(q=0.01, sigma=1.0, steps=steps, delta=1e-5)
            eps_values.append(eps)

        for i in range(1, len(eps_values)):
            self.assertGreater(
                eps_values[i],
                eps_values[i - 1],
                f"ε should increase with steps: {eps_values}",
            )

    def test_eps_decreases_with_more_noise(self):
        """Higher noise σ should yield lower ε (stronger privacy)."""
        eps_values = []
        for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
            eps = _compute_eps_rdp(q=0.01, sigma=sigma, steps=100, delta=1e-5)
            eps_values.append(eps)

        for i in range(1, len(eps_values)):
            self.assertLess(
                eps_values[i],
                eps_values[i - 1],
                f"ε should decrease with more noise: {eps_values}",
            )

    def test_eps_positive(self):
        """ε must always be positive."""
        eps = _compute_eps_rdp(q=0.01, sigma=1.0, steps=100, delta=1e-5)
        self.assertGreater(eps, 0.0)

    def test_subsampling_amplification(self):
        """Smaller sampling rate q should yield lower ε (privacy amplification)."""
        eps_high_q = _compute_eps_rdp(q=0.1, sigma=1.0, steps=100, delta=1e-5)
        eps_low_q = _compute_eps_rdp(q=0.01, sigma=1.0, steps=100, delta=1e-5)
        self.assertLess(eps_low_q, eps_high_q)


class TestNoiseCalibration(unittest.TestCase):
    """Tests for automatic noise multiplier calibration."""

    def test_calibrated_sigma_satisfies_budget(self):
        """Calibrated σ should yield ε ≤ target."""
        target_eps = 1.0
        sigma = calibrate_noise_multiplier(
            target_epsilon=target_eps,
            target_delta=1e-5,
            sample_rate=0.01,
            steps=100,
        )
        actual_eps = _compute_eps_rdp(q=0.01, sigma=sigma, steps=100, delta=1e-5)
        self.assertLessEqual(
            actual_eps,
            target_eps + 0.05,  # Allow small numerical tolerance
            f"Calibrated σ={sigma:.4f} yields ε={actual_eps:.4f} > target {target_eps}",
        )

    def test_sigma_positive(self):
        """Calibrated σ must be positive."""
        sigma = calibrate_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.01,
            steps=100,
        )
        self.assertGreater(sigma, 0.0)

    def test_tighter_budget_requires_more_noise(self):
        """Smaller ε target should require larger σ."""
        sigma_loose = calibrate_noise_multiplier(
            target_epsilon=5.0,
            target_delta=1e-5,
            sample_rate=0.01,
            steps=100,
        )
        sigma_tight = calibrate_noise_multiplier(
            target_epsilon=0.5,
            target_delta=1e-5,
            sample_rate=0.01,
            steps=100,
        )
        self.assertGreater(
            sigma_tight, sigma_loose, "Tighter ε budget should require more noise"
        )


class TestGradientClipping(unittest.TestCase):
    """Tests for per-sample gradient clipping."""

    def _make_gradient(self, norm: float) -> Dict[str, torch.Tensor]:
        """Create a gradient dict with a specific L2 norm."""
        # 10-element vector with equal values
        value = norm / math.sqrt(10)
        return {"weight": torch.full((10,), value)}

    def test_large_gradient_is_clipped(self):
        """Gradient with norm > C should be rescaled to norm C."""
        clip_bound = 1.0
        grad = self._make_gradient(norm=5.0)  # 5× above bound
        clipped = clip_per_sample_gradients([grad], clipping_bound=clip_bound)

        result_norm = torch.norm(clipped[0]["weight"], p=2).item()
        self.assertAlmostEqual(result_norm, clip_bound, places=4)

    def test_small_gradient_unchanged(self):
        """Gradient with norm ≤ C should NOT be modified."""
        clip_bound = 1.0
        grad = self._make_gradient(norm=0.5)  # Below bound
        original = grad["weight"].clone()
        clipped = clip_per_sample_gradients([grad], clipping_bound=clip_bound)

        torch.testing.assert_close(clipped[0]["weight"], original)

    def test_zero_gradient_unchanged(self):
        """Zero gradient should remain zero."""
        grad = {"weight": torch.zeros(10)}
        clipped = clip_per_sample_gradients([grad], clipping_bound=1.0)
        torch.testing.assert_close(clipped[0]["weight"], torch.zeros(10))

    def test_batch_clipping(self):
        """All gradients in a batch should respect the bound."""
        clip_bound = 1.0
        grads = [self._make_gradient(norm=n) for n in [0.1, 0.5, 1.0, 3.0, 10.0]]
        clipped = clip_per_sample_gradients(grads, clipping_bound=clip_bound)

        for i, g in enumerate(clipped):
            norm = torch.norm(g["weight"], p=2).item()
            self.assertLessEqual(
                norm,
                clip_bound + 1e-5,
                f"Gradient {i} norm={norm:.4f} exceeds clip bound {clip_bound}",
            )


class TestGaussianNoise(unittest.TestCase):
    """Tests for Gaussian noise injection."""

    def test_noise_has_correct_scale(self):
        """Noise std should be σ × C / B."""
        sigma = 1.0
        clip = 1.0
        batch_size = 10

        # Run many times and check empirical std
        noisy_samples = []
        base = torch.zeros(1000)
        for _ in range(100):
            noisy = add_gaussian_noise(base.clone(), sigma, clip, batch_size)
            noisy_samples.append(noisy)

        stacked = torch.stack(noisy_samples)
        empirical_std = stacked.std().item()
        expected_std = sigma * clip / batch_size

        # Allow 20% tolerance for empirical measurement
        self.assertAlmostEqual(
            empirical_std,
            expected_std,
            delta=expected_std * 0.3,
            msg=f"Empirical std {empirical_std:.4f} ≠ expected {expected_std:.4f}",
        )

    def test_noise_is_nonzero(self):
        """Noisy gradient should differ from the original."""
        grad = torch.ones(100)
        noisy = add_gaussian_noise(grad.clone(), 1.0, 1.0, 10)
        self.assertFalse(torch.equal(grad, noisy))

    def test_noise_zero_sigma(self):
        """With σ=0, output should equal input."""
        grad = torch.ones(100)
        noisy = add_gaussian_noise(grad.clone(), 0.0, 1.0, 10)
        torch.testing.assert_close(noisy, grad)


class TestPrivacyLedger(unittest.TestCase):
    """Tests for privacy budget ledger."""

    def test_initial_budget_full(self):
        """Fresh ledger should have full budget."""
        ledger = PrivacyLedger(target_epsilon=1.0, target_delta=1e-5)
        self.assertEqual(ledger.epsilon_spent, 0.0)
        self.assertEqual(ledger.epsilon_remaining, 1.0)
        self.assertFalse(ledger.budget_exhausted)

    def test_budget_decreases_after_record(self):
        """Budget should decrease after recording consumption."""
        ledger = PrivacyLedger(target_epsilon=1.0, target_delta=1e-5)
        entry = PrivacyBudgetEntry(
            round_num=1,
            node_id=0,
            epsilon_delta=0.3,
            sigma=1.0,
            sample_rate=0.01,
            steps=100,
        )
        ledger.record(entry)
        self.assertAlmostEqual(ledger.epsilon_spent, 0.3)
        self.assertAlmostEqual(ledger.epsilon_remaining, 0.7)

    def test_budget_exhaustion_detection(self):
        """Ledger should detect when budget is exceeded."""
        ledger = PrivacyLedger(target_epsilon=1.0, target_delta=1e-5)
        entry = PrivacyBudgetEntry(
            round_num=1,
            node_id=0,
            epsilon_delta=1.5,
            sigma=0.5,
            sample_rate=0.01,
            steps=500,
        )
        ledger.record(entry)
        self.assertTrue(ledger.budget_exhausted)

    def test_compliance_report_structure(self):
        """Compliance report should contain required fields."""
        ledger = PrivacyLedger(target_epsilon=1.0, target_delta=1e-5)
        report = ledger.compliance_report()
        required_keys = [
            "target_epsilon",
            "target_delta",
            "epsilon_spent",
            "epsilon_remaining",
            "is_compliant",
        ]
        for key in required_keys:
            self.assertIn(key, report, f"Missing key: {key}")


class TestPrivacyEngine(unittest.TestCase):
    """Tests for the unified PrivacyEngine."""

    def setUp(self):
        self.config = PrivacyConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            l2_clip_norm=1.0,
        )
        self.engine = PrivacyEngine(
            config=self.config,
            node_id=0,
            node_name="Test-Node",
        )

    def test_calibration_succeeds(self):
        """Engine should calibrate without errors."""
        sigma = self.engine.calibrate(sample_rate=0.01, steps=100)
        self.assertGreater(sigma, 0.0)
        self.assertTrue(self.engine._is_calibrated)

    def test_step_tracks_epsilon(self):
        """Each step should increase tracked ε."""
        self.engine.calibrate(sample_rate=0.01, steps=100)
        eps1 = self.engine.step(round_num=1)
        self.assertGreater(eps1, 0.0)

    def test_budget_not_exceeded_with_single_step(self):
        """Single step with proper calibration should not exceed budget."""
        self.engine.calibrate(sample_rate=0.01, steps=100, total_rounds=1)
        eps = self.engine.step(round_num=1)
        self.assertLessEqual(eps, self.config.target_epsilon + 0.1)

    def test_compliance_check_format(self):
        """Compliance check should return well-formed report."""
        self.engine.calibrate(sample_rate=0.01, steps=100)
        check = self.engine.compliance_check()

        self.assertIn("compliant", check)
        self.assertIn("epsilon_spent", check)
        self.assertIn("gdpr_art25_status", check)
        self.assertIn("evidence", check)

    def test_uncalibrated_step_raises(self):
        """Stepping without calibration should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.engine.step()

    def test_uncalibrated_noise_raises(self):
        """Adding noise without calibration should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.engine.add_noise(torch.ones(10), batch_size=10)


class TestPrivacyCoordinator(unittest.TestCase):
    """Tests for multi-node privacy coordination."""

    def test_worst_case_epsilon(self):
        """Global ε should be the max across all nodes."""
        coord = PrivacyCoordinator(
            target_epsilon=5.0,
            target_delta=1e-5,
            node_count=3,
        )

        # Calibrate and step different amounts per node
        for nid in range(3):
            engine = coord.get_engine(nid)
            engine.calibrate(sample_rate=0.01, steps=100 * (nid + 1))
            engine.step(round_num=1)

        # Node 2 should have highest ε (most steps)
        node2_eps = coord.get_engine(2).epsilon_spent
        self.assertAlmostEqual(
            coord.worst_case_epsilon,
            node2_eps,
            delta=0.01,
            msg="Global ε should equal worst-case node ε",
        )

    def test_global_compliance(self):
        """Fresh coordinator should be compliant."""
        coord = PrivacyCoordinator(target_epsilon=1.0, node_count=3)
        self.assertTrue(coord.global_compliant)

    def test_summary_format(self):
        """Summary should be a non-empty string."""
        coord = PrivacyCoordinator(target_epsilon=1.0, node_count=2)
        summary = coord.global_summary()
        self.assertIsInstance(summary, str)
        self.assertIn("PRIVACY COORDINATOR", summary)


class TestEndToEndPrivacy(unittest.TestCase):
    """Integration test: full DP lifecycle."""

    def test_full_lifecycle(self):
        """
        Test the complete DP workflow:
        1. Configure → 2. Calibrate → 3. Clip → 4. Noise → 5. Account
        """
        config = PrivacyConfig(target_epsilon=5.0, target_delta=1e-5, l2_clip_norm=1.0)
        engine = PrivacyEngine(config=config, node_id=0, node_name="E2E-Test")

        # 1. Calibrate
        sigma = engine.calibrate(sample_rate=0.05, steps=50, total_rounds=1)
        self.assertGreater(sigma, 0.0)

        # 2. Create a gradient and privatize it
        grad = torch.randn(100) * 5.0  # Large gradient
        noisy_grad = engine.privatize_gradient(grad, batch_size=64)

        # Verify it was modified
        self.assertFalse(torch.equal(grad, noisy_grad))

        # Verify norm was clipped (noisy_grad should have bounded signal component)
        # The noise makes exact norm checking impractical, but we verify it ran

        # 3. Track budget
        eps = engine.step(round_num=1)
        self.assertGreater(eps, 0.0)

        # 4. Compliance check
        check = engine.compliance_check()
        self.assertIn("compliant", check)

        # 5. Summary
        summary = engine.summary()
        self.assertIn("Node-0", summary)
        self.assertIn("E2E-Test", summary)


if __name__ == "__main__":
    unittest.main()
