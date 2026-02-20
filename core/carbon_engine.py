"""
carbon_engine.py — The Renewable Oracle
═══════════════════════════════════════
Multi-factor renewable energy forecasting engine that transforms geographic,
temporal, and stochastic data into actionable carbon-aware scheduling decisions.

The oracle models three energy sources:
  • Solar: Diurnal sinusoidal cycle with latitude-corrected day-length and cloud noise
  • Wind:  Weibull-distributed generation with temporal autocorrelation
  • Grid:  Historical carbon intensity blended with real-time availability fraction

Each call emits a rich RenewableSnapshot encapsulating energy mix breakdown,
forecast confidence, and look-ahead predictions for oracle scheduling.

Design Philosophy — "Computational Ecology":
  Treat renewable energy as a first-class scheduling constraint, not an
  afterthought.  The oracle owns the moral authority to veto training if
  doing so violates the node's carbon conscience.

Complexity: O(N × L) for N nodes × L look-ahead rounds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnergyMix:
    """Fractional breakdown of generation sources at a simulated instant."""

    solar_fraction: float  # 0–1, fraction of load from solar
    wind_fraction: float  # 0–1, fraction of load from wind
    hydro_fraction: float  # 0–1 (derived: max 0 for non-hydro nodes)
    fossil_fraction: float  # 1 - (solar + wind + hydro)

    @property
    def renewable_fraction(self) -> float:
        return self.solar_fraction + self.wind_fraction + self.hydro_fraction


@dataclass
class RenewableSnapshot:
    """
    Complete renewable energy characterisation for a node at a given round.

    Attributes:
        node_id:            Integer node identifier.
        round_num:          FL communication round (1-indexed).
        renewable_score:    Composite score in [0, 1] used for scheduling.
        carbon_intensity:   Grid carbon intensity g CO₂/kWh.
        energy_mix:         EnergyMix breakdown.
        forecast_confidence: Sigma of Gaussian uncertainty in renewable_score.
        can_train:          Whether the oracle permits training this round.
        reason:             Human-readable decision rationale.
        lookahead:          List of predicted renewable_scores for next rounds.
    """

    node_id: int
    round_num: int
    renewable_score: float
    carbon_intensity: float
    energy_mix: EnergyMix
    forecast_confidence: float
    can_train: bool
    reason: str
    lookahead: List[float] = field(default_factory=list)

    @property
    def energy_kwh(self) -> float:
        """Simulated energy consumed if this node trains one local epoch (kWh-eq)."""
        return 1.05  # constant per epoch; refined by CarbonLedger

    def __str__(self) -> str:
        flag = "🌱 TRAIN " if self.can_train else "⛔ IDLE  "
        return (
            f"{flag} (Renewable: {self.renewable_score:.2f} | "
            f"Carbon: {self.carbon_intensity:.0f}g CO₂/kWh | "
            f"Mix: ☀{self.energy_mix.solar_fraction:.0%} "
            f"🌬{self.energy_mix.wind_fraction:.0%} "
            f"🪨{self.energy_mix.fossil_fraction:.0%})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Node Geographic Profile
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NodeGeography:
    """Immutable geographic and grid profile for one federated node."""

    node_id: int
    name: str
    country: str
    latitude: float
    longitude: float
    timezone_offset_hours: float
    solar_capacity: float  # Peak solar production fraction (0–1)
    wind_capacity: float  # Peak wind production fraction (0–1)
    grid_carbon_intensity: float  # Baseline grid carbon g CO₂/kWh
    weibull_shape: float = 2.0  # Weibull k, wind distribution shape
    wind_scale: float = 0.5  # Weibull λ, normalised


# ──────────────────────────────────────────────────────────────────────────────
# Renewable Oracle
# ──────────────────────────────────────────────────────────────────────────────


class RenewableOracle:
    """
    Multi-factor renewable energy forecasting system.

    Simulates realistic energy availability using:
      1. Diurnal solar cycles with latitude-corrected day-length
      2. Stochastic wind via Weibull distribution with temporal autocorrelation
      3. Grid carbon intensity blended from node baseline and renewable fraction
      4. Seasonal adjustment (±20% amplitude across a 52-round pseudo-year)

    The oracle exposes two scheduling modes:
      • Naive:  Threshold gate — train iff renewable_score ≥ threshold
      • Oracle: Look-ahead over next `lookahead` rounds, schedule greedily

    Args:
        nodes:              List of NodeGeography profiles.
        threshold:          Renewable threshold for naive arm.
        lookahead_rounds:   Number of future rounds to predict for oracle arm.
        seed:               Random seed for reproducible stochastic components.
        solar_noise_std:    Std of cloud-cover Gaussian perturbation.
        wind_noise_std:     Temporal autocorrelation noise on wind.
    """

    def __init__(
        self,
        nodes: List[NodeGeography],
        threshold: float = 0.6,
        lookahead_rounds: int = 3,
        seed: int = 42,
        solar_noise_std: float = 0.08,
        wind_noise_std: float = 0.10,
    ) -> None:
        self._nodes: Dict[int, NodeGeography] = {n.node_id: n for n in nodes}
        self._threshold = threshold
        self._lookahead = lookahead_rounds
        self._solar_noise_std = solar_noise_std
        self._wind_noise_std = wind_noise_std
        self._rng = np.random.default_rng(seed)

        # Persistent wind state — simulates temporal autocorrelation
        self._wind_state: Dict[int, float] = {n.node_id: 0.5 for n in nodes}

        # Cache of computed snapshots for deterministic lookahead
        self._snapshot_cache: Dict[Tuple[int, int], RenewableSnapshot] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(
        self,
        node_id: int,
        round_num: int,
        mode: str = "naive",
    ) -> RenewableSnapshot:
        """
        Compute the renewable energy snapshot for a node at a given round.

        Args:
            node_id:   Target node.
            round_num: Current FL round (1-indexed).
            mode:      "naive" | "oracle" — affects can_train decision.

        Returns:
            :class:`RenewableSnapshot` with full energy characterisation.

        Complexity: O(L) where L = lookahead_rounds for oracle mode.
        """
        cache_key = (node_id, round_num)
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]

        node = self._nodes[node_id]
        solar = self._solar_score(node, round_num)
        wind = self._wind_score(node, round_num)
        seasonal = self._seasonal_factor(round_num)

        # Composite renewable score
        renewable_score = min(
            1.0,
            solar * node.solar_capacity + wind * node.wind_capacity + seasonal * 0.05,
        )

        # Effective carbon intensity scales inversely with renewables
        effective_carbon = node.grid_carbon_intensity * (1.0 - 0.8 * renewable_score)

        # Energy mix breakdown
        solar_frac = min(solar * node.solar_capacity, renewable_score)
        wind_frac = min(wind * node.wind_capacity, renewable_score - solar_frac)
        fossil = max(0.0, 1.0 - solar_frac - wind_frac)
        mix = EnergyMix(
            solar_fraction=round(solar_frac, 3),
            wind_fraction=round(wind_frac, 3),
            hydro_fraction=0.0,
            fossil_fraction=round(fossil, 3),
        )

        # Look-ahead for oracle arm
        lookahead_scores: List[float] = []
        if mode == "oracle":
            for future_round in range(round_num + 1, round_num + self._lookahead + 1):
                f_solar = self._solar_score(node, future_round)
                f_wind = self._wind_score_deterministic(node, future_round)
                f_score = min(
                    1.0,
                    f_solar * node.solar_capacity
                    + f_wind * node.wind_capacity
                    + self._seasonal_factor(future_round) * 0.05,
                )
                lookahead_scores.append(round(f_score, 3))

        # Scheduling decision
        can_train, reason = self._decide(renewable_score, lookahead_scores, mode)

        # Forecast confidence (lower when renewable is near threshold)
        confidence = 1.0 - abs(renewable_score - self._threshold) * 0.5

        snap = RenewableSnapshot(
            node_id=node_id,
            round_num=round_num,
            renewable_score=round(renewable_score, 4),
            carbon_intensity=round(effective_carbon, 2),
            energy_mix=mix,
            forecast_confidence=round(confidence, 3),
            can_train=can_train,
            reason=reason,
            lookahead=lookahead_scores,
        )
        self._snapshot_cache[cache_key] = snap
        return snap

    def schedule_round(
        self,
        round_num: int,
        mode: str = "naive",
    ) -> Dict[int, RenewableSnapshot]:
        """
        Return scheduling snapshots for all nodes for a given round.

        Args:
            round_num: FL round (1-indexed).
            mode:      "standard" | "naive" | "oracle".

        Returns:
            Dict mapping node_id → RenewableSnapshot.
        """
        return {nid: self.snapshot(nid, round_num, mode=mode) for nid in self._nodes}

    # ── Decision Logic ────────────────────────────────────────────────────────

    def _decide(
        self,
        score: float,
        lookahead: List[float],
        mode: str,
    ) -> Tuple[bool, str]:
        """Determine whether a node should train given the scheduling mode."""
        if mode == "standard":
            return True, "Standard FL: always train"

        if mode == "naive":
            if score >= self._threshold:
                return True, f"Naive: score {score:.2f} ≥ threshold {self._threshold}"
            return False, f"Naive: score {score:.2f} < threshold {self._threshold}"

        if mode == "oracle":
            best_future = max(lookahead) if lookahead else 0.0
            soft_threshold = self._threshold * 0.65  # allow training at 65% of ideal

            if score >= self._threshold:
                # Above threshold: train now if not significantly better window ahead
                if best_future <= score * 1.1:
                    return (
                        True,
                        f"Oracle: now ({score:.2f}) is optimal vs future ({best_future:.2f})",
                    )
                return (
                    False,
                    f"Oracle: deferring — better window ahead ({best_future:.2f})",
                )

            if score >= soft_threshold:
                # Soft zone: train unless a clearly better window is coming
                if best_future > score * 1.25:
                    return (
                        False,
                        f"Oracle: soft-zone defer (future={best_future:.2f} >> now={score:.2f})",
                    )
                return (
                    True,
                    f"Oracle: soft-zone training ({score:.2f} ≥ {soft_threshold:.2f})",
                )

            # Below soft threshold: only train if no better window exists at all
            if best_future < soft_threshold:
                return (
                    True,
                    f"Oracle: best-available training ({score:.2f}, no better window)",
                )
            return False, f"Oracle: waiting for better window ({best_future:.2f})"

        return True, "Unknown mode — defaulting to train"

    # ── Renewable Simulation Kernels ─────────────────────────────────────────

    def _solar_score(self, node: NodeGeography, round_num: int) -> float:
        """
        Compute solar availability for a node at a given round.

        Models:
          • Simulated hour-of-day from timezone offset + round position
          • Sinusoidal diurnal curve (peak 12:00, zero at night)
          • Latitude correction: shorter days at high latitudes in winter
          • Stochastic cloud cover (Gaussian noise)

        Complexity: O(1)
        """
        # Map round to simulated hour (each round ≈ 2.4 h in a 24-h cycle)
        local_hour = (round_num * 2.4 + node.timezone_offset_hours) % 24.0

        # Daylight hours depend on latitude (simplified)
        day_frac = 0.5 + 0.2 * math.cos(math.radians(node.latitude))  # 0.3–0.7
        dawn = 12.0 - day_frac * 12
        dusk = 12.0 + day_frac * 12

        if local_hour < dawn or local_hour > dusk:
            base_solar = 0.0
        else:
            # Sinusoidal bell peak at noon
            angle = math.pi * (local_hour - dawn) / (dusk - dawn)
            base_solar = math.sin(angle) ** 2

        # Gaussian cloud-cover noise
        noise = self._rng.normal(0, self._solar_noise_std)
        return float(np.clip(base_solar + noise, 0.0, 1.0))

    def _wind_score(self, node: NodeGeography, round_num: int) -> float:
        """
        Simulate wind availability with Weibull distribution and temporal
        autocorrelation (AR(1) process with α=0.7).

        Complexity: O(1) with state update.
        """
        # Weibull sample for current round
        weibull_sample = self._rng.weibull(node.weibull_shape) * node.wind_scale

        # AR(1) autocorrelation: blend with previous state
        alpha = 0.7
        new_state = (
            alpha * self._wind_state[node.node_id] + (1 - alpha) * weibull_sample
        )

        # Temporal noise
        noise = self._rng.normal(0, self._wind_noise_std)
        new_state = float(np.clip(new_state + noise, 0.0, 1.0))
        self._wind_state[node.node_id] = new_state
        return new_state

    def _wind_score_deterministic(
        self, node: NodeGeography, future_round: int
    ) -> float:
        """
        Deterministic wind estimate for look-ahead (no state mutation).
        Uses mean of Weibull distribution: λ·Γ(1 + 1/k).

        Complexity: O(1)
        """
        k = node.weibull_shape
        expected_weibull = node.wind_scale * math.gamma(1 + 1 / k)
        return float(np.clip(expected_weibull, 0.0, 1.0))

    def _seasonal_factor(self, round_num: int) -> float:
        """
        Seasonal adjustment: sinusoidal over a 52-round pseudo-year.
        Peak in summer (+20%), trough in winter (-20%).

        Complexity: O(1)
        """
        angle = 2 * math.pi * round_num / 52
        return 0.2 * math.sin(angle)
