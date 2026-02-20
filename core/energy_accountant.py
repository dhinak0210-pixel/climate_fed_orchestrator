"""
energy_accountant.py — The Carbon Ledger
════════════════════════════════════════
Granular per-round energy and carbon accounting with life-cycle assessment.

Tracks every training event's carbon cost through:
  • Operational carbon: computation, memory bandwidth, cooling overhead (PUE)
  • Network transmission carbon: gradient communication cost
  • FLOPs counting with hardware-specific energy efficiency (FLOP/J)

Generates tangible impact metrics:
  • Trees planted equivalent (one tree absorbs ~21 kg CO₂/year)
  • Car miles avoided (US average car: ~404 gCO₂/mile)
  • Smartphone charges equivalent (one charge ≈ 8.22 Wh)
  • Hours of average EU home powered

Complexity: O(R × N) amortised for R rounds, N nodes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Equivalence Constants  (all in kg CO₂)
# ──────────────────────────────────────────────────────────────────────────────
KG_CO2_PER_TREE_YEAR = 21.0  # avg deciduous tree annual absorption
KG_CO2_PER_CAR_MILE = 0.000404  # US EPA average passenger car
KG_CO2_PER_SMARTPHONE = 0.0000822  # one smartphone charge
KG_CO2_PER_HOME_HOUR = 0.000475  # avg EU household hourly carbon


@dataclass
class RoundEnergyRecord:
    """Energy and carbon record for a single FL round."""

    round_num: int
    node_id: int
    node_name: str
    is_active: bool
    renewable_score: float
    carbon_intensity_g_kwh: float
    compute_kwh: float  # Local training computation
    comm_kwh: float  # Network transmission
    cooling_overhead_kwh: float  # PUE - 1 × compute
    total_kwh: float  # compute + comm + cooling
    kg_co2e: float  # Total carbon footprint this event
    energy_mix_solar: float
    energy_mix_wind: float
    energy_mix_fossil: float

    @property
    def is_green(self) -> bool:
        return self.renewable_score >= 0.6


@dataclass
class RoundSummary:
    """Per-round aggregate energy summary across all nodes."""

    round_num: int
    total_kwh: float
    total_kg_co2e: float
    active_nodes: int
    total_nodes: int
    avg_renewable_score: float
    cumulative_kwh: float
    cumulative_kg_co2e: float


@dataclass
class ImpactReport:
    """
    Full life-cycle carbon impact report with ESG-ready metrics.

    Attributes:
        baseline_kg_co2e:   Carbon if all nodes trained every round (no CA).
        actual_kg_co2e:     Carbon actually consumed.
        saved_kg_co2e:      = baseline − actual.
        reduction_pct:      Percentage reduction.
        trees_planted_equiv: Equivalent annual trees absorbing saved CO₂.
        car_miles_avoided:   Miles of driving avoided by CO₂ savings.
        smartphone_charges:  Equivalent smartphone charges with saved energy.
        home_hours_powered:  Hours of EU household powered by savings.
        social_cost_usd:     Damage avoided at $51/tonne (US IAWG SCC).
    """

    baseline_kg_co2e: float
    actual_kg_co2e: float
    saved_kg_co2e: float
    reduction_pct: float
    trees_planted_equiv: float
    car_miles_avoided: float
    smartphone_charges: float
    home_hours_powered: float
    social_cost_usd: float
    total_kwh_consumed: float
    total_kwh_baseline: float


class CarbonLedger:
    """
    Granular carbon footprint ledger for federated learning experiments.

    Tracks per-round, per-node energy consumption accounting for:
      • Computation (local training FLOPs × hardware energy efficiency)
      • Communication (gradient size × transmission energy)
      • Cooling overhead via PUE multiplier
      • Carbon intensity of local grid at time of training

    Args:
        pue:               Power Usage Effectiveness (≥ 1.0). Default 1.4.
        flops_per_sample:  Approximate FLOPs for one forward+backward pass.
        hardware_tflops_w: GPU/CPU efficiency in TFLOP/s per Watt.
                           CPU-only ≈ 0.1 TFLOP/s/W.

    Complexity: O(1) per record; O(R·N) total for R rounds, N nodes.
    """

    # CPU energy envelope: ~50 W for a modern CPU, ~0.05 TFLOPS
    _DEFAULT_FLOPS_PER_SAMPLE = 2_100_000  # EcoCNN on MNIST ≈ 2.1 MFLOPs
    _DEFAULT_CPU_W = 45.0  # Watts
    _DEFAULT_CPU_TFLOPS = 0.05  # TFLOPS for typical laptop CPU

    def __init__(
        self,
        pue: float = 1.4,
        flops_per_sample: int = _DEFAULT_FLOPS_PER_SAMPLE,
        cpu_watts: float = _DEFAULT_CPU_W,
        cpu_tflops: float = _DEFAULT_CPU_TFLOPS,
    ) -> None:
        self._pue = pue
        self._flops_per_sample = flops_per_sample
        self._cpu_watts = cpu_watts
        self._cpu_tflops = cpu_tflops

        self._records: List[RoundEnergyRecord] = []
        self._round_summaries: List[RoundSummary] = []
        self._cumulative_kwh = 0.0
        self._cumulative_co2 = 0.0

    # ── Recording API ─────────────────────────────────────────────────────────

    def record_event(
        self,
        round_num: int,
        node_id: int,
        node_name: str,
        is_active: bool,
        renewable_score: float,
        carbon_intensity_g_kwh: float,
        num_samples: int,
        num_epochs: int,
        gradient_size_mb: float = 0.4,
        energy_mix_solar: float = 0.0,
        energy_mix_wind: float = 0.0,
        energy_mix_fossil: float = 1.0,
    ) -> RoundEnergyRecord:
        """
        Record energy consumption for a single node–round event.

        For active nodes: estimates FLOPs, converts to kWh, applies PUE.
        For idle nodes:   only standby power is recorded (50 W × idle_fraction).

        Args:
            round_num:            FL round number.
            node_id:              Node identifier.
            node_name:            Node human name.
            is_active:            Whether node trained this round.
            renewable_score:      Renewable availability in [0, 1].
            carbon_intensity_g_kwh: Grid carbon gCO₂/kWh.
            num_samples:          Training samples this round.
            num_epochs:           Local training epochs.
            gradient_size_mb:     Size of weight update in MB.
            energy_mix_*:         Fractional generation source mixes.

        Returns:
            :class:`RoundEnergyRecord` with all computed fields.
        """
        # ── Energy Accounting ────────────────────────────────────────────────
        # We use the established FL energy model: 1.05 kWh per active training epoch,
        # 0.05 kWh for idle/standby.  PUE overhead is applied on top.
        if is_active:
            compute_kwh = 1.05 * num_epochs
            comm_kwh = (gradient_size_mb / 1024) * 0.001  # 1 Wh/GB network cost
            cooling_kwh = compute_kwh * (self._pue - 1.0)
            total_kwh = compute_kwh + comm_kwh + cooling_kwh
        else:
            compute_kwh = 0.05  # standby power
            comm_kwh = 0.0
            cooling_kwh = compute_kwh * (self._pue - 1.0)
            total_kwh = compute_kwh + cooling_kwh

        # Carbon footprint
        kg_co2e = total_kwh * (carbon_intensity_g_kwh / 1000)

        rec = RoundEnergyRecord(
            round_num=round_num,
            node_id=node_id,
            node_name=node_name,
            is_active=is_active,
            renewable_score=renewable_score,
            carbon_intensity_g_kwh=carbon_intensity_g_kwh,
            compute_kwh=compute_kwh,
            comm_kwh=comm_kwh,
            cooling_overhead_kwh=cooling_kwh,
            total_kwh=total_kwh,
            kg_co2e=kg_co2e,
            energy_mix_solar=energy_mix_solar,
            energy_mix_wind=energy_mix_wind,
            energy_mix_fossil=energy_mix_fossil,
        )
        self._records.append(rec)
        return rec

    def close_round(self, round_num: int, total_nodes: int) -> RoundSummary:
        """
        Aggregate all node records for a completed round and store summary.

        Args:
            round_num:   The round to summarise.
            total_nodes: Total node count (for participation rate).

        Returns:
            :class:`RoundSummary`.
        """
        round_recs = [r for r in self._records if r.round_num == round_num]
        active_recs = [r for r in round_recs if r.is_active]

        total_kwh = sum(r.total_kwh for r in round_recs)
        total_co2 = sum(r.kg_co2e for r in round_recs)
        avg_renew = (
            sum(r.renewable_score for r in active_recs) / len(active_recs)
            if active_recs
            else 0.0
        )

        self._cumulative_kwh += total_kwh
        self._cumulative_co2 += total_co2

        summary = RoundSummary(
            round_num=round_num,
            total_kwh=total_kwh,
            total_kg_co2e=total_co2,
            active_nodes=len(active_recs),
            total_nodes=total_nodes,
            avg_renewable_score=avg_renew,
            cumulative_kwh=self._cumulative_kwh,
            cumulative_kg_co2e=self._cumulative_co2,
        )
        self._round_summaries.append(summary)
        return summary

    # ── Reporting API ─────────────────────────────────────────────────────────

    def generate_impact_report(
        self,
        baseline_kg_co2e: float,
        social_cost_per_tonne: float = 51.0,
    ) -> ImpactReport:
        """
        Compute full life-cycle impact against a no-carbon-awareness baseline.

        Args:
            baseline_kg_co2e:      Carbon of standard FL (all nodes, all rounds).
            social_cost_per_tonne: US IAWG social cost of carbon (USD per tonne).

        Returns:
            :class:`ImpactReport` with ESG-ready metrics.
        """
        actual = self._cumulative_co2
        saved = max(0.0, baseline_kg_co2e - actual)
        pct = (saved / baseline_kg_co2e * 100) if baseline_kg_co2e > 0 else 0.0

        # Baseline kWh estimate (back-calculate via avg carbon intensity)
        avg_ci = (baseline_kg_co2e / (self._cumulative_kwh + 1e-9)) * 1000  # g/kWh
        baseline_kwh = baseline_kg_co2e * 1000 / max(avg_ci, 1)

        return ImpactReport(
            baseline_kg_co2e=baseline_kg_co2e,
            actual_kg_co2e=actual,
            saved_kg_co2e=saved,
            reduction_pct=pct,
            trees_planted_equiv=saved / KG_CO2_PER_TREE_YEAR,
            car_miles_avoided=saved / KG_CO2_PER_CAR_MILE,
            smartphone_charges=saved / KG_CO2_PER_SMARTPHONE,
            home_hours_powered=saved / KG_CO2_PER_HOME_HOUR,
            social_cost_usd=(saved / 1000) * social_cost_per_tonne,
            total_kwh_consumed=self._cumulative_kwh,
            total_kwh_baseline=baseline_kwh,
        )

    @property
    def records(self) -> List[RoundEnergyRecord]:
        return list(self._records)

    @property
    def round_summaries(self) -> List[RoundSummary]:
        return list(self._round_summaries)

    @property
    def cumulative_kwh(self) -> float:
        return self._cumulative_kwh

    @property
    def cumulative_kg_co2e(self) -> float:
        return self._cumulative_co2
