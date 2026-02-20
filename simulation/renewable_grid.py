"""
renewable_grid.py — Real-World Energy Pattern Simulation
════════════════════════════════════════════════════════
Node grid profiles factory for the three experimental sites.

Wraps the RenewableOracle configuration into geographic profiles that mirror
real-world energy grid characteristics:

  Oslo, Norway      — Nordic hydro dominant, minimal solar in winter, strong wind
  Melbourne, AUS    — High solar irradiance, dirty coal-heavy Victorian grid
  San José, CR      — Almost 100% hydroelectric + geothermal, very low carbon

The phase offsets from timezone differences create temporal diversity:
when Melbourne is at noon solar peak, Oslo and Costa Rica are at night,
naturally staggering renewable availability windows across rounds.
"""

from __future__ import annotations

from typing import List

from climate_fed_orchestrator.core.carbon_engine import NodeGeography


def build_node_geographies(node_configs: List[dict]) -> List[NodeGeography]:
    """
    Construct :class:`NodeGeography` instances from YAML config dicts.

    Args:
        node_configs: Parsed list from config['nodes'].

    Returns:
        List of :class:`NodeGeography` instances, one per node.
    """
    return [
        NodeGeography(
            node_id=cfg["id"],
            name=cfg["name"],
            country=cfg["country"],
            latitude=cfg["latitude"],
            longitude=cfg["longitude"],
            timezone_offset_hours=cfg["timezone_offset_hours"],
            solar_capacity=cfg["solar_capacity"],
            wind_capacity=cfg["wind_capacity"],
            grid_carbon_intensity=cfg["grid_carbon_intensity"],
        )
        for cfg in node_configs
    ]


# ── Canonical Reference Profiles ─────────────────────────────────────────────
# Used for display and ESG reporting, independent of YAML config.

REFERENCE_GRID_PROFILES = {
    "Oslo": {
        "emoji": "🌬",
        "color": "#B0E0E6",  # glacier blue
        "grid_label": "Nordic Hydro-Wind",
        "co2_range": "20–200 g/kWh",
    },
    "Melbourne": {
        "emoji": "☀️",
        "color": "#FFD700",  # solar gold
        "grid_label": "Victorian Coal-Solar",
        "co2_range": "600–900 g/kWh",
    },
    "San José": {
        "emoji": "🌿",
        "color": "#4CAF50",  # forest green
        "grid_label": "Costa Rican Hydro-Geo",
        "co2_range": "20–80 g/kWh",
    },
}
