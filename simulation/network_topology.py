"""
network_topology.py — Heterogeneous Network Simulation
════════════════════════════════════════════════════════
Modeling realistic latency and bandwidth constraints for federated nodes.

In global federated learning, nodes are distributed across varying network
conditions. A node in Oslo might have high-speed fiber, while a node in a rural
area might be on 4G or satellite.

This module provides:
  1. Latency Modeling: Log-normal distribution of RTT (Round Trip Time).
  2. Bandwidth Constraints: Simulation of upload/download speeds.
  3. Packet Loss: Stochastic model for unreliable connections.
  4. Transmission Time Calculation: For gradient uploads.

These network factors contribute to the "Communication Energy" in the CarbonLedger.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("climate_fed.network")


@dataclass
class NetworkProfile:
    """Network capabilities of a specific node."""

    node_id: int
    name: str
    avg_latency_ms: float
    bandwidth_mbps: float  # Upload bandwidth
    packet_loss_rate: float = 0.001
    unstable: bool = False


class NetworkTopology:
    """
    Simulation of global network conditions for federated learning.

    Args:
        profiles: List of :class:`NetworkProfile` for each node.
        seed:     RNG seed for reproducibility.
    """

    def __init__(
        self,
        profiles: List[NetworkProfile],
        seed: int = 42,
    ) -> None:
        self._profiles = {p.node_id: p for p in profiles}
        self._rng = np.random.default_rng(seed)

    def get_transmission_delay(self, node_id: int, payload_size_mb: float) -> float:
        """
        Calculate stochastic transmission delay in seconds for a given payload.

        Formula:
          Total Delay = Latency + (Size / Bandwidth) + Unstability Noise

        Args:
            node_id:         Identifier of the node.
            payload_size_mb: Size of the gradient/model in Megabytes.

        Returns:
            Total delay in seconds.
        """
        if node_id not in self._profiles:
            return 0.1  # Default fallback

        p = self._profiles[node_id]

        # 1. Latency (Log-Normal distribution for RTT)
        # We assume sigma is 0.25 for moderate jitter
        mu = np.log(p.avg_latency_ms / 1000.0)
        latency = self._rng.lognormal(mean=mu, sigma=0.25)

        # 2. Bandwidth Delay
        bw_delay = (payload_size_mb * 8) / max(p.bandwidth_mbps, 0.1)

        # 3. Packet Loss / Retransmission Overhead
        # Simplistic model: overhead scales with loss rate
        loss_overhead = 1.0 + (p.packet_loss_rate * 5.0)

        total_delay = (latency + bw_delay) * loss_overhead

        # 4. Unstability noise
        if p.unstable:
            total_delay *= self._rng.uniform(1.0, 2.5)

        return total_delay

    def get_comm_energy_kwh(self, node_id: int, payload_size_mb: float) -> float:
        """
        Calculate energy consumption of data transmission.

        Based on a model where energy ~ bandwidth usage and duration.
        Reference: 0.1 - 0.5 kWh / GB for global mobile/fixed networks.

        Args:
            node_id:         Node identifier.
            payload_size_mb: Data size.

        Returns:
            Energy in kWh.
        """
        # Roughly 0.05 Wh per MB (50 Wh per GB)
        base_rate = 0.00005  # kWh per MB
        return payload_size_mb * base_rate


def build_network_topology(node_configs: List[dict]) -> NetworkTopology:
    """
    Factory to build a NetworkTopology from YAML config.

    Args:
        node_configs: List of node dicts from config['nodes'].

    Returns:
        :class:`NetworkTopology` instance.
    """
    profiles = []
    for cfg in node_configs:
        profiles.append(
            NetworkProfile(
                node_id=cfg["id"],
                name=cfg["name"],
                avg_latency_ms=cfg.get("network", {}).get("latency_ms", 50),
                bandwidth_mbps=cfg.get("network", {}).get("bandwidth_mbps", 100),
                packet_loss_rate=cfg.get("network", {}).get("packet_loss", 0.001),
                unstable=cfg.get("network", {}).get("unstable", False),
            )
        )
    return NetworkTopology(profiles)
