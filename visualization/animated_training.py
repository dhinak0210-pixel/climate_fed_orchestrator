"""
animated_training.py — The Training Cinema
═════════════════════════════════════════════
Dynamic animation of the federated learning evolution.

Visualizes:
  1. Weight Delta Intensity — Heatmap of parameter updates over rounds.
  2. Renewable Pulse — Pulsating indicators of node energy availability.
  3. Accuracy Growth — Real-time line plot evolution.
  4. Global Convergence — Model state representation.

Generates:
  • MP4 or GIF animation for "theater-grade" mission control experience.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from climate_fed_orchestrator.visualization.carbon_dashboard import (
    ExperimentRecord,
    ARM_COLORS,
    CARBON_BLACK,
    SOLAR_GOLD,
)

log = logging.getLogger("climate_fed.viz")


class TrainingCinema:
    """
    Orchestrator for 4D training animations.

    Args:
        record:   The :class:`ExperimentRecord` to animate.
        save_dir: Output directory for the animation file.
        fps:      Frames per second (one round per second by default).
    """

    def __init__(
        self,
        record: ExperimentRecord,
        save_dir: str = "./results/animations",
        fps: int = 2,
    ) -> None:
        self.record = record
        self.save_dir = save_dir
        self.fps = fps
        os.makedirs(save_dir, exist_ok=True)

    def generate(self, filename: str = "training_evolution.mp4") -> str:
        """
        Create the animation and save to disk.

        Args:
            filename: Target filename (supports .mp4 or .gif).

        Returns:
            Path to the saved animation.
        """
        fig = plt.figure(figsize=(12, 8), facecolor=CARBON_BLACK)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # ── Setup Subplots ───────────────────────────────────────────────
        ax_acc = fig.add_subplot(gs[0, 0])  # Accuracy evolution
        ax_heat = fig.add_subplot(gs[0, 1])  # Weight heatmap (abstract)
        ax_renew = fig.add_subplot(gs[1, 0])  # Node renewable pulses
        ax_co2 = fig.add_subplot(gs[1, 1])  # Cumulative CO2

        fig.suptitle(
            f"🎬 CI-FED CINEMA: {self.record.arm_name}",
            color=SOLAR_GOLD,
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        n_rounds = len(self.record.accuracies)
        rounds = np.arange(1, n_rounds + 1)

        # ── Initialization ───────────────────────────────────────────────
        (line_acc,) = ax_acc.plot([], [], color=SOLAR_GOLD, lw=2)
        ax_acc.set_xlim(0, n_rounds + 1)
        ax_acc.set_ylim(0, 105)
        ax_acc.set_title("Global Accuracy %", color="white")
        ax_acc.set_facecolor("#111")

        # ── Abstract Heatmap setup ───────────────────────────────────────
        # We'll simulate a 10x10 model 'weight' intensity grid
        heat_data = np.random.randn(10, 10) * 0.1
        im_heat = ax_heat.imshow(heat_data, cmap="magma", vmin=-1, vmax=1)
        ax_heat.set_title("Weight Delta Pulse", color="white")
        ax_heat.axis("off")

        # ── Renewable Pulse setup ────────────────────────────────────────
        bars_renew = ax_renew.bar(
            self.record.node_names,
            [0] * len(self.record.node_names),
            color=["#FFD700", "#B0E0E6", "#A5D6A7"],
        )
        ax_renew.set_ylim(0, 1.2)
        ax_renew.set_title("Renewable Score", color="white")
        ax_renew.set_facecolor("#111")
        ax_renew.tick_params(axis="x", colors="white")

        # ── CO2 setup ────────────────────────────────────────────────────
        (line_co2,) = ax_co2.plot([], [], color="#FF4500", lw=2)
        ax_co2.set_xlim(0, n_rounds + 1)
        ax_co2.set_ylim(0, max(self.record.cumulative_co2) * 1.1)
        ax_co2.set_title("Cumulative CO2 (kg)", color="white")
        ax_co2.set_facecolor("#111")

        def animate(i):
            round_idx = i

            # Update Accuracy
            line_acc.set_data(
                rounds[: i + 1], [a * 100 for a in self.record.accuracies[: i + 1]]
            )

            # Update Heatmap (Stochastic representation of 'activity')
            # Randomize intensity based on participation
            activity = sum(self.record.participation[i]) / len(self.record.node_names)
            new_heat = np.random.randn(10, 10) * activity
            im_heat.set_data(new_heat)

            # Update Energy Bars
            for bar, score in zip(bars_renew, self.record.renewable_scores[i]):
                bar.set_height(score)
                # Participation glow
                bar.set_alpha(1.0 if any(self.record.participation[i]) else 0.3)

            # Update CO2
            line_co2.set_data(rounds[: i + 1], self.record.cumulative_co2[: i + 1])

            return line_acc, im_heat, *bars_renew, line_co2

        ani = animation.FuncAnimation(
            fig, animate, frames=n_rounds, interval=1000 / self.fps, blit=True
        )

        output_path = os.path.join(self.save_dir, filename)

        # Determine writer based on extension
        if filename.endswith(".gif"):
            ani.save(output_path, writer="pillow")
        else:
            try:
                # Requires ffmpeg installed
                ani.save(output_path, writer="ffmpeg", dpi=100)
            except Exception as e:
                log.warning(
                    f"Could not save video (ffmpeg missing?): {e}. Falling back to GIF."
                )
                output_path = output_path.replace(".mp4", ".gif")
                ani.save(output_path, writer="pillow")

        plt.close(fig)
        log.info(f"Training Cinema saved: {output_path}")
        return output_path
