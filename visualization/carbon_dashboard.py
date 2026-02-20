"""
carbon_dashboard.py — Carbon Observatory Visualisation Suite
════════════════════════════════════════════════════════════
Publication-quality 6-panel matplotlib masterpiece documenting the
three-arm carbon-aware FL experiment.

Panel Layout:
  A. Accuracy Theater        — accuracy & renewable participation over rounds
  B. Renewable Energy Topology — 3D-style heatmap: Node × Round × Score
  C. Energy Accounting Ledger  — stacked area: baseline vs carbon-aware consumption
  D. Node Participation Matrix — heatmap with participation decisions
  E. Carbon Impact Infographic — CO₂ savings translated to tangible equivalents
  F. Convergence Efficiency    — energy vs accuracy Pareto frontier

Color Palette — "Carbon-Neutral Brutalism":
  • Deep forest green  #0D3B1A
  • Solar gold         #FFD700
  • Warning amber      #FF6B35
  • Glacier blue       #B0E0E6
  • Carbon black       #0A0A0A
  • Moss               #2E7D32
  • Sky                #64B5F6
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams.update(
    {
        "font.family": "monospace",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "#0A0A0A",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#333333",
        "text.color": "#E0E0E0",
        "axes.labelcolor": "#CCCCCC",
        "xtick.color": "#999999",
        "ytick.color": "#999999",
        "grid.color": "#222222",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

log = logging.getLogger("climate_fed.viz")

# ── Palette ───────────────────────────────────────────────────────────────────
FOREST_GREEN = "#0D3B1A"
SOLAR_GOLD = "#FFD700"
AMBER = "#FF6B35"
GLACIER_BLUE = "#B0E0E6"
CARBON_BLACK = "#0A0A0A"
MOSS = "#2E7D32"
SKY_BLUE = "#64B5F6"
LIGHT_GREEN = "#A5D6A7"

ARM_COLORS = {
    "Standard FL": AMBER,
    "Naive Carbon-Aware": SKY_BLUE,
    "Oracle Carbon-Aware": LIGHT_GREEN,
}

NODE_COLORS = ["#FFD700", "#B0E0E6", "#A5D6A7"]  # Oslo, Melbourne, San José

# ── Custom colormaps ──────────────────────────────────────────────────────────
CARBON_CM = LinearSegmentedColormap.from_list(
    "carbon_neutral", ["#1A0A00", "#FF6B35", "#FFD700", "#A5D6A7", "#0D3B1A"]
)
PARTICIPATION_CM = LinearSegmentedColormap.from_list(
    "participation", ["#1A1A1A", "#FF6B35", "#FFD700", "#2E7D32"]
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Carrier
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExperimentRecord:
    """All data required for a single experimental arm's plots."""

    arm_name: str
    accuracies: List[float]  # per-round global accuracy
    energies: List[float]  # per-round total kWh
    cumulative_energy: List[float]  # cumulative kWh
    co2_kg: List[float]  # per-round kg CO₂e
    cumulative_co2: List[float]  # cumulative kg CO₂e
    participation: List[List[int]]  # [round][node] = 0/1
    renewable_scores: List[List[float]]  # [round][node]
    final_accuracy: float
    total_kwh: float
    total_co2_kg: float
    node_names: List[str]


# ──────────────────────────────────────────────────────────────────────────────
# Master Dashboard Generator
# ──────────────────────────────────────────────────────────────────────────────


def render_carbon_observatory(
    records: List[ExperimentRecord],
    impact: Optional[Dict] = None,
    save_dir: str = "./results/plots",
    dpi: int = 150,
    threshold: float = 0.6,
) -> str:
    """
    Generate the complete 6-panel Carbon Observatory dashboard.

    Args:
        records:   List of :class:`ExperimentRecord` — one per experimental arm.
        impact:    Optional impact dict from CarbonLedger.generate_impact_report().
        save_dir:  Directory to save output PNG.
        dpi:       Output resolution.
        threshold: Renewable gate threshold (for green-window shading).

    Returns:
        Absolute path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 14), facecolor=CARBON_BLACK)
    fig.suptitle(
        "🌍  CLIMATE-FED ORCHESTRATOR  ─  Carbon Observatory",
        fontsize=18,
        fontweight="bold",
        color=SOLAR_GOLD,
        fontfamily="monospace",
        y=0.97,
    )

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.40,
        wspace=0.35,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    ax_A = fig.add_subplot(gs[0, :2])  # Accuracy Theater (wide)
    ax_B = fig.add_subplot(gs[0, 2])  # Renewable Score Heatmap
    ax_C = fig.add_subplot(gs[1, :2])  # Energy Accounting Ledger (wide)
    ax_D = fig.add_subplot(gs[1, 2])  # Node Participation Matrix
    ax_E = fig.add_subplot(gs[2, :1])  # Carbon Impact Infographic
    ax_F = fig.add_subplot(gs[2, 1:])  # Convergence Efficiency

    rounds = list(range(1, len(records[0].accuracies) + 1))

    _panel_A_accuracy_theater(ax_A, records, rounds, threshold)
    _panel_B_renewable_heatmap(ax_B, records, rounds)
    _panel_C_energy_ledger(ax_C, records, rounds)
    _panel_D_participation_matrix(ax_D, records, rounds)
    _panel_E_carbon_impact(ax_E, records, impact)
    _panel_F_convergence_frontier(ax_F, records)

    path = os.path.join(save_dir, "carbon_observatory.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=CARBON_BLACK)
    plt.close(fig)
    log.info(f"[Dashboard] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Individual Panel Renderers
# ──────────────────────────────────────────────────────────────────────────────


def _panel_A_accuracy_theater(
    ax: plt.Axes,
    records: List[ExperimentRecord],
    rounds: List[int],
    threshold: float,
) -> None:
    """
    Panel A: Global Accuracy Theater.
    Dual-axis: accuracy (left, solid) + avg renewable score (right, dashed).
    Green shading marks "green training windows" where renewables ≥ threshold.
    """
    ax2 = ax.twinx()

    for rec in records:
        color = ARM_COLORS.get(rec.arm_name, "#FFFFFF")
        ax.plot(
            rounds,
            [a * 100 for a in rec.accuracies],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=rec.arm_name,
            zorder=5,
        )
        # Avg renewable score across nodes per round
        avg_renew = [np.mean(row) for row in rec.renewable_scores]
        ax2.plot(
            rounds,
            avg_renew,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.5,
        )
        # Shade green training windows (Oracle arm reference)
        if rec.arm_name == "Oracle Carbon-Aware":
            for r_idx, r in enumerate(rounds):
                if avg_renew[r_idx] >= threshold:
                    ax.axvspan(
                        r - 0.5,
                        r + 0.5,
                        color="#0D3B1A",
                        alpha=0.25,
                        zorder=0,
                        linewidth=0,
                    )

    ax.axhline(y=90, color="#888888", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(rounds[-1] * 0.05, 90.5, "90% target", color="#888888", fontsize=7)

    ax.set_title("A │ Global Accuracy Theater", color=SOLAR_GOLD, pad=8)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)", color="#E0E0E0")
    ax2.set_ylabel("Avg Renewable Score", color="#888888")
    ax2.set_ylim(0, 1.3)
    ax2.tick_params(colors="#888888")

    legend = ax.legend(
        loc="lower right", framealpha=0.2, labelcolor="#E0E0E0", fontsize=8
    )
    legend.get_frame().set_edgecolor("#444444")
    ax.set_xlim(0.5, len(rounds) + 0.5)
    ax.set_ylim(0, 105)
    ax.grid(True)

    # Milestone annotations
    for rec in records:
        for i, acc in enumerate(rec.accuracies):
            if acc >= 0.90:
                ax.annotate(
                    f"R{rounds[i]}→90%",
                    xy=(rounds[i], acc * 100),
                    xytext=(rounds[i] + 0.3, acc * 100 - 4),
                    fontsize=6,
                    color=ARM_COLORS.get(rec.arm_name, "#FFF"),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
                )
                break


def _panel_B_renewable_heatmap(
    ax: plt.Axes,
    records: List[ExperimentRecord],
    rounds: List[int],
) -> None:
    """
    Panel B: Renewable Energy Topology Heatmap.
    Node × Round grid showing renewable score for the Oracle arm.
    """
    oracle = next((r for r in records if "Oracle" in r.arm_name), records[-1])
    n_nodes = len(oracle.node_names)
    matrix = np.array(oracle.renewable_scores).T  # (nodes, rounds)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=CARBON_CM,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([str(r) for r in rounds], fontsize=6)
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels(oracle.node_names, fontsize=7)
    ax.set_title("B │ Renewable Topology", color=SOLAR_GOLD, pad=8)
    ax.set_xlabel("Round")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Renewable Score", color="#999", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#999")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#999")

    # Mark training decisions
    for row_i, row in enumerate(oracle.participation):
        for col_i, active in enumerate(row):
            marker = "●" if active else "○"
            color = "#FFFFFF" if active else "#555555"
            ax.text(
                col_i, row_i, marker, ha="center", va="center", fontsize=6, color=color
            )


def _panel_C_energy_ledger(
    ax: plt.Axes,
    records: List[ExperimentRecord],
    rounds: List[int],
) -> None:
    """
    Panel C: Energy Accounting Ledger.
    Cumulative energy consumption comparison across all arms.
    Shaded areas distinguish arms.
    """
    ax.set_title("C │ Energy Accounting Ledger", color=SOLAR_GOLD, pad=8)

    for rec in records:
        color = ARM_COLORS.get(rec.arm_name, "#FFFFFF")
        ax.plot(
            rounds,
            rec.cumulative_energy,
            color=color,
            linewidth=2.5,
            label=rec.arm_name,
        )
        ax.fill_between(rounds, rec.cumulative_energy, alpha=0.12, color=color)

    # Annotate savings at final round
    if len(records) >= 2:
        baseline = records[0]
        for rec in records[1:]:
            if rec.total_kwh < baseline.total_kwh:
                savings = baseline.total_kwh - rec.total_kwh
                pct = savings / baseline.total_kwh * 100
                mid = len(rounds) // 2
                ax.annotate(
                    f"−{pct:.0f}%",
                    xy=(rounds[mid], rec.cumulative_energy[mid]),
                    xytext=(
                        rounds[mid] + 0.5,
                        rec.cumulative_energy[mid] + savings * 0.1,
                    ),
                    fontsize=9,
                    fontweight="bold",
                    color=ARM_COLORS.get(rec.arm_name, "#FFF"),
                    arrowprops=dict(arrowstyle="->", color="#888"),
                )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative Energy (kWh-eq)")
    legend = ax.legend(
        loc="upper left", framealpha=0.2, labelcolor="#E0E0E0", fontsize=8
    )
    legend.get_frame().set_edgecolor("#444444")
    ax.grid(True)


def _panel_D_participation_matrix(
    ax: plt.Axes,
    records: List[ExperimentRecord],
    rounds: List[int],
) -> None:
    """
    Panel D: Node × Round participation matrix for Naive or Oracle arm.
    Green = trained, Red = skipped.
    """
    rec = next((r for r in records if "Carbon" in r.arm_name), records[0])
    n_nodes = len(rec.node_names)
    # participation: list[round][node] 0/1
    matrix = np.array(
        [[rec.participation[r][n] for r in range(len(rounds))] for n in range(n_nodes)],
        dtype=float,
    )

    ax.imshow(
        matrix,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("part", ["#3B0000", "#0D3B1A"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([str(r) for r in rounds], fontsize=6)
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels(rec.node_names, fontsize=7)
    ax.set_title("D │ Participation Matrix", color=SOLAR_GOLD, pad=8)
    ax.set_xlabel("Round")

    # Annotate cells
    for ni in range(n_nodes):
        row = matrix[ni]
        rate = row.mean() * 100
        ax.text(
            len(rounds) + 0.1,
            ni,
            f"{rate:.0f}%",
            va="center",
            fontsize=7,
            color=NODE_COLORS[ni % len(NODE_COLORS)],
        )

    trained_patch = mpatches.Patch(color="#0D3B1A", label="Trained")
    skipped_patch = mpatches.Patch(color="#3B0000", label="Skipped")
    ax.legend(
        handles=[trained_patch, skipped_patch],
        loc="lower right",
        fontsize=6,
        framealpha=0.2,
        labelcolor="#E0E0E0",
    ).get_frame().set_edgecolor("#444")


def _panel_E_carbon_impact(
    ax: plt.Axes,
    records: List[ExperimentRecord],
    impact: Optional[Dict],
) -> None:
    """
    Panel E: Carbon Impact Infographic.
    Split bar chart comparing CO₂ across arms + tangible impact icons.
    """
    ax.set_title("E │ Carbon Impact", color=SOLAR_GOLD, pad=8)
    ax.set_facecolor(CARBON_BLACK)

    names = [r.arm_name.replace(" ", "\n") for r in records]
    totals = [r.total_co2_kg * 1000 for r in records]  # → grams for scale
    colors = [ARM_COLORS.get(r.arm_name, "#888") for r in records]

    bars = ax.bar(
        names, totals, color=colors, edgecolor="#333", linewidth=0.8, width=0.5
    )

    for bar, val in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(totals) * 0.01,
            f"{val:.1f}g",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#E0E0E0",
            fontweight="bold",
        )

    ax.set_ylabel("Total CO₂e (grams)")
    ax.grid(axis="y", alpha=0.3)

    # Tangible impact footer
    if impact and len(records) >= 2:
        saved_g = max(0, totals[0] - min(totals[1:]))
        lines = [
            f"🌳 {saved_g / 21000:.2f} trees/year",
            f"🚗 {saved_g / 0.404:.0f} m avoided",
            f"📱 {saved_g / 0.0822:.0f} charges",
        ]
        ax.text(
            0.5,
            -0.32,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=LIGHT_GREEN,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0D3B1A",
                edgecolor="#2E7D32",
                alpha=0.8,
            ),
        )


def _panel_F_convergence_frontier(
    ax: plt.Axes,
    records: List[ExperimentRecord],
) -> None:
    """
    Panel F: Energy vs Accuracy Pareto Frontier.
    Scatter: x=cumulative energy at round, y=accuracy at round.
    Optimal region is bottom-right (high acc, low energy).
    """
    ax.set_title("F │ Convergence Efficiency Frontier", color=SOLAR_GOLD, pad=8)
    ax.set_xlabel("Cumulative Energy (kWh-eq)")
    ax.set_ylabel("Global Accuracy (%)")

    for rec in records:
        color = ARM_COLORS.get(rec.arm_name, "#888")
        x = rec.cumulative_energy
        y = [a * 100 for a in rec.accuracies]
        ax.scatter(x, y, color=color, s=40, alpha=0.85, zorder=5, label=rec.arm_name)
        ax.plot(x, y, color=color, linewidth=1.2, alpha=0.5, linestyle="--")

        # Mark endpoint
        ax.scatter(
            x[-1],
            y[-1],
            color=color,
            s=120,
            marker="*",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )
        ax.text(
            x[-1] * 1.01,
            y[-1] - 1.5,
            f"{y[-1]:.1f}%",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

    # Pareto frontier annotation box
    ax.text(
        0.03,
        0.07,
        "← Better efficiency\n   (same accuracy, less energy)",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#888",
        style="italic",
    )

    legend = ax.legend(
        loc="lower right", framealpha=0.2, labelcolor="#E0E0E0", fontsize=8
    )
    legend.get_frame().set_edgecolor("#444444")
    ax.grid(True)
