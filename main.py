"""
main.py — Single-Command Climate-Fed Orchestrator
═══════════════════════════════════════════════════════
Mission control for the Carbon-Aware Federated Learning Orchestration Platform.

Three-arm experiment protocol:
  Arm 1 — Standard FL:       All nodes train every round, energy-agnostic.
  Arm 2 — Naive Carbon-Aware: Skip nodes with renewable_score < threshold.
  Arm 3 — Oracle Carbon-Aware: Look-ahead scheduling + adaptive LR + renewable²-
                               weighted aggregation.

Usage:
  python main.py                        # full comparison, 10 rounds
  python main.py --rounds 20 --viz      # more rounds, save visualisations
  python main.py --mode oracle --rounds 50
  python main.py --mode naive  --rounds 15

The module is importable; call `run_experiment(args)` programmatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from torchvision import datasets, transforms

# ── Package imports ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from core.carbon_engine import NodeGeography, RenewableOracle
from core.energy_accountant import CarbonLedger, ImpactReport
from core.federated_node import CarbonAwareNode
from core.aggregation_server import CarbonAwareAggregator
from data.mnist_partitioner import MNISTPartitioner
from models.mnist_cnn import EcoCNN
from simulation.renewable_grid import build_node_geographies
from visualization.carbon_dashboard import (
    ExperimentRecord,
    render_carbon_observatory,
)
from visualization.animated_training import TrainingCinema
from visualization.report_generator import (
    generate_markdown_report,
    print_console_summary,
)


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("climate_fed")
    log.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "orchestrator.log"))
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ──────────────────────────────────────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Config Loading
# ──────────────────────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Data Setup
# ──────────────────────────────────────────────────────────────────────────────


def _build_datasets(
    data_root: str,
    node_configs: list,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """Download MNIST, apply non-IID partitioning, return per-node loaders + test loader."""
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tf)

    partitioner = MNISTPartitioner(
        dataset=train_ds,
        node_configs=node_configs,
        alpha=0.5,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    node_loaders = partitioner.partition()

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
    )
    return node_loaders, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Single Arm Runner
# ──────────────────────────────────────────────────────────────────────────────


def run_arm(
    arm_name: str,
    mode: str,  # "standard" | "naive" | "oracle"
    num_rounds: int,
    node_configs: list,
    node_geographies: List[NodeGeography],
    node_loaders: Dict[int, DataLoader],
    test_loader: DataLoader,
    oracle: RenewableOracle,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
) -> ExperimentRecord:
    """
    Execute one experimental arm end-to-end and return an ExperimentRecord.

    Args:
        arm_name:           Human-readable arm label.
        mode:               Scheduling mode passed to oracle.snapshot().
        num_rounds:         Total FL communication rounds.
        node_configs:       YAML node config list.
        node_geographies:   Geographic profiles.
        node_loaders:       Per-node DataLoader dict.
        test_loader:        Global test set loader.
        oracle:             RenewableOracle (stateful — use fresh per arm).
        cfg:                Full YAML config dict.
        device:             Torch device.
        logger:             Logger instance.

    Returns:
        :class:`ExperimentRecord` with all round-level metrics.
    """
    train_cfg = cfg.get("training", {})
    agg_cfg = cfg.get("aggregation", {})
    carbon_cfg = cfg.get("carbon", {})
    model_cfg = cfg.get("model", {})

    base_lr = train_cfg.get("base_lr", 0.01)
    momentum = train_cfg.get("momentum", 0.9)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    local_epochs = train_cfg.get("local_epochs", 1)
    adaptive_lr = train_cfg.get("adaptive_lr", True) and mode == "oracle"
    threshold = carbon_cfg.get("renewable_threshold", 0.6)
    pue = carbon_cfg.get("pue", 1.4)

    strategy = agg_cfg.get("strategy", "renewable_weighted")
    if mode == "standard":
        strategy = "fedavg"  # Standard FL uses plain FedAvg

    # Fresh model and aggregator per arm (no cross-contamination)
    global_model = EcoCNN(
        input_channels=model_cfg.get("input_channels", 1),
        num_classes=model_cfg.get("num_classes", 10),
        conv1_filters=model_cfg.get("conv1_filters", 16),
        conv2_filters=model_cfg.get("conv2_filters", 32),
        fc_hidden=model_cfg.get("fc_hidden", 64),
        dropout=model_cfg.get("dropout", 0.25),
    ).to(device)

    aggregator = CarbonAwareAggregator(
        model=global_model,
        test_loader=test_loader,
        strategy=strategy,
        trimmed_frac=agg_cfg.get("trimmed_fraction", 0.10),
        server_momentum=(
            agg_cfg.get("server_momentum", 0.9)
            if agg_cfg.get("use_server_momentum", True)
            else 0.0
        ),
        device=device,
    )

    # Build nodes
    nodes: Dict[int, CarbonAwareNode] = {}
    for ncfg in node_configs:
        nid = ncfg["id"]
        if nid not in node_loaders:
            continue
        nodes[nid] = CarbonAwareNode(
            node_id=nid,
            node_name=ncfg["name"],
            data_loader=node_loaders[nid],
            model=global_model,
            base_lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            adaptive_lr=adaptive_lr,
            sparsify_threshold=300.0,
            sparsify_ratio=0.10,
        )

    ledger = CarbonLedger(pue=pue)

    # Per-round record holders
    accuracies: List[float] = []
    energies: List[float] = []
    cumulative_energy: List[float] = []
    co2_kg: List[float] = []
    cumulative_co2: List[float] = []
    participation: List[List[int]] = []
    renewable_scores: List[List[float]] = []

    node_ids = list(nodes.keys())
    node_names = [
        n for cfg in node_configs if (n := cfg["name"]) and cfg["id"] in nodes
    ]

    # Print arm banner
    banner_color = {"standard": "🟡", "naive": "🔵", "oracle": "🟢"}.get(mode, "⚪")
    logger.info(f"\n{'═'*70}")
    logger.info(f"  {banner_color} Arm: {arm_name}")
    logger.info(f"{'═'*70}")

    arm_oracle = RenewableOracle(
        nodes=node_geographies,
        threshold=threshold,
        lookahead_rounds=carbon_cfg.get("oracle_lookahead_rounds", 3),
        seed=42,
        solar_noise_std=carbon_cfg.get("solar_noise_std", 0.08),
        wind_noise_std=carbon_cfg.get("wind_noise_std", 0.10),
    )

    for round_num in range(1, num_rounds + 1):
        schedule = arm_oracle.schedule_round(round_num, mode=mode)
        node_results = []
        round_part = []
        round_renew = []

        logger.info(f"\nRound {round_num:02d}/{num_rounds} │ {arm_name}")
        global_w = aggregator.global_weights

        for nid, node in nodes.items():
            snap = schedule[nid]
            ncfg = next(c for c in node_configs if c["id"] == nid)
            round_renew.append(snap.renewable_score)

            if snap.can_train:
                result = node.train_round(
                    global_weights=global_w,
                    num_epochs=local_epochs,
                    renewable_score=snap.renewable_score,
                    carbon_intensity=snap.carbon_intensity,
                )
                round_part.append(1)
                logger.info(
                    f"  Node {snap.node_id} {ncfg['name']:10s} " f"{str(snap)[:80]}"
                )
                logger.info(
                    f"    └─ Acc={result.local_accuracy:.4f} | "
                    f"Loss={result.local_loss:.4f} | "
                    f"Compress={result.compression_ratio:.2f}"
                )
            else:
                result = node.skip_round(snap.renewable_score)
                round_part.append(0)
                logger.info(
                    f"  Node {snap.node_id} {ncfg['name']:10s} "
                    f"⛔ IDLE ({snap.reason})"
                )

            # Record energy
            ledger.record_event(
                round_num=round_num,
                node_id=nid,
                node_name=ncfg["name"],
                is_active=snap.can_train,
                renewable_score=snap.renewable_score,
                carbon_intensity_g_kwh=snap.carbon_intensity,
                num_samples=result.num_samples,
                num_epochs=local_epochs,
                energy_mix_solar=snap.energy_mix.solar_fraction,
                energy_mix_wind=snap.energy_mix.wind_fraction,
                energy_mix_fossil=snap.energy_mix.fossil_fraction,
            )
            node_results.append(result)

        # Aggregate
        agg_result = aggregator.aggregate(node_results, round_num)
        rs = ledger.close_round(round_num, len(nodes))

        # Store round metrics
        accuracies.append(agg_result.global_accuracy)
        energies.append(rs.total_kwh)
        cumulative_energy.append(rs.cumulative_kwh)
        co2_kg.append(rs.total_kg_co2e)
        cumulative_co2.append(rs.cumulative_kg_co2e)
        participation.append(round_part)
        renewable_scores.append(round_renew)

        logger.info(
            f"  ► Global Acc={agg_result.global_accuracy:.4f} | "
            f"Participants={agg_result.num_participants}/{len(nodes)} | "
            f"Energy={rs.total_kwh:.4f} kWh | "
            f"CO₂={rs.total_kg_co2e*1000:.2f}g"
        )

    # Carbon savings vs "all-active" reference for this arm's ledger
    # Reference = actual records with all nodes active every round
    all_active_co2 = (
        sum(ledger.records[0].kg_co2e for r in ledger.records if r.is_active)
        if ledger.records
        else 0
    )  # placeholder; full baseline passed from caller

    return ExperimentRecord(
        arm_name=arm_name,
        accuracies=accuracies,
        energies=energies,
        cumulative_energy=cumulative_energy,
        co2_kg=co2_kg,
        cumulative_co2=cumulative_co2,
        participation=participation,
        renewable_scores=renewable_scores,
        final_accuracy=accuracies[-1],
        total_kwh=cumulative_energy[-1],
        total_co2_kg=cumulative_co2[-1],
        node_names=node_names,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="climate-fed",
        description="🌍 Climate-Fed Orchestrator — Carbon-Aware Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(_HERE / "config" / "simulation_params.yaml"))
    p.add_argument(
        "--mode",
        choices=["full", "standard", "naive", "oracle"],
        default="full",
        help="Experiment mode",
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--viz",
        action="store_true",
        default=True,
        help="Generate visualisation dashboard",
    )
    p.add_argument("--no-viz", dest="viz", action="store_false")
    p.add_argument(
        "--animate",
        action="store_true",
        help="Generate animated training cinema (requires ffmpeg or pillow)",
    )
    p.add_argument("--out", default="./results", help="Root output directory")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = _load_config(args.config)

    # Override YAML with CLI
    cfg["experiment"]["num_rounds"] = args.rounds
    cfg["experiment"]["seed"] = args.seed

    # Output directories
    out = Path(args.out)
    log_dir = str(out / "logs")
    plots_dir = str(out / "plots")
    reports_dir = str(out / "reports")
    metrics_dir = str(out / "metrics")

    for d in [log_dir, plots_dir, reports_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)

    logger = _setup_logging(log_dir)
    _seed_everything(args.seed)

    device = torch.device("cpu")  # CPU-first; carbon-conscious compute

    # ── Boot Banner ────────────────────────────────────────────────────────
    logger.info("━" * 70)
    logger.info("  🌍  CLIMATE-FED ORCHESTRATOR v2.0  ─  Carbon-Aware FL Platform")
    logger.info("━" * 70)
    logger.info(
        f"  Mode: {args.mode.upper()} | Rounds: {args.rounds} | Seed: {args.seed}"
    )
    logger.info(f"  Device: {device}")

    node_configs = cfg["nodes"]
    node_geographies = build_node_geographies(node_configs)

    logger.info("\n  Initializing Renewable Grid...")
    for ng in node_geographies:
        nm = next(c for c in node_configs if c["id"] == ng.node_id)
        logger.info(
            f"    ✓ Node-{ng.node_id}: {ng.name}, {ng.country} "
            f"(Solar: {ng.solar_capacity:.0%}, Wind: {ng.wind_capacity:.0%}, "
            f"Grid: {ng.grid_carbon_intensity}g CO₂/kWh)"
        )

    # ── Model Info ──────────────────────────────────────────────────────────
    sample_model = EcoCNN(
        **{
            k: cfg["model"][k]
            for k in [
                "input_channels",
                "num_classes",
                "conv1_filters",
                "conv2_filters",
                "fc_hidden",
                "dropout",
            ]
        }
    )
    logger.info(
        f"\n  GreenNet-Mini: {sample_model.num_parameters:,} parameters "
        f"| ~2.1 MFLOPs per forward pass"
    )
    del sample_model

    # ── Data Setup ──────────────────────────────────────────────────────────
    logger.info("\n  Partitioning MNIST (Dirichlet α=0.5, Non-IID)...")
    train_cfg = cfg["training"]
    data_cfg = cfg.get("data", {})
    node_loaders, test_loader = _build_datasets(
        data_root=data_cfg.get("data_dir", "./data"),
        node_configs=node_configs,
        batch_size=train_cfg.get("batch_size", 64),
        num_workers=train_cfg.get("num_workers", 0),
        seed=args.seed,
    )

    # ── Build shared oracle (one per arm to avoid state cross-contamination)
    oracle_factory = lambda: RenewableOracle(
        nodes=node_geographies,
        threshold=cfg["carbon"].get("renewable_threshold", 0.6),
        lookahead_rounds=cfg["carbon"].get("oracle_lookahead_rounds", 3),
        seed=args.seed,
        solar_noise_std=cfg["carbon"].get("solar_noise_std", 0.08),
        wind_noise_std=cfg["carbon"].get("wind_noise_std", 0.10),
    )

    # ── Experiment Arms ─────────────────────────────────────────────────────
    arm_spec = {
        "full": [
            ("Standard FL", "standard"),
            ("Naive Carbon-Aware", "naive"),
            ("Oracle Carbon-Aware", "oracle"),
        ],
        "standard": [("Standard FL", "standard")],
        "naive": [("Naive Carbon-Aware", "naive")],
        "oracle": [("Oracle Carbon-Aware", "oracle")],
    }

    t_start = time.time()
    records: List[ExperimentRecord] = []

    for arm_name, mode in arm_spec[args.mode]:
        _seed_everything(args.seed)
        record = run_arm(
            arm_name=arm_name,
            mode=mode,
            num_rounds=args.rounds,
            node_configs=node_configs,
            node_geographies=node_geographies,
            node_loaders=node_loaders,
            test_loader=test_loader,
            oracle=oracle_factory(),
            cfg=cfg,
            device=device,
            logger=logger,
        )
        records.append(record)

    # ── Console + Report ────────────────────────────────────────────────────
    print_console_summary(records)

    report_path = generate_markdown_report(records, cfg, reports_dir)
    logger.info(f"\n  📝 Technical report → {report_path}")

    # ── Visualisation ───────────────────────────────────────────────────────
    if args.viz and records:
        # Dashboard
        logger.info("  🎨 Rendering Carbon Observatory dashboard...")
        dash_path = render_carbon_observatory(
            records=records,
            save_dir=plots_dir,
            dpi=cfg.get("visualization", {}).get("dpi", 150),
            threshold=cfg["carbon"].get("renewable_threshold", 0.6),
        )
        logger.info(f"  🖼  Dashboard → {dash_path}")

    if args.animate and records:
        # Animations
        anim_dir = os.path.join(args.out, "animations")
        os.makedirs(anim_dir, exist_ok=True)
        logger.info(f"  🎬 Generating Training Cinema for {len(records)} arms...")
        for rec in records:
            cinema = TrainingCinema(record=rec, save_dir=anim_dir)
            fname = f"{rec.arm_name.lower().replace(' ', '_')}_evolution.gif"
            path = cinema.generate(filename=fname)
            logger.info(f"    ✓ {rec.arm_name} → {path}")

    # ── Save metrics JSON ───────────────────────────────────────────────────
    metrics_out = {
        "config": {
            "mode": args.mode,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        "arms": [
            {
                "name": r.arm_name,
                "final_accuracy": r.final_accuracy,
                "total_kwh": r.total_kwh,
                "total_co2_kg": r.total_co2_kg,
                "accuracies": r.accuracies,
                "cumulative_energy": r.cumulative_energy,
                "cumulative_co2": r.cumulative_co2,
                "participation": r.participation,
                "renewable_scores": r.renewable_scores,
            }
            for r in records
        ],
    }
    json_path = os.path.join(metrics_dir, "experiment_results.json")
    with open(json_path, "w") as jf:
        json.dump(metrics_out, jf, indent=2)
    logger.info(f"  📋 Metrics JSON → {json_path}")

    elapsed = time.time() - t_start
    logger.info(f"\n  ⏱  Total execution: {elapsed:.1f}s")
    logger.info("━" * 70)
    logger.info("  🌱 Training complete. The planet thanks you.")
    logger.info("━" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
