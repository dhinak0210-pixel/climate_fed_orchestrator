"""
dp_main.py — DP + Live Carbon Orchestrator Entry Point
════════════════════════════════════════════════════════════════════════════
Privacy-Preserving, Carbon-Aware Federated Learning platform.

Three-arm DP experiment protocol:
  Arm 1 — Standard FL (no DP, no carbon):    Pure FedAvg baseline.
  Arm 2 — DP-Only FL:                         DP-SGD but no carbon gating.
  Arm 3 — DP + Oracle Carbon-Aware:           DP-SGD + live carbon scheduling.

CLI:
  python3 -m climate_fed_orchestrator.dp_main --mode full --rounds 10 --epsilon 1.0
  python3 -m climate_fed_orchestrator.dp_main --mode dp_oracle --rounds 20 --live-carbon
  python3 -m climate_fed_orchestrator.dp_main --mode dp_only --rounds 15 --delta 1e-5
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision import datasets, transforms

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from climate_fed_orchestrator.core.carbon_engine import NodeGeography, RenewableOracle
from climate_fed_orchestrator.core.dp_sgd import (
    PrivacyLedger,
    calibrate_noise_multiplier,
)
from climate_fed_orchestrator.core.dp_federated_node import (
    PrivateCarbonNode,
    PrivateNodeRoundResult,
)
from climate_fed_orchestrator.core.live_carbon_api import CarbonAPIManager
from climate_fed_orchestrator.core.aggregation_server import CarbonAwareAggregator
from climate_fed_orchestrator.data.mnist_partitioner import MNISTPartitioner
from climate_fed_orchestrator.models.mnist_cnn import EcoCNN
from climate_fed_orchestrator.simulation.renewable_grid import build_node_geographies
from climate_fed_orchestrator.visualization.carbon_dashboard import ExperimentRecord
from climate_fed_orchestrator.visualization.compliance_dashboard import (
    ComplianceDashboard,
    RoundCarbonMetric,
    RoundPrivacyMetric,
)
from climate_fed_orchestrator.visualization.report_generator import (
    generate_markdown_report,
    print_console_summary,
)


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("climate_fed")
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "dp_orchestrator.log"))
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers (Byzantine-aware)
# ──────────────────────────────────────────────────────────────────────────────


def _fedavg_aggregate(
    results: List[PrivateNodeRoundResult],
    global_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Weighted FedAvg on weight deltas from DP-trained nodes."""
    active = [r for r in results if r.is_active and r.weight_delta]
    if not active:
        return global_weights

    total_samples = sum(r.num_samples for r in active)
    new_weights: Dict[str, torch.Tensor] = {}

    for key in global_weights:
        weighted_delta = sum(
            (r.num_samples / total_samples) * r.weight_delta[key]
            for r in active
            if key in r.weight_delta
        )
        new_weights[key] = global_weights[key] + weighted_delta

    return new_weights


def _renewable_weighted_aggregate(
    results: List[PrivateNodeRoundResult],
    global_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Aggregation weights = samples × renewable_score² (greener nodes → more influence)."""
    active = [r for r in results if r.is_active and r.weight_delta]
    if not active:
        return global_weights

    raw_weights = [r.num_samples * (r.renewable_score**2) for r in active]
    total_w = sum(raw_weights) or 1.0

    new_weights: Dict[str, torch.Tensor] = {}
    for key in global_weights:
        weighted_delta = sum(
            (w / total_w) * r.weight_delta[key]
            for r, w in zip(active, raw_weights)
            if key in r.weight_delta
        )
        new_weights[key] = global_weights[key] + weighted_delta

    return new_weights


def _evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# One DP Arm
# ──────────────────────────────────────────────────────────────────────────────


def run_dp_arm(
    arm_name: str,
    mode: str,  # "standard" | "dp_only" | "dp_oracle"
    num_rounds: int,
    cfg: dict,
    node_configs: list,
    node_geographies: List[NodeGeography],
    node_loaders: Dict[int, torch.utils.data.DataLoader],
    test_loader,
    carbon_api: CarbonAPIManager,
    privacy_ledger: PrivacyLedger,
    compliance: ComplianceDashboard,
    device: torch.device,
    logger: logging.Logger,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    l2_norm_clip: float = 1.0,
    use_live_carbon: bool = False,
) -> ExperimentRecord:

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    carbon_cfg = cfg.get("carbon", {})
    threshold = carbon_cfg.get("renewable_threshold", 0.45)
    local_epochs = train_cfg.get("local_epochs", 1)

    # Build oracle for renewable simulation
    oracle = RenewableOracle(
        nodes=node_geographies,
        threshold=threshold,
        lookahead_rounds=carbon_cfg.get("oracle_lookahead_rounds", 3),
        seed=42,
    )

    # Build global model
    global_model = EcoCNN(
        input_channels=model_cfg.get("input_channels", 1),
        num_classes=model_cfg.get("num_classes", 10),
        conv1_filters=model_cfg.get("conv1_filters", 16),
        conv2_filters=model_cfg.get("conv2_filters", 32),
        fc_hidden=model_cfg.get("fc_hidden", 64),
        dropout=model_cfg.get("dropout", 0.25),
    ).to(device)

    # Build DP nodes
    nodes: Dict[int, PrivateCarbonNode] = {}
    for ncfg in node_configs:
        nid = ncfg["id"]
        if nid not in node_loaders:
            continue
        nodes[nid] = PrivateCarbonNode(
            node_id=nid,
            node_name=ncfg["name"],
            lat=ncfg["latitude"],
            lon=ncfg["longitude"],
            zone=ncfg.get("zone", ncfg["country"][:2].upper()),
            data_loader=node_loaders[nid],
            model=global_model,
            base_lr=train_cfg.get("base_lr", 0.01),
            momentum=train_cfg.get("momentum", 0.9),
            device=device,
            carbon_api=carbon_api if use_live_carbon else None,
            privacy_ledger=privacy_ledger if mode != "standard" else None,
            target_epsilon=epsilon,
            target_delta=delta,
            l2_norm_clip=l2_norm_clip,
            renewable_threshold=threshold if mode != "dp_only" else 0.0,
            adaptive_lr=(mode == "dp_oracle"),
            num_rounds=num_rounds,
        )

    # Metrics storage
    accuracies: List[float] = []
    energies: List[float] = []
    cumulative_energy: List[float] = []
    co2_kg: List[float] = []
    cumulative_co2: List[float] = []
    participation: List[List[int]] = []
    renewable_scores: List[List[float]] = []
    total_kwh = 0.0
    total_co2 = 0.0

    banner = {"standard": "🟡", "dp_only": "🔵", "dp_oracle": "🟢"}.get(mode, "⚪")
    logger.info(f"\n{'═'*68}")
    logger.info(f"  {banner} Arm: {arm_name}")
    logger.info(f"{'═'*68}")

    for round_num in range(1, num_rounds + 1):
        global_w = {k: v.clone() for k, v in global_model.state_dict().items()}

        # Get carbon schedule (simulated oracle or live API)
        schedule = oracle.schedule_round(
            round_num, mode=("oracle" if mode == "dp_oracle" else "standard")
        )

        round_results: List[PrivateNodeRoundResult] = []
        round_part: List[int] = []
        round_renew: List[float] = []

        logger.info(f"\nRound {round_num:02d}/{num_rounds} │ {arm_name}")

        for nid, node in nodes.items():
            snap = schedule[nid]
            round_renew.append(snap.renewable_score)
            ncfg = next(c for c in node_configs if c["id"] == nid)

            # Get live carbon data (or fall back to simulated snapshot)
            from climate_fed_orchestrator.core.live_carbon_api import LiveCarbonData
            from climate_fed_orchestrator.core.live_carbon_api import SimulationFallback

            if use_live_carbon:
                live_cd = nodes[nid].get_carbon_data()
            else:
                # Use oracle snapshot values wrapped in LiveCarbonData
                sim = SimulationFallback(
                    base_carbon_g_kwh=ncfg.get("grid_carbon_intensity", 250),
                    solar_capacity=ncfg.get("solar_capacity", 0.3),
                    wind_capacity=ncfg.get("wind_capacity", 0.4),
                    lat=ncfg["latitude"],
                )
                live_cd = sim.get(ncfg.get("zone", "SIM"))
                live_cd.renewable_score = snap.renewable_score
                live_cd.carbon_intensity_g_kwh = snap.carbon_intensity

            result = node.train_private_round(
                global_weights=global_w,
                round_num=round_num,
                num_epochs=local_epochs,
                carbon_data=live_cd,
            )
            round_results.append(result)
            round_part.append(1 if result.is_active else 0)

            # Log to compliance dashboard
            compliance.record_carbon(
                RoundCarbonMetric(
                    round_num=round_num,
                    node_id=nid,
                    node_name=ncfg["name"],
                    renewable_score=result.renewable_score,
                    carbon_intensity_g_kwh=result.carbon_intensity,
                    data_source=live_cd.data_source,
                    is_simulated=live_cd.is_simulated,
                    is_active=result.is_active,
                )
            )
            if result.is_active and mode != "standard":
                compliance.record_privacy(
                    RoundPrivacyMetric(
                        round_num=round_num,
                        node_id=nid,
                        node_name=ncfg["name"],
                        epsilon_consumed=result.epsilon_consumed,
                        epsilon_total=result.epsilon_total,
                        dp_satisfied=result.dp_satisfied,
                        noise_multiplier=result.noise_multiplier,
                        sigma=result.sigma,
                    )
                )

            if result.is_active:
                logger.info(
                    f"  Node {nid} {ncfg['name']:10s} ✅ PRIVATE "
                    f"| Acc={result.local_accuracy:.4f} | ε={result.epsilon_consumed:.4f} "
                    f"| σ={result.sigma:.4f} | CO₂={result.carbon_intensity:.0f}g/kWh"
                )
            else:
                logger.info(
                    f"  Node {nid} {ncfg['name']:10s} ⛔ SKIP  — {result.skip_reason}"
                )

        # Aggregate
        if mode == "dp_oracle":
            new_weights = _renewable_weighted_aggregate(round_results, global_w)
        else:
            new_weights = _fedavg_aggregate(round_results, global_w)

        global_model.load_state_dict(new_weights, strict=False)

        acc = _evaluate(global_model, test_loader, device)

        # Energy accounting (simplified)
        round_kwh = sum(1.05 if r.is_active else 0.05 for r in round_results)
        round_co2 = sum(
            1.05 * r.carbon_intensity / 1000.0 if r.is_active else 0.0
            for r in round_results
        )
        total_kwh += round_kwh
        total_co2 += round_co2

        accuracies.append(acc)
        energies.append(round_kwh)
        cumulative_energy.append(total_kwh)
        co2_kg.append(round_co2)
        cumulative_co2.append(total_co2)
        participation.append(round_part)
        renewable_scores.append(round_renew)

        eps_info = (
            f"ε={privacy_ledger.epsilon_spent:.4f}" if mode != "standard" else "no-DP"
        )
        logger.info(
            f"  ► Global Acc={acc:.4f} | Energy={round_kwh:.2f}kWh | CO₂={round_co2*1000:.1f}g | {eps_info}"
        )

    node_names = [c["name"] for c in node_configs if c["id"] in nodes]
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dp-climate-fed",
        description="🌍🔒 DP + Carbon-Aware Federated Learning Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(_HERE / "config" / "simulation_params.yaml"))
    p.add_argument(
        "--mode", choices=["full", "standard", "dp_only", "dp_oracle"], default="full"
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="DP privacy budget ε (≤1.0 for strong privacy)",
    )
    p.add_argument("--delta", type=float, default=1e-5, help="DP failure probability δ")
    p.add_argument(
        "--clip", type=float, default=1.0, help="L2 gradient clipping bound C"
    )
    p.add_argument(
        "--live-carbon",
        action="store_true",
        help="Fetch live carbon intensity via APIs",
    )
    p.add_argument("--viz", action="store_true", default=True)
    p.add_argument("--no-viz", dest="viz", action="store_false")
    p.add_argument("--out", default="./results_dp")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["num_rounds"] = args.rounds
    cfg["experiment"]["seed"] = args.seed

    out = Path(args.out)
    for d in ["logs", "plots", "reports", "metrics", "compliance"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(str(out / "logs"))
    _seed_everything(args.seed)
    device = torch.device("cpu")

    logger.info("━" * 68)
    logger.info("  🌍🔒  DP-CLIMATE-FED ORCHESTRATOR v1.0  — Privacy × Planet")
    logger.info("━" * 68)
    logger.info(
        f"  Mode: {args.mode.upper()} | Rounds: {args.rounds} | ε={args.epsilon} | δ={args.delta}"
    )
    logger.info(
        f"  Live Carbon APIs: {'ON' if args.live_carbon else 'OFF (simulation)'}"
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
    node_configs = cfg["nodes"]
    node_geographies = build_node_geographies(node_configs)

    # Carbon API
    carbon_api = CarbonAPIManager(
        electricity_maps_key=os.environ.get("ELECTRICITY_MAPS_API_KEY"),
        watttime_username=os.environ.get("WATTTIME_USERNAME"),
        watttime_password=os.environ.get("WATTTIME_PASSWORD"),
        simulation_configs={
            ncfg.get("zone", ncfg["country"][:2].upper()): {
                "base_carbon": ncfg["grid_carbon_intensity"],
                "solar_capacity": ncfg["solar_capacity"],
                "wind_capacity": ncfg["wind_capacity"],
            }
            for ncfg in node_configs
        },
    )

    # Shared Privacy Ledger
    privacy_ledger = PrivacyLedger(
        target_epsilon=args.epsilon,
        target_delta=args.delta,
    )

    # Data
    train_cfg = cfg["training"]
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

    partitioner = MNISTPartitioner(
        dataset=train_ds,
        node_configs=node_configs,
        alpha=0.5,
        batch_size=train_cfg.get("batch_size", 64),
        num_workers=train_cfg.get("num_workers", 0),
        seed=args.seed,
    )
    node_loaders = partitioner.partition()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # ── Arms ─────────────────────────────────────────────────────────────────
    arm_spec = {
        "full": [
            ("Standard FL", "standard"),
            ("DP-Only FL", "dp_only"),
            ("DP + Oracle Carbon-Aware", "dp_oracle"),
        ],
        "standard": [("Standard FL", "standard")],
        "dp_only": [("DP-Only FL", "dp_only")],
        "dp_oracle": [("DP + Oracle Carbon-Aware", "dp_oracle")],
    }

    compliance = ComplianceDashboard(
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        baseline_kwh=len(arm_spec[args.mode]) * args.rounds * 3 * 1.05,
        experiment_id=f"dp_climate_fed_{args.mode}_{args.rounds}r",
    )

    t_start = time.time()
    records = []

    for arm_name, mode in arm_spec[args.mode]:
        _seed_everything(args.seed)
        # Fresh privacy ledger per arm (separate accounting)
        arm_ledger = PrivacyLedger(target_epsilon=args.epsilon, target_delta=args.delta)
        record = run_dp_arm(
            arm_name=arm_name,
            mode=mode,
            num_rounds=args.rounds,
            cfg=cfg,
            node_configs=node_configs,
            node_geographies=node_geographies,
            node_loaders=node_loaders,
            test_loader=test_loader,
            carbon_api=carbon_api,
            privacy_ledger=arm_ledger,
            compliance=compliance,
            device=device,
            logger=logger,
            epsilon=args.epsilon,
            delta=args.delta,
            l2_norm_clip=args.clip,
            use_live_carbon=args.live_carbon,
        )
        records.append(record)

    # ── Reports ──────────────────────────────────────────────────────────────
    print_console_summary(records)
    compliance._api_health = carbon_api.api_health_report()
    compliance.print_summary()

    md_path, json_path = compliance.generate_report(str(out / "compliance"))
    logger.info(f"\n  🔒 Compliance report → {md_path}")
    logger.info(f"  📋 Compliance JSON  → {json_path}")

    report_path = generate_markdown_report(records, cfg, str(out / "reports"))
    logger.info(f"  📝 Technical report → {report_path}")

    # Save metrics JSON
    metrics = {
        "config": {
            "mode": args.mode,
            "rounds": args.rounds,
            "epsilon": args.epsilon,
            "delta": args.delta,
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
        "api_health": carbon_api.api_health_report(),
    }
    jp = out / "metrics" / "dp_experiment_results.json"
    with open(jp, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  📊 Metrics JSON     → {jp}")

    elapsed = time.time() - t_start
    logger.info(f"\n  ⏱  Total execution: {elapsed:.1f}s")
    logger.info("━" * 68)
    logger.info("  🌱🔒 Training complete. Privacy preserved. Planet thanked.")
    logger.info("━" * 68 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
