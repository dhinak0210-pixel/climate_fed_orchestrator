import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# Experiment configuration
SEEDS = [42, 123, 456, 789, 101112]
ROUNDS = 10
RESULTS_DIR = "results_experiment"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiments():
    print("🚀 Starting 15-Run Publication Experiment...")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Condition A: Standard FL
        # Using main.py default mode=standard (which matches 'standard' here)
        print(f"Running Condition A (Standard) | Seed {seed}")
        subprocess.run(
            [
                "python3",
                "main.py",
                "--mode",
                "standard",
                "--rounds",
                str(ROUNDS),
                "--seed",
                str(seed),
                "--no-viz",
                "--out",
                f"{RESULTS_DIR}/A_{seed}",
            ],
            check=True,
        )

        # Condition B: Carbon-Aware Only
        # Using main.py mode=naive (matches "threshold=0.6" description)
        print(f"Running Condition B (Carbon-Aware) | Seed {seed}")
        subprocess.run(
            [
                "python3",
                "main.py",
                "--mode",
                "naive",
                "--rounds",
                str(ROUNDS),
                "--seed",
                str(seed),
                "--no-viz",
                "--out",
                f"{RESULTS_DIR}/B_{seed}",
            ],
            check=True,
        )

        # Condition C: Full System (Carbon + Privacy)
        # Using dp_main.py mode=dp_oracle (matches "Full System")
        # Note: dp_oracle uses Oracle scheduling, which is slightly advanced compared to Naive.
        # But this represents the best "System".
        print(f"Running Condition C (Full System) | Seed {seed}")
        subprocess.run(
            [
                "python3",
                "dp_main.py",
                "--mode",
                "dp_oracle",
                "--rounds",
                str(ROUNDS),
                "--seed",
                str(seed),
                "--epsilon",
                "2.0",
                "--no-viz",
                "--out",
                f"{RESULTS_DIR}/C_{seed}",
            ],
            check=True,
        )


def collect_data():
    records = []

    for seed in SEEDS:
        # Load A
        with open(f"{RESULTS_DIR}/A_{seed}/metrics/experiment_results.json") as f:
            data = json.load(f)["arms"][0]
            records.append(
                {
                    "Condition": "Standard FL",
                    "Seed": seed,
                    "Final Accuracy": data["final_accuracy"],
                    "Total CO2 (kg)": data["total_co2_kg"],
                    "Privacy Budget": "\u221e",  # Infinity
                    "Rounds to 90%": next(
                        (
                            i + 1
                            for i, acc in enumerate(data["accuracies"])
                            if acc >= 0.90
                        ),
                        None,
                    ),
                    "Accuracies": data["accuracies"],
                    "Cumulative CO2": data["cumulative_co2"],
                }
            )

        # Load B
        with open(f"{RESULTS_DIR}/B_{seed}/metrics/experiment_results.json") as f:
            data = json.load(f)["arms"][0]
            records.append(
                {
                    "Condition": "Carbon-Aware",
                    "Seed": seed,
                    "Final Accuracy": data["final_accuracy"],
                    "Total CO2 (kg)": data["total_co2_kg"],
                    "Privacy Budget": "\u221e",
                    "Rounds to 90%": next(
                        (
                            i + 1
                            for i, acc in enumerate(data["accuracies"])
                            if acc >= 0.90
                        ),
                        None,
                    ),
                    "Accuracies": data["accuracies"],
                    "Cumulative CO2": data["cumulative_co2"],
                }
            )

        # Load C
        with open(f"{RESULTS_DIR}/C_{seed}/metrics/dp_experiment_results.json") as f:
            data = json.load(f)["arms"][0]
            records.append(
                {
                    "Condition": "Full System (C+P)",
                    "Seed": seed,
                    "Final Accuracy": data["final_accuracy"],
                    "Total CO2 (kg)": data["total_co2_kg"],
                    "Privacy Budget": "2.0",
                    "Rounds to 90%": next(
                        (
                            i + 1
                            for i, acc in enumerate(data["accuracies"])
                            if acc >= 0.90
                        ),
                        None,
                    ),
                    "Accuracies": data["accuracies"],
                    "Cumulative CO2": data["cumulative_co2"],
                }
            )

    # Save aggregated raw data
    with open(f"{RESULTS_DIR}/experimental_data.json", "w") as f:
        json.dump(records, f, indent=2)

    return pd.DataFrame(records)


def generate_visualizations(df):
    sns.set_theme(style="whitegrid")

    # 1. Convergence Comparison
    plt.figure(figsize=(10, 6))

    # We need to explode the lists of accuracies
    plot_data = []
    for _, row in df.iterrows():
        for r, acc in enumerate(row["Accuracies"]):
            plot_data.append(
                {"Round": r + 1, "Accuracy": acc, "Condition": row["Condition"]}
            )
    plot_df = pd.DataFrame(plot_data)

    sns.lineplot(
        data=plot_df,
        x="Round",
        y="Accuracy",
        hue="Condition",
        palette=["#ffcc00", "#00aaff", "#00cc66"],
        linewidth=2.5,
    )
    plt.title("Convergence Comparison (Mean ± 95% CI)", fontsize=14)
    plt.ylabel("Test Accuracy")
    plt.savefig(f"{RESULTS_DIR}/convergence_comparison.png", dpi=300)
    plt.close()

    # 2. Carbon Savings
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x="Condition",
        y="Total CO2 (kg)",
        palette=["#ffcc00", "#00aaff", "#00cc66"],
        ci=95,
        capsize=0.1,
    )
    plt.title("Carbon Footprint per Condition", fontsize=14)
    plt.ylabel("Total CO2 Emissions (kg)")
    plt.savefig(f"{RESULTS_DIR}/carbon_savings.png", dpi=300)
    plt.close()

    # 3. Tables
    # Create Summary Table
    summary = (
        df.groupby("Condition")
        .agg(
            {
                "Final Accuracy": ["mean", "std"],
                "Total CO2 (kg)": ["mean", "std"],
                "Rounds to 90%": ["mean"],
            }
        )
        .round(4)
    )

    summary.to_csv(f"{RESULTS_DIR}/summary_stats.csv")

    # Markdown Table
    md = "| Condition | Accuracy (Mean ± Std) | CO2 (kg) (Mean ± Std) | Rounds to 90% |\n"
    md += "|---|---|---|---|\n"
    for idx, row in summary.iterrows():
        acc = f"{row[('Final Accuracy', 'mean')]:.4f} ± {row[('Final Accuracy', 'std')]:.3f}"
        co2 = f"{row[('Total CO2 (kg)', 'mean')]:.4f} ± {row[('Total CO2 (kg)', 'std')]:.3f}"
        rnd = f"{row[('Rounds to 90%', 'mean')]:.1f}"
        md += f"| {idx} | {acc} | {co2} | {rnd} |\n"

    with open(f"{RESULTS_DIR}/comparison_table.md", "w") as f:
        f.write("# Comparison Table: 15-Run Experiment\n\n")
        f.write(md)


if __name__ == "__main__":
    # Ensure correct working directory
    # If this script is in orchestrator root, call run_experiments
    run_experiments()
    df = collect_data()
    generate_visualizations(df)
    print("✅ Experiment Complete. Results in results_experiment/")
