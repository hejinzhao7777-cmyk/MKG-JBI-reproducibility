"""Summarize unbiased outer-holdout gate performance from 18 train-only runs.

Each source run reconstructed scaling, null-Cox residuals, all graph layers,
routing and Top-20 selection inside a 70% training partition, then evaluated a
ridge-Cox score on the untouched 30% test partition.  This script contrasts the
train-only routed selector with the train-only zero-graph selector and therefore
separates development routing scores from held-out performance evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
SEEDS = [42, 2025, 7301]


def exact_cluster_bootstrap(values: dict[str, float]) -> tuple[float, float]:
    ordered = np.asarray([values[cancer] for cancer in CANCERS], dtype=float)
    means = np.empty(len(CANCERS) ** len(CANCERS), dtype=float)
    for position, indices in enumerate(itertools.product(range(len(CANCERS)), repeat=len(CANCERS))):
        means[position] = ordered[list(indices)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_run(root: Path, cancer: str, seed: int) -> dict:
    path = root / f"{cancer}_seed{seed}" / "CMPB_REPEATED_TRAIN_ONLY_GRAPH_AUDIT.json"
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(run: dict, margin_key: str) -> dict:
    reconstructed = run["reconstructed"]["margins"][margin_key]
    fixed = run["fixed"]["margins"][margin_key]
    baseline = run["baseline"]
    positive_layers = [
        name for name, value in run["reconstructed"]["delta"].items() if float(value) > 0.0
    ]
    selected_layers = [
        name for name, value in reconstructed["weights"].items() if float(value) > 0.0
    ]
    reconstructed_gain = float(reconstructed["test_cindex"] - baseline["test_cindex"])
    fixed_gain = float(fixed["test_cindex"] - baseline["test_cindex"])
    return {
        "Cancer": run["cancer"],
        "Split seed": int(run["split_seed"]),
        "Margin": float(margin_key),
        "Train n": int(run["train_n"]),
        "Test n": int(run["test_n"]),
        "Test events": int(run["test_events"]),
        "Zero-graph test C-index": float(baseline["test_cindex"]),
        "Reconstructed-route test C-index": float(reconstructed["test_cindex"]),
        "Reconstructed route minus zero": reconstructed_gain,
        "Fixed-route test C-index": float(fixed["test_cindex"]),
        "Fixed route minus zero": fixed_gain,
        "Reconstructed mode": reconstructed["mode"],
        "Fixed mode": fixed["mode"],
        "Reconstructed selected layers": ";".join(selected_layers) if selected_layers else "none",
        "Positive development-score layers": ";".join(positive_layers) if positive_layers else "none",
        "Maximum development delta": float(max(run["reconstructed"]["delta"].values())),
        "Graph admitted": bool(selected_layers),
        "Harmful held-out admission": bool(selected_layers and reconstructed_gain < 0.0),
        "Reconstructed Top20": ";".join(reconstructed["top20"]),
        "Zero-graph Top20": ";".join(baseline["top20"]),
    }


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    primary = raw[np.isclose(raw["Margin"], 0.0)].copy()
    by_cancer = (
        primary.groupby("Cancer", sort=False)
        .agg(
            splits=("Split seed", "count"),
            zero_graph_test_cindex=("Zero-graph test C-index", "mean"),
            reconstructed_route_test_cindex=("Reconstructed-route test C-index", "mean"),
            reconstructed_route_minus_zero=("Reconstructed route minus zero", "mean"),
            fixed_route_minus_zero=("Fixed route minus zero", "mean"),
            graph_admission_rate=("Graph admitted", "mean"),
            harmful_heldout_admission_rate=("Harmful held-out admission", "mean"),
        )
        .reset_index()
    )
    cancer_gain = dict(
        zip(by_cancer["Cancer"], by_cancer["reconstructed_route_minus_zero"], strict=True)
    )
    low, high = exact_cluster_bootstrap(cancer_gain)
    margin_rows: list[dict] = []
    for margin, frame in raw.groupby("Margin", sort=True):
        margin_by_cancer = frame.groupby("Cancer")["Reconstructed route minus zero"].mean().to_dict()
        low_margin, high_margin = exact_cluster_bootstrap(margin_by_cancer)
        admitted = frame[frame["Graph admitted"]]
        margin_rows.append(
            {
                "Margin": float(margin),
                "Mean reconstructed route minus zero": float(np.mean(list(margin_by_cancer.values()))),
                "Cluster bootstrap low": low_margin,
                "Cluster bootstrap high": high_margin,
                "Graph admission runs": int(frame["Graph admitted"].sum()),
                "Graph rejection runs": int((~frame["Graph admitted"]).sum()),
                "Harmful held-out admissions": int(frame["Harmful held-out admission"].sum()),
                "Harmful fraction among admissions": (
                    float(admitted["Harmful held-out admission"].mean()) if len(admitted) else np.nan
                ),
            }
        )
    by_margin = pd.DataFrame(margin_rows)
    admitted = primary[primary["Graph admitted"]]
    overall = {
        "estimand": (
            "untouched-test C-index difference between the train-only reconstructed route "
            "and the train-only zero-graph selector"
        ),
        "primary_margin": 0.0,
        "n_outer_holdouts": int(len(primary)),
        "n_cancers": int(primary["Cancer"].nunique()),
        "mean_difference_cancer_first": float(np.mean(list(cancer_gain.values()))),
        "exact_cancer_cluster_bootstrap_95pct_ci": [low, high],
        "median_run_difference": float(primary["Reconstructed route minus zero"].median()),
        "graph_admission_runs": int(primary["Graph admitted"].sum()),
        "rejection_runs": int((~primary["Graph admitted"]).sum()),
        "harmful_heldout_admissions": int(primary["Harmful held-out admission"].sum()),
        "harmful_fraction_among_admissions": (
            float(admitted["Harmful held-out admission"].mean()) if len(admitted) else np.nan
        ),
        "development_delta_test_gain_spearman": float(
            primary[["Maximum development delta", "Reconstructed route minus zero"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "boundary": (
            "The outer holdouts provide unbiased performance checks for the fitted train-only "
            "pipeline. The internal cross-fitted deltas remain development routing scores, not "
            "stand-alone unbiased estimates of generalization gain."
        ),
        "margin_sensitivity": margin_rows,
    }
    return by_cancer, by_margin, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        flatten(load_run(args.input_root, cancer, seed), margin)
        for cancer in CANCERS
        for seed in SEEDS
        for margin in ["0.000", "0.005", "0.010"]
    ]
    raw = pd.DataFrame(rows)
    by_cancer, by_margin, overall = summarize(raw)
    raw.to_csv(args.output_dir / "CMPB_OUTER_HOLDOUT_GATE_RAW.csv", index=False)
    by_cancer.to_csv(args.output_dir / "CMPB_OUTER_HOLDOUT_GATE_BY_CANCER.csv", index=False)
    by_margin.to_csv(args.output_dir / "CMPB_OUTER_HOLDOUT_GATE_BY_MARGIN.csv", index=False)
    (args.output_dir / "CMPB_OUTER_HOLDOUT_GATE_SUMMARY.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(by_cancer.to_string(index=False))
    print(by_margin.to_string(index=False))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
