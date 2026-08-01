"""Conditional stability of Cox baselines using development-CV-selected alphas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from cmpb_conditional_fused_stability import CANCERS, pairwise_summary
import final_config_comparison as F


METHODS = {"CV-Cox-Lasso": 1.0, "CV-Cox-EN": 0.5}


def load_data(cancer: str):
    directory = F.ROOT / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    outcomes = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expression.index) & set(outcomes.index))
    expression = expression.loc[common].fillna(expression.loc[common].mean()).fillna(0.0)
    return (
        expression.to_numpy(dtype=float),
        outcomes.loc[common, "OS_time"].to_numpy(dtype=float),
        outcomes.loc[common, "OS"].to_numpy(dtype=int),
        np.asarray(expression.columns, dtype=str),
    )


def fixed_alpha_ranking(X, time, event, l1_ratio: float, alpha: float):
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha], max_iter=10000, tol=1e-7)
    model.fit(X, Surv.from_arrays(event.astype(bool), time))
    coefficient = model.coef_[:, 0]
    return np.argsort(np.abs(coefficient))[::-1][: F.TOP_K], int(np.sum(np.abs(coefficient) > 1e-12))


def run_cancer(cancer: str, alpha_dir: Path, output_dir: Path, repeats: int, fraction: float, seed: int):
    X, time, event, genes = load_data(cancer)
    alpha_result = json.loads((alpha_dir / f"{cancer}_CV_TUNED_COX_BASELINES.json").read_text(encoding="utf-8"))
    selected = {
        method: float(alpha_result[method]["signature"]["alpha"])
        for method in METHODS
    }
    rng = np.random.RandomState(seed)
    sample_n = int(np.floor(fraction * len(X)))
    rankings = {method: [] for method in METHODS}
    rows = []
    for repeat in range(repeats):
        indices = rng.choice(len(X), sample_n, replace=False)
        for method, ratio in METHODS.items():
            top, nonzero = fixed_alpha_ranking(X[indices], time[indices], event[indices], ratio, selected[method])
            rankings[method].append(top.tolist())
            rows.append({
                "Cancer": cancer,
                "Repeat": repeat + 1,
                "Method": method,
                "Selected alpha": selected[method],
                "Subsample n": sample_n,
                "Event n": int(event[indices].sum()),
                "Nonzero": nonzero,
                "Top20 indices": ";".join(map(str, top)),
                "Top20 genes": ";".join(genes[top]),
            })
        print(f"[{cancer}] tuned-Cox repeat {repeat + 1}/{repeats}", flush=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{cancer}_CV_TUNED_COX_STABILITY_RAW.csv", index=False)
    summary = pd.DataFrame([
        {
            "Cancer": cancer,
            "Method": method,
            "Selected alpha": selected[method],
            **pairwise_summary(rankings[method]),
            "Stability estimand": "method ranking conditional on development-CV-selected alpha",
            "Sampling": f"{repeats} x {fraction:.0%} without replacement",
        }
        for method in METHODS
    ])
    summary.to_csv(output_dir / f"{cancer}_CV_TUNED_COX_STABILITY_SUMMARY.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cancer in args.cancers:
        run_cancer(cancer, args.alpha_dir, args.output_dir, args.repeats, args.fraction, args.seed)
    files = [args.output_dir / f"{cancer}_CV_TUNED_COX_STABILITY_SUMMARY.csv" for cancer in CANCERS]
    if all(path.exists() for path in files):
        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        combined.to_csv(args.output_dir / "CMPB_CV_TUNED_COX_STABILITY_ALL6.csv", index=False)


if __name__ == "__main__":
    main()
