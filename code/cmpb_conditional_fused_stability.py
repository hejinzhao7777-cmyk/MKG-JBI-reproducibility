"""Conditional fused-ranking stability audit for the CMPB revision.

The route (graph weights) and the two-engine QP weights are frozen from the
development lock.  In each of 30 identical 80% subsamples without replacement,
the graph-Lasso and random-forest engines are refitted and their rank-normalized
feature scores are fused.  This measures selector perturbation after routing;
route variability is reported by the separate repeated-routing audit.

All comparator methods use the same subsample indices.  Raw Top-20 lists are
written so that every reported pairwise RBO/Jaccard value can be regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
METHODS = ["MKG", "Uni-Cox", "Cox-Lasso", "Cox-EN", "CGBoost", "RSF"]
LAYER_FILES = {
    "coexpr": "L_coexpr.npz",
    "meth": "L_meth_expr.npz",
    "cnv": "L_cnv.npz",
}


def normalized_rbo(left: list[int], right: list[int], persistence: float = 0.9) -> float:
    depth = min(len(left), len(right))
    if depth == 0:
        return 0.0
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    weighted = 0.0
    for d in range(1, depth + 1):
        seen_left.add(left[d - 1])
        seen_right.add(right[d - 1])
        agreement = len(seen_left & seen_right) / d
        weighted += (persistence ** (d - 1)) * agreement
    finite_rbo = (1.0 - persistence) * weighted
    return float(finite_rbo / (1.0 - persistence**depth))


def jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return float(len(set(left) & set(right)) / len(union)) if union else 0.0


def pairwise_summary(lists: list[list[int]]) -> dict[str, float | int]:
    rbo_values: list[float] = []
    jaccard_values: list[float] = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            rbo_values.append(normalized_rbo(lists[i], lists[j]))
            jaccard_values.append(jaccard(lists[i], lists[j]))
    return {
        "n_subsamples": len(lists),
        "n_pairs": len(rbo_values),
        "normalized_RBO20": float(np.mean(rbo_values)) if rbo_values else np.nan,
        "Jaccard": float(np.mean(jaccard_values)) if jaccard_values else np.nan,
    }


def cox_score_ranking(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Vectorized univariate Cox score-test ranking at beta=0.

    Breslow handling is used for tied event times.  The statistic is the
    diagonal score statistic U^2 / I; it avoids thousands of separate Cox fits
    while retaining a genuine survival-risk-set definition.
    """

    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    order = np.argsort(-time, kind="mergesort")
    Xs = X[order]
    ts = time[order]
    es = event[order]
    cumulative_x = np.cumsum(Xs, axis=0)
    cumulative_x2 = np.cumsum(Xs * Xs, axis=0)
    score = np.zeros(X.shape[1], dtype=np.float64)
    information = np.zeros(X.shape[1], dtype=np.float64)

    _, first, counts = np.unique(ts, return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        stop = start + count
        d = int(es[start:stop].sum())
        if d == 0:
            continue
        risk_n = stop
        risk_sum = cumulative_x[stop - 1]
        risk_sum2 = cumulative_x2[stop - 1]
        mean = risk_sum / risk_n
        variance = np.maximum(risk_sum2 / risk_n - mean * mean, 0.0)
        score += Xs[start:stop][es[start:stop]].sum(axis=0) - d * mean
        information += d * variance
    statistic = score * score / np.maximum(information, 1e-12)
    return np.argsort(statistic)[::-1]


def fit_mkg_conditional(
    X: np.ndarray,
    residual: np.ndarray,
    laplacian: sparse.csr_matrix,
    qp_weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    beta = F.graph_lasso_pgd(X, residual, laplacian)
    rf = RandomForestRegressor(
        n_estimators=F.STAGE1_RF_TREES,
        max_depth=5,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=F.RF_JOBS,
    )
    rf.fit(X, residual)
    p = X.shape[1]
    linear_rank = rankdata(np.abs(beta), method="min") / p
    forest_rank = rankdata(rf.feature_importances_, method="min") / p
    fused = qp_weights[0] * linear_rank + qp_weights[1] * forest_rank
    return np.argsort(fused)[::-1][: F.TOP_K]


def fit_coxnet(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l1_ratio: float,
) -> np.ndarray:
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=50, max_iter=3000)
    model.fit(X, Surv.from_arrays(event.astype(bool), time))
    coefficient = model.coef_[:, -1]
    return np.argsort(np.abs(coefficient))[::-1][: F.TOP_K]


def fit_cgboost(X: np.ndarray, time: np.ndarray, event: np.ndarray, seed: int) -> np.ndarray:
    model = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X, Surv.from_arrays(event.astype(bool), time))
    coefficient = np.asarray(model.coef_).reshape(-1)
    return np.argsort(np.abs(coefficient))[::-1][: F.TOP_K]


def fit_rsf(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    seed: int,
    screen: int = 150,
) -> np.ndarray:
    prescreen = cox_score_ranking(X, time, event)[:screen]
    y = Surv.from_arrays(event.astype(bool), time)
    model = RandomSurvivalForest(
        n_estimators=80,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=seed,
        n_jobs=min(F.RF_JOBS, 4),
    )
    model.fit(X[:, prescreen], y)
    importance = permutation_importance(
        model,
        X[:, prescreen],
        y,
        n_repeats=1,
        random_state=seed,
        n_jobs=1,
    ).importances_mean
    return prescreen[np.argsort(importance)[::-1][: F.TOP_K]]


def load_cancer(cancer: str) -> dict:
    cancer_dir = F.ROOT / cancer
    expression = pd.read_csv(cancer_dir / "expr_final.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(cancer_dir / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expression.index) & set(survival.index))
    expression = expression.loc[common]
    expression = expression.fillna(expression.mean()).fillna(0.0)
    X = expression.to_numpy(dtype=np.float64)
    residual = survival.loc[common, "deviance_residual"].to_numpy(dtype=np.float64)
    time_values = survival.loc[common, "OS_time"].to_numpy(dtype=np.float64)
    event_values = survival.loc[common, "OS"].to_numpy(dtype=int)

    lock = json.loads((F.OUT / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]
    p = X.shape[1]
    layers = {
        layer: F.normalize_laplacian(sparse.load_npz(cancer_dir / "graph" / filename), p)
        for layer, filename in LAYER_FILES.items()
    }
    routed = sum(float(lock["omics_weights"][layer]) * layers[layer] for layer in layers)
    locked_signature = lock["method_signatures"]["GR-SAFS_v2"]
    if "w_qp" in locked_signature:
        qp = np.asarray(locked_signature["w_qp"], dtype=np.float64)
        qp_source = "stored submission lock"
    else:
        # Early result JSON files retained the fused Top-20 scores but omitted
        # this intermediate.  Recompute it deterministically from the complete
        # development cohort and record the provenance in every output file.
        _, _, _, qp, _ = F.stage1_select(X, residual, routed)
        qp = np.asarray(qp, dtype=np.float64)
        qp_source = "deterministically recomputed from full development cohort"
    return {
        "X": X,
        "residual": residual,
        "time": time_values,
        "event": event_values,
        "genes": np.asarray(expression.columns, dtype=str),
        "laplacian": routed.tocsr(),
        "route_weights": lock["omics_weights"],
        "qp_weights": qp,
        "qp_source": qp_source,
    }


def run_cancer(cancer: str, output_dir: Path, repeats: int, fraction: float, seed: int) -> None:
    payload = load_cancer(cancer)
    X = payload["X"]
    residual = payload["residual"]
    time_values = payload["time"]
    event_values = payload["event"]
    genes = payload["genes"]
    n = X.shape[0]
    subsample_n = int(np.floor(fraction * n))
    rng = np.random.RandomState(seed)
    rankings: dict[str, list[list[int]]] = {method: [] for method in METHODS}
    raw_rows: list[dict] = []
    started = time.time()

    for repeat in range(repeats):
        indices = rng.choice(n, subsample_n, replace=False)
        Xi = X[indices]
        ri = residual[indices]
        ti = time_values[indices]
        ei = event_values[indices]
        repeat_seed = seed + repeat
        fitted = {
            "MKG": lambda: fit_mkg_conditional(
                Xi, ri, payload["laplacian"], payload["qp_weights"], repeat_seed
            ),
            "Uni-Cox": lambda: cox_score_ranking(Xi, ti, ei)[: F.TOP_K],
            "Cox-Lasso": lambda: fit_coxnet(Xi, ti, ei, 1.0),
            "Cox-EN": lambda: fit_coxnet(Xi, ti, ei, 0.5),
            "CGBoost": lambda: fit_cgboost(Xi, ti, ei, repeat_seed),
            "RSF": lambda: fit_rsf(Xi, ti, ei, repeat_seed),
        }
        for method in METHODS:
            method_started = time.time()
            try:
                top = np.asarray(fitted[method](), dtype=int)
                status = "ok"
                error = ""
                rankings[method].append(top.tolist())
            except Exception as exc:  # retain a complete failure record
                top = np.asarray([], dtype=int)
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            raw_rows.append(
                {
                    "Cancer": cancer,
                    "Repeat": repeat + 1,
                    "Seed": repeat_seed,
                    "Method": method,
                    "Status": status,
                    "Error": error,
                    "Subsample n": subsample_n,
                    "Event n": int(ei.sum()),
                    "Top20 indices": ";".join(map(str, top.tolist())),
                    "Top20 genes": ";".join(genes[top].tolist()) if len(top) else "",
                    "Elapsed sec": time.time() - method_started,
                }
            )
        pd.DataFrame(raw_rows).to_csv(output_dir / f"{cancer}_conditional_stability_raw.csv", index=False)
        print(f"[{cancer}] repeat {repeat + 1}/{repeats} complete", flush=True)

    summary_rows = []
    for method in METHODS:
        summary_rows.append(
            {
                "Cancer": cancer,
                "Method": method,
                **pairwise_summary(rankings[method]),
                "Route weights": json.dumps(payload["route_weights"], sort_keys=True),
                "QP weights": json.dumps(payload["qp_weights"].tolist()),
                "QP source": payload["qp_source"],
                "Stability estimand": "conditional fused ranking after route/hyperparameter lock"
                if method == "MKG"
                else "method ranking under locked hyperparameters",
                "Sampling": f"{repeats} x {fraction:.0%} without replacement",
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{cancer}_conditional_stability_summary.csv", index=False)
    metadata = {
        "cancer": cancer,
        "n": n,
        "p": int(X.shape[1]),
        "repeats": repeats,
        "fraction": fraction,
        "seed": seed,
        "route_weights": payload["route_weights"],
        "qp_weights": payload["qp_weights"].tolist(),
        "qp_source": payload["qp_source"],
        "elapsed_min": (time.time() - started) / 60.0,
        "interpretation": (
            "MKG refits both graph-Lasso and random-forest ranking engines while keeping "
            "the development-lock route and QP fusion weights fixed. Route variability is "
            "quantified by the separate full-configuration repeated-routing audit."
        ),
    }
    (output_dir / f"{cancer}_conditional_stability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def combine(output_dir: Path) -> None:
    summary_files = [output_dir / f"{cancer}_conditional_stability_summary.csv" for cancer in CANCERS]
    if not all(path.exists() for path in summary_files):
        return
    combined = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    combined.to_csv(output_dir / "CMPB_CONDITIONAL_FUSED_STABILITY_ALL6.csv", index=False)
    method_summary = (
        combined.groupby("Method", sort=False)[["normalized_RBO20", "Jaccard"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    method_summary.to_csv(output_dir / "CMPB_CONDITIONAL_FUSED_STABILITY_METHOD_SUMMARY.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("MKG_REVISION_OUTPUT", "outputs/conditional_fused_stability")),
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cancer in args.cancers:
        run_cancer(cancer, args.output_dir, args.repeats, args.fraction, args.seed)
    combine(args.output_dir)


if __name__ == "__main__":
    main()
