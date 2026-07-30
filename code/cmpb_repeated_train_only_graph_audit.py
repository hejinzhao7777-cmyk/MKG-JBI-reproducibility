"""Repeated train-only graph reconstruction audit for the CMPB manuscript.

For one cancer and one prespecified split seed, the script rebuilds expression
scaling, null-Cox residuals, co-expression, methylation-expression, and CNV
graphs using training samples only. It then compares reconstructed graphs with
the fixed full-cohort graph structures on the same untouched test split.

The random-forest branch of the stage-1 selector is graph-independent. It is
computed once per split and reused exactly across all candidate and routed
Laplacians. This is algebraically identical to repeated stage1_select() calls
and substantially reduces the cost of repeated leakage audits.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sksurv.util import Surv


HERE = Path(__file__).resolve()
CODE_DIR = HERE.parent
REPOSITORY_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import final_config_comparison as F  # noqa: E402
import full_train_only_graph_audit as A  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MKG_DATA_ROOT",
        REPOSITORY_ROOT / "data" / "processed_data",
    )
)
DEFAULT_OUTDIR = Path(
    os.environ.get(
        "MKG_OUTPUT_ROOT",
        REPOSITORY_ROOT / "results" / "reruns",
    )
) / "repeated_train_only_graph_audit"
GATE_MARGINS = (0.0, 0.005, 0.010)
LAYER_WORKERS = 1


def configure(args: argparse.Namespace) -> None:
    global LAYER_WORKERS
    A.DATA = Path(args.data_root)
    A.SEED = int(args.split_seed)
    A.TEST_SIZE = float(args.test_size)
    F.N_BOOTSTRAP = int(args.bootstrap)
    F.STAGE1_RF_TREES = int(args.rf_trees)
    F.PGD_MAX_ITER = int(args.pgd_iterations)
    F.RF_JOBS = int(args.rf_jobs)
    LAYER_WORKERS = max(1, int(args.layer_workers))


def stage1_context(
    expression: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """Compute the graph-independent stage-1 random-forest branch once."""

    n, p = expression.shape
    folds = list(KFold(F.K_FOLDS, shuffle=True, random_state=F.SEED).split(expression))
    prediction = np.zeros(n, dtype=float)
    importance = np.zeros(p, dtype=float)
    for train, validation in folds:
        forest = RandomForestRegressor(
            n_estimators=F.STAGE1_RF_TREES,
            max_depth=5,
            min_samples_leaf=3,
            random_state=F.SEED,
            n_jobs=F.RF_JOBS,
        )
        forest.fit(expression[train], residual[train])
        prediction[validation] = forest.predict(expression[validation])
        importance += forest.feature_importances_
    return {
        "folds": folds,
        "rf_prediction": prediction,
        "rf_importance": importance,
    }


def stage1_top_cached(
    expression: np.ndarray,
    residual: np.ndarray,
    laplacian: sparse.csr_matrix,
    context: dict,
) -> np.ndarray:
    """Return the exact stage-1 Top-K while reusing the RF branch."""

    n, p = expression.shape
    linear_prediction = np.zeros(n, dtype=float)
    linear_importance = np.zeros(p, dtype=float)
    for train, validation in context["folds"]:
        coefficient = F.graph_lasso_pgd(
            expression[train],
            residual[train],
            laplacian,
            max_iter=F.PGD_MAX_ITER,
        )
        linear_prediction[validation] = expression[validation] @ coefficient
        linear_importance += np.abs(coefficient)
    predictions = np.column_stack(
        [linear_prediction, context["rf_prediction"]]
    )
    ensemble = F.solve_qp_weights(predictions, residual)
    scores = (
        ensemble[0] * rankdata(linear_importance, method="min") / p
        + ensemble[1] * rankdata(context["rf_importance"], method="min") / p
    )
    return np.argsort(scores)[::-1][: F.TOP_K]


def fit_and_test_top(
    expression_train: np.ndarray,
    expression_test: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    names: np.ndarray,
    top: np.ndarray,
) -> tuple[float, list[str]]:
    selected = names[top].tolist()
    train_frame = pd.DataFrame(expression_train[:, top], columns=selected)
    train_frame["T"] = time_train
    train_frame["E"] = event_train
    model = CoxPHFitter(penalizer=A.RIDGE_PENALIZER, l1_ratio=0.0)
    model.fit(train_frame, duration_col="T", event_col="E")
    test_frame = pd.DataFrame(expression_test[:, top], columns=selected)
    risk = test_frame[model.params_.index].to_numpy() @ model.params_.to_numpy()
    value = concordance_index(time_test, -risk, event_test)
    return float(value), selected


def layer_diagnostics(
    expression: np.ndarray,
    residual: np.ndarray,
    survival,
    clinical: pd.DataFrame,
    layers: dict[str, sparse.csr_matrix],
    baseline_cindex: float,
    context: dict,
) -> tuple[dict, dict, dict]:
    def compute(item):
        name, laplacian = item
        stability_value = F.bootstrap_stability(
            expression,
            residual,
            laplacian,
            seed=F.SEED,
        )
        top = stage1_top_cached(expression, residual, laplacian, context)
        delta_value = (
            F.stage2_oof_ci(expression[:, top], clinical, survival)
            - baseline_cindex
        )
        return name, stability_value, delta_value, top

    items = list(layers.items())
    if LAYER_WORKERS == 1:
        computed = [compute(item) for item in items]
    else:
        with ThreadPoolExecutor(
            max_workers=min(LAYER_WORKERS, len(items))
        ) as executor:
            computed = list(executor.map(compute, items))
    stability = {name: value for name, value, _, _ in computed}
    delta = {name: value for name, _, value, _ in computed}
    top_by_layer = {name: top for name, _, _, top in computed}
    return stability, delta, top_by_layer


def route(
    layers: dict[str, sparse.csr_matrix],
    stability: dict[str, float],
    delta: dict[str, float],
    margin: float,
) -> tuple[sparse.csr_matrix, dict[str, float], str]:
    utility = {
        name: stability[name] * delta[name] if delta[name] > margin else 0.0
        for name in layers
    }
    total = float(sum(utility.values()))
    if total <= 1e-15:
        weights = {name: 0.0 for name in layers}
        combined = sparse.csr_matrix(next(iter(layers.values())).shape, dtype=np.float32)
        mode = "reject_all_graphs"
    else:
        weights = {name: utility[name] / total for name in layers}
        combined = sum(weights[name] * layers[name] for name in layers)
        mode = "dual_driven"
    return combined.tocsr(), weights, mode


def evaluate_graph_set(
    label: str,
    layers: dict[str, sparse.csr_matrix],
    expression_train: np.ndarray,
    expression_test: np.ndarray,
    residual_train: np.ndarray,
    survival_train,
    clinical_train: pd.DataFrame,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    names: np.ndarray,
    baseline_top: np.ndarray,
    baseline_cindex: float,
    context: dict,
) -> dict:
    print(f"[{label}] computing layer stability and OOF utility", flush=True)
    stability, delta, top_by_layer = layer_diagnostics(
        expression_train,
        residual_train,
        survival_train,
        clinical_train,
        layers,
        baseline_cindex,
        context,
    )
    margins = {}
    for margin in GATE_MARGINS:
        combined, weights, mode = route(layers, stability, delta, margin)
        top = stage1_top_cached(
            expression_train,
            residual_train,
            combined,
            context,
        )
        test_cindex, selected = fit_and_test_top(
            expression_train,
            expression_test,
            time_train,
            event_train,
            time_test,
            event_test,
            names,
            top,
        )
        margins[f"{margin:.3f}"] = {
            "margin": margin,
            "mode": mode,
            "weights": weights,
            "test_cindex": test_cindex,
            "top20": selected,
        }
    return {
        "stability": stability,
        "delta": delta,
        "candidate_top20": {
            name: names[top].tolist() for name, top in top_by_layer.items()
        },
        "margins": margins,
    }


def run(args: argparse.Namespace) -> dict:
    configure(args)
    started = time.time()
    cancer = args.cancer
    print(f"[{cancer} seed={args.split_seed}] loading matrices", flush=True)
    data = A.load_cancer(cancer)
    time_values = data["time"]
    event = data["event"]
    index = np.arange(len(time_values))
    train, test = train_test_split(
        index,
        test_size=A.TEST_SIZE,
        random_state=A.SEED,
        stratify=event,
    )

    expression_raw = data["expression"].to_numpy(dtype=np.float32)
    expression_train, expression_test = A.standardize_from_train(
        expression_raw[train],
        expression_raw[test],
    )
    methylation_train, _ = A.fill_train_values(
        data["methylation"].to_numpy(dtype=np.float32)[train],
        data["methylation"].to_numpy(dtype=np.float32)[test],
    )
    copy_train, _ = A.fill_train_values(
        data["copy_number"].to_numpy(dtype=np.float32)[train],
        data["copy_number"].to_numpy(dtype=np.float32)[test],
    )
    residual_train = A.null_cox_deviance(
        time_values[train],
        event[train],
    )
    survival_train = Surv.from_arrays(
        event[train].astype(bool),
        time_values[train],
    )
    clinical_train = data["clinical"].iloc[train]
    p = expression_train.shape[1]
    zero = sparse.csr_matrix((p, p), dtype=np.float32)

    print(f"[{cancer}] reconstructing train-only relation graphs", flush=True)
    coexpr_adjacency, coexpr_meta = A.symmetric_correlation_graph(
        expression_train,
        A.COEXPR_TAU,
        A.COEXPR_POWER,
    )
    meth_adjacency, meth_meta = A.methylation_expression_graph(
        methylation_train,
        expression_train,
    )
    cnv_adjacency, cnv_meta = A.symmetric_correlation_graph(
        copy_train,
        A.CNV_TAU,
        A.CNV_POWER,
    )
    reconstructed = {
        "coexpr": F.normalize_laplacian(
            A.adjacency_to_laplacian(coexpr_adjacency),
            p,
        ),
        "meth": F.normalize_laplacian(
            A.adjacency_to_laplacian(meth_adjacency),
            p,
        ),
        "cnv": F.normalize_laplacian(
            A.adjacency_to_laplacian(cnv_adjacency),
            p,
        ),
    }
    del coexpr_adjacency, meth_adjacency, cnv_adjacency
    gc.collect()

    graph_directory = data["directory"] / "graph"
    fixed = {
        name: F.normalize_laplacian(
            sparse.load_npz(graph_directory / filename),
            p,
        )
        for name, filename in [
            ("coexpr", "L_coexpr.npz"),
            ("meth", "L_meth_expr.npz"),
            ("cnv", "L_cnv.npz"),
        ]
    }

    print(f"[{cancer}] caching graph-independent RF branch", flush=True)
    context = stage1_context(expression_train, residual_train)
    baseline_top = stage1_top_cached(
        expression_train,
        residual_train,
        zero,
        context,
    )
    baseline_cindex = F.stage2_oof_ci(
        expression_train[:, baseline_top],
        clinical_train,
        survival_train,
    )
    baseline_test_cindex, baseline_selected = fit_and_test_top(
        expression_train,
        expression_test,
        time_values[train],
        event[train],
        time_values[test],
        event[test],
        data["genes"],
        baseline_top,
    )

    reconstructed_result = evaluate_graph_set(
        "reconstructed",
        reconstructed,
        expression_train,
        expression_test,
        residual_train,
        survival_train,
        clinical_train,
        time_values[train],
        event[train],
        time_values[test],
        event[test],
        data["genes"],
        baseline_top,
        baseline_cindex,
        context,
    )
    fixed_result = evaluate_graph_set(
        "fixed",
        fixed,
        expression_train,
        expression_test,
        residual_train,
        survival_train,
        clinical_train,
        time_values[train],
        event[train],
        time_values[test],
        event[test],
        data["genes"],
        baseline_top,
        baseline_cindex,
        context,
    )

    for margin in GATE_MARGINS:
        key = f"{margin:.3f}"
        reconstructed_top = set(reconstructed_result["margins"][key]["top20"])
        fixed_top = set(fixed_result["margins"][key]["top20"])
        overlap = len(reconstructed_top & fixed_top)
        reconstructed_result["margins"][key]["fixed_overlap_count"] = overlap
        reconstructed_result["margins"][key]["fixed_jaccard"] = overlap / (40 - overlap)
        reconstructed_result["margins"][key]["fixed_minus_reconstructed_cindex"] = (
            fixed_result["margins"][key]["test_cindex"]
            - reconstructed_result["margins"][key]["test_cindex"]
        )

    return {
        "cancer": cancer,
        "split_seed": int(args.split_seed),
        "algorithm_seed": int(F.SEED),
        "n": len(index),
        "train_n": len(train),
        "test_n": len(test),
        "test_events": int(event[test].sum()),
        "test_size": A.TEST_SIZE,
        "configuration": {
            "bootstrap_B": F.N_BOOTSTRAP,
            "stage1_rf_trees": F.STAGE1_RF_TREES,
            "stage1_pgd_iterations": F.PGD_MAX_ITER,
            "stability_pgd_iterations": 300,
            "rf_jobs": F.RF_JOBS,
            "layer_workers": LAYER_WORKERS,
            "top_k": F.TOP_K,
            "gate_margins": list(GATE_MARGINS),
            "rf_cache_exact_reuse": True,
        },
        "baseline": {
            "oof_cindex": baseline_cindex,
            "test_cindex": baseline_test_cindex,
            "top20": baseline_selected,
        },
        "graph_metadata": {
            "coexpr": coexpr_meta,
            "meth": meth_meta,
            "cnv": cnv_meta,
        },
        "reconstructed": reconstructed_result,
        "fixed": fixed_result,
        "runtime_seconds": time.time() - started,
    }


def flatten(result: dict) -> list[dict]:
    rows = []
    for margin_key, reconstructed in result["reconstructed"]["margins"].items():
        fixed = result["fixed"]["margins"][margin_key]
        rows.append(
            {
                "Cancer": result["cancer"],
                "Split seed": result["split_seed"],
                "Train n": result["train_n"],
                "Test n": result["test_n"],
                "Test events": result["test_events"],
                "Gate margin": float(margin_key),
                "No-graph test C-index": result["baseline"]["test_cindex"],
                "Reconstructed C-index": reconstructed["test_cindex"],
                "Fixed C-index": fixed["test_cindex"],
                "Fixed minus reconstructed": reconstructed[
                    "fixed_minus_reconstructed_cindex"
                ],
                "Top-20 overlap": reconstructed["fixed_overlap_count"],
                "Top-20 Jaccard": reconstructed["fixed_jaccard"],
                "Reconstructed mode": reconstructed["mode"],
                "Fixed mode": fixed["mode"],
                "Reconstructed weights": json.dumps(
                    reconstructed["weights"],
                    sort_keys=True,
                ),
                "Fixed weights": json.dumps(
                    fixed["weights"],
                    sort_keys=True,
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--test-size", default=0.30, type=float)
    parser.add_argument("--bootstrap", default=10, type=int)
    parser.add_argument("--rf-trees", default=100, type=int)
    parser.add_argument("--pgd-iterations", default=300, type=int)
    parser.add_argument("--rf-jobs", default=2, type=int)
    parser.add_argument("--layer-workers", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    outdir = Path(args.outdir) / f"{args.cancer}_seed{args.split_seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "CMPB_REPEATED_TRAIN_ONLY_GRAPH_AUDIT.json"
    csv_path = outdir / "CMPB_REPEATED_TRAIN_ONLY_GRAPH_AUDIT.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(flatten(result)).to_csv(csv_path, index=False)
    print(f"saved {json_path}", flush=True)
    print(f"saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
