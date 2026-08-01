"""High-accuracy Stage-1 routing sensitivity for the CMPB submission lock.

This audit isolates numerical optimization from resampling uncertainty.  It
keeps each cancer's locked 30-subsample layer-stability values, but recomputes
the zero-graph and candidate-graph Stage-1 rankings with converged FISTA fits
in every development fold.  Stage-2 routing scores, graph eligibility, the
final routed signature, and frozen external C-indices are then recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sksurv.util import Surv

try:
    import final_config_comparison as F
except ModuleNotFoundError:
    core_dir = Path(__file__).resolve().parents[1] / "01_核心模型与锁定流程"
    sys.path.insert(0, str(core_dir))
    import final_config_comparison as F

from cmpb_fista_component_audit import fista
from cmpb_pgd_convergence_audit import largest_eigenvalue


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
LAYER_FILES = {
    "coexpr": "L_coexpr.npz",
    "meth": "L_meth_expr.npz",
    "cnv": "L_cnv.npz",
}


def load_development(cancer: str):
    directory = F.ROOT / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    expression.columns = expression.columns.astype(str)
    outcomes = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    clinical_path = directory / "clinical_covariates.tsv"
    clinical = (
        pd.read_csv(clinical_path, sep="\t", index_col=0)
        if clinical_path.exists()
        else pd.DataFrame(index=expression.index)
    )
    common = sorted(set(expression.index) & set(outcomes.index) & set(clinical.index))
    expression = expression.loc[common].fillna(expression.loc[common].mean()).fillna(0.0)
    X = expression.to_numpy(dtype=float)
    residual = outcomes.loc[common, "deviance_residual"].to_numpy(dtype=float)
    times = outcomes.loc[common, "OS_time"].to_numpy(dtype=float)
    events = outcomes.loc[common, "OS"].to_numpy(dtype=int)
    survival = Surv.from_arrays(events.astype(bool), times)
    return directory, expression, X, residual, times, events, survival, clinical.loc[common]


def stage1_fista(X: np.ndarray, residual: np.ndarray, L: sparse.csr_matrix):
    n, p = X.shape
    predictions = np.zeros((n, 2), dtype=float)
    linear_importance = np.zeros(p, dtype=float)
    signed_linear = np.zeros(p, dtype=float)
    forest_importance = np.zeros(p, dtype=float)
    diagnostics = []
    folds = KFold(F.K_FOLDS, shuffle=True, random_state=F.SEED)
    for fold, (train, validation) in enumerate(folds.split(X), start=1):
        lip = largest_eigenvalue(X[train], L)
        beta, fit_diag = fista(X[train], residual[train], L, lip)
        if not fit_diag["FISTA converged"]:
            beta, fit_diag = fista(
                X[train], residual[train], L, lip, max_iter=10000, tolerance=1e-6
            )
        if not fit_diag["FISTA converged"]:
            raise RuntimeError(f"FISTA did not converge for fold {fold}")
        diagnostics.append({"fold": fold, "largest_eigenvalue": lip, **fit_diag})
        predictions[validation, 0] = X[validation] @ beta
        linear_importance += np.abs(beta)
        signed_linear += beta
        forest = RandomForestRegressor(
            n_estimators=F.STAGE1_RF_TREES,
            max_depth=5,
            min_samples_leaf=3,
            random_state=F.SEED,
            n_jobs=F.RF_JOBS,
        )
        forest.fit(X[train], residual[train])
        predictions[validation, 1] = forest.predict(X[validation])
        forest_importance += forest.feature_importances_
    weights = F.solve_qp_weights(predictions, residual)
    scores = (
        weights[0] * rankdata(linear_importance, method="min") / p
        + weights[1] * rankdata(forest_importance, method="min") / p
    )
    top = np.argsort(scores)[::-1][: F.TOP_K]
    return {
        "top": top,
        "directions": np.sign(signed_linear / F.K_FOLDS),
        "scores": scores,
        "qp_weights": weights,
        "fold_diagnostics": diagnostics,
    }


def signature_from_fit(fit: dict, names: np.ndarray) -> dict:
    top = fit["top"]
    return {
        "genes": names[top].tolist(),
        "directions": fit["directions"][top].tolist(),
        "scores": fit["scores"][top].tolist(),
        "w_qp": fit["qp_weights"].tolist(),
    }


def external_results(directory: Path, signature: dict) -> dict:
    results = {}
    for cohort in F.EXTERNAL.get(directory.name, []):
        cohort_dir = directory / "external" / cohort
        if not (cohort_dir / "expr.tsv").exists():
            continue
        expression = pd.read_csv(cohort_dir / "expr.tsv", sep="\t", index_col=0)
        expression.columns = expression.columns.astype(str)
        outcome = pd.read_csv(cohort_dir / "survival.tsv", sep="\t")
        outcome.columns = [column.lower().replace(".", "_") for column in outcome.columns]
        event_column = next(column for column in outcome.columns if column in {"os", "event", "status", "os_event"})
        time_column = next(column for column in outcome.columns if column in {"os_time", "time", "survival_time"})
        outcome = outcome.set_index(outcome.columns[0])
        common = sorted(set(expression.index) & set(outcome.index))
        event = pd.to_numeric(outcome.loc[common, event_column], errors="coerce").to_numpy()
        times = pd.to_numeric(outcome.loc[common, time_column], errors="coerce").to_numpy()
        eligible = np.isfinite(event) & np.isfinite(times) & (times > 0)
        risk, matched = F.frozen_risk(
            expression.loc[np.asarray(common)[eligible]],
            signature["genes"],
            signature["directions"],
            signature["scores"],
        )
        results[cohort] = {
            **F.evaluate(risk, times[eligible], event[eligible].astype(int)),
            "n_matched": matched,
        }
    return results


def run_cancer(cancer: str) -> dict:
    started = time.time()
    directory, expression, X, residual, times, events, survival, clinical = load_development(cancer)
    names = np.asarray(expression.columns)
    p = X.shape[1]
    layers = {
        name: F.normalize_laplacian(sparse.load_npz(directory / "graph" / filename), p)
        for name, filename in LAYER_FILES.items()
    }
    zero = sparse.csr_matrix((p, p), dtype=np.float32)
    lock = json.loads((F.OUT / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]

    fits = {"zero": stage1_fista(X, residual, zero)}
    development_scores = {
        "zero": F.stage2_oof_ci(X[:, fits["zero"]["top"]], clinical, survival)
    }
    for name, layer in layers.items():
        fits[name] = stage1_fista(X, residual, layer)
        development_scores[name] = F.stage2_oof_ci(X[:, fits[name]["top"]], clinical, survival)

    deltas = {name: development_scores[name] - development_scores["zero"] for name in layers}
    utilities = {
        name: float(lock["stabilities"][name]) * max(deltas[name], 0.0)
        for name in layers
    }
    total = sum(utilities.values())
    weights = {name: (utilities[name] / total if total > 1e-15 else 0.0) for name in layers}
    route_mode = "dual_driven" if total > 1e-15 else "reject_all_graphs"
    routed = sum(weights[name] * layers[name] for name in layers).tocsr()
    routed_fit = stage1_fista(X, residual, routed)
    signature = signature_from_fit(routed_fit, names)
    training_risk, _ = F.frozen_risk(expression, signature["genes"], signature["directions"], signature["scores"])
    old_signature = lock["method_signatures"]["GR-SAFS_v2"]
    old_genes, new_genes = set(old_signature["genes"]), set(signature["genes"])
    return {
        "Cancer": cancer,
        "estimand": "routing conditional on locked layer stability, with converged FISTA Stage-1 fits",
        "old_route_mode": lock["weight_mode"],
        "old_weights": lock["omics_weights"],
        "fista_route_mode": route_mode,
        "fista_weights": weights,
        "development_routing_scores": development_scores,
        "candidate_minus_zero_deltas": deltas,
        "utilities": utilities,
        "candidate_fit_diagnostics": {
            name: fit["fold_diagnostics"] for name, fit in fits.items()
        },
        "routed_fit_diagnostics": routed_fit["fold_diagnostics"],
        "signature": signature,
        "top20_overlap_with_lock": len(old_genes & new_genes),
        "top20_jaccard_with_lock": len(old_genes & new_genes) / len(old_genes | new_genes),
        "training": F.evaluate(training_risk, times, events),
        "external": external_results(directory, signature),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }


def summary_row(result: dict) -> dict:
    return {
        "Cancer": result["Cancer"],
        "Old route": result["old_route_mode"],
        "FISTA route": result["fista_route_mode"],
        "Old coexpr/meth/cnv": "/".join(f"{result['old_weights'].get(name, 0):.3f}" for name in LAYER_FILES),
        "FISTA coexpr/meth/cnv": "/".join(f"{result['fista_weights'].get(name, 0):.3f}" for name in LAYER_FILES),
        "Top20 overlap": result["top20_overlap_with_lock"],
        "Top20 Jaccard": result["top20_jaccard_with_lock"],
        **{f"{cohort} C-index": value["c_index"] for cohort, value in result["external"].items()},
        "Elapsed minutes": result["elapsed_minutes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--lock-output-root", type=Path)
    args = parser.parse_args()
    if args.data_root is not None:
        F.ROOT = args.data_root
    if args.lock_output_root is not None:
        F.OUT = args.lock_output_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cancer in args.cancers:
        print(f"[{cancer}] converged-FISTA routing sensitivity", flush=True)
        result = run_cancer(cancer)
        (args.output_dir / f"{cancer}_FISTA_ROUTING_SENSITIVITY.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("*_FISTA_ROUTING_SENSITIVITY.json"))
    ]
    pd.DataFrame(summary_row(result) for result in results).to_csv(
        args.output_dir / "CMPB_FISTA_ROUTING_SENSITIVITY_SUMMARY.csv", index=False
    )


if __name__ == "__main__":
    main()
