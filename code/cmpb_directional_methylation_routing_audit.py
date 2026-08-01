"""Formal routing rerun with the directional-maximum methylation graph.

All tuning constants, 30 repeated 80% subsamples, five-fold development
routing score, random-forest size and frozen Top-20 scoring rule are inherited
unchanged from final_config_comparison.py.  Only the methylation graph is
replaced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sksurv.util import Surv

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
LAYER_FILES = {"coexpr": "L_coexpr.npz", "cnv": "L_cnv.npz"}


def laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    adjacency = adjacency.tocsr().astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree, format="csr") - adjacency).astype(np.float32)


def load_development(cancer: str):
    directory = F.ROOT / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    outcomes = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    clinical_path = directory / "clinical_covariates.tsv"
    clinical = pd.read_csv(clinical_path, sep="\t", index_col=0) if clinical_path.exists() else pd.DataFrame(index=expression.index)
    common = sorted(set(expression.index) & set(outcomes.index) & set(clinical.index))
    expression = expression.loc[common].fillna(expression.loc[common].mean()).fillna(0.0)
    X = expression.to_numpy(dtype=float)
    residual = outcomes.loc[common, "deviance_residual"].to_numpy(dtype=float)
    time_values = outcomes.loc[common, "OS_time"].to_numpy(dtype=float)
    events = outcomes.loc[common, "OS"].to_numpy(dtype=int)
    survival = Surv.from_arrays(events.astype(bool), time_values)
    return directory, expression, X, residual, time_values, events, survival, clinical.loc[common], np.asarray(expression.columns)


def external_results(directory: Path, signature: dict) -> dict:
    results = {}
    for cohort in F.EXTERNAL.get(directory.name, []):
        cohort_dir = directory / "external" / cohort
        if not (cohort_dir / "expr.tsv").exists():
            continue
        expression = pd.read_csv(cohort_dir / "expr.tsv", sep="\t", index_col=0)
        expression.columns = [str(column) for column in expression.columns]
        survival = pd.read_csv(cohort_dir / "survival.tsv", sep="\t")
        survival.columns = [column.lower().replace(".", "_") for column in survival.columns]
        event_column = next(column for column in survival.columns if column in {"os", "event", "status", "os_event"})
        time_column = next(column for column in survival.columns if column in {"os_time", "time", "survival_time"})
        survival = survival.set_index(survival.columns[0])
        common = sorted(set(expression.index) & set(survival.index))
        event = pd.to_numeric(survival.loc[common, event_column], errors="coerce").to_numpy()
        time_values = pd.to_numeric(survival.loc[common, time_column], errors="coerce").to_numpy()
        eligible = np.isfinite(event) & np.isfinite(time_values) & (time_values > 0)
        risk, matched = F.frozen_risk(
            expression.loc[np.asarray(common)[eligible]], signature["genes"], signature["directions"], signature["scores"]
        )
        results[cohort] = {**F.evaluate(risk, time_values[eligible], event[eligible].astype(int)), "n_matched": matched}
    return results


def run_cancer(cancer: str, graph_dir: Path) -> dict:
    started = time.time()
    directory, expression, X, residual, times, events, survival, clinical, names = load_development(cancer)
    p = X.shape[1]
    layers = {
        name: F.normalize_laplacian(sparse.load_npz(directory / "graph" / filename), p)
        for name, filename in LAYER_FILES.items()
    }
    corrected_adjacency = sparse.load_npz(graph_dir / f"{cancer}_A_meth_expr_directional_max.npz")
    layers["meth"] = F.normalize_laplacian(laplacian(corrected_adjacency), p)
    layers = {name: layers[name] for name in ("coexpr", "meth", "cnv")}
    zero = sparse.csr_matrix((p, p), dtype=np.float32)

    graph_stability = {name: F.bootstrap_stability(X, residual, layer) for name, layer in layers.items()}
    zero_top, _, _, _, _ = F.stage1_select(X, residual, zero)
    zero_score = F.stage2_oof_ci(X[:, zero_top], clinical, survival)
    candidate_score = {}
    deltas = {}
    for name, layer in layers.items():
        top, _, _, _, _ = F.stage1_select(X, residual, layer)
        candidate_score[name] = F.stage2_oof_ci(X[:, top], clinical, survival)
        deltas[name] = candidate_score[name] - zero_score
    utility = {name: graph_stability[name] * max(deltas[name], 0.0) for name in layers}
    total = sum(utility.values())
    weights = {name: (utility[name] / total if total > 1e-15 else 0.0) for name in layers}
    mode = "dual_driven" if total > 1e-15 else "reject_all_graphs"
    routed = sum(weights[name] * layers[name] for name in layers).tocsr()
    signature = F.sig_grsafs(X, residual, routed, names)
    training_risk, _ = F.frozen_risk(expression, signature["genes"], signature["directions"], signature["scores"])

    lock_path = F.OUT / f"final_config_comparison_{cancer}.json"
    old = json.loads(lock_path.read_text(encoding="utf-8"))[cancer]
    old_signature = old["method_signatures"]["GR-SAFS_v2"]
    old_genes, new_genes = set(old_signature["genes"]), set(signature["genes"])
    return {
        "Cancer": cancer,
        "variant": "directional-maximum methylation--expression graph",
        "sampling": "30 independent 80% subsamples without replacement",
        "zero_graph_development_routing_score": zero_score,
        "candidate_development_routing_scores": candidate_score,
        "candidate_minus_zero_deltas": deltas,
        "graph_stabilities": graph_stability,
        "utilities": utility,
        "route_mode": mode,
        "weights": weights,
        "signature": signature,
        "training": F.evaluate(training_risk, times, events),
        "external": external_results(directory, signature),
        "old_route_mode": old["weight_mode"],
        "old_weights": old["omics_weights"],
        "old_signature": old_signature,
        "top20_overlap_with_original": len(old_genes & new_genes),
        "top20_jaccard_with_original": len(old_genes & new_genes) / len(old_genes | new_genes),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }


def summary_row(result: dict) -> dict:
    external = result["external"]
    old_weights = result["old_weights"]
    return {
        "Cancer": result["Cancer"],
        "Corrected route": result["route_mode"],
        "Old coexpr/meth/cnv": "/".join(f"{old_weights.get(name, 0):.3f}" for name in ("coexpr", "meth", "cnv")),
        "Corrected coexpr/meth/cnv": "/".join(f"{result['weights'].get(name, 0):.3f}" for name in ("coexpr", "meth", "cnv")),
        "Top20 overlap": result["top20_overlap_with_original"],
        "Top20 Jaccard": result["top20_jaccard_with_original"],
        **{f"{cohort} C-index": values["c_index"] for cohort, values in external.items()},
        **{f"{cohort} matched": values["n_matched"] for cohort, values in external.items()},
        "Elapsed minutes": result["elapsed_minutes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cancer in args.cancers:
        print(f"[{cancer}] formal corrected-graph routing rerun", flush=True)
        result = run_cancer(cancer, args.graph_dir)
        (args.output_dir / f"{cancer}_DIRECTIONAL_METHYLATION_ROUTING.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    results = [
        json.loads((args.output_dir / f"{cancer}_DIRECTIONAL_METHYLATION_ROUTING.json").read_text(encoding="utf-8"))
        for cancer in CANCERS
        if (args.output_dir / f"{cancer}_DIRECTIONAL_METHYLATION_ROUTING.json").exists()
    ]
    pd.DataFrame(summary_row(result) for result in results).to_csv(
        args.output_dir / "CMPB_DIRECTIONAL_METHYLATION_ROUTING_SUMMARY.csv", index=False
    )


if __name__ == "__main__":
    main()
