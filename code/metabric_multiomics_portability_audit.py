"""Independent end-to-end METABRIC multi-omics portability audit for MKG.

For each prespecified split, every data-dependent operation is confined to the
training partition: imputation/scaling, construction of all three graph layers,
route selection, Top-20 selection, and reduced Cox fitting.  The untouched test
partition is used once to compare MKG with the zero-graph reference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import sparse
from sklearn.model_selection import train_test_split
from sksurv.util import Surv


HERE = Path(__file__).resolve().parent
if (HERE / "final_config_comparison.py").is_file() and (
    HERE / "full_train_only_graph_audit.py"
).is_file():
    # Public repository and supplementary-source archive: dependencies are
    # distributed beside this script.
    sys.path.insert(0, str(HERE))
else:
    # Internal project layout.
    CORE = HERE.parent / "01_核心模型与锁定流程"
    AUDIT = HERE.parent / "03_统计与泄漏审计"
    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(AUDIT))

import final_config_comparison as F  # noqa: E402
import full_train_only_graph_audit as A  # noqa: E402


DEFAULT_SEEDS = (11, 29, 42, 71, 113)
RIDGE_PENALIZER = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_reduced_cox(
    expression_train: np.ndarray,
    expression_test: np.ndarray,
    residual_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    genes: np.ndarray,
    laplacian: sparse.csr_matrix,
) -> tuple[float, list[str], list[float]]:
    top, _, scores, _, _ = F.stage1_select(
        expression_train, residual_train, laplacian
    )
    selected = genes[top].tolist()
    train_frame = pd.DataFrame(expression_train[:, top], columns=selected)
    train_frame["T"] = time_train
    train_frame["E"] = event_train
    model = CoxPHFitter(penalizer=RIDGE_PENALIZER, l1_ratio=0.0)
    model.fit(train_frame, duration_col="T", event_col="E")
    test_frame = pd.DataFrame(expression_test[:, top], columns=selected)
    coefficients = model.params_.reindex(selected).to_numpy(dtype=float)
    risk = test_frame[selected].to_numpy(dtype=float) @ coefficients
    value = concordance_index(time_test, -risk, event_test)
    return float(value), selected, scores[top].astype(float).tolist()


def load_metabric(directory: Path) -> dict:
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    methylation = pd.read_csv(
        directory / "meth_gene_level.tsv", sep="\t", index_col=0
    )
    copy_number = pd.read_csv(directory / "cnv_aligned.tsv", sep="\t", index_col=0)
    survival = pd.read_csv(
        directory / "deviance_residuals.tsv", sep="\t", index_col=0
    )
    clinical = pd.read_csv(
        directory / "clinical_covariates.tsv", sep="\t", index_col=0
    )
    common_samples = sorted(
        set(expression.index)
        & set(methylation.index)
        & set(copy_number.index)
        & set(survival.index)
        & set(clinical.index)
    )
    common_genes = sorted(
        set(expression.columns)
        & set(methylation.columns)
        & set(copy_number.columns)
    )
    survival = survival.loc[common_samples]
    valid = (
        pd.to_numeric(survival["OS_time"], errors="coerce").notna()
        & (pd.to_numeric(survival["OS_time"], errors="coerce") > 0)
        & pd.to_numeric(survival["OS"], errors="coerce").isin([0, 1])
    )
    common_samples = survival.index[valid].tolist()
    return {
        "samples": np.asarray(common_samples),
        "genes": np.asarray(common_genes),
        "expression": expression.loc[common_samples, common_genes].to_numpy(
            dtype=np.float32
        ),
        "methylation": methylation.loc[common_samples, common_genes].to_numpy(
            dtype=np.float32
        ),
        "copy_number": copy_number.loc[common_samples, common_genes].to_numpy(
            dtype=np.float32
        ),
        "time": survival.loc[common_samples, "OS_time"].to_numpy(dtype=float),
        "event": survival.loc[common_samples, "OS"].to_numpy(dtype=int),
        "clinical": clinical.loc[common_samples],
    }


def run_split(data: dict, seed: int, test_size: float) -> dict:
    time = data["time"]
    event = data["event"]
    indices = np.arange(len(time))
    train, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=event
    )

    expression_train, expression_test = A.standardize_from_train(
        data["expression"][train], data["expression"][test]
    )
    methylation_train, _ = A.fill_train_values(
        data["methylation"][train], data["methylation"][test]
    )
    copy_train, _ = A.fill_train_values(
        data["copy_number"][train], data["copy_number"][test]
    )
    residual_train = A.null_cox_deviance(time[train], event[train])
    survival_train = Surv.from_arrays(event[train].astype(bool), time[train])
    clinical_train = data["clinical"].iloc[train].copy()
    clinical_train.index = np.arange(len(train))
    p = expression_train.shape[1]
    zero = sparse.csr_matrix((p, p), dtype=np.float32)

    print(f"[METABRIC seed={seed}] co-expression graph", flush=True)
    coexpr_a, coexpr_meta = A.symmetric_correlation_graph(
        expression_train, A.COEXPR_TAU, A.COEXPR_POWER
    )
    print(f"[METABRIC seed={seed}] methylation-expression graph", flush=True)
    meth_a, meth_meta = A.methylation_expression_graph(
        methylation_train, expression_train
    )
    print(f"[METABRIC seed={seed}] copy-number graph", flush=True)
    cnv_a, cnv_meta = A.symmetric_correlation_graph(
        copy_train, A.CNV_TAU, A.CNV_POWER
    )
    laplacians = {
        "coexpr": F.normalize_laplacian(A.adjacency_to_laplacian(coexpr_a), p),
        "meth": F.normalize_laplacian(A.adjacency_to_laplacian(meth_a), p),
        "cnv": F.normalize_laplacian(A.adjacency_to_laplacian(cnv_a), p),
    }
    del coexpr_a, meth_a, cnv_a
    gc.collect()

    print(f"[METABRIC seed={seed}] train-only routing", flush=True)
    (
        combined,
        weights,
        mode,
        stability,
        delta,
        baseline_oof,
    ) = A.learn_weights(
        expression_train,
        residual_train,
        survival_train,
        clinical_train,
        laplacians,
        zero,
    )
    mkg_ci, mkg_top, mkg_scores = fit_reduced_cox(
        expression_train,
        expression_test,
        residual_train,
        time[train],
        event[train],
        time[test],
        event[test],
        data["genes"],
        combined,
    )
    zero_ci, zero_top, zero_scores = fit_reduced_cox(
        expression_train,
        expression_test,
        residual_train,
        time[train],
        event[train],
        time[test],
        event[test],
        data["genes"],
        zero,
    )
    result = {
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "events_train": int(event[train].sum()),
        "events_test": int(event[test].sum()),
        "route_mode": mode,
        "weights": weights,
        "stability": stability,
        "routing_oof_delta": delta,
        "zero_graph_oof_cindex": baseline_oof,
        "test_cindex_mkg": mkg_ci,
        "test_cindex_zero_graph": zero_ci,
        "test_delta_mkg_minus_zero": mkg_ci - zero_ci,
        "mkg_top20": mkg_top,
        "mkg_top20_scores": mkg_scores,
        "zero_top20": zero_top,
        "zero_top20_scores": zero_scores,
        "graph_metadata": {
            "coexpr": coexpr_meta,
            "meth": meth_meta,
            "cnv": cnv_meta,
        },
    }
    del laplacians, combined
    gc.collect()
    print(
        f"[METABRIC seed={seed}] MKG={mkg_ci:.4f}; "
        f"zero={zero_ci:.4f}; delta={mkg_ci-zero_ci:+.4f}; route={weights}",
        flush=True,
    )
    return result


def rbo_from_gene_lists(left: list[str], right: list[str]) -> float:
    return float(F.rbo_score(left, right))


def summarize(data: dict, results: list[dict]) -> dict:
    mkg = np.asarray([row["test_cindex_mkg"] for row in results], dtype=float)
    zero = np.asarray(
        [row["test_cindex_zero_graph"] for row in results], dtype=float
    )
    delta = mkg - zero
    rbo = [
        rbo_from_gene_lists(results[i]["mkg_top20"], results[j]["mkg_top20"])
        for i, j in combinations(range(len(results)), 2)
    ]
    jaccard = []
    for i, j in combinations(range(len(results)), 2):
        left = set(results[i]["mkg_top20"])
        right = set(results[j]["mkg_top20"])
        jaccard.append(len(left & right) / len(left | right))
    dominant = []
    for row in results:
        if row["route_mode"] == "reject_all_graphs":
            dominant.append("none")
        else:
            dominant.append(max(row["weights"], key=row["weights"].get))
    return {
        "cohort": "METABRIC",
        "role": "independent end-to-end multi-omics portability audit",
        "n": int(len(data["time"])),
        "events": int(data["event"].sum()),
        "genes": int(len(data["genes"])),
        "prespecified_seeds": [row["seed"] for row in results],
        "test_fraction": float(results[0]["n_test"] / len(data["time"])),
        "test_cindex_mkg_mean": float(mkg.mean()),
        "test_cindex_mkg_range": [float(mkg.min()), float(mkg.max())],
        "test_cindex_zero_graph_mean": float(zero.mean()),
        "test_cindex_zero_graph_range": [float(zero.min()), float(zero.max())],
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_range": [float(delta.min()), float(delta.max())],
        "mkg_better_split_count": int((delta > 0).sum()),
        "mean_pairwise_top20_rbo": float(np.mean(rbo)),
        "mean_pairwise_top20_jaccard": float(np.mean(jaccard)),
        "dominant_route_counts": {
            name: dominant.count(name) for name in sorted(set(dominant))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--pgd-iterations", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    F.N_BOOTSTRAP = args.bootstrap
    F.STAGE1_RF_TREES = args.rf_trees
    F.PGD_MAX_ITER = args.pgd_iterations
    F.RF_JOBS = args.jobs
    A.BLOCK = 512

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "METABRIC_MULTIOMICS_PORTABILITY_AUDIT.json"
    payload = {
        "configuration": {
            "test_size": args.test_size,
            "bootstrap": args.bootstrap,
            "rf_trees": args.rf_trees,
            "pgd_iterations": args.pgd_iterations,
            "jobs": args.jobs,
            "thresholds": {
                "coexpression_tau": A.COEXPR_TAU,
                "coexpression_power": A.COEXPR_POWER,
                "methylation_tau": A.METH_TAU,
                "methylation_fdr": A.METH_FDR,
                "copy_number_tau": A.CNV_TAU,
                "copy_number_power": A.CNV_POWER,
            },
        },
        "splits": [],
    }
    if args.resume and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    data_directory = args.data.resolve()
    preparation_manifest = data_directory / "METABRIC_PREPARATION_MANIFEST.json"
    payload["input_manifest"] = {
        "path": preparation_manifest.name,
        "sha256": sha256(preparation_manifest),
    }
    data = load_metabric(data_directory)
    completed = {row["seed"] for row in payload["splits"]}
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        if seed in completed:
            print(f"[METABRIC seed={seed}] retained from existing result", flush=True)
            continue
        payload["splits"].append(run_split(data, seed, args.test_size))
        payload["summary"] = summarize(data, payload["splits"])
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["splits"] = sorted(payload["splits"], key=lambda row: seeds.index(row["seed"]))
    payload["summary"] = summarize(data, payload["splits"])
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for row in payload["splits"]:
        rows.append(
            {
                "Seed": row["seed"],
                "Train n": row["n_train"],
                "Test n": row["n_test"],
                "Test events": row["events_test"],
                "Route mode": row["route_mode"],
                "Coexpression stability": row["stability"]["coexpr"],
                "Methylation stability": row["stability"]["meth"],
                "Copy-number stability": row["stability"]["cnv"],
                "Coexpression OOF delta": row["routing_oof_delta"]["coexpr"],
                "Methylation OOF delta": row["routing_oof_delta"]["meth"],
                "Copy-number OOF delta": row["routing_oof_delta"]["cnv"],
                "Coexpression weight": row["weights"]["coexpr"],
                "Methylation weight": row["weights"]["meth"],
                "Copy-number weight": row["weights"]["cnv"],
                "MKG test C-index": row["test_cindex_mkg"],
                "Zero-graph test C-index": row["test_cindex_zero_graph"],
                "MKG minus zero": row["test_delta_mkg_minus_zero"],
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "TableS_METABRIC_multiomics_portability.csv", index=False
    )
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
