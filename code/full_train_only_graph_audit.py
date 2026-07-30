"""Representative from-scratch train-only graph reconstruction audit.

For LUAD, COAD, and LIHC, this audit:
1. creates a stratified development/test split;
2. recomputes expression scaling and Null-Cox deviance residuals in development;
3. reconstructs all three omics relation graphs from development samples only;
4. learns routing weights and selects Top-20 genes in development only; and
5. evaluates a reduced-space ridge-Cox score on the untouched test split.

The fixed-graph contrast uses the same split, train-only response, routing,
selection, and test evaluation, but substitutes the submission's precomputed
full-cohort graph structures. The contrast therefore isolates graph-construction
leakage rather than weight-learning leakage.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import sparse
from scipy.stats import rankdata, t as student_t
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

# graph_lasso_pgd binds its default iteration cap when the core module is
# imported, so propagate the audit setting before that import.
os.environ.setdefault(
    "FINAL_PGD_MAX_ITER",
    os.environ.get("FULL_GRAPH_AUDIT_PGD_ITER", "300"),
)
import final_config_comparison as F


ROOT = Path(__file__).resolve().parents[1]

# Public-package paths can be overridden without editing source.
DATA = Path(os.environ.get("MKG_DATA_ROOT", ROOT / "data" / "processed_data"))
OUT = Path(
    os.environ.get(
        "MKG_FULL_GRAPH_AUDIT_OUTPUT",
        ROOT / "results" / "full_train_only_graph_audit",
    )
)
TABLES = Path(os.environ.get("MKG_TABLE_OUTPUT", ROOT / "results" / "source_tables"))

CANCERS = [
    cancer.strip()
    for cancer in os.environ.get("FULL_GRAPH_AUDIT_CANCERS", "LUAD,COAD,LIHC").split(",")
    if cancer.strip()
]
SEED = int(os.environ.get("FULL_GRAPH_AUDIT_SEED", "42"))
TEST_SIZE = float(os.environ.get("FULL_GRAPH_AUDIT_TEST_SIZE", "0.30"))
BLOCK = int(os.environ.get("FULL_GRAPH_AUDIT_BLOCK", "512"))
F.N_BOOTSTRAP = int(os.environ.get("FULL_GRAPH_AUDIT_BOOTSTRAP", "10"))
F.STAGE1_RF_TREES = int(os.environ.get("FULL_GRAPH_AUDIT_RF_TREES", "100"))
F.PGD_MAX_ITER = int(os.environ.get("FULL_GRAPH_AUDIT_PGD_ITER", "300"))
F.RF_JOBS = int(os.environ.get("FULL_GRAPH_AUDIT_JOBS", str(F.RF_JOBS)))

COEXPR_TAU = 0.30
COEXPR_POWER = 6
METH_TAU = 0.15
METH_FDR = 0.05
CNV_TAU = 0.25
CNV_POWER = 4
RIDGE_PENALIZER = 0.1


def fill_train_values(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mean = np.nanmean(train, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    train = np.where(np.isfinite(train), train, mean)
    test = np.where(np.isfinite(test), test, mean)
    return train, test


def standardize_from_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train, test = fill_train_values(train, test)
    mean = train.mean(axis=0, dtype=np.float64)
    sd = train.std(axis=0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    return ((train - mean) / sd).astype(np.float32), ((test - mean) / sd).astype(np.float32)


def null_cox_deviance(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    event_times = np.unique(time[event == 1])
    cumulative = 0.0
    increments: list[tuple[float, float]] = []
    for current in np.sort(event_times):
        at_risk = int(np.sum(time >= current))
        deaths = int(np.sum((time == current) & (event == 1)))
        if at_risk:
            cumulative += deaths / at_risk
        increments.append((current, cumulative))
    grid = np.asarray([value[0] for value in increments], dtype=float)
    hazard = np.asarray([value[1] for value in increments], dtype=float)
    position = np.searchsorted(grid, time, side="right") - 1
    h0 = np.where(position >= 0, hazard[np.maximum(position, 0)], 0.0)
    martingale = event - h0
    log_term = np.zeros_like(h0)
    observed = event == 1
    log_term[observed] = np.log(np.maximum(h0[observed], 1e-12))
    inside = np.maximum(-2.0 * (martingale + event * log_term), 0.0)
    return np.sign(martingale) * np.sqrt(inside)


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(matrix, dtype=np.float32)
    for column in range(matrix.shape[1]):
        ranked[:, column] = rankdata(matrix[:, column], method="average").astype(np.float32)
    ranked -= ranked.mean(axis=0, keepdims=True)
    sd = ranked.std(axis=0, ddof=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ranked /= sd
    return ranked


def adjacency_to_laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree, format="csr") - adjacency).astype(np.float32)


def symmetric_correlation_graph(
    matrix: np.ndarray, tau: float, power: int, block: int = BLOCK
) -> tuple[sparse.csr_matrix, dict]:
    n, p = matrix.shape
    standardized, _ = standardize_from_train(matrix, matrix)
    denominator = float(max(n - 1, 1))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, p, block):
        end = min(start + block, p)
        correlation = (standardized[:, start:end].T @ standardized) / denominator
        absolute = np.minimum(np.abs(correlation), 1.0)
        local_row, column = np.nonzero(absolute > tau)
        global_row = local_row + start
        keep = column > global_row
        if np.any(keep):
            global_row = global_row[keep].astype(np.int32)
            column = column[keep].astype(np.int32)
            weight = np.power(absolute[local_row[keep], column], power).astype(np.float32)
            rows.append(global_row)
            cols.append(column)
            values.append(weight)
    if rows:
        row = np.concatenate(rows)
        column = np.concatenate(cols)
        value = np.concatenate(values)
        upper = sparse.coo_matrix((value, (row, column)), shape=(p, p), dtype=np.float32)
        adjacency = (upper + upper.T).tocsr()
    else:
        adjacency = sparse.csr_matrix((p, p), dtype=np.float32)
    edges = adjacency.nnz // 2
    return adjacency, {
        "n_samples": n,
        "n_genes": p,
        "undirected_edges": int(edges),
        "density": float(edges / (p * (p - 1) / 2)),
        "tau": tau,
        "power": power,
    }


def pvalues_from_abs_rho(rho_abs: np.ndarray, n: int) -> np.ndarray:
    rho_abs = np.minimum(rho_abs, 0.999999)
    statistic = rho_abs * np.sqrt((n - 2) / np.maximum(1.0 - rho_abs**2, 1e-12))
    return (2.0 * student_t.sf(statistic, n - 2)).astype(np.float32)


def bh_cutoff(selected_pvalues: np.ndarray, total_tests: int, alpha: float) -> float:
    if selected_pvalues.size == 0:
        return 0.0
    ordered = np.sort(selected_pvalues.astype(np.float64))
    rank = np.arange(1, len(ordered) + 1, dtype=float)
    accepted = ordered <= alpha * rank / float(total_tests)
    if not np.any(accepted):
        return 0.0
    return float(ordered[np.flatnonzero(accepted)[-1]])


def methylation_expression_graph(
    methylation: np.ndarray, expression: np.ndarray, block: int = BLOCK
) -> tuple[sparse.csr_matrix, dict]:
    n, p = expression.shape
    methylation, _ = fill_train_values(methylation, methylation)
    expression, _ = fill_train_values(expression, expression)
    methylation_rank = rank_columns(methylation)
    expression_rank = rank_columns(expression)
    denominator = float(max(n - 1, 1))

    selected: list[np.ndarray] = []
    for start in range(0, p, block):
        end = min(start + block, p)
        rho = (methylation_rank[:, start:end].T @ expression_rank) / denominator
        absolute = np.minimum(np.abs(rho), 1.0)
        local_row, column = np.nonzero(absolute > METH_TAU)
        global_row = local_row + start
        keep = column > global_row
        if np.any(keep):
            selected.append(pvalues_from_abs_rho(absolute[local_row[keep], column[keep]], n))
    selected_pvalues = np.concatenate(selected) if selected else np.empty(0, dtype=np.float32)
    total_tests = p * (p - 1) // 2
    cutoff = bh_cutoff(selected_pvalues, total_tests, METH_FDR)
    del selected, selected_pvalues
    gc.collect()

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, p, block):
        end = min(start + block, p)
        rho = (methylation_rank[:, start:end].T @ expression_rank) / denominator
        absolute = np.minimum(np.abs(rho), 1.0)
        local_row, column = np.nonzero(absolute > METH_TAU)
        global_row = local_row + start
        keep = column > global_row
        if not np.any(keep):
            continue
        local_row = local_row[keep]
        column = column[keep]
        global_row = global_row[keep]
        pvalue = pvalues_from_abs_rho(absolute[local_row, column], n)
        keep_fdr = pvalue <= cutoff
        if np.any(keep_fdr):
            rows.append(global_row[keep_fdr].astype(np.int32))
            cols.append(column[keep_fdr].astype(np.int32))
            values.append(
                (absolute[local_row[keep_fdr], column[keep_fdr]] - METH_TAU).astype(
                    np.float32
                )
            )
    if rows:
        row = np.concatenate(rows)
        column = np.concatenate(cols)
        value = np.concatenate(values)
        upper = sparse.coo_matrix((value, (row, column)), shape=(p, p), dtype=np.float32)
        adjacency = (upper + upper.T).tocsr()
    else:
        adjacency = sparse.csr_matrix((p, p), dtype=np.float32)
    edges = adjacency.nnz // 2
    return adjacency, {
        "n_samples": n,
        "n_genes": p,
        "undirected_edges": int(edges),
        "density": float(edges / (p * (p - 1) / 2)),
        "tau": METH_TAU,
        "fdr": METH_FDR,
        "bh_p_cutoff": cutoff,
        "directional_convention": "upper-index meth_i versus expr_j, then mirrored",
    }


def learn_weights(
    expression: np.ndarray,
    residual: np.ndarray,
    survival,
    clinical: pd.DataFrame,
    laplacians: dict[str, sparse.csr_matrix],
    zero: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, dict, str, dict, dict, float]:
    stability = {
        name: F.bootstrap_stability(expression, residual, laplacian)
        for name, laplacian in laplacians.items()
    }
    baseline_top, _, _, _, _ = F.stage1_select(expression, residual, zero)
    baseline_ci = F.stage2_oof_ci(expression[:, baseline_top], clinical, survival)
    delta: dict[str, float] = {}
    for name, laplacian in laplacians.items():
        top, _, _, _, _ = F.stage1_select(expression, residual, laplacian)
        delta[name] = (
            F.stage2_oof_ci(expression[:, top], clinical, survival) - baseline_ci
        )
    utility = {
        name: stability[name] * max(delta[name], 0.0) for name in laplacians
    }
    total = sum(utility.values())
    if total < 1e-15:
        weight = {name: 0.0 for name in laplacians}
        mode = "reject_all_graphs"
    else:
        weight = {name: utility[name] / total for name in laplacians}
        mode = "dual_driven"
    combined = sum(weight[name] * laplacians[name] for name in laplacians)
    return combined, weight, mode, stability, delta, baseline_ci


def fit_and_test(
    expression_train: np.ndarray,
    expression_test: np.ndarray,
    residual_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    names: np.ndarray,
    laplacian: sparse.csr_matrix,
) -> tuple[float, list[str]]:
    top, _, _, _, _ = F.stage1_select(expression_train, residual_train, laplacian)
    selected = names[top].tolist()
    train_frame = pd.DataFrame(expression_train[:, top], columns=selected)
    train_frame["T"] = time_train
    train_frame["E"] = event_train
    model = CoxPHFitter(penalizer=RIDGE_PENALIZER, l1_ratio=0.0)
    model.fit(train_frame, duration_col="T", event_col="E")
    test_frame = pd.DataFrame(expression_test[:, top], columns=selected)
    risk = test_frame[model.params_.index].to_numpy() @ model.params_.to_numpy()
    value = concordance_index(time_test, -risk, event_test)
    return float(value), selected


def load_cancer(cancer: str) -> dict:
    directory = DATA / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    methylation = pd.read_csv(directory / "meth_gene_level.tsv", sep="\t", index_col=0)
    copy_number = pd.read_csv(directory / "cnv_aligned.tsv", sep="\t", index_col=0)
    survival = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    clinical_path = directory / "clinical_covariates.tsv"
    clinical = (
        pd.read_csv(clinical_path, sep="\t", index_col=0)
        if clinical_path.exists()
        else pd.DataFrame(index=expression.index)
    )
    common = sorted(
        set(expression.index)
        & set(methylation.index)
        & set(copy_number.index)
        & set(survival.index)
        & set(clinical.index)
    )
    genes = expression.columns.astype(str)
    methylation = methylation.loc[common, genes]
    copy_number = copy_number.loc[common, genes]
    return {
        "directory": directory,
        "expression": expression.loc[common, genes],
        "methylation": methylation,
        "copy_number": copy_number,
        "time": survival.loc[common, "OS_time"].to_numpy(dtype=float),
        "event": survival.loc[common, "OS"].to_numpy(dtype=int),
        "clinical": clinical.loc[common],
        "genes": genes.to_numpy(),
    }


def run_cancer(cancer: str) -> dict:
    print(f"\n[{cancer}] loading aligned matrices", flush=True)
    data = load_cancer(cancer)
    time = data["time"]
    event = data["event"]
    index = np.arange(len(time))
    train, test = train_test_split(
        index, test_size=TEST_SIZE, random_state=SEED, stratify=event
    )

    expression_train_raw = data["expression"].to_numpy(dtype=np.float32)[train]
    expression_test_raw = data["expression"].to_numpy(dtype=np.float32)[test]
    expression_train, expression_test = standardize_from_train(
        expression_train_raw, expression_test_raw
    )
    methylation_train, _ = fill_train_values(
        data["methylation"].to_numpy(dtype=np.float32)[train],
        data["methylation"].to_numpy(dtype=np.float32)[test],
    )
    copy_train, _ = fill_train_values(
        data["copy_number"].to_numpy(dtype=np.float32)[train],
        data["copy_number"].to_numpy(dtype=np.float32)[test],
    )
    residual_train = null_cox_deviance(time[train], event[train])
    survival_train = Surv.from_arrays(event[train].astype(bool), time[train])
    clinical_train = data["clinical"].iloc[train]
    p = expression_train.shape[1]
    zero = sparse.csr_matrix((p, p), dtype=np.float32)

    print(f"[{cancer}] reconstructing co-expression graph", flush=True)
    coexpr_a, coexpr_meta = symmetric_correlation_graph(
        expression_train, COEXPR_TAU, COEXPR_POWER
    )
    print(f"[{cancer}] reconstructing methylation-expression graph", flush=True)
    meth_a, meth_meta = methylation_expression_graph(
        methylation_train, expression_train
    )
    print(f"[{cancer}] reconstructing CNV graph", flush=True)
    cnv_a, cnv_meta = symmetric_correlation_graph(copy_train, CNV_TAU, CNV_POWER)

    reconstructed = {
        "coexpr": F.normalize_laplacian(adjacency_to_laplacian(coexpr_a), p),
        "meth": F.normalize_laplacian(adjacency_to_laplacian(meth_a), p),
        "cnv": F.normalize_laplacian(adjacency_to_laplacian(cnv_a), p),
    }
    del coexpr_a, meth_a, cnv_a
    gc.collect()

    print(f"[{cancer}] learning weights on reconstructed train-only graphs", flush=True)
    (
        reconstructed_combined,
        reconstructed_weight,
        reconstructed_mode,
        reconstructed_stability,
        reconstructed_delta,
        reconstructed_baseline,
    ) = learn_weights(
        expression_train,
        residual_train,
        survival_train,
        clinical_train,
        reconstructed,
        zero,
    )
    reconstructed_ci, reconstructed_top = fit_and_test(
        expression_train,
        expression_test,
        residual_train,
        time[train],
        event[train],
        time[test],
        event[test],
        data["genes"],
        reconstructed_combined,
    )
    del reconstructed, reconstructed_combined
    gc.collect()

    print(f"[{cancer}] learning weights on precomputed full-cohort graphs", flush=True)
    graph_directory = data["directory"] / "graph"
    fixed = {
        name: F.normalize_laplacian(sparse.load_npz(graph_directory / filename), p)
        for name, filename in [
            ("coexpr", "L_coexpr.npz"),
            ("meth", "L_meth_expr.npz"),
            ("cnv", "L_cnv.npz"),
        ]
    }
    (
        fixed_combined,
        fixed_weight,
        fixed_mode,
        fixed_stability,
        fixed_delta,
        fixed_baseline,
    ) = learn_weights(
        expression_train,
        residual_train,
        survival_train,
        clinical_train,
        fixed,
        zero,
    )
    fixed_ci, fixed_top = fit_and_test(
        expression_train,
        expression_test,
        residual_train,
        time[train],
        event[train],
        time[test],
        event[test],
        data["genes"],
        fixed_combined,
    )
    overlap = len(set(reconstructed_top) & set(fixed_top))
    result = {
        "Cancer": cancer,
        "n": len(time),
        "Train n": len(train),
        "Test n": len(test),
        "Test events": int(event[test].sum()),
        "Bootstrap B": F.N_BOOTSTRAP,
        "RF trees": F.STAGE1_RF_TREES,
        "PGD iterations": F.PGD_MAX_ITER,
        "Reconstructed graph C-index": reconstructed_ci,
        "Fixed full-graph C-index": fixed_ci,
        "Fixed minus reconstructed": fixed_ci - reconstructed_ci,
        "Top-20 overlap count": overlap,
        "Top-20 Jaccard": overlap / (40 - overlap),
        "Reconstructed mode": reconstructed_mode,
        "Fixed graph mode": fixed_mode,
        "Reconstructed weights": reconstructed_weight,
        "Fixed graph weights": fixed_weight,
        "Reconstructed stability": reconstructed_stability,
        "Fixed graph stability": fixed_stability,
        "Reconstructed delta": reconstructed_delta,
        "Fixed graph delta": fixed_delta,
        "Reconstructed baseline C-index": reconstructed_baseline,
        "Fixed graph baseline C-index": fixed_baseline,
        "Reconstructed Top-20": reconstructed_top,
        "Fixed graph Top-20": fixed_top,
        "Graph metadata": {
            "coexpr": coexpr_meta,
            "meth": meth_meta,
            "cnv": cnv_meta,
        },
    }
    print(
        f"[{cancer}] reconstructed={reconstructed_ci:.4f}, "
        f"fixed={fixed_ci:.4f}, difference={fixed_ci - reconstructed_ci:+.4f}",
        flush=True,
    )
    return result


def flatten(result: dict) -> dict:
    return {
        "Cancer": result["Cancer"],
        "n": result["n"],
        "Train n": result["Train n"],
        "Test n": result["Test n"],
        "Test events": result["Test events"],
        "Bootstrap B": result["Bootstrap B"],
        "Reconstructed graph C-index": result["Reconstructed graph C-index"],
        "Fixed full-graph C-index": result["Fixed full-graph C-index"],
        "Fixed minus reconstructed": result["Fixed minus reconstructed"],
        "Top-20 overlap count": result["Top-20 overlap count"],
        "Top-20 Jaccard": result["Top-20 Jaccard"],
        "Reconstructed mode": result["Reconstructed mode"],
        "Fixed graph mode": result["Fixed graph mode"],
        "Reconstructed weights": json.dumps(result["Reconstructed weights"], sort_keys=True),
        "Fixed graph weights": json.dumps(result["Fixed graph weights"], sort_keys=True),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    output_json = OUT / "MKG_FULL_TRAIN_ONLY_GRAPH_AUDIT.json"
    results: dict[str, dict] = {}
    if output_json.exists() and os.environ.get("FULL_GRAPH_AUDIT_RESUME", "1") == "1":
        results = json.loads(output_json.read_text(encoding="utf-8"))
    for cancer in CANCERS:
        if cancer in results:
            print(f"[{cancer}] existing result retained", flush=True)
            continue
        results[cancer] = run_cancer(cancer)
        output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    frame = pd.DataFrame([flatten(results[cancer]) for cancer in CANCERS])
    frame.to_csv(OUT / "MKG_FULL_TRAIN_ONLY_GRAPH_AUDIT.csv", index=False)
    frame.to_csv(TABLES / "TableS_full_train_only_graph_audit.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

