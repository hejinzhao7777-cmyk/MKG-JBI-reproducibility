"""Build and audit order-invariant methylation--expression graphs for all cancers.

The directed score for methylation gene i and expression gene j is
abs(Spearman(meth_i, expr_j)) - tau after the prespecified correlation and
directional BH filters.  An undirected edge uses max(D_ij, D_ji), which avoids
the row/column-order dependence of upper-triangle symmetrisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr, t as student_t


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
TAU = 0.15
FDR = 0.05
BLOCK = 512
MAX_CORRELATION_SAMPLE = 1_000_000


def read_genes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def standardised_ranks(values: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        ranked[:, column] = rankdata(values[:, column], method="average").astype(np.float32)
    ranked -= ranked.mean(axis=0, keepdims=True)
    scale = ranked.std(axis=0, ddof=1, keepdims=True)
    scale[scale == 0] = np.nan
    return ranked / scale


def p_values(abs_rho: np.ndarray, sample_size: int) -> np.ndarray:
    clipped = np.minimum(abs_rho, 0.999999)
    statistic = clipped * np.sqrt((sample_size - 2) / np.maximum(1.0 - clipped**2, 1e-12))
    return (2.0 * student_t.sf(statistic, sample_size - 2)).astype(np.float32)


def correlation_blocks(methylation: np.ndarray, expression: np.ndarray):
    denominator = float(methylation.shape[0] - 1)
    for start in range(0, methylation.shape[1], BLOCK):
        end = min(start + BLOCK, methylation.shape[1])
        yield start, end, (methylation[:, start:end].T @ expression / denominator).astype(np.float32)


def directional_bh_cutoff(methylation: np.ndarray, expression: np.ndarray) -> float:
    selected: list[np.ndarray] = []
    sample_size, genes = methylation.shape
    for start, end, rho in correlation_blocks(methylation, expression):
        absolute = np.abs(rho)
        absolute[np.arange(end - start), np.arange(start, end)] = 0.0
        mask = absolute > TAU
        if mask.any():
            selected.append(p_values(absolute[mask], sample_size))
    if not selected:
        return 0.0
    ordered = np.sort(np.concatenate(selected).astype(np.float64))
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    accepted = ordered <= FDR * ranks / float(genes * (genes - 1))
    return float(ordered[np.flatnonzero(accepted)[-1]]) if accepted.any() else 0.0


def build_graph(methylation: np.ndarray, expression: np.ndarray) -> tuple[sparse.csr_matrix, float, int]:
    sample_size, genes = methylation.shape
    cutoff = directional_bh_cutoff(methylation, expression)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    retained = 0
    for start, end, rho in correlation_blocks(methylation, expression):
        absolute = np.abs(rho)
        absolute[np.arange(end - start), np.arange(start, end)] = 0.0
        mask = absolute > TAU
        if cutoff > 0 and mask.any():
            local_p = np.ones_like(absolute, dtype=np.float32)
            local_p[mask] = p_values(absolute[mask], sample_size)
            mask &= local_p <= cutoff
        else:
            mask[:] = False
        if mask.any():
            local_rows, local_columns = np.nonzero(mask)
            rows.append((local_rows + start).astype(np.int32))
            columns.append(local_columns.astype(np.int32))
            weights.append((absolute[mask] - TAU).astype(np.float32))
            retained += int(mask.sum())
    if rows:
        directed = sparse.coo_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
            shape=(genes, genes), dtype=np.float32,
        ).tocsr()
    else:
        directed = sparse.csr_matrix((genes, genes), dtype=np.float32)
    adjacency = directed.maximum(directed.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency, cutoff, retained


def upper_edge_keys(adjacency: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    upper = sparse.triu(adjacency, k=1, format="coo")
    return upper.row.astype(np.int64) * adjacency.shape[0] + upper.col, upper.data


def compare_graphs(old: sparse.csr_matrix, new: sparse.csr_matrix, seed: int) -> dict:
    old_keys, _ = upper_edge_keys(old)
    new_keys, _ = upper_edge_keys(new)
    intersection = np.intersect1d(old_keys, new_keys, assume_unique=False).size
    union = np.union1d(old_keys, new_keys)
    rng = np.random.default_rng(seed)
    if union.size > MAX_CORRELATION_SAMPLE:
        union = rng.choice(union, MAX_CORRELATION_SAMPLE, replace=False)
    row, column = union // old.shape[0], union % old.shape[0]
    old_values = np.asarray(old[row, column]).ravel()
    new_values = np.asarray(new[row, column]).ravel()
    correlation = spearmanr(old_values, new_values).correlation if union.size > 2 else np.nan
    difference = (new - old).tocsr()
    old_norm = float(np.sqrt(old.multiply(old).sum()))
    return {
        "edge_intersection": int(intersection),
        "edge_union": int(np.union1d(old_keys, new_keys).size),
        "edge_jaccard": float(intersection / np.union1d(old_keys, new_keys).size),
        "sampled_union_n": int(union.size),
        "sampled_weight_spearman": float(correlation),
        "relative_frobenius_difference": float(np.sqrt(difference.multiply(difference).sum()) / old_norm),
    }


def run_cancer(cancer: str, data_root: Path, output_dir: Path) -> dict:
    directory = data_root / cancer
    genes = read_genes(directory / "graph" / "graph_genes.txt")
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0).loc[:, genes]
    methylation = pd.read_csv(directory / "meth_gene_level.tsv", sep="\t", index_col=0).loc[expression.index, genes]
    expression_rank = standardised_ranks(expression.to_numpy(dtype=np.float32))
    methylation_rank = standardised_ranks(methylation.to_numpy(dtype=np.float32))
    new, cutoff, retained = build_graph(methylation_rank, expression_rank)
    old = sparse.load_npz(directory / "graph" / "A_meth_expr.npz").tocsr().astype(np.float32)
    sparse.save_npz(output_dir / f"{cancer}_A_meth_expr_directional_max.npz", new, compressed=True)
    pairs = len(genes) * (len(genes) - 1) / 2
    return {
        "Cancer": cancer,
        "n": int(expression.shape[0]),
        "p": len(genes),
        "directional_hypotheses": len(genes) * (len(genes) - 1),
        "BH cutoff": cutoff,
        "retained directional edges": retained,
        "old undirected edges": old.nnz // 2,
        "directional-max undirected edges": new.nnz // 2,
        "old density": (old.nnz / 2) / pairs,
        "directional-max density": (new.nnz / 2) / pairs,
        **compare_graphs(old, new, 20260801 + sum(map(ord, cancer))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cancer in args.cancers:
        print(f"[{cancer}] building directional-maximum graph", flush=True)
        rows.append(run_cancer(cancer, args.data_root, args.output_dir))
        pd.DataFrame(rows).to_csv(args.output_dir / "CMPB_DIRECTIONAL_METHYLATION_GRAPH_AUDIT.csv", index=False)
    (args.output_dir / "CMPB_DIRECTIONAL_METHYLATION_GRAPH_AUDIT.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
