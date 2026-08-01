"""Numerical convergence and step-size audit for submission-lock graph-Lasso fits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
LAYER_FILES = {
    "coexpr": "L_coexpr.npz",
    "meth": "L_meth_expr.npz",
    "cnv": "L_cnv.npz",
}


def objective(X, y, L, beta, l1=F.LAMBDA1, l2=F.LAMBDA2) -> float:
    residual = X @ beta - y
    return float(
        0.5 * (residual @ residual) / len(y)
        + 0.5 * l2 * (beta @ (L @ beta))
        + l1 * np.abs(beta).sum()
    )


def largest_eigenvalue(X: np.ndarray, L: sparse.csr_matrix, l2: float = F.LAMBDA2) -> float:
    n, p = X.shape

    def matvec(vector):
        return X.T @ (X @ vector) / n + l2 * (L @ vector)

    operator = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    value = eigsh(operator, k=1, which="LA", return_eigenvectors=False, tol=1e-5, maxiter=1000)[0]
    return float(value)


def robust_fit(X, y, L, lipschitz, max_iter=3000, tol=1e-7):
    n, p = X.shape
    step = 1.0 / (1.01 * lipschitz)
    Xty = X.T @ y / n

    def hessian(vector):
        return X.T @ (X @ vector) / n + F.LAMBDA2 * (L @ vector)

    beta = np.zeros(p, dtype=float)
    relative = np.inf
    converged = False
    for iteration in range(max_iter):
        updated = F.soft_threshold(
            beta - step * (hessian(beta) - Xty), step * F.LAMBDA1
        )
        relative = np.linalg.norm(updated - beta) / max(np.linalg.norm(beta), 1e-10)
        beta = updated
        if relative < tol and iteration > 10:
            converged = True
            break
    gradient = hessian(beta) - Xty
    prox = F.soft_threshold(beta - step * gradient, step * F.LAMBDA1)
    mapping = (beta - prox) / step
    return beta, {
        "robust_converged": converged,
        "robust_iterations": iteration + 1,
        "robust_relative_change": float(relative),
        "robust_objective": objective(X, y, L, beta),
        "robust_prox_gradient_linf": float(np.max(np.abs(mapping))),
        "robust_nonzero": int(np.sum(np.abs(beta) > 1e-8)),
    }


def load_cancer(cancer: str):
    directory = F.ROOT / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expression.index) & set(survival.index))
    expression = expression.loc[common].fillna(expression.loc[common].mean()).fillna(0.0)
    X = expression.to_numpy(dtype=float)
    y = survival.loc[common, "deviance_residual"].to_numpy(dtype=float)
    lock = json.loads((F.OUT / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]
    p = X.shape[1]
    layers = {
        name: F.normalize_laplacian(sparse.load_npz(directory / "graph" / filename), p)
        for name, filename in LAYER_FILES.items()
    }
    zero = sparse.csr_matrix((p, p), dtype=np.float32)
    routed = sum(float(lock["omics_weights"][name]) * layers[name] for name in layers)
    candidates = {"zero": zero, **layers, "routed": routed.tocsr()}
    return X, y, candidates, lock["omics_weights"]


def audit_cancer(cancer: str) -> list[dict]:
    X, y, candidates, route = load_cancer(cancer)
    rows = []
    for name, laplacian in candidates.items():
        started = time.time()
        beta, diagnostics = F.graph_lasso_pgd(X, y, laplacian, return_diagnostics=True)
        exact_lipschitz = largest_eigenvalue(X, laplacian)
        robust_beta, robust = robust_fit(X, y, laplacian, exact_lipschitz)
        current_top = set(np.argsort(np.abs(beta))[::-1][: F.TOP_K].tolist())
        robust_top = set(np.argsort(np.abs(robust_beta))[::-1][: F.TOP_K].tolist())
        rows.append(
            {
                "Cancer": cancer,
                "Laplacian": name,
                "n": X.shape[0],
                "p": X.shape[1],
                "Route weights": json.dumps(route, sort_keys=True),
                **diagnostics,
                "exact_largest_eigenvalue": exact_lipschitz,
                "estimated_to_exact_ratio": diagnostics["lipschitz_estimate"] / exact_lipschitz,
                "current_objective_recomputed": objective(X, y, laplacian, beta),
                **robust,
                "current_minus_robust_objective": objective(X, y, laplacian, beta)
                - robust["robust_objective"],
                "Top20 overlap": len(current_top & robust_top),
                "Top20 Jaccard": len(current_top & robust_top) / len(current_top | robust_top),
                "Elapsed sec": time.time() - started,
            }
        )
        print(f"[{cancer}] {name} complete", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--lock-output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root is not None:
        F.ROOT = args.data_root
    if args.lock_output_root is not None:
        F.OUT = args.lock_output_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cancer in args.cancers:
        rows.extend(audit_cancer(cancer))
        pd.DataFrame(rows).to_csv(args.output_dir / "CMPB_PGD_CONVERGENCE_AUDIT.csv", index=False)
    summary = {
        "fits": len(rows),
        "submission_lock_converged": int(sum(row["converged"] for row in rows)),
        "submission_lock_hit_cap": int(sum(row["hit_iteration_cap"] for row in rows)),
        "minimum_estimated_to_exact_lipschitz_ratio": float(
            min(row["estimated_to_exact_ratio"] for row in rows)
        ),
        "minimum_top20_overlap": int(min(row["Top20 overlap"] for row in rows)),
        "median_top20_overlap": float(np.median([row["Top20 overlap"] for row in rows])),
        "maximum_current_minus_robust_objective": float(
            max(row["current_minus_robust_objective"] for row in rows)
        ),
    }
    (args.output_dir / "CMPB_PGD_CONVERGENCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
