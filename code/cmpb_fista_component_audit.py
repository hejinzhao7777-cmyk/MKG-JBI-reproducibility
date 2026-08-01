"""Accelerated high-accuracy sensitivity for locked graph-Lasso components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import final_config_comparison as F
except ModuleNotFoundError:  # workspace layout: core code is a sibling folder
    core_dir = Path(__file__).resolve().parents[1] / "01_核心模型与锁定流程"
    sys.path.insert(0, str(core_dir))
    import final_config_comparison as F
from cmpb_pgd_convergence_audit import load_cancer, objective


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]


def fista(X, y, L, lipschitz, max_iter=3000, tolerance=1e-6):
    n, p = X.shape
    step = 1.0 / (1.01 * lipschitz)
    Xty = X.T @ y / n

    def gradient(vector):
        return X.T @ (X @ vector) / n + F.LAMBDA2 * (L @ vector) - Xty

    x = np.zeros(p, dtype=float)
    momentum_point = x.copy()
    acceleration = 1.0
    relative = np.inf
    prox_linf = np.inf
    for iteration in range(max_iter):
        updated = F.soft_threshold(
            momentum_point - step * gradient(momentum_point), step * F.LAMBDA1
        )
        relative = np.linalg.norm(updated - x) / max(np.linalg.norm(x), 1e-10)
        new_acceleration = (1.0 + np.sqrt(1.0 + 4.0 * acceleration**2)) / 2.0
        extrapolated = updated + ((acceleration - 1.0) / new_acceleration) * (updated - x)
        # Gradient restart suppresses oscillation in this strongly convex problem.
        if np.dot(momentum_point - updated, updated - x) > 0:
            new_acceleration = 1.0
            extrapolated = updated.copy()
        x = updated
        momentum_point = extrapolated
        acceleration = new_acceleration
        if (iteration + 1) % 10 == 0 or iteration + 1 == max_iter:
            prox = F.soft_threshold(x - step * gradient(x), step * F.LAMBDA1)
            mapping = (x - prox) / step
            prox_linf = float(np.max(np.abs(mapping)))
            if prox_linf <= tolerance and relative <= tolerance:
                break
    return x, {
        "FISTA iterations": iteration + 1,
        "FISTA converged": prox_linf <= tolerance,
        "FISTA relative change": float(relative),
        "FISTA prox-gradient Linf": prox_linf,
        "FISTA objective": objective(X, y, L, x),
        "FISTA nonzero": int(np.sum(np.abs(x) > 1e-8)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--pgd-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--lock-output-root", type=Path)
    args = parser.parse_args()
    if args.data_root is not None:
        F.ROOT = args.data_root
    if args.lock_output_root is not None:
        F.OUT = args.lock_output_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = pd.read_csv(args.pgd_audit)
    rows = []
    for cancer in args.cancers:
        X, y, candidates, _ = load_cancer(cancer)
        for name in ("zero", "routed"):
            reference = diagnostic[(diagnostic.Cancer == cancer) & (diagnostic.Laplacian == name)].iloc[0]
            locked, locked_diag = F.graph_lasso_pgd(X, y, candidates[name], return_diagnostics=True)
            accelerated, fista_diag = fista(X, y, candidates[name], float(reference["exact_largest_eigenvalue"]))
            locked_top = set(np.argsort(np.abs(locked))[::-1][: F.TOP_K])
            fista_top = set(np.argsort(np.abs(accelerated))[::-1][: F.TOP_K])
            rows.append({
                "Cancer": cancer,
                "Laplacian": name,
                "Exact largest eigenvalue": float(reference["exact_largest_eigenvalue"]),
                **locked_diag,
                **fista_diag,
                "Locked minus FISTA objective": objective(X, y, candidates[name], locked) - fista_diag["FISTA objective"],
                "Top20 overlap": len(locked_top & fista_top),
                "Top20 Jaccard": len(locked_top & fista_top) / len(locked_top | fista_top),
            })
            pd.DataFrame(rows).to_csv(args.output_dir / "CMPB_FISTA_COMPONENT_AUDIT.csv", index=False)
            print(f"[{cancer}] {name} FISTA complete", flush=True)
    summary = {
        "fits": len(rows),
        "converged": int(sum(row["FISTA converged"] for row in rows)),
        "minimum_top20_overlap": int(min(row["Top20 overlap"] for row in rows)),
        "median_top20_overlap": float(np.median([row["Top20 overlap"] for row in rows])),
        "maximum_objective_gap": float(max(row["Locked minus FISTA objective"] for row in rows)),
        "maximum_FISTA_prox_gradient_Linf": float(max(row["FISTA prox-gradient Linf"] for row in rows)),
    }
    (args.output_dir / "CMPB_FISTA_COMPONENT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
