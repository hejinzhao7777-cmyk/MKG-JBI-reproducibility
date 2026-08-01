"""Cross-validated Coxnet baselines under the frozen Top-20 transfer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
COHORTS = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}


def load_development(cancer: str) -> dict:
    directory = F.ROOT / cancer
    expression = pd.read_csv(directory / "expr_final.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(directory / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expression.index) & set(survival.index))
    expression = expression.loc[common].fillna(expression.loc[common].mean()).fillna(0.0)
    return {
        "X": expression.to_numpy(dtype=np.float64),
        "genes": np.asarray(expression.columns, dtype=str),
        "time": survival.loc[common, "OS_time"].to_numpy(dtype=float),
        "event": survival.loc[common, "OS"].to_numpy(dtype=int),
    }


def select_alpha(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l1_ratio: float,
    folds: int = 5,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    y = Surv.from_arrays(event.astype(bool), time)
    grid_model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=40, max_iter=5000)
    grid_model.fit(X, y)
    alphas = np.asarray(grid_model.alphas_, dtype=float)
    nonzero = np.sum(np.abs(grid_model.coef_) > 1e-10, axis=0)
    eligible = nonzero >= F.TOP_K
    if not np.any(eligible):
        eligible[:] = True

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_scores = np.full((folds, len(alphas)), np.nan, dtype=float)
    for fold, (train, validation) in enumerate(splitter.split(X, event)):
        model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, max_iter=5000)
        model.fit(X[train], Surv.from_arrays(event[train].astype(bool), time[train]))
        for alpha_index, alpha in enumerate(alphas):
            prediction = model.predict(X[validation], alpha=alpha)
            fold_scores[fold, alpha_index] = concordance_index_censored(
                event[validation].astype(bool), time[validation], prediction
            )[0]
    mean_score = np.nanmean(fold_scores, axis=0)
    candidate_score = np.where(eligible, mean_score, -np.inf)
    best = int(np.nanargmax(candidate_score))
    rows = []
    for alpha_index, alpha in enumerate(alphas):
        row = {
            "alpha": float(alpha),
            "full_fit_nonzero": int(nonzero[alpha_index]),
            "eligible_at_least_top20": bool(eligible[alpha_index]),
            "mean_validation_cindex": float(mean_score[alpha_index]),
            "selected": alpha_index == best,
        }
        for fold in range(folds):
            row[f"fold_{fold + 1}_cindex"] = float(fold_scores[fold, alpha_index])
        rows.append(row)
    return float(alphas[best]), pd.DataFrame(rows)


def final_signature(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    genes: np.ndarray,
    l1_ratio: float,
    alpha: float,
) -> dict:
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha], max_iter=5000)
    model.fit(X, Surv.from_arrays(event.astype(bool), time))
    coefficient = model.coef_[:, 0]
    top = np.argsort(np.abs(coefficient))[::-1][: F.TOP_K]
    return {
        "genes": genes[top].tolist(),
        "directions": np.sign(coefficient[top]).astype(float).tolist(),
        "scores": np.abs(coefficient[top]).astype(float).tolist(),
        "nonzero": int(np.sum(np.abs(coefficient) > 1e-10)),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
    }


def external_evaluation(cancer: str, signature: dict) -> dict:
    directory = F.ROOT / cancer / "external" / COHORTS[cancer]
    expression = pd.read_csv(directory / "expr.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(directory / "survival.tsv", sep="\t")
    survival.columns = [column.lower().replace(".", "_") for column in survival.columns]
    event_column = next(column for column in survival.columns if column in ["os", "event", "status", "os_event"])
    time_column = next(column for column in survival.columns if column in ["os_time", "time", "survival_time"])
    survival = survival.set_index(survival.columns[0])
    common = sorted(set(expression.index) & set(survival.index))
    expression = expression.loc[common]
    event = pd.to_numeric(survival.loc[common, event_column], errors="coerce").to_numpy()
    time = pd.to_numeric(survival.loc[common, time_column], errors="coerce").to_numpy()
    usable = np.isfinite(event) & np.isfinite(time) & (time > 0)
    risk, matched = F.frozen_risk(
        expression.loc[usable], signature["genes"], signature["directions"], signature["scores"]
    )
    if risk is None:
        return {"c_index": np.nan, "n_matched": 0}
    return {**F.evaluate(risk, time[usable], event[usable].astype(int)), "n_matched": matched}


def run_cancer(cancer: str, output_dir: Path) -> list[dict]:
    data = load_development(cancer)
    rows: list[dict] = []
    payload: dict[str, dict] = {}
    for method, ratio in [("CV-Cox-Lasso", 1.0), ("CV-Cox-EN", 0.5)]:
        alpha, trace = select_alpha(data["X"], data["time"], data["event"], ratio)
        trace.insert(0, "Cancer", cancer)
        trace.insert(1, "Method", method)
        trace.to_csv(output_dir / f"{cancer}_{method}_alpha_trace.csv", index=False)
        signature = final_signature(
            data["X"], data["time"], data["event"], data["genes"], ratio, alpha
        )
        external = external_evaluation(cancer, signature)
        selected_trace = trace[trace["selected"]].iloc[0]
        payload[method] = {
            "signature": signature,
            "cv_mean_cindex": float(selected_trace["mean_validation_cindex"]),
            "external": external,
            "cohort": COHORTS[cancer],
        }
        rows.append(
            {
                "Cancer": cancer,
                "Cohort": COHORTS[cancer],
                "Method": method,
                "l1_ratio": ratio,
                "selected_alpha": alpha,
                "full_fit_nonzero": signature["nonzero"],
                "CV mean C-index": float(selected_trace["mean_validation_cindex"]),
                "External C-index": external["c_index"],
                "Matched genes": external["n_matched"],
            }
        )
    (output_dir / f"{cancer}_CV_TUNED_COX_BASELINES.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cancers", nargs="*", default=CANCERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cancer in args.cancers:
        print(f"[{cancer}] CV-tuned Cox baselines", flush=True)
        rows.extend(run_cancer(cancer, args.output_dir))
        pd.DataFrame(rows).to_csv(args.output_dir / "CMPB_CV_TUNED_COX_BASELINES.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
