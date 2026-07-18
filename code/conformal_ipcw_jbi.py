"""
JBI-priority conformal audit: IPCW risk-stratified survival calibration.

This script keeps the original held-out train/cal/test design, but adds a
censoring-aware weighted quantile and subgroup miscoverage diagnostics. It is
intended as a methodological validity upgrade over the earlier naive horizon
subsetting analysis, not as a drop-in replacement for all survival conformal
theory.

Outputs:
  - conformal_ipcw_jbi_results.json
  - conformal_ipcw_jbi_table.csv
  - figures/FigJBI_conformal_ipcw_ccd.{png,pdf}
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "03_数据与结果" / "processed_data"
OUT = PROJECT / "06_新实验结果"
FIG = PROJECT / "07_论文初稿" / "figures"

CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
TIME_POINTS = [365.0, 1095.0, 1825.0]
TIME_LABELS = ["1y", "3y", "5y"]
ALPHA = 0.10
N_GROUPS = 3
TOP_K = 20
SEED = 42
MIN_GROUP_CAL = 8
EPS = 1e-4


def load_signature(cancer: str) -> list[str]:
    p = OUT / f"final_config_comparison_{cancer}.json"
    if p.exists():
        obj = json.load(open(p, encoding="utf-8"))
        block = obj.get(cancer, obj)
        sig = block.get("method_signatures", {}).get("GR-SAFS_v2")
        if sig and sig.get("genes"):
            return [str(g) for g in sig["genes"][:TOP_K]]
    cdir = ROOT / cancer
    return (
        pd.read_csv(cdir / "results" / "top_genes.csv")
        .head(TOP_K)["Gene"]
        .astype(str)
        .tolist()
    )


def clinical_features(clin: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=clin.index)
    if "age" in clin.columns:
        age = pd.to_numeric(clin["age"], errors="coerce")
        X["age"] = age.fillna(age.median())
    if "gender" in clin.columns:
        X["gender_male"] = (
            clin["gender"].astype(str).str.upper() == "MALE"
        ).astype(float)
    if "stage" in clin.columns:
        s = clin["stage"].astype(str).str.upper()
        X["stage_late"] = s.str.contains("III|IV", regex=True).astype(float)
    return X


def load_data(cancer: str):
    cdir = ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    clin = pd.read_csv(cdir / "clinical_covariates.tsv", sep="\t", index_col=0)
    common = sorted(set(expr.index) & set(dev.index) & set(clin.index))
    genes = [g for g in load_signature(cancer) if g in expr.columns]
    Xg = expr.loc[common, genes].astype(float).copy()
    Xg = Xg.fillna(Xg.mean()).fillna(0.0)
    Xc = clinical_features(clin.loc[common])
    X = pd.concat([Xg, Xc], axis=1)
    X.columns = [str(c) for c in X.columns]
    t = pd.to_numeric(dev.loc[common, "OS_time"], errors="coerce").values.astype(float)
    e = (
        pd.to_numeric(dev.loc[common, "OS"], errors="coerce")
        .fillna(0)
        .values.astype(bool)
    )
    valid = np.isfinite(t) & (t > 0)
    return X.iloc[valid], t[valid], e[valid], genes


def km_censor_survival(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate of G(t)=P(C >= t), treating censoring as event."""
    times = np.asarray(times, dtype=float)
    censor_events = ~np.asarray(events, dtype=bool)
    event_times = np.sort(np.unique(times[censor_events]))
    surv = []
    g = 1.0
    for tt in event_times:
        at_risk = np.sum(times >= tt)
        d = np.sum((times == tt) & censor_events)
        if at_risk > 0:
            g *= max(0.0, 1.0 - d / at_risk)
        surv.append((float(tt), float(g)))
    event_times = np.array([x for x, _ in surv], dtype=float)
    surv_vals = np.array([y for _, y in surv], dtype=float)

    def ghat(x):
        if len(event_times) == 0:
            return 1.0
        idx = np.searchsorted(event_times, x, side="right") - 1
        if idx < 0:
            return 1.0
        return float(max(surv_vals[idx], EPS))

    return ghat


def surv_at(sf, t: float) -> float:
    if t <= sf.x.max():
        return float(sf(t))
    return float(sf(sf.x.max()))


def horizon_arrays(surv_funcs, times, events, horizon: float, ghat):
    s_hat = np.array([surv_at(sf, horizon) for sf in surv_funcs], dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    alive_known = times > horizon
    dead_known = (times <= horizon) & events
    usable = alive_known | dead_known

    truth = np.full(len(times), np.nan)
    truth[alive_known] = 1.0
    truth[dead_known] = 0.0
    scores = np.full(len(times), np.nan)
    scores[usable] = np.abs(truth[usable] - s_hat[usable])

    weights = np.full(len(times), np.nan)
    for i in np.where(usable)[0]:
        wt_time = horizon if alive_known[i] else times[i]
        weights[i] = 1.0 / max(ghat(wt_time), EPS)
    return scores, s_hat, truth, weights, usable


def unweighted_quantile(scores, alpha=ALPHA):
    v = np.asarray(scores, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, 0
    lvl = min(np.ceil((1.0 - alpha) * (len(v) + 1)) / len(v), 1.0)
    return float(np.quantile(v, lvl)), int(len(v))


def weighted_quantile(scores, weights, alpha=ALPHA):
    v = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan, 0, np.nan
    v = v[m]
    w = w[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = (1.0 - alpha) * w.sum()
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(idx, len(v) - 1)
    ess = float((w.sum() ** 2) / np.sum(w**2))
    return float(v[idx]), int(len(v)), ess


def eval_interval(s_hat, truth, weights, q):
    usable = np.isfinite(truth) & np.isfinite(s_hat) & np.isfinite(q)
    if usable.sum() == 0:
        return {
            "n": 0,
            "coverage": np.nan,
            "weighted_coverage": np.nan,
            "width": np.nan,
        }
    lo = np.maximum(0.0, s_hat[usable] - q)
    hi = np.minimum(1.0, s_hat[usable] + q)
    hit = (truth[usable] >= lo) & (truth[usable] <= hi)
    width = hi - lo
    ww = np.asarray(weights[usable], dtype=float)
    wm = np.isfinite(ww) & (ww > 0)
    wcov = np.average(hit[wm], weights=ww[wm]) if wm.any() else np.nan
    return {
        "n": int(usable.sum()),
        "coverage": float(hit.mean()),
        "weighted_coverage": float(wcov),
        "width": float(width.mean()),
    }


def group_metrics(s_hat, truth, weights, groups, q_by_group):
    rows = []
    for g in range(N_GROUPS):
        mask = groups == g
        q = q_by_group.get(g, np.nan)
        met = eval_interval(s_hat[mask], truth[mask], weights[mask], q)
        met["group"] = int(g)
        met["q"] = float(q) if np.isfinite(q) else np.nan
        rows.append(met)
    return rows


def aggregate_group_metrics(group_rows):
    usable = [r for r in group_rows if r["n"] > 0 and np.isfinite(r["coverage"])]
    if not usable:
        return np.nan, np.nan, 0
    n = sum(r["n"] for r in usable)
    cov = sum(r["coverage"] * r["n"] for r in usable) / n
    wid = sum(r["width"] * r["n"] for r in usable) / n
    return float(cov), float(wid), int(n)


def coverage_disparity(group_rows, target=1.0 - ALPHA):
    vals = [r["coverage"] for r in group_rows if r["n"] > 0 and np.isfinite(r["coverage"])]
    if not vals:
        return np.nan, np.nan
    vals = np.asarray(vals)
    return float(np.max(np.abs(vals - target))), float(vals.max() - vals.min())


def run_cancer(cancer: str):
    X, t, e, genes = load_data(cancer)
    n = len(X)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_train = int(0.50 * n)
    n_cal = int(0.25 * n)
    tr = idx[:n_train]
    cal = idx[n_train : n_train + n_cal]
    te = idx[n_train + n_cal :]

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=6,
        random_state=SEED,
        n_jobs=-1,
    )
    rsf.fit(X.iloc[tr], Surv.from_arrays(e[tr], t[tr]))

    risk_train = rsf.predict(X.iloc[tr])
    thresholds = np.percentile(risk_train, np.linspace(0, 100, N_GROUPS + 1))[1:-1]

    sf_cal = rsf.predict_survival_function(X.iloc[cal])
    sf_test = rsf.predict_survival_function(X.iloc[te])
    grp_cal = np.digitize(rsf.predict(X.iloc[cal]), thresholds)
    grp_test = np.digitize(rsf.predict(X.iloc[te]), thresholds)
    ghat = km_censor_survival(t[tr], e[tr])

    rows = []
    details = {}
    for horizon, label in zip(TIME_POINTS, TIME_LABELS):
        sc_cal, _, _, wt_cal, _ = horizon_arrays(sf_cal, t[cal], e[cal], horizon, ghat)
        _, s_test, truth_test, wt_test, _ = horizon_arrays(sf_test, t[te], e[te], horizon, ghat)

        q_naive_global, m_naive = unweighted_quantile(sc_cal)
        q_ipcw_global, m_ipcw, ess_ipcw = weighted_quantile(sc_cal, wt_cal)

        q_naive_local = {}
        q_ipcw_local = {}
        local_cal_info = {}
        for g in range(N_GROUPS):
            mask = grp_cal == g
            qn, mn = unweighted_quantile(sc_cal[mask])
            qi, mi, essi = weighted_quantile(sc_cal[mask], wt_cal[mask])
            if (not np.isfinite(qn)) or mn < MIN_GROUP_CAL:
                qn = q_naive_global
                mn = m_naive
            if (not np.isfinite(qi)) or mi < MIN_GROUP_CAL:
                qi = q_ipcw_global
                mi = m_ipcw
                essi = ess_ipcw
            q_naive_local[g] = qn
            q_ipcw_local[g] = qi
            local_cal_info[str(g)] = {
                "q_naive": float(qn),
                "q_ipcw": float(qi),
                "m_naive": int(mn),
                "m_ipcw": int(mi),
                "ess_ipcw": float(essi) if np.isfinite(essi) else np.nan,
            }

        methods = {
            "naive_global": {"q": q_naive_global, "local": False, "ipcw": False},
            "naive_risk_stratified": {
                "q": q_naive_local,
                "local": True,
                "ipcw": False,
            },
            "ipcw_global": {"q": q_ipcw_global, "local": False, "ipcw": True},
            "ipcw_risk_stratified": {
                "q": q_ipcw_local,
                "local": True,
                "ipcw": True,
            },
        }

        details[label] = {
            "q_naive_global": float(q_naive_global),
            "q_ipcw_global": float(q_ipcw_global),
            "m_naive_global": int(m_naive),
            "m_ipcw_global": int(m_ipcw),
            "ess_ipcw_global": float(ess_ipcw) if np.isfinite(ess_ipcw) else np.nan,
            "local_calibration": local_cal_info,
        }

        for method, spec in methods.items():
            if spec["local"]:
                group_rows = group_metrics(
                    s_test, truth_test, wt_test, grp_test, spec["q"]
                )
                cov, wid, n_usable = aggregate_group_metrics(group_rows)
                ccd_abs, ccd_range = coverage_disparity(group_rows)
                weighted_covs = [
                    r["weighted_coverage"]
                    for r in group_rows
                    if r["n"] > 0 and np.isfinite(r["weighted_coverage"])
                ]
                weighted_coverage = (
                    float(np.mean(weighted_covs)) if weighted_covs else np.nan
                )
            else:
                met = eval_interval(s_test, truth_test, wt_test, spec["q"])
                cov, wid, n_usable = met["coverage"], met["width"], met["n"]
                weighted_coverage = met["weighted_coverage"]
                group_rows = group_metrics(
                    s_test,
                    truth_test,
                    wt_test,
                    grp_test,
                    {g: spec["q"] for g in range(N_GROUPS)},
                )
                ccd_abs, ccd_range = coverage_disparity(group_rows)

            rows.append(
                {
                    "cancer": cancer,
                    "horizon": label,
                    "method": method,
                    "n_total": n,
                    "n_train": len(tr),
                    "n_cal": len(cal),
                    "n_test": len(te),
                    "m_cal_global": m_ipcw if spec["ipcw"] else m_naive,
                    "ess_cal_global": ess_ipcw if spec["ipcw"] else np.nan,
                    "m_test_usable": n_usable,
                    "coverage": cov,
                    "weighted_coverage": weighted_coverage,
                    "width": wid,
                    "ccd_abs": ccd_abs,
                    "ccd_range": ccd_range,
                    "n_signature_genes": len(genes),
                    "group_metrics": group_rows,
                }
            )

    return rows, {
        "n_total": n,
        "n_train": len(tr),
        "n_cal": len(cal),
        "n_test": len(te),
        "signature_genes": genes,
        "risk_thresholds": thresholds.tolist(),
        "horizons": details,
    }


def plot_results(df: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    method_order = [
        "naive_global",
        "naive_risk_stratified",
        "ipcw_global",
        "ipcw_risk_stratified",
    ]
    labels = ["Naive\nGlobal", "Naive\nRisk-strat.", "IPCW\nGlobal", "IPCW\nRisk-strat."]
    colors = ["#bdbdbd", "#74add1", "#fdae61", "#2b8cbe"]

    one = df[df["horizon"] == "1y"].copy()
    summary = (
        one.groupby("method")[["coverage", "width", "ccd_abs"]]
        .mean()
        .reindex(method_order)
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    x = np.arange(len(method_order))
    axes[0].bar(x, summary["coverage"], color=colors)
    axes[0].axhline(1 - ALPHA, ls="--", lw=1.2, color="#333333")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0.82, 1.0)
    axes[0].set_ylabel("Mean observed coverage")
    axes[0].set_title("(A) 1-year coverage")

    axes[1].bar(x, summary["width"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean interval width")
    axes[1].set_title("(B) 1-year width")

    axes[2].bar(x, summary["ccd_abs"], color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Mean CCD")
    axes[2].set_title("(C) Conditional coverage disparity")

    fig.tight_layout()
    fig.savefig(FIG / "FigJBI_conformal_ipcw_ccd.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigJBI_conformal_ipcw_ccd.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    details = {}
    for cancer in CANCERS:
        print(f"[ipcw conformal] {cancer}", flush=True)
        rows, det = run_cancer(cancer)
        all_rows.extend(rows)
        details[cancer] = det

    df = pd.DataFrame(all_rows)
    flat = df.drop(columns=["group_metrics"]).copy()
    flat.to_csv(OUT / "conformal_ipcw_jbi_table.csv", index=False)
    with open(OUT / "conformal_ipcw_jbi_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": ALPHA,
                "note": (
                    "IPCW quantiles are used as a censoring-aware empirical "
                    "calibration upgrade. Reported test coverage is empirical "
                    "on horizon-usable subjects; weighted_coverage uses "
                    "inverse censoring weights on the same subjects."
                ),
                "rows": all_rows,
                "details": details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    plot_results(flat)

    view = flat[
        ["cancer", "horizon", "method", "m_test_usable", "coverage", "width", "ccd_abs"]
    ]
    print(view.to_string(index=False))
    print("saved conformal_ipcw_jbi_results.json, conformal_ipcw_jbi_table.csv")
    print("saved FigJBI_conformal_ipcw_ccd.{png,pdf}")


if __name__ == "__main__":
    main()

