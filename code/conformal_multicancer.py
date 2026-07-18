"""
Multi-cancer conformal survival calibration table.

This script strengthens the conformal evidence beyond the original LUAD case
study. It uses a train/calibration/test split per cancer:
  - train: fit RSF on the frozen MKG Top-20 signature + clinical covariates
  - calibration: estimate global and locally adaptive conformal quantiles
  - test: report empirical coverage and average band width at 1/3/5 years

Outputs:
  conformal_multicancer_results.json
  conformal_multicancer_table.csv
  figures/Fig10_conformal_multicancer.{png,pdf}
"""
import json
import os
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

ROOT = Path(os.environ.get("MKG_DATA_ROOT", "data/processed"))
OUT = Path(os.environ.get("MKG_OUTPUT_ROOT", "outputs"))
FIG = Path(os.environ.get("MKG_FIGURE_ROOT", "outputs/figures"))
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
TIME_POINTS = [365, 1095, 1825]
TIME_LABELS = ["1y", "3y", "5y"]
ALPHA = 0.10
N_GROUPS = 3
TOP_K = 20
SEED = 42


def load_signature(cancer):
    """Prefer the latest final-config signature; fall back to top_genes.csv."""
    p = OUT / f"final_config_comparison_{cancer}.json"
    if p.exists():
        obj = json.load(open(p, encoding="utf-8"))
        block = obj.get(cancer, obj)
        sig = block.get("method_signatures", {}).get("GR-SAFS_v2")
        if sig and sig.get("genes"):
            return [str(g) for g in sig["genes"][:TOP_K]]
    cdir = ROOT / cancer
    return pd.read_csv(cdir / "results" / "top_genes.csv").head(TOP_K)["Gene"].astype(str).tolist()


def clinical_features(clin):
    X = pd.DataFrame(index=clin.index)
    if "age" in clin.columns:
        age = pd.to_numeric(clin["age"], errors="coerce")
        X["age"] = age.fillna(age.median())
    if "gender" in clin.columns:
        X["gender_male"] = (clin["gender"].astype(str).str.upper() == "MALE").astype(float)
    if "stage" in clin.columns:
        s = clin["stage"].astype(str).str.upper()
        X["stage_late"] = s.str.contains("III|IV", regex=True).astype(float)
    return X


def load_data(cancer):
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
    e = pd.to_numeric(dev.loc[common, "OS"], errors="coerce").fillna(0).values.astype(bool)
    valid = np.isfinite(t) & (t > 0)
    return X.iloc[valid], t[valid], e[valid], genes


def surv_at(sf, t):
    if t <= sf.x.max():
        return float(sf(t))
    return float(sf(sf.x.max()))


def scores_at(surv_funcs, times, events, horizon):
    scores = np.full(len(times), np.nan)
    s_hat = np.array([surv_at(sf, horizon) for sf in surv_funcs], dtype=float)
    alive_known = times > horizon
    dead_known = (times <= horizon) & events
    scores[alive_known] = np.abs(1.0 - s_hat[alive_known])
    scores[dead_known] = np.abs(0.0 - s_hat[dead_known])
    truth = np.full(len(times), np.nan)
    truth[alive_known] = 1.0
    truth[dead_known] = 0.0
    return scores, s_hat, truth


def conformal_quantile(scores, alpha=ALPHA):
    v = np.asarray(scores, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, 0
    lvl = min(np.ceil((1 - alpha) * (len(v) + 1)) / len(v), 1.0)
    return float(np.quantile(v, lvl)), int(len(v))


def coverage_and_width(s_hat, truth, q):
    usable = np.isfinite(truth) & np.isfinite(s_hat) & np.isfinite(q)
    if usable.sum() == 0:
        return np.nan, np.nan, 0
    lo = np.maximum(0.0, s_hat[usable] - q)
    hi = np.minimum(1.0, s_hat[usable] + q)
    cov = ((truth[usable] >= lo) & (truth[usable] <= hi)).mean()
    width = (hi - lo).mean()
    return float(cov), float(width), int(usable.sum())


def run_cancer(cancer):
    X, t, e, genes = load_data(cancer)
    n = len(X)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_train = int(0.50 * n)
    n_cal = int(0.25 * n)
    tr = idx[:n_train]
    cal = idx[n_train:n_train + n_cal]
    te = idx[n_train + n_cal:]

    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_split=10, min_samples_leaf=5,
        max_depth=6, random_state=SEED, n_jobs=-1
    )
    rsf.fit(X.iloc[tr], Surv.from_arrays(e[tr], t[tr]))

    risk_train = rsf.predict(X.iloc[tr])
    thresholds = np.percentile(risk_train, np.linspace(0, 100, N_GROUPS + 1))[1:-1]

    sf_cal = rsf.predict_survival_function(X.iloc[cal])
    sf_test = rsf.predict_survival_function(X.iloc[te])
    grp_cal = np.digitize(rsf.predict(X.iloc[cal]), thresholds)
    grp_test = np.digitize(rsf.predict(X.iloc[te]), thresholds)

    rows = []
    details = {}
    for horizon, label in zip(TIME_POINTS, TIME_LABELS):
        sc_cal, _, _ = scores_at(sf_cal, t[cal], e[cal], horizon)
        sc_test, s_test, truth_test = scores_at(sf_test, t[te], e[te], horizon)
        q_global, m_cal = conformal_quantile(sc_cal)
        cov_g, wid_g, m_test = coverage_and_width(s_test, truth_test, q_global)

        local_q = {}
        local_width_num = []
        local_cov_num = []
        local_usable_total = 0
        for g in range(N_GROUPS):
            q_g, m_g = conformal_quantile(sc_cal[grp_cal == g])
            if not np.isfinite(q_g) or m_g < 8:
                q_g = q_global
                m_g = int(np.isfinite(sc_cal).sum())
            local_q[str(g)] = {"q": float(q_g), "m_cal": int(m_g)}
            mask = grp_test == g
            cov_l, wid_l, m_l = coverage_and_width(s_test[mask], truth_test[mask], q_g)
            if m_l:
                local_cov_num.append(cov_l * m_l)
                local_width_num.append(wid_l * m_l)
                local_usable_total += m_l
        cov_l = sum(local_cov_num) / local_usable_total if local_usable_total else np.nan
        wid_l = sum(local_width_num) / local_usable_total if local_usable_total else np.nan

        rows.append({
            "cancer": cancer,
            "horizon": label,
            "n_total": n,
            "n_train": len(tr),
            "n_cal": len(cal),
            "n_test": len(te),
            "m_cal_global": m_cal,
            "m_test_usable": m_test,
            "coverage_global": cov_g,
            "width_global": wid_g,
            "coverage_local": cov_l,
            "width_local": wid_l,
            "width_reduction": wid_g - wid_l if np.isfinite(wid_g) and np.isfinite(wid_l) else np.nan,
            "n_signature_genes": len(genes),
        })
        details[label] = {"q_global": q_global, "local_q": local_q}

    return rows, {
        "n_total": n, "n_train": len(tr), "n_cal": len(cal), "n_test": len(te),
        "signature_genes": genes, "risk_thresholds": thresholds.tolist(), "horizons": details
    }


def plot_table(df):
    FIG.mkdir(parents=True, exist_ok=True)
    cancer_order = CANCERS
    horizon_order = TIME_LABELS
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    mat_cov = df.pivot(index="cancer", columns="horizon", values="coverage_local").loc[cancer_order, horizon_order]
    im = axes[0].imshow(mat_cov.values, vmin=0.80, vmax=1.00, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(len(horizon_order))); axes[0].set_xticklabels(horizon_order)
    axes[0].set_yticks(range(len(cancer_order))); axes[0].set_yticklabels(cancer_order)
    axes[0].set_title("(A) Local conformal coverage")
    for i in range(mat_cov.shape[0]):
        for j in range(mat_cov.shape[1]):
            v = mat_cov.iloc[i, j]
            axes[0].text(j, i, "NA" if pd.isna(v) else f"{v:.2f}",
                         ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = np.arange(len(cancer_order))
    sub = df[df["horizon"] == "1y"].set_index("cancer").loc[cancer_order]
    axes[1].bar(x - 0.18, sub["width_global"], 0.36, label="Global", color="#bdbdbd")
    axes[1].bar(x + 0.18, sub["width_local"], 0.36, label="Local", color="#2c7fb8")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cancer_order)
    axes[1].set_ylabel("Mean band width")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(B) 1-year band width")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG / "Fig10_conformal_multicancer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Fig10_conformal_multicancer.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    details = {}
    for cancer in CANCERS:
        print(f"[conformal] {cancer}", flush=True)
        rows, det = run_cancer(cancer)
        all_rows.extend(rows)
        details[cancer] = det
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "conformal_multicancer_table.csv", index=False)
    with open(OUT / "conformal_multicancer_results.json", "w", encoding="utf-8") as f:
        json.dump({"alpha": ALPHA, "rows": all_rows, "details": details},
                  f, indent=2, ensure_ascii=False)
    plot_table(df)
    print(df[["cancer", "horizon", "m_test_usable", "coverage_local", "width_local", "width_reduction"]])
    print("saved conformal_multicancer_results.json, conformal_multicancer_table.csv, Fig10_conformal_multicancer")


if __name__ == "__main__":
    main()

