"""
补充实验(对标 DASEN/JBHI 的统计严谨性):
(1) 显著性检验 + 效应量:
    - 跨 6 癌配对检验 MKG vs 基线(外部 C-index、稳定性):Wilcoxon + paired t + Cohen's d
    - 每癌外部队列 Bootstrap(B=2000)给 95% CI 与配对 bootstrap p(MKG vs Uni-Cox / Cox-Lasso)
(2) 最差子群鲁棒性:跨癌种 worst-case(最差癌)外部 C-index 与稳定性
复用已存签名(method_genes)+ 外部数据;重拟合 20 基因岭-Cox 取系数算冻结风险。
输出 significance_robustness.json + Fig8 森林图。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lci
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("MKG_DATA_ROOT", "data/processed"))
OUT = Path(os.environ.get("MKG_OUTPUT_ROOT", "outputs"))
FIG = Path(os.environ.get("MKG_FIGURE_ROOT", "outputs/figures"))
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
EXT = {"LUAD": "GSE31210", "LIHC": "GSE14520", "KIRC": "GSE29609",
       "COAD": "GSE39582", "STAD": "GSE84437", "HNSC": "GSE65858"}
METHODS = ["GR-SAFS_v2", "Uni-Cox", "Cox-Lasso", "Cox-EN", "RSF", "CGBoost"]
LABEL = {"GR-SAFS_v2": "MKG"}
PEN, SEED, B = 0.1, 42, 2000
rng = np.random.RandomState(SEED)


def cohen_d_paired(diff):
    diff = np.asarray(diff, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else np.nan


def fit_coef(Xtr, yt, ye):
    mu, sd = Xtr.mean(0), Xtr.std(0).replace(0, 1.0)
    d = ((Xtr - mu) / sd).copy(); d["T"] = yt; d["E"] = ye
    cph = CoxPHFitter(penalizer=PEN, l1_ratio=0.0).fit(d, duration_col="T", event_col="E")
    return cph.params_


def ext_risk(cancer, genes):
    """返回外部队列上的冻结风险向量 + 生存"""
    cdir = ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0); expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expr.index) & set(dev.index))
    exprc = expr.loc[common]
    if exprc.isna().any().any():
        exprc = exprc.fillna(exprc.mean()).fillna(0.0)
    g = [x for x in genes if x in exprc.columns]
    if len(g) < 3:
        return None
    coef = fit_coef(exprc[g], dev.loc[common, "OS_time"].values.astype(float),
                    dev.loc[common, "OS"].values.astype(int))
    ed = cdir / "external" / EXT[cancer]
    ee = pd.read_csv(ed / "expr.tsv", sep="\t", index_col=0); ee.columns = [str(c) for c in ee.columns]
    es = pd.read_csv(ed / "survival.tsv", sep="\t"); es.columns = [c.lower().replace(".", "_") for c in es.columns]
    sc = next((c for c in es.columns if c in ["os", "event", "status", "os_event"]), None)
    tc = next((c for c in es.columns if c in ["os_time", "time", "survival_time"]), None)
    es = es.set_index(es.columns[0]); cm = sorted(set(ee.index) & set(es.index)); ee = ee.loc[cm]
    eve = pd.to_numeric(es.loc[cm, sc], errors="coerce").values
    tve = pd.to_numeric(es.loc[cm, tc], errors="coerce").values
    v = (~np.isnan(eve)) & (~np.isnan(tve)) & (tve > 0)
    ee, eve, tve = ee.loc[v], eve[v].astype(int), tve[v]
    av = [x for x in g if x in ee.columns]
    Xe = ee[av].copy()
    if Xe.isna().any().any():
        Xe = Xe.fillna(Xe.mean()).fillna(0.0)
    Xz = (Xe - Xe.mean(0)) / Xe.std(0).replace(0, 1.0)
    risk = Xz.values @ coef[av].values
    return risk, tve, eve


def main():
    # 读签名 + 稳定性
    sig = {c: json.load(open(OUT / f"final_config_comparison_{c}.json", encoding="utf-8"))[c]["method_genes"] for c in CANCERS}
    stab = json.load(open(OUT / "stability_and_weighted_results.json", encoding="utf-8"))

    # ---- 每癌外部风险 + Bootstrap ----
    point, boot = {m: {} for m in METHODS}, {m: {} for m in METHODS}
    surv = {}
    for c in CANCERS:
        for m in METHODS:
            if m not in sig[c]:
                continue
            r = ext_risk(c, sig[c][m])
            if r is None:
                continue
            risk, tve, eve = r
            surv[c] = (tve, eve)
            point[m][c] = float(lci(tve, -risk, eve))
            point[m].setdefault("_risk", {})[c] = risk
    # bootstrap per cancer
    res_boot = {m: {} for m in METHODS}
    n_per = {}
    for c in CANCERS:
        tve, eve = surv[c]; n = len(tve); n_per[c] = n
        idxs = [rng.randint(0, n, n) for _ in range(B)]
        for m in METHODS:
            if c not in point[m]:
                continue
            risk = point[m]["_risk"][c]; vals = []
            for idx in idxs:
                try:
                    vals.append(lci(tve[idx], -risk[idx], eve[idx]))
                except Exception:
                    pass
            res_boot[m][c] = np.array(vals)

    # ---- 每癌配对 bootstrap p(MKG vs Uni-Cox / Cox-Lasso)----
    out = {"per_cancer_external": {}, "across_cancer_tests": {}, "stability_test": {},
           "worst_group": {}, "bootstrap_B": B}
    for c in CANCERS:
        entry = {"n_external": n_per[c]}
        for m in METHODS:
            if c in res_boot[m]:
                v = res_boot[m][c]
                entry[m] = {"c_index": round(point[m][c], 4),
                            "ci95": [round(np.percentile(v, 2.5), 4), round(np.percentile(v, 97.5), 4)]}
        mk = res_boot["GR-SAFS_v2"].get(c)
        for base in ["Uni-Cox", "Cox-Lasso"]:
            bv = res_boot[base].get(c)
            if mk is not None and bv is not None:
                L = min(len(mk), len(bv)); d = mk[:L] - bv[:L]
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                entry[f"MKG_vs_{base}_bootP"] = round(float(p), 4)
                entry[f"MKG_vs_{base}_cohend"] = round(cohen_d_paired(d), 3)
        out["per_cancer_external"][c] = entry

    # ---- 跨癌配对检验(外部 point C-index;含/不含 KIRC)----
    for scope, cs in [("all6", CANCERS), ("excl_KIRC", [c for c in CANCERS if c != "KIRC"])]:
        d = {}
        for base in ["Uni-Cox", "Cox-Lasso", "Cox-EN", "RSF", "CGBoost"]:
            mk = [point["GR-SAFS_v2"][c] for c in cs if c in point["GR-SAFS_v2"] and c in point[base]]
            bs = [point[base][c] for c in cs if c in point["GR-SAFS_v2"] and c in point[base]]
            mk, bs = np.array(mk), np.array(bs); diff = mk - bs
            try:
                w_p = stats.wilcoxon(mk, bs, alternative="greater").pvalue
            except Exception:
                w_p = np.nan
            t_p = stats.ttest_rel(mk, bs).pvalue
            d[base] = {"n": len(mk), "mean_MKG": round(mk.mean(), 4), "mean_base": round(bs.mean(), 4),
                       "wins": int((diff > 0).sum()), "wilcoxon_greater_p": round(float(w_p), 4),
                       "paired_t_p": round(float(t_p), 4), "cohen_d": round(cohen_d_paired(diff), 3)}
        out["across_cancer_tests"][scope] = d

    # ---- 稳定性跨癌配对检验(MKG vs Cox-Lasso, RBO & Jaccard)----
    for metric in ["RBO", "Jaccard"]:
        mk = np.array([stab[c]["stability"]["GR-SAFS"][metric] for c in CANCERS])
        ls = np.array([stab[c]["stability"]["Cox-Lasso"][metric] for c in CANCERS])
        diff = mk - ls
        out["stability_test"][metric] = {
            "n": len(mk), "mean_MKG": round(mk.mean(), 4), "mean_Lasso": round(ls.mean(), 4),
            "wins": int((diff > 0).sum()),
            "wilcoxon_greater_p": round(float(stats.wilcoxon(mk, ls, alternative="greater").pvalue), 4),
            "paired_t_p": round(float(stats.ttest_rel(mk, ls).pvalue), 4),
            "cohen_d": round(cohen_d_paired(diff), 3)}

    # ---- 最差子群(最差癌)鲁棒性 ----
    cs_ek = [c for c in CANCERS if c != "KIRC"]
    for m in METHODS:
        ext_all = [point[m][c] for c in CANCERS if c in point[m]]
        ext_ek = [point[m][c] for c in cs_ek if c in point[m]]
        out["worst_group"][m] = {
            "worst_cancer_external_all6": round(min(ext_all), 4),
            "worst_cancer_external_exclKIRC": round(min(ext_ek), 4),
            "mean_external_exclKIRC": round(float(np.mean(ext_ek)), 4)}
    out["worst_group"]["stability_RBO"] = {
        "MKG_worst": round(min(stab[c]["stability"]["GR-SAFS"]["RBO"] for c in CANCERS), 4),
        "CoxLasso_worst": round(min(stab[c]["stability"]["Cox-Lasso"]["RBO"] for c in CANCERS), 4)}

    json.dump(out, open(OUT / "significance_robustness.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- 打印关键结论 ----
    print("== 跨癌配对检验:外部 C-index, MKG vs 基线 (excl_KIRC) ==")
    for base, v in out["across_cancer_tests"]["excl_KIRC"].items():
        print(f"  MKG vs {base:9}: 胜{v['wins']}/{v['n']}  均值 {v['mean_MKG']} vs {v['mean_base']}  "
              f"Wilcoxon p={v['wilcoxon_greater_p']}  t p={v['paired_t_p']}  d={v['cohen_d']}")
    print("== 稳定性跨癌配对检验 MKG vs Cox-Lasso ==")
    for mtr, v in out["stability_test"].items():
        print(f"  {mtr}: 胜{v['wins']}/{v['n']}  {v['mean_MKG']} vs {v['mean_Lasso']}  "
              f"Wilcoxon p={v['wilcoxon_greater_p']}  d={v['cohen_d']}")
    print("== 最差癌(worst-group)外部 C-index (excl KIRC) ==")
    for m in METHODS:
        print(f"  {LABEL.get(m,m):9}: worst={out['worst_group'][m]['worst_cancer_external_exclKIRC']}  "
              f"mean={out['worst_group'][m]['mean_external_exclKIRC']}")

    # ---- Fig8 森林图:每癌外部 C-index ± 95% CI (MKG/Uni-Cox/Cox-Lasso) ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    keym = [("GR-SAFS_v2", "MKG", "#2c7fb8"), ("Uni-Cox", "Uni-Cox", "#756bb1"), ("Cox-Lasso", "Cox-Lasso", "#d95f0e")]
    yb = np.arange(len(CANCERS)); off = {0: 0.24, 1: 0.0, 2: -0.24}
    for j, (m, lab, col) in enumerate(keym):
        for i, c in enumerate(CANCERS):
            if c not in out["per_cancer_external"] or m not in out["per_cancer_external"][c]:
                continue
            e = out["per_cancer_external"][c][m]; ci = e["ci95"]
            ax.errorbar(e["c_index"], yb[i] + off[j], xerr=[[e["c_index"]-ci[0]], [ci[1]-e["c_index"]]],
                        fmt="o", color=col, capsize=3, ms=5, label=lab if i == 0 else "")
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_yticks(yb); ax.set_yticklabels(CANCERS); ax.set_xlabel("External C-index (bootstrap 95% CI)")
    ax.legend(frameon=False, fontsize=9); ax.set_title("Per-cancer external C-index with 95% CI")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"Fig8_external_CI.{ext}", dpi=300, bbox_inches="tight")
    print("saved Fig8_external_CI + significance_robustness.json")


if __name__ == "__main__":
    main()

