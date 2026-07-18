"""
Nested de-leakage audit for JBI revision.

For each cancer, split TCGA into train/test. The nested path learns graph-layer
weights, performs selection, and refits the ridge-Cox model using only train.
The leakage-contrast path learns only the graph-layer weights on the full cohort,
then uses the same train-only selection/refit and test evaluation. The difference
quantifies whether full-cohort weight learning creates measurable optimism.

Environment variables:
  NESTED_CANCERS=LUAD,LIHC,KIRC,COAD,STAD,HNSC
  NESTED_BOOTSTRAP=30
  NESTED_RESUME=1
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lci
from sksurv.util import Surv
import final_config_comparison as F

warnings.filterwarnings("ignore")

ROOT, OUT = F.ROOT, F.OUT
CANCERS = [
    c.strip()
    for c in os.environ.get(
        "NESTED_CANCERS", "LUAD,LIHC,KIRC,COAD,STAD,HNSC"
    ).split(",")
    if c.strip()
]
F.N_BOOTSTRAP = int(os.environ.get("NESTED_BOOTSTRAP", F.N_BOOTSTRAP))
PEN, SEED = 0.1, 42


def learn_weights(G, y, ysurv, clin, Ls, Lid):
    """Learn stability-utility graph weights and return the combined Laplacian."""
    stab = {k: F.bootstrap_stability(G, y, L) for k, L in Ls.items()}
    base_top, _, _, _, _ = F.stage1_select(G, y, Lid)
    ci_base = F.stage2_oof_ci(G[:, base_top], clin, ysurv)

    deltas = {}
    for k, L in Ls.items():
        tk, _, _, _, _ = F.stage1_select(G, y, L)
        deltas[k] = F.stage2_oof_ci(G[:, tk], clin, ysurv) - ci_base

    raw = {k: stab[k] * max(deltas[k], 0) for k in Ls}
    total = sum(raw.values())
    if total < 1e-15:
        w = {k: 0.0 for k in Ls}
        mode = "reject_all_graphs"
    else:
        w = {k: v / total for k, v in raw.items()}
        mode = "dual_driven"
    return sum(w[k] * Ls[k] for k in Ls), w, mode, stab, deltas, ci_base


def fit_coef(X, yt, ye):
    mu, sd = X.mean(0), X.std(0).replace(0, 1.0)
    d = ((X - mu) / sd).copy()
    d["T"] = yt
    d["E"] = ye
    cph = CoxPHFitter(penalizer=PEN, l1_ratio=0.0)
    return cph.fit(d, duration_col="T", event_col="E").params_, mu, sd


def load_cancer(cancer):
    cdir = ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(x) for x in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    clinp = cdir / "clinical_covariates.tsv"
    clin = pd.read_csv(clinp, sep="\t", index_col=0) if clinp.exists() else pd.DataFrame(index=expr.index)

    common = sorted(set(expr.index) & set(dev.index) & set(clin.index))
    exprc = expr.loc[common]
    if exprc.isna().any().any():
        exprc = exprc.fillna(exprc.mean()).fillna(0.0)

    G = exprc.values.astype(np.float64)
    y = dev.loc[common, "deviance_residual"].values.astype(np.float64)
    yt = dev.loc[common, "OS_time"].values.astype(float)
    ye = dev.loc[common, "OS"].values.astype(int)
    names = np.array([str(g) for g in expr.columns])
    p = G.shape[1]
    clin_c = clin.loc[common]
    Ls = {
        k: F.normalize_laplacian(sparse.load_npz(str(cdir / "graph" / fn)), p)
        for k, fn in [
            ("coexpr", "L_coexpr.npz"),
            ("meth", "L_meth_expr.npz"),
            ("cnv", "L_cnv.npz"),
        ]
    }
    return G, y, yt, ye, names, clin_c, Ls, sparse.csr_matrix((p, p), dtype=np.float32)


def run_cancer(cancer):
    print(f"\n[{cancer}] loading data and graphs...", flush=True)
    G, y, yt, ye, names, clin_c, Ls, Lid = load_cancer(cancer)

    idx = np.arange(len(yt))
    tr, te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=ye)
    ysurv_tr = Surv.from_arrays(ye[tr].astype(bool), yt[tr])

    def eval_on_test(L_opt):
        top, _, _, _, _ = F.stage1_select(G[tr], y[tr], L_opt)
        genes = names[top]
        Xtr = pd.DataFrame(G[np.ix_(tr, top)], columns=genes)
        coef, mu, sd = fit_coef(Xtr, yt[tr], ye[tr])
        Xte = (pd.DataFrame(G[np.ix_(te, top)], columns=genes) - mu) / sd
        risk = Xte[coef.index].values @ coef.values
        return float(lci(yt[te], -risk, ye[te]))

    print(f"[{cancer}] learning train-only weights...", flush=True)
    Lopt_tr, w_tr, mode_tr, stab_tr, delta_tr, ci_base_tr = learn_weights(
        G[tr], y[tr], ysurv_tr, clin_c.iloc[tr], Ls, Lid
    )
    print(f"[{cancer}] evaluating nested path...", flush=True)
    ci_nested = eval_on_test(Lopt_tr)

    print(f"[{cancer}] learning full-cohort weights for leakage contrast...", flush=True)
    ysurv_all = Surv.from_arrays(ye.astype(bool), yt)
    Lopt_all, w_all, mode_all, stab_all, delta_all, ci_base_all = learn_weights(
        G, y, ysurv_all, clin_c, Ls, Lid
    )
    print(f"[{cancer}] evaluating full-weight path...", flush=True)
    ci_leaky = eval_on_test(Lopt_all)

    result = {
        "n": len(yt),
        "n_test": int(len(te)),
        "bootstrap_B": int(F.N_BOOTSTRAP),
        "ci_test_nested": round(ci_nested, 4),
        "ci_test_leaky_weights": round(ci_leaky, 4),
        "leakage_gap": round(ci_leaky - ci_nested, 4),
        "weight_mode_train": mode_tr,
        "weight_mode_full": mode_all,
        "w_train": {k: round(w_tr[k], 3) for k in w_tr},
        "w_full": {k: round(w_all[k], 3) for k in w_all},
        "train_stability": {k: round(stab_tr[k], 4) for k in stab_tr},
        "train_delta": {k: round(delta_tr[k], 4) for k in delta_tr},
        "train_baseline_ci": round(ci_base_tr, 4),
        "full_stability": {k: round(stab_all[k], 4) for k in stab_all},
        "full_delta": {k: round(delta_all[k], 4) for k in delta_all},
        "full_baseline_ci": round(ci_base_all, 4),
    }
    print(
        f"{cancer}: nested C={ci_nested:.4f} | full-weight C={ci_leaky:.4f} | "
        f"gap={ci_leaky - ci_nested:+.4f}",
        flush=True,
    )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "nested_deleakage_results.json"
    res = {}
    if os.environ.get("NESTED_RESUME", "0") == "1" and out_path.exists():
        try:
            res = json.load(open(out_path, "r", encoding="utf-8"))
        except Exception:
            res = {}

    print(f"Nested de-leakage audit: cancers={CANCERS}, bootstrap B={F.N_BOOTSTRAP}", flush=True)
    for cancer in CANCERS:
        if os.environ.get("NESTED_RESUME", "0") == "1" and cancer in res:
            print(f"{cancer}: skip existing result", flush=True)
            continue
        res[cancer] = run_cancer(cancer)
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    gaps = [res[c]["leakage_gap"] for c in CANCERS if c in res]
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nMean leakage gap = {np.mean(gaps):+.4f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

