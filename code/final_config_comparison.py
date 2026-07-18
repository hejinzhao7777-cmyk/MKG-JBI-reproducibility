"""
实验补全①：最终确认配置下的完整方法对比表
================================================================================
统一口径（以导师汇报 PDF 为准）：
  - 谱归一化拉普拉斯 (normalize_laplacian)
  - λ₂ = 50.0
  - 稳定-预测双驱动权重 w_k ∝ S_k·max(Δ_k,0) (Bootstrap B=30, RBO p=0.9)
  - QP 加权 γ=10, eCDF rank 融合, TOP_K=20
对比方法：GR-SAFS_v2(双驱动) / GR-SAFS_v1(共表达) / v2_equal(等权) /
          Cox-Lasso / Cox-EN / Uni-Cox / RSF / DeepSurv
评估：训练集 + LUAD 外部(GSE31210/GSE50081) + 泛化衰减 + Nomogram
用法：python final_config_comparison.py [CANCER1 CANCER2 ...]   (默认 LUAD)
"""
import os, sys, json, time, warnings, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # 强制 CPU，避免 DeepSurv 设备不匹配（""在新版torch无效，须用-1）
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lci
from lifelines.statistics import logrank_test

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path(os.environ.get("MKG_DATA_ROOT", "data/processed"))
OUT = Path(os.environ.get("MKG_OUTPUT_ROOT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
LAMBDA1, LAMBDA2, GAMMA = 0.2, 50.0, 10.0       # 报告最终配置
K_FOLDS, TOP_K, SEED = 5, int(os.environ.get("FINAL_TOP_K", "20")), 42
RF_JOBS = int(os.environ.get("FINAL_RF_JOBS", "8"))  # 16核取8：内存已用float32+del解决，恢复并行提速（2太慢）
N_BOOTSTRAP = int(os.environ.get("FINAL_BOOTSTRAP", "30"))
PGD_MAX_ITER = int(os.environ.get("FINAL_PGD_MAX_ITER", "300"))
PGD_POWER_ITER = int(os.environ.get("FINAL_PGD_POWER_ITER", "8"))
STAGE1_RF_TREES = int(os.environ.get("FINAL_STAGE1_RF_TREES", "300"))
RSF_TREES = int(os.environ.get("FINAL_RSF_TREES", "100"))
DEEPSURV_EPOCHS = int(os.environ.get("FINAL_DEEPSURV_EPOCHS", "50"))
BOOTSTRAP_RATIO, RBO_P = 0.8, 0.9
EXTERNAL = {"LUAD": ["GSE31210", "GSE50081"],
            "LIHC": ["GSE76427", "GSE14520"], "KIRC": ["GSE29609", "E-MTAB-1980"],
            "COAD": ["GSE39582"], "STAD": ["GSE84437"], "HNSC": ["GSE65858"]}


# ---------------- 核心数值 ----------------
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def estimate_lipschitz(G, L, l2=LAMBDA2, n_iter=PGD_POWER_ITER):
    """Fast deterministic power-iteration estimate for the PGD step size."""
    n, p = G.shape
    def H(v): return G.T @ (G @ v) / n + l2 * (L @ v)
    rng = np.random.RandomState(SEED)
    v = rng.normal(size=p)
    v /= max(np.linalg.norm(v), 1e-12)
    Lip = 1.0
    for _ in range(max(2, n_iter)):
        Hv = H(v)
        norm = np.linalg.norm(Hv)
        if not np.isfinite(norm) or norm < 1e-12:
            break
        v = Hv / norm
        Lip = float(v @ H(v))
    return max(Lip * 1.05, 1e-8)


def graph_lasso_pgd(G, y, L, l1=LAMBDA1, l2=LAMBDA2, max_iter=PGD_MAX_ITER, tol=1e-5):
    n, p = G.shape
    def H(v): return G.T @ (G @ v) / n + l2 * (L @ v)
    Lip = estimate_lipschitz(G, L, l2=l2)
    eta = 1.0 / Lip
    Gty = G.T @ y / n
    beta = np.zeros(p)
    for k in range(max_iter):
        bn = soft_threshold(beta - eta * (H(beta) - Gty), eta * l1)
        rel = np.linalg.norm(bn - beta) / max(np.linalg.norm(beta), 1e-10)
        beta = bn
        if rel < tol and k > 10:
            break
    return beta


def normalize_laplacian(L, p):
    Ld = L.tocsr().astype(np.float64) if sparse.issparse(L) else sparse.csr_matrix(L.astype(np.float64))
    if Ld.shape[0] > p:
        Ld = Ld[:p, :p].tocsr()
    elif Ld.shape[0] < p:
        pad = p - Ld.shape[0]
        Ld = sparse.block_diag((Ld, sparse.eye(pad, format="csr")), format="csr")
    Ld = Ld + 1e-6 * sparse.eye(p, format="csr")
    try:
        me = eigsh(Ld, k=1, which='LM', return_eigenvectors=False)[0]
    except Exception:
        me = np.max(np.asarray(Ld.sum(axis=1)).ravel())
    return (Ld / (me + 1e-10)).astype(np.float32).tocsr()


def solve_qp_weights(P, y, gamma=GAMMA):
    n, K = P.shape
    res = y[:, None] - P
    S = np.cov(res.T); mu = np.trace(S) / K
    Omega = 0.5 * S + 0.5 * mu * np.eye(K)
    Hm = (P.T @ P) / n + gamma * Omega
    f = -(P.T @ y) / n
    r = minimize(lambda w: 0.5 * w @ Hm @ w + f @ w, np.ones(K) / K,
                 jac=lambda w: Hm @ w + f, method='SLSQP',
                 bounds=[(0.10, 1.0)] * K,
                 constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                 options={'maxiter': 1000, 'ftol': 1e-12})
    w = np.maximum(r.x, 0)
    return w / w.sum()


# ---------------- RBO / 稳定性 ----------------
def rbo_score(l1, l2, p=RBO_P):
    k = min(len(l1), len(l2))
    if k == 0:
        return 0.0
    s1, s2, agree = set(), set(), []
    for d in range(1, k + 1):
        s1.add(l1[d - 1]); s2.add(l2[d - 1]); agree.append(len(s1 & s2) / d)
    denom = 1 - p ** k
    if denom <= 0:
        return 0.0
    return (1 - p) * sum((p ** d) * agree[d] for d in range(k)) / denom


def pairwise_rbo(rk, p=RBO_P):
    n = len(rk)
    if n < 2:
        return 1.0
    tot = c = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += rbo_score(rk[i], rk[j], p); c += 1
    return tot / c if c else 0.0


def bootstrap_stability(G, y, L, seed=SEED):
    n, p = G.shape
    ns = int(n * BOOTSTRAP_RATIO)
    rng = np.random.RandomState(seed)
    rk = []
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, ns, replace=False)
        beta = graph_lasso_pgd(G[idx], y[idx], L, max_iter=300, tol=1e-4)
        rk.append(np.argsort(np.abs(beta))[::-1][:TOP_K].tolist())
    return pairwise_rbo(rk)


# ---------------- Stage1 / Stage2 ----------------
def clin_features(df, base_index):
    X = pd.DataFrame(index=base_index)
    if "age" in df.columns:
        X["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(60.0)
    if "gender" in df.columns:
        X["gender_male"] = (df["gender"].str.upper() == "MALE").astype(float)
    if "stage" in df.columns:
        sm = {s: (1.0 if ("III" in str(s).upper() or "IV" in str(s).upper()) else 0.0)
              for s in df["stage"].dropna().unique()}
        X["stage_late"] = df["stage"].map(sm).fillna(0)
    return X


def stage1_select(G, y, L):
    n, p = G.shape
    P = np.zeros((n, 2)); igl = np.zeros(p); irf = np.zeros(p); igs = np.zeros(p)
    kf = KFold(K_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in kf.split(G):
        beta = graph_lasso_pgd(G[tr], y[tr], L)
        P[va, 0] = G[va] @ beta; igl += np.abs(beta); igs += beta
        rf = RandomForestRegressor(n_estimators=STAGE1_RF_TREES, max_depth=5, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=RF_JOBS)
        rf.fit(G[tr], y[tr]); P[va, 1] = rf.predict(G[va]); irf += rf.feature_importances_
    w = solve_qp_weights(P, y)
    scores = w[0] * rankdata(igl, method='min') / p + w[1] * rankdata(irf, method='min') / p
    top = np.argsort(scores)[::-1][:TOP_K]
    return top, np.sign(igs / K_FOLDS), scores, w, int(np.sum(np.abs(igl / K_FOLDS) > 1e-8))


def stage2_oof_ci(Gtop, clin_df, y_surv):
    X = pd.DataFrame(Gtop, index=clin_df.index)
    X = pd.concat([X, clin_features(clin_df, clin_df.index)], axis=1)
    X.columns = X.columns.astype(str)
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=5,
                               max_depth=6, random_state=SEED, n_jobs=RF_JOBS)
    kf = KFold(K_FOLDS, shuffle=True, random_state=SEED); cis = []
    for tr, va in kf.split(X):
        rsf.fit(X.iloc[tr], y_surv[tr])
        cis.append(concordance_index_censored(
            y_surv[va]["event"], y_surv[va]["time"], rsf.predict(X.iloc[va]))[0])
    return float(np.mean(cis))


# ---------------- 评估 ----------------
def frozen_risk(expr_df, genes, dirs, scores):
    av = [g for g in genes if g in expr_df.columns]
    if not av:
        return None, 0
    idx = [genes.index(g) for g in av]
    G = expr_df[av].values.astype(float)
    mu = np.nanmean(G, 0); sd = np.nanstd(G, 0); sd[sd < 1e-8] = 1.0
    Gn = np.nan_to_num((G - mu) / sd, nan=0.0)   # 缺失值→中性0，避免NaN污染风险评分
    return Gn @ (np.asarray(dirs)[idx] * np.asarray(scores)[idx]), len(av)


def binmetrics(risk, yt, ye, t=365):
    yb = ((yt <= t) & (ye == 1)).astype(int)
    m = (yt > t) | (ye == 1)
    o = {"auc_1y": None, "f1": None, "precision": None, "recall": None}
    if m.sum() < 10 or yb[m].sum() < 3:
        return o
    try:
        o["auc_1y"] = roc_auc_score(yb[m], risk[m])
    except Exception:
        pass
    yp = (risk[m] >= np.median(risk[m])).astype(int)
    try:
        o["f1"] = f1_score(yb[m], yp, zero_division=0)
        o["precision"] = precision_score(yb[m], yp, zero_division=0)
        o["recall"] = recall_score(yb[m], yp, zero_division=0)
    except Exception:
        pass
    return o


def evaluate(risk, yt, ye):
    # 风险中性化：坏方法(签名分数含NaN)退化为常数风险~0.5，而非让整个癌种崩
    risk = np.nan_to_num(np.asarray(risk, float), nan=0.0, posinf=0.0, neginf=0.0)
    yt = np.asarray(yt, float); ye = np.asarray(ye)
    m = np.isfinite(yt) & np.isfinite(ye)
    risk, yt, ye = risk[m], yt[m], ye[m]
    try:
        ci = float(lci(yt, -risk, ye))
    except ZeroDivisionError:
        ci = float("nan")
    hi = risk >= np.median(risk)
    p = 1.0
    if hi.sum() and (~hi).sum():
        p = logrank_test(yt[hi], yt[~hi], event_observed_A=ye[hi], event_observed_B=ye[~hi]).p_value
    return {"c_index": ci, "km_p": float(p), **binmetrics(risk, yt, ye)}


# ---------------- 基线签名 ----------------
def sig_grsafs(G, y, L, names):
    top, dirs_full, scores, w, nnz = stage1_select(G, y, L)
    return {"genes": names[top].tolist(), "directions": dirs_full[top].tolist(),
            "scores": scores[top].tolist(), "nnz": nnz, "w_qp": w.tolist()}


def sig_coxnet(G, yt, ye, names, l1):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    m = CoxnetSurvivalAnalysis(l1_ratio=l1, n_alphas=50, max_iter=1000)
    m.fit(G, Surv.from_arrays(ye.astype(bool), yt))
    c = m.coef_[:, -1]
    t = np.argsort(np.abs(c))[::-1][:TOP_K]
    return {"genes": names[t].tolist(), "directions": np.sign(c[t]).tolist(),
            "scores": np.abs(c[t]).tolist()}


def sig_unicox(G, yt, ye, names):
    p = G.shape[1]; pv = np.ones(p); co = np.zeros(p)
    for j in range(p):
        try:
            cph = CoxPHFitter().fit(pd.DataFrame({"T": yt, "E": ye, "g": G[:, j]}),
                                    duration_col="T", event_col="E")
            pv[j] = cph.summary["p"].values[0]; co[j] = cph.summary["coef"].values[0]
        except Exception:
            pass
    t = np.argsort(pv)[:TOP_K]
    return {"genes": names[t].tolist(), "directions": np.sign(co[t]).tolist(),
            "scores": (1 - pv[t]).tolist()}


def sig_rsf(G, yt, ye, names):
    from sksurv.ensemble import RandomSurvivalForest as RSF
    ys = Surv.from_arrays(ye.astype(bool), yt)
    r = RSF(n_estimators=RSF_TREES, max_depth=5, min_samples_leaf=5, random_state=SEED, n_jobs=RF_JOBS).fit(G, ys)
    pre = np.argsort(np.var(G, 0))[::-1][:500]
    base = r.score(G, ys); imp = np.zeros(G.shape[1])
    for idx in pre:
        sc = []
        for _ in range(3):
            Gp = G.copy(); np.random.shuffle(Gp[:, idx]); sc.append(r.score(Gp, ys))
        imp[idx] = base - np.mean(sc)
    t = np.argsort(imp)[::-1][:TOP_K]
    rp = r.predict(G)
    d = [np.sign(np.corrcoef(G[:, i], rp)[0, 1]) if not np.isnan(np.corrcoef(G[:, i], rp)[0, 1]) else 1.0 for i in t]
    return {"genes": names[t].tolist(), "directions": list(map(float, d)), "scores": imp[t].tolist()}


def sig_deepsurv(G, yt, ye, names):
    import torch, torchtuples as tt
    from pycox.models import CoxPH
    torch.manual_seed(SEED)
    x = G.astype(np.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.BatchNorm1d(64), torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.BatchNorm1d(32), torch.nn.Dropout(0.3),
        torch.nn.Linear(32, 1)).cpu()
    m = CoxPH(net, tt.optim.Adam); m.optimizer.set_lr(0.001)
    bs = 64
    while x.shape[0] % bs == 1:           # 避免最后一个 batch 只有1个样本导致 BatchNorm 报错(如HNSC 513)
        bs -= 1
    m.fit(x, (yt.astype(np.float32), ye.astype(np.float32)), batch_size=bs, epochs=DEEPSURV_EPOCHS, verbose=False)
    net.eval()
    xt = torch.tensor(x); base = torch.zeros_like(xt); ig = torch.zeros(x.shape[1])
    for a in np.linspace(0, 1, 50):
        xs = (base + a * (xt - base)).requires_grad_(True)
        net(xs).sum().backward(); ig += xs.grad.abs().mean(0); net.zero_grad()
    grads = (ig / 50).detach().numpy() * np.abs(x).mean(0)
    grads = np.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)   # 深度梯度偶发NaN，中性化
    t = np.argsort(grads)[::-1][:TOP_K]
    with torch.no_grad():
        rp = np.nan_to_num(net(torch.tensor(x)).squeeze().numpy(), nan=0.0)
    d = [np.sign(np.corrcoef(G[:, i], rp)[0, 1]) if not np.isnan(np.corrcoef(G[:, i], rp)[0, 1]) else 1.0 for i in t]
    return {"genes": names[t].tolist(), "directions": list(map(float, d)), "scores": grads[t].tolist()}


def nomogram(expr_c, dev_c, clin_c, sig):
    risk, _ = frozen_risk(expr_c, sig["genes"], sig["directions"], sig["scores"])
    df = pd.DataFrame(index=expr_c.index)
    df["MolRisk"] = risk; df["T"] = dev_c["OS_time"].values; df["E"] = dev_c["OS"].values
    df = pd.concat([df, clin_features(clin_c, expr_c.index)], axis=1).dropna()
    cc = [c for c in df.columns if c not in ["MolRisk", "T", "E"]]
    out = {}
    for name, cols in [("mol_only", ["MolRisk"]), ("clin_only", cc), ("joint", ["MolRisk"] + cc)]:
        try:
            cph = CoxPHFitter(penalizer=0.01).fit(df[cols + ["T", "E"]], duration_col="T", event_col="E")
            out[name] = float(cph.concordance_index_)
            if name == "joint" and "MolRisk" in cph.summary.index:
                out["mol_HR"] = float(np.exp(cph.summary.loc["MolRisk", "coef"]))
                out["mol_p"] = float(cph.summary.loc["MolRisk", "p"])
        except Exception:
            pass
    return out


# ---------------- 单癌种 ----------------
def run(cancer):
    t0 = time.time()
    cdir = ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    clin_p = cdir / "clinical_covariates.tsv"
    clin = pd.read_csv(clin_p, sep="\t", index_col=0) if clin_p.exists() else pd.DataFrame(index=expr.index)
    common = sorted(set(expr.index) & set(dev.index) & set(clin.index))
    expr_c = expr.loc[common]
    if expr_c.isna().any().any():               # 部分癌种(如COAD)表达含缺失，统一用列均值填补，整列缺失填0
        expr_c = expr_c.fillna(expr_c.mean()).fillna(0.0)
        print(f"  [清洗] 表达矩阵含缺失，已用列均值填补", flush=True)
    G = expr_c.values.astype(np.float64)
    y = dev.loc[common, "deviance_residual"].values.astype(np.float64)
    yt = dev.loc[common, "OS_time"].values.astype(float)
    ye = dev.loc[common, "OS"].values.astype(int)
    ysurv = Surv.from_arrays(ye.astype(bool), yt)
    clin_c = clin.loc[common]
    names = np.array([str(g) for g in expr.columns]); n, p = G.shape
    print(f"\n===== {cancer}: {n}x{p}, 事件率={ye.mean():.1%} =====", flush=True)

    # 谱归一化层
    Ls = {}
    for nm, fn in [("coexpr", "L_coexpr.npz"), ("meth", "L_meth_expr.npz"), ("cnv", "L_cnv.npz")]:
        if (cdir / "graph" / fn).exists():
            Ls[nm] = normalize_laplacian(sparse.load_npz(str(cdir / "graph" / fn)), p)
    Lzero = sparse.csr_matrix((p, p), dtype=np.float32)
    Lid = Lzero  # compatibility for legacy cleanup line; zero Laplacian is the true no-graph baseline

    # 双驱动权重
    print("  [权重] Bootstrap 稳定性 + Stage2 增益...", flush=True)
    s_base = bootstrap_stability(G, y, Lzero)
    stab = {k: bootstrap_stability(G, y, L) for k, L in Ls.items()}
    base_top, _, _, _, _ = stage1_select(G, y, Lzero)
    ci_base = stage2_oof_ci(G[:, base_top], clin_c, ysurv)
    deltas, s2 = {}, {}
    for k, L in Ls.items():
        tk, _, _, _, _ = stage1_select(G, y, L)
        s2[k] = stage2_oof_ci(G[:, tk], clin_c, ysurv); deltas[k] = s2[k] - ci_base
    raw = {k: stab[k] * max(deltas[k], 0) for k in Ls}
    tot = sum(raw.values())
    if tot < 1e-15:
        w_opt = {k: 0.0 for k in Ls}; mode = "reject_all_graphs"
    else:
        w_opt = {k: v / tot for k, v in raw.items()}; mode = "dual_driven"
    L_opt = sum(w_opt[k] * Ls[k] for k in Ls)
    print(f"  权重模式={mode}: " + ", ".join(f"{k}={w_opt[k]:.3f}" for k in Ls), flush=True)

    # 签名
    methods = {}
    print("  [签名] GR-SAFS v2/v1/equal ...", flush=True)
    methods["GR-SAFS_v2"] = sig_grsafs(G, y, L_opt, names)
    methods["GR-SAFS_v1"] = sig_grsafs(G, y, Ls["coexpr"], names)
    L_eq = sum((1.0 / len(Ls)) * Ls[k] for k in Ls)
    methods["v2_equal"] = sig_grsafs(G, y, L_eq, names)
    del Ls, L_opt, L_eq, Lid; gc.collect()      # 释放稠密拉普拉斯，降低后续峰值内存
    print("  [签名] 基线 ...", flush=True)
    for nm, fn in [("Cox-Lasso", lambda: sig_coxnet(G, yt, ye, names, 1.0)),
                   ("Cox-EN", lambda: sig_coxnet(G, yt, ye, names, 0.5)),
                   ("Uni-Cox", lambda: sig_unicox(G, yt, ye, names)),
                   ("RSF", lambda: sig_rsf(G, yt, ye, names)),
                   ("DeepSurv", lambda: sig_deepsurv(G, yt, ye, names))]:
        try:
            methods[nm] = fn(); print(f"    {nm} ok", flush=True)
        except Exception as e:
            print(f"    {nm} 失败: {e}", flush=True)

    # 训练集评估
    train = {}
    for nm, s in methods.items():
        if not np.all(np.isfinite(np.asarray(s["scores"], float))):
            print(f"    [警告] {nm} 签名分数含NaN(将中性化)", flush=True)
        r, _ = frozen_risk(expr_c, s["genes"], s["directions"], s["scores"])
        if r is not None:
            try:
                train[nm] = evaluate(r, yt, ye)
            except Exception as e:
                print(f"    [跳过] {nm} 训练评估失败: {e}", flush=True)

    # 外部
    ext = {}
    for ds in EXTERNAL.get(cancer, []):
        ed = cdir / "external" / ds
        if not (ed / "expr.tsv").exists():
            continue
        ee = pd.read_csv(ed / "expr.tsv", sep="\t", index_col=0); ee.columns = [str(c) for c in ee.columns]
        es = pd.read_csv(ed / "survival.tsv", sep="\t")
        es.columns = [c.lower().replace(".", "_") for c in es.columns]
        sc = next((c for c in es.columns if c in ["os", "event", "status", "os_event"]), None)
        tc = next((c for c in es.columns if c in ["os_time", "time", "survival_time"]), None)
        es = es.set_index(es.columns[0])
        cm = sorted(set(ee.index) & set(es.index)); ee = ee.loc[cm]
        eve = pd.to_numeric(es.loc[cm, sc], errors="coerce").values
        tve = pd.to_numeric(es.loc[cm, tc], errors="coerce").values
        v = (~np.isnan(eve)) & (~np.isnan(tve)) & (tve > 0)
        ee, eve, tve = ee.loc[v], eve[v].astype(int), tve[v]
        er = {}
        for nm, s in methods.items():
            r, nmatch = frozen_risk(ee, s["genes"], s["directions"], s["scores"])
            if r is not None:
                try:
                    er[nm] = {**evaluate(r, tve, eve), "n_matched": nmatch}
                except Exception as e:
                    print(f"    [跳过] {ds}/{nm} 外部评估失败: {e}", flush=True)
        ext[ds] = er

    nomo = nomogram(expr_c, dev.loc[common], clin_c, methods["GR-SAFS_v2"]) if cancer == "LUAD" else None
    res = {"cancer": cancer, "n": n, "p": p, "event_rate": float(ye.mean()),
           "weight_mode": mode, "omics_weights": w_opt, "stabilities": stab,
           "stage2_cis": s2, "deltas": deltas, "baseline_ci": ci_base,
           "training": train, "external": ext, "nomogram": nomo,
           "method_genes": {k: v["genes"] for k, v in methods.items()},
           "method_signatures": {k: {"genes": v["genes"], "directions": v["directions"],
                                     "scores": v["scores"]} for k, v in methods.items()},
           "elapsed_min": (time.time() - t0) / 60}
    return res


def main():
    cancers = sys.argv[1:] if len(sys.argv) > 1 else ["LUAD"]
    OUT.mkdir(parents=True, exist_ok=True)
    allr = {}
    for c in cancers:
        try:
            allr[c] = run(c)
        except Exception as e:
            import traceback; traceback.print_exc(); allr[c] = {"error": str(e)}
    tag = "_".join(cancers)
    with open(OUT / f"final_config_comparison_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(allr, f, indent=2, ensure_ascii=False)
    # 摘要表
    print("\n==== 最终配置对比摘要 ====")
    print(f"{'癌种':<6}{'方法':<14}{'训练CI':>9}{'外部CI':>9}{'衰减':>9}")
    for c, r in allr.items():
        if "error" in r:
            print(f"{c} ERROR {r['error']}"); continue
        ds0 = next(iter(r["external"]), None)
        for m in ["GR-SAFS_v2", "GR-SAFS_v1", "v2_equal", "Cox-Lasso", "Cox-EN", "Uni-Cox", "RSF", "DeepSurv"]:
            if m in r["training"]:
                tci = r["training"][m]["c_index"]
                eci = r["external"][ds0][m]["c_index"] if ds0 and m in r["external"][ds0] else None
                es = f"{eci:.4f}" if eci else "N/A"
                dc = f"{tci - eci:+.4f}" if eci else "N/A"
                print(f"{c:<6}{m:<14}{tci:>9.4f}{es:>9}{dc:>9}")
    print(f"\n保存: final_config_comparison_{tag}.json")


if __name__ == "__main__":
    main()

