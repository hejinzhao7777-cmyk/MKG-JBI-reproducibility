import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
FIG = OUT.parent / "07_论文初稿" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
P = 300
N = 240
TOP_K = 20
BOOT = 20
RBO_P = 0.9


def rbo_at_k(a, b, p=RBO_P):
    k = min(len(a), len(b))
    s1, s2 = set(), set()
    val = 0.0
    for d in range(1, k + 1):
        s1.add(int(a[d - 1]))
        s2.add(int(b[d - 1]))
        val += (p ** (d - 1)) * (len(s1 & s2) / d)
    return (1 - p) * val / (1 - p ** k)


def pairwise_rbo(rankings):
    vals = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            vals.append(rbo_at_k(rankings[i], rankings[j]))
    return float(np.mean(vals)) if vals else 1.0


def module_graph(nodes, p=P, weight=1.0):
    A = np.zeros((p, p), dtype=float)
    nodes = list(nodes)
    for i in nodes:
        for j in nodes:
            if i != j:
                A[i, j] = weight
    return A


def random_noise_graph(nodes, p=P, degree=8):
    A = np.zeros((p, p), dtype=float)
    nodes = list(nodes)
    for i in nodes:
        js = RNG.choice(nodes, size=min(degree, len(nodes)), replace=False)
        for j in js:
            if i != j:
                A[i, j] = A[j, i] = 1.0
    return A


def normalize(A):
    row = A.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return A / row


def topk_scores(X, y, A):
    base = np.abs(X.T @ y) / len(y)
    smooth = base + 0.65 * (normalize(A) @ base)
    return np.argsort(smooth)[::-1][:TOP_K], smooth


def ridge_predictive_score(X_train, y_train, X_val, y_val, top, alpha=1.0):
    """Observable validation score used for routing; no true support is used."""
    cols = np.asarray(top, dtype=int)
    Xt = X_train[:, cols]
    Xv = X_val[:, cols]
    gram = Xt.T @ Xt + alpha * np.eye(len(cols))
    try:
        coef = np.linalg.solve(gram, Xt.T @ y_train)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ Xt.T @ y_train
    pred = Xv @ coef
    mse = float(np.mean((y_val - pred) ** 2))
    return -mse


def recovery(top, truth):
    """Hidden diagnostic only, never used by the routing rule."""
    return len(set(map(int, top)) & set(truth)) / len(truth)


def evaluate_layer(X, y, A, baseline_score, train_idx, val_idx):
    rankings = []
    for _ in range(BOOT):
        idx = RNG.choice(len(y), size=int(0.8 * len(y)), replace=False)
        top, score = topk_scores(X[idx], y[idx], A)
        rankings.append(top)
    stability = pairwise_rbo(rankings)
    top_train, _ = topk_scores(X[train_idx], y[train_idx], A)
    layer_score = ridge_predictive_score(
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], top_train
    )
    gain = layer_score - baseline_score
    top_full, _ = topk_scores(X, y, A)
    hidden_recovery = recovery(top_full, set(range(20)))
    return stability, gain, hidden_recovery


def route(stability, gain):
    util = {k: stability[k] * max(gain[k], 0) for k in stability}
    total = sum(util.values())
    if total <= 1e-12:
        w = {k: 0.0 for k in stability}
        w["no_relation"] = 1.0
    else:
        w = {k: util[k] / total for k in stability}
        w["no_relation"] = 0.0
    return w


def data():
    X = RNG.normal(size=(N, P))
    beta = np.zeros(P)
    beta[:10] = 1.0
    beta[10:20] = -0.8
    y = X @ beta + RNG.normal(scale=2.2, size=N)
    y = (y - y.mean()) / y.std()
    return X, y


def scenario_graphs(name):
    true1 = range(0, 10)
    true2 = range(10, 20)
    noise = range(80, 160)
    weak_noise = range(180, 240)
    if name == "R1_one_reliable":
        return {
            "reliable": module_graph(range(0, 20)),
            "neutral": random_noise_graph(weak_noise),
            "adversarial": random_noise_graph(noise),
        }
    if name == "R2_complementary":
        return {
            "module_A": module_graph(true1),
            "module_B": module_graph(true2),
            "adversarial": random_noise_graph(noise),
        }
    if name == "R3_adversarial":
        return {
            "weak_reliable": module_graph(range(0, 8), weight=0.5),
            "adversarial": random_noise_graph(noise),
            "neutral": random_noise_graph(weak_noise),
        }
    if name == "R4_all_harmful":
        return {
            "harmful_A": random_noise_graph(noise),
            "harmful_B": random_noise_graph(weak_noise),
            "harmful_C": random_noise_graph(range(240, 300)),
        }
    raise ValueError(name)


def run():
    rows = []
    for scen in ["R1_one_reliable", "R2_complementary", "R3_adversarial", "R4_all_harmful"]:
        weights = []
        gains = []
        stabs = []
        for rep in range(50):
            X, y = data()
            perm = RNG.permutation(len(y))
            train_idx = perm[: int(0.7 * len(y))]
            val_idx = perm[int(0.7 * len(y)) :]
            top0, score0 = topk_scores(X[train_idx], y[train_idx], np.zeros((P, P)))
            baseline = ridge_predictive_score(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], top0
            )
            graphs = scenario_graphs(scen)
            stability, gain, hidden = {}, {}, {}
            for name, A in graphs.items():
                stability[name], gain[name], hidden[name] = evaluate_layer(
                    X, y, A, baseline, train_idx, val_idx
                )
            w = route(stability, gain)
            weights.append(w)
            gains.append(gain)
            stabs.append(stability)
        keys = sorted(set().union(*[w.keys() for w in weights]))
        for key in keys:
            rows.append({
                "scenario": scen,
                "layer": key,
                "mean_weight": float(np.mean([w.get(key, 0.0) for w in weights])),
                "sd_weight": float(np.std([w.get(key, 0.0) for w in weights])),
                "mean_gain": float(np.mean([g.get(key, np.nan) for g in gains if key in g])) if key != "no_relation" else np.nan,
                "mean_stability": float(np.mean([s.get(key, np.nan) for s in stabs if key in s])) if key != "no_relation" else np.nan,
                "routing_gain_definition": "validation ridge negative-MSE gain versus no-relation baseline; no true support used",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "routing_reliability_simulation.csv", index=False)
    json.dump(rows, open(OUT / "routing_reliability_simulation.json", "w", encoding="utf-8"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    piv = df.pivot(index="scenario", columns="layer", values="mean_weight").fillna(0)
    order = ["R1_one_reliable", "R2_complementary", "R3_adversarial", "R4_all_harmful"]
    piv = piv.loc[order]
    colors = {
        "reliable": "#2E7D32", "module_A": "#2E7D32", "module_B": "#66A61E",
        "weak_reliable": "#80B1D3", "neutral": "#9E9E9E", "adversarial": "#C62828",
        "harmful_A": "#D95F02", "harmful_B": "#E6AB02", "harmful_C": "#A6761D",
        "no_relation": "#4A4A4A"
    }
    bottom = np.zeros(len(piv))
    for layer in piv.columns:
        vals = piv[layer].values
        axes[0].bar(range(len(piv)), vals, bottom=bottom, label=layer, color=colors.get(layer, "#7570B3"))
        bottom += vals
    axes[0].set_xticks(range(len(piv)))
    axes[0].set_xticklabels(["R1\none reliable", "R2\ncomplementary", "R3\nadversarial", "R4\nall harmful"])
    axes[0].set_ylabel("Mean routing weight")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.28))
    axes[0].set_title("A. Routing weights")

    gain_df = df[df["layer"] != "no_relation"].copy()
    gain_df["scenario_short"] = gain_df["scenario"].map({
        "R1_one_reliable": "R1", "R2_complementary": "R2",
        "R3_adversarial": "R3", "R4_all_harmful": "R4"
    })
    x = np.arange(len(gain_df))
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].bar(x, gain_df["mean_gain"], color=[colors.get(v, "#7570B3") for v in gain_df["layer"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{a}\n{b}" for a, b in zip(gain_df["scenario_short"], gain_df["layer"])],
                            rotation=90, fontsize=7)
    axes[1].set_ylabel("Mean validation prediction gain")
    axes[1].set_title("B. Utility signal")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "FigJBI_routing_reliability_simulation.pdf")
    fig.savefig(FIG / "FigJBI_routing_reliability_simulation.png", dpi=300)


if __name__ == "__main__":
    run()

