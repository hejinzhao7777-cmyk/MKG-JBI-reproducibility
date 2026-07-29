"""
Independent-test synthetic ablation of the MKG routing rule.

Routing statistics are estimated on training/validation data only. The routed
graph is then frozen, the selector is refit on training plus validation data,
and performance is evaluated on an untouched test split. Hidden support
recovery is recorded only as a diagnostic and is never used for routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path(__file__).resolve().parent.parent / "10_CMPB_SUBMISSION_LOCK"
P = 300
N = 240
TOP_K = 20
BOOT = 20
RBO_P = 0.9
REPETITIONS = 50
SCENARIOS = [
    "R1_one_reliable",
    "R2_complementary",
    "R3_adversarial",
    "R4_all_harmful",
    "R5_unstable_lure",
]
SCHEMES = ["no_graph", "equal", "stability_only", "utility_only", "joint"]


def rbo_at_k(a, b, p=RBO_P):
    k = min(len(a), len(b))
    s1, s2 = set(), set()
    value = 0.0
    for depth in range(1, k + 1):
        s1.add(int(a[depth - 1]))
        s2.add(int(b[depth - 1]))
        value += (p ** (depth - 1)) * (len(s1 & s2) / depth)
    return (1 - p) * value / (1 - p ** k)


def pairwise_rbo(rankings):
    values = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            values.append(rbo_at_k(rankings[i], rankings[j]))
    return float(np.mean(values)) if values else 1.0


def module_graph(nodes, weight=1.0):
    A = np.zeros((P, P), dtype=float)
    nodes = list(nodes)
    A[np.ix_(nodes, nodes)] = weight
    np.fill_diagonal(A, 0.0)
    return A


def random_noise_graph(rng, nodes, degree=8):
    A = np.zeros((P, P), dtype=float)
    nodes = list(nodes)
    for i in nodes:
        candidates = [j for j in nodes if j != i]
        chosen = rng.choice(
            candidates, size=min(degree, len(candidates)), replace=False
        )
        A[i, chosen] = 1.0
        A[chosen, i] = 1.0
    return A


def normalize(A):
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return A / row_sum


def topk_scores(X, y, A):
    base = np.abs(X.T @ y) / len(y)
    smooth = base + 0.65 * (A @ base)
    return np.argsort(smooth)[::-1][:TOP_K], smooth


def ridge_score(X_train, y_train, X_eval, y_eval, top, alpha=1.0):
    columns = np.asarray(top, dtype=int)
    Xt = X_train[:, columns]
    Xe = X_eval[:, columns]
    gram = Xt.T @ Xt + alpha * np.eye(len(columns))
    try:
        coef = np.linalg.solve(gram, Xt.T @ y_train)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ Xt.T @ y_train
    prediction = Xe @ coef
    return -float(np.mean((y_eval - prediction) ** 2))


def data(rng):
    X = rng.normal(size=(N, P))
    beta = np.zeros(P)
    beta[:10] = 1.0
    beta[10:20] = -0.8
    y = X @ beta + rng.normal(scale=2.2, size=N)
    return X, (y - y.mean()) / y.std()


def scenario_graphs(name, rng):
    true_a = range(0, 10)
    true_b = range(10, 20)
    noise = range(80, 160)
    weak_noise = range(180, 240)
    if name == "R1_one_reliable":
        graphs = {
            "reliable": module_graph(range(0, 20)),
            "neutral": random_noise_graph(rng, weak_noise),
            "adversarial": random_noise_graph(rng, noise),
        }
    elif name == "R2_complementary":
        graphs = {
            "module_A": module_graph(true_a),
            "module_B": module_graph(true_b),
            "adversarial": random_noise_graph(rng, noise),
        }
    elif name == "R3_adversarial":
        graphs = {
            "weak_reliable": module_graph(range(0, 8), weight=0.5),
            "adversarial": random_noise_graph(rng, noise),
            "neutral": random_noise_graph(rng, weak_noise),
        }
    elif name == "R4_all_harmful":
        graphs = {
            "harmful_A": random_noise_graph(rng, noise),
            "harmful_B": random_noise_graph(rng, weak_noise),
            "harmful_C": random_noise_graph(rng, range(240, 300)),
        }
    elif name == "R5_unstable_lure":
        graphs = {
            "stable_partial": module_graph(true_a),
            "unstable_lure": random_noise_graph(
                rng, list(range(0, 20)) + list(range(80, 160)), degree=8
            ),
            "adversarial": random_noise_graph(rng, weak_noise),
        }
    else:
        raise ValueError(name)
    return {layer: normalize(graph) for layer, graph in graphs.items()}


def evaluate_layer(X, y, A, baseline_score, train_idx, val_idx, rng):
    rankings = []
    for _ in range(BOOT):
        sampled = rng.choice(train_idx, size=int(0.8 * len(train_idx)), replace=False)
        top, _ = topk_scores(X[sampled], y[sampled], A)
        rankings.append(top)
    stability = pairwise_rbo(rankings)
    top_train, _ = topk_scores(X[train_idx], y[train_idx], A)
    layer_score = ridge_score(
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], top_train
    )
    return stability, layer_score - baseline_score


def normalized(values):
    total = sum(values.values())
    if total <= 1e-12:
        return {k: 0.0 for k in values}, 1.0
    return {k: values[k] / total for k in values}, 0.0


def route(scheme, stability, gain):
    if scheme == "no_graph":
        return {k: 0.0 for k in stability}, 1.0
    if scheme == "equal":
        return {k: 1.0 / len(stability) for k in stability}, 0.0
    if scheme == "stability_only":
        return normalized({k: stability[k] for k in stability})
    if scheme == "utility_only":
        return normalized({k: max(gain[k], 0.0) for k in gain})
    if scheme == "joint":
        return normalized(
            {k: stability[k] * max(gain[k], 0.0) for k in stability}
        )
    raise ValueError(scheme)


def weighted_graph(graphs, weights):
    if sum(weights.values()) <= 1e-12:
        return np.zeros((P, P), dtype=float)
    return sum(weights[k] * graphs[k] for k in graphs)


def recovery(top):
    return len(set(map(int, top)) & set(range(20))) / 20.0


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    weight_rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for repetition in range(REPETITIONS):
            rng = np.random.default_rng(42000 + 1000 * scenario_index + repetition)
            X, y = data(rng)
            order = rng.permutation(N)
            train_idx = order[: int(0.60 * N)]
            val_idx = order[int(0.60 * N) : int(0.80 * N)]
            test_idx = order[int(0.80 * N) :]
            fit_idx = np.concatenate([train_idx, val_idx])
            graphs = scenario_graphs(scenario, rng)

            top0_train, _ = topk_scores(
                X[train_idx], y[train_idx], np.zeros((P, P))
            )
            validation_baseline = ridge_score(
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                top0_train,
            )
            stability, gain = {}, {}
            for layer, graph in graphs.items():
                stability[layer], gain[layer] = evaluate_layer(
                    X,
                    y,
                    graph,
                    validation_baseline,
                    train_idx,
                    val_idx,
                    rng,
                )

            top0_fit, _ = topk_scores(
                X[fit_idx], y[fit_idx], np.zeros((P, P))
            )
            test_baseline = ridge_score(
                X[fit_idx], y[fit_idx], X[test_idx], y[test_idx], top0_fit
            )

            for scheme in SCHEMES:
                weights, no_relation = route(scheme, stability, gain)
                routed_graph = weighted_graph(graphs, weights)
                top, _ = topk_scores(X[fit_idx], y[fit_idx], routed_graph)
                test_score = ridge_score(
                    X[fit_idx], y[fit_idx], X[test_idx], y[test_idx], top
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "repetition": repetition,
                        "scheme": scheme,
                        "test_gain_vs_no_graph": test_score - test_baseline,
                        "support_recovery": recovery(top),
                        "no_relation": no_relation,
                    }
                )
                for layer in graphs:
                    weight_rows.append(
                        {
                            "scenario": scenario,
                            "repetition": repetition,
                            "scheme": scheme,
                            "layer": layer,
                            "weight": weights[layer],
                            "layer_stability": stability[layer],
                            "validation_gain": gain[layer],
                        }
                    )

    raw = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    summary = (
        raw.groupby(["scenario", "scheme"], as_index=False)
        .agg(
            mean_test_gain=("test_gain_vs_no_graph", "mean"),
            sd_test_gain=("test_gain_vs_no_graph", "std"),
            mean_support_recovery=("support_recovery", "mean"),
            sd_support_recovery=("support_recovery", "std"),
            no_relation_rate=("no_relation", "mean"),
        )
    )
    raw.to_csv(OUTDIR / "CMPB_SYNTHETIC_FIVE_ARM_RAW.csv", index=False)
    weights.to_csv(OUTDIR / "CMPB_SYNTHETIC_ROUTING_WEIGHTS_RAW.csv", index=False)
    summary.to_csv(OUTDIR / "CMPB_SYNTHETIC_FIVE_ARM_SUMMARY.csv", index=False)
    save = {
        "design": {
            "n": N,
            "p": P,
            "top_k": TOP_K,
            "bootstrap": BOOT,
            "repetitions": REPETITIONS,
            "split": "60% routing-train / 20% routing-validation / 20% untouched test",
            "routing_uses_oracle_support": False,
        },
        "summary": summary.to_dict(orient="records"),
    }
    (OUTDIR / "CMPB_SYNTHETIC_FIVE_ARM_SUMMARY.json").write_text(
        json.dumps(save, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
