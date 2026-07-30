"""
Minimum-gain gate sensitivity under right-censored survival simulation.

This experiment is intentionally a routing-mechanism stress test rather than
an end-to-end benchmark of the complete MKG pipeline. It preserves the core
decision contract used by MKG:

1. estimate Top-K selection stability on routing-training data;
2. estimate validation C-index gain relative to a zero-graph selector;
3. freeze graph weights before evaluating an untouched test set.

The analysis varies a prespecified minimum validation-gain margin. Hidden
support is recorded only as a diagnostic and is never used by routing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.util import Surv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    os.environ.get(
        "MKG_OUTPUT_ROOT",
        REPOSITORY_ROOT / "results" / "reruns",
    )
)
OUTDIR = OUTPUT_ROOT / "conservative_gate_sensitivity"

P = 300
N = 360
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
GATE_MARGINS = {
    "joint_m0": 0.000,
    "joint_m005": 0.005,
    "joint_m010": 0.010,
    "joint_m020": 0.020,
}
SCHEMES = list(GATE_MARGINS)


def rbo_at_k(a, b, p=RBO_P):
    k = min(len(a), len(b))
    s1, s2 = set(), set()
    value = 0.0
    for depth in range(1, k + 1):
        s1.add(int(a[depth - 1]))
        s2.add(int(b[depth - 1]))
        value += (p ** (depth - 1)) * (len(s1 & s2) / depth)
    return (1 - p) * value / (1 - p**k)


def pairwise_rbo(rankings):
    values = [
        rbo_at_k(rankings[i], rankings[j])
        for i in range(len(rankings))
        for j in range(i + 1, len(rankings))
    ]
    return float(np.mean(values)) if values else 1.0


def module_graph(nodes, weight=1.0):
    adjacency = np.zeros((P, P), dtype=float)
    nodes = list(nodes)
    adjacency[np.ix_(nodes, nodes)] = weight
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def random_noise_graph(rng, nodes, degree=8):
    adjacency = np.zeros((P, P), dtype=float)
    nodes = list(nodes)
    for i in nodes:
        candidates = [j for j in nodes if j != i]
        chosen = rng.choice(
            candidates, size=min(degree, len(candidates)), replace=False
        )
        adjacency[i, chosen] = 1.0
        adjacency[chosen, i] = 1.0
    return adjacency


def normalize(adjacency):
    row_sum = adjacency.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adjacency / row_sum


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


def generate_survival_data(rng):
    x = rng.normal(size=(N, P))
    beta = np.zeros(P)
    beta[:10] = 0.70
    beta[10:20] = -0.55
    linear_predictor = x @ beta
    linear_predictor = linear_predictor / np.std(linear_predictor)

    event_rate = 0.045 * np.exp(0.75 * linear_predictor)
    event_time = -np.log(rng.uniform(size=N)) / event_rate
    censor_time = -np.log(rng.uniform(size=N)) / 0.030
    observed_time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    return x, observed_time, event, beta


def null_cox_deviance_residual(time, event):
    unique_time, cumulative_hazard = nelson_aalen_estimator(event, time)
    index = np.searchsorted(unique_time, time, side="right") - 1
    hazard = np.where(index >= 0, cumulative_hazard[np.maximum(index, 0)], 0.0)
    martingale = event.astype(float) - hazard
    log_hazard = np.log(np.maximum(hazard, 1e-12))
    inside = -2.0 * (martingale + event.astype(float) * log_hazard)
    return np.sign(martingale) * np.sqrt(np.maximum(inside, 0.0))


def topk_scores(x, residual, adjacency):
    base = np.abs(x.T @ residual) / len(residual)
    smooth = base + 0.65 * (adjacency @ base)
    return np.argsort(smooth)[::-1][:TOP_K], smooth


def survival_cindex(x_train, time_train, event_train, x_eval, time_eval, event_eval, top):
    columns = np.asarray(top, dtype=int)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train[:, columns])
    eval_scaled = scaler.transform(x_eval[:, columns])
    model = CoxPHSurvivalAnalysis(alpha=1.0, n_iter=100)
    model.fit(
        train_scaled,
        Surv.from_arrays(event_train.astype(bool), time_train.astype(float)),
    )
    risk = model.predict(eval_scaled)
    return float(
        concordance_index_censored(
            event_eval.astype(bool), time_eval.astype(float), risk
        )[0]
    )


def evaluate_layer(
    x,
    time,
    event,
    residual,
    adjacency,
    baseline_cindex,
    train_idx,
    val_idx,
    rng,
):
    rankings = []
    for _ in range(BOOT):
        sampled = rng.choice(
            train_idx, size=int(0.8 * len(train_idx)), replace=False
        )
        top, _ = topk_scores(x[sampled], residual[sampled], adjacency)
        rankings.append(top)
    stability = pairwise_rbo(rankings)
    top_train, _ = topk_scores(x[train_idx], residual[train_idx], adjacency)
    layer_cindex = survival_cindex(
        x[train_idx],
        time[train_idx],
        event[train_idx],
        x[val_idx],
        time[val_idx],
        event[val_idx],
        top_train,
    )
    return stability, layer_cindex - baseline_cindex


def normalized(values):
    total = sum(values.values())
    if total <= 1e-12:
        return {key: 0.0 for key in values}, 1.0
    return {key: values[key] / total for key in values}, 0.0


def route(scheme, stability, gain):
    margin = GATE_MARGINS[scheme]
    return normalized(
        {
            key: stability[key] * gain[key] if gain[key] > margin else 0.0
            for key in stability
        }
    )


def weighted_graph(graphs, weights):
    if sum(weights.values()) <= 1e-12:
        return np.zeros((P, P), dtype=float)
    return sum(weights[key] * graphs[key] for key in graphs)


def support_recovery(top):
    return len(set(map(int, top)) & set(range(20))) / 20.0


def confidence_summary(raw):
    summary = (
        raw.groupby(["scenario", "scheme"], as_index=False)
        .agg(
            n=("test_cindex_gain", "size"),
            mean_test_cindex=("test_cindex", "mean"),
            mean_test_gain=("test_cindex_gain", "mean"),
            sd_test_gain=("test_cindex_gain", "std"),
            mean_support_recovery=("support_recovery", "mean"),
            sd_support_recovery=("support_recovery", "std"),
            no_relation_rate=("no_relation", "mean"),
            mean_censoring_fraction=("censoring_fraction", "mean"),
        )
    )
    se = summary["sd_test_gain"] / np.sqrt(summary["n"])
    summary["test_gain_ci_low"] = summary["mean_test_gain"] - 1.96 * se
    summary["test_gain_ci_high"] = summary["mean_test_gain"] + 1.96 * se
    return summary


def paired_gate_summary(raw, bootstrap_repetitions=50000):
    pivot = raw.pivot_table(
        index=["scenario", "repetition"],
        columns="scheme",
        values="test_cindex_gain",
    )
    rng = np.random.default_rng(20260729)
    rows = []
    for scenario, sub in pivot.groupby(level=0):
        for scheme in SCHEMES[1:]:
            difference = (sub[scheme] - sub["joint_m0"]).to_numpy()
            bootstrap_index = rng.integers(
                0,
                len(difference),
                size=(bootstrap_repetitions, len(difference)),
            )
            bootstrap_means = difference[bootstrap_index].mean(axis=1)
            rows.append(
                {
                    "scenario": scenario,
                    "scheme": scheme,
                    "reference": "joint_m0",
                    "n": len(difference),
                    "mean_difference": float(difference.mean()),
                    "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                    "bootstrap_repetitions": bootstrap_repetitions,
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    palette = {
        "joint_m0": "#496A9B",
        "joint_m005": "#4E9A8D",
        "joint_m010": "#D9B36C",
        "joint_m020": "#B45C5C",
    }
    labels = {
        "joint_m0": "Margin 0",
        "joint_m005": "Margin 0.005",
        "joint_m010": "Margin 0.010",
        "joint_m020": "Margin 0.020",
    }
    scenario_labels = ["R1", "R2", "R3", "R4", "R5"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )
    x = np.arange(len(SCENARIOS))
    offsets = np.linspace(-0.24, 0.24, len(SCHEMES))
    for offset, scheme in zip(offsets, SCHEMES):
        sub = summary.set_index(["scenario", "scheme"]).loc[
            [(scenario, scheme) for scenario in SCENARIOS]
        ]
        y = sub["mean_test_gain"].to_numpy()
        lower = y - sub["test_gain_ci_low"].to_numpy()
        upper = sub["test_gain_ci_high"].to_numpy() - y
        axes[0].errorbar(
            x + offset,
            y,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            ms=3.5,
            capsize=2,
            color=palette[scheme],
            label=labels[scheme],
            linewidth=1,
        )
    axes[0].axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, scenario_labels)
    axes[0].set_ylabel("Untouched-test C-index gain\nrelative to no graph")
    axes[0].set_xlabel("Controlled graph-reliability scenario")
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))

    indexed = summary.set_index(["scenario", "scheme"])
    rejection = np.asarray(
        [
            [
                indexed.loc[(scenario, scheme), "no_relation_rate"]
                for scenario in SCENARIOS
            ]
            for scheme in SCHEMES
        ]
    )
    heatmap = axes[1].imshow(
        rejection,
        vmin=0,
        vmax=1,
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(rejection.shape[0]):
        for column in range(rejection.shape[1]):
            value = rejection[row, column]
            axes[1].text(
                column,
                row,
                f"{int(round(100 * value))}%",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=6.5,
            )
    axes[1].set_xticks(x, scenario_labels)
    axes[1].set_yticks(np.arange(len(SCHEMES)), [labels[item] for item in SCHEMES])
    axes[1].set_xlabel("Controlled graph-reliability scenario")
    axes[1].set_ylabel("Minimum validation-gain gate")
    colorbar = fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("No-relation activation rate")

    axes[0].text(-0.13, 1.12, "a", transform=axes[0].transAxes, weight="bold", size=9)
    axes[1].text(-0.20, 1.12, "b", transform=axes[1].transAxes, weight="bold", size=9)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.98, wspace=0.34)

    stem = OUTDIR / "Fig_CMPB_conservative_gate_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    weight_rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for repetition in range(REPETITIONS):
            rng = np.random.default_rng(96000 + 1000 * scenario_index + repetition)
            x, time, event, _ = generate_survival_data(rng)
            order = rng.permutation(N)
            train_idx = order[: int(0.60 * N)]
            val_idx = order[int(0.60 * N) : int(0.80 * N)]
            test_idx = order[int(0.80 * N) :]
            fit_idx = np.concatenate([train_idx, val_idx])
            graphs = scenario_graphs(scenario, rng)

            residual = np.zeros(N, dtype=float)
            residual[train_idx] = null_cox_deviance_residual(
                time[train_idx], event[train_idx]
            )
            top0_train, _ = topk_scores(
                x[train_idx],
                residual[train_idx],
                np.zeros((P, P), dtype=float),
            )
            validation_baseline = survival_cindex(
                x[train_idx],
                time[train_idx],
                event[train_idx],
                x[val_idx],
                time[val_idx],
                event[val_idx],
                top0_train,
            )

            stability, gain = {}, {}
            for layer, graph in graphs.items():
                stability[layer], gain[layer] = evaluate_layer(
                    x,
                    time,
                    event,
                    residual,
                    graph,
                    validation_baseline,
                    train_idx,
                    val_idx,
                    rng,
                )

            residual_fit = null_cox_deviance_residual(
                time[fit_idx], event[fit_idx]
            )
            top0_fit, _ = topk_scores(
                x[fit_idx],
                residual_fit,
                np.zeros((P, P), dtype=float),
            )
            test_baseline = survival_cindex(
                x[fit_idx],
                time[fit_idx],
                event[fit_idx],
                x[test_idx],
                time[test_idx],
                event[test_idx],
                top0_fit,
            )

            for scheme in SCHEMES:
                weights, no_relation = route(scheme, stability, gain)
                routed_graph = weighted_graph(graphs, weights)
                top, _ = topk_scores(x[fit_idx], residual_fit, routed_graph)
                test_cindex = survival_cindex(
                    x[fit_idx],
                    time[fit_idx],
                    event[fit_idx],
                    x[test_idx],
                    time[test_idx],
                    event[test_idx],
                    top,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "repetition": repetition,
                        "scheme": scheme,
                        "test_cindex": test_cindex,
                        "test_cindex_gain": test_cindex - test_baseline,
                        "support_recovery": support_recovery(top),
                        "no_relation": no_relation,
                        "censoring_fraction": 1.0 - float(np.mean(event)),
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
                            "validation_cindex_gain": gain[layer],
                        }
                    )

            if (repetition + 1) % 10 == 0:
                print(
                    f"{scenario}: completed {repetition + 1}/{REPETITIONS}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    summary = confidence_summary(raw)
    paired = paired_gate_summary(raw)
    raw.to_csv(OUTDIR / "CMPB_CONSERVATIVE_GATE_RAW.csv", index=False)
    weights.to_csv(
        OUTDIR / "CMPB_CONSERVATIVE_GATE_WEIGHTS.csv", index=False
    )
    summary.to_csv(
        OUTDIR / "CMPB_CONSERVATIVE_GATE_SUMMARY.csv", index=False
    )
    paired.to_csv(
        OUTDIR / "CMPB_CONSERVATIVE_GATE_PAIRED.csv", index=False
    )
    metadata = {
        "design": {
            "n": N,
            "p": P,
            "top_k": TOP_K,
            "bootstrap": BOOT,
            "repetitions": REPETITIONS,
            "split": "60% routing-train / 20% routing-validation / 20% untouched test",
            "outcome": "right-censored proportional-hazards survival time",
            "routing_uses_oracle_support": False,
            "gate_margins": GATE_MARGINS,
            "eligibility_rule": "validation C-index gain strictly exceeds margin",
            "interval": "normal-approximation 95% Monte Carlo interval across repetitions",
        },
        "summary": summary.to_dict(orient="records"),
    }
    (OUTDIR / "CMPB_CONSERVATIVE_GATE_SUMMARY.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_summary(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
