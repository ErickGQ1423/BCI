#!/usr/bin/env python3
"""Evaluate counterfactual online decision rules from saved event logs.

This script does not touch the real online driver, FES, robot, model files, or
configuration.  It only parses the probabilities already printed during online
runs and asks: "what would have happened if this rule had controlled?"
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_ROOT = Path(
    "/home/lab-admin/Documents/CNVStudy/"
    "sub-CNV_PILOT_SUBJ_025/ses-S001_ONLINE/logs"
)
DEFAULT_OUTPUT = Path(
    "/home/lab-admin/Documents/CNVStudy/"
    "sub-CNV_PILOT_SUBJ_025/ses-S001_ONLINE/reports/"
    "pseudo_online_decision_rules_SUBJ025.csv"
)

MODEL_NAMES = ["MDM", "LDA", "LDA3", "LR", "SVM"]

STEP_RE = re.compile(
    r"\[M2_step\]\s+paso=(?P<step>\d+)/(?P<n_steps>\d+)\s+"
    r"t=(?P<time>[+-]?\d+\.\d+)s\s+.*?"
    r"MDM_PMI=(?P<MDM>NA|[0-9.]+)\s+"
    r"LDA=(?P<LDA>NA|[0-9.]+)\s+"
    r"LDA3=(?P<LDA3>NA|[0-9.]+)\s+"
    r"LR=(?P<LR>NA|[0-9.]+)\s+"
    r"SVM=(?P<SVM>NA|[0-9.]+)"
)
TRIAL_START_RE = re.compile(r"--- Trial (?P<trial>\d+)/\d+ START ---")
TRIGGER_RE = re.compile(r"\[TRIGGER\] Sent opcode: (?P<code>100|110|200|210)")
FULL_RE = re.compile(r"\[FULL_WINDOW_OBSERVERS\] trial=(?P<trial>\d+) (?P<rest>.+)")
FULL_TARGET_RE = re.compile(r"target=(100|200)")
FULL_SCORE_RE = re.compile(r"(?P<model>MDM|LDA_shrink|LDA_shrink_3ch|LR|SVM)_PMI=(?P<p>[0-9.]+)")


@dataclass
class Trial:
    run: str
    trial: int
    target: int | None = None
    steps: list[dict[str, float]] = field(default_factory=list)
    full: dict[str, float] = field(default_factory=dict)


def parse_probability(value: str) -> float | None:
    if value == "NA":
        return None
    return float(value)


def parse_event_log(log_path: Path) -> list[Trial]:
    run_name = log_path.parent.name
    trials: list[Trial] = []
    current: Trial | None = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := TRIAL_START_RE.search(line):
            current = Trial(run=run_name, trial=int(match.group("trial")))
            trials.append(current)
            continue

        if current is None:
            continue

        if match := TRIGGER_RE.search(line):
            code = int(match.group("code"))
            if code == 210:
                current.target = 200
            elif code == 110:
                current.target = 100
            continue

        if match := STEP_RE.search(line):
            row = {
                "step": float(match.group("step")),
                "time": float(match.group("time")),
            }
            for model in MODEL_NAMES:
                probability = parse_probability(match.group(model))
                if probability is not None:
                    row[model] = probability
            current.steps.append(row)
            continue

        if match := FULL_RE.search(line):
            if int(match.group("trial")) != current.trial:
                continue
            rest = match.group("rest")
            if target_match := FULL_TARGET_RE.search(rest):
                current.target = int(target_match.group(1))
            for score_match in FULL_SCORE_RE.finditer(rest):
                model = score_match.group("model")
                if model == "LDA_shrink":
                    model = "LDA"
                elif model == "LDA_shrink_3ch":
                    model = "LDA3"
                current.full[model] = float(score_match.group("p"))

    return [trial for trial in trials if trial.target in (100, 200)]


def load_trials(log_root: Path) -> list[Trial]:
    log_paths = sorted(log_root.glob("ONLINE_*_run-*/event_log.txt"))
    if not log_paths:
        raise FileNotFoundError(f"No event_log.txt files found under {log_root}")
    trials: list[Trial] = []
    for log_path in log_paths:
        trials.extend(parse_event_log(log_path))
    return trials


def classify_p_mi(
    p_mi: float | None,
    mi_threshold: float,
    rest_threshold: float,
) -> int | None:
    if p_mi is None or math.isnan(p_mi):
        return None
    if p_mi >= mi_threshold:
        return 200
    if p_mi <= rest_threshold:
        return 100
    return None


def endpoint_probability(
    trial: Trial,
    model: str,
    endpoint: float,
) -> float | None:
    candidates = [row for row in trial.steps if model in row]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(row["time"] - endpoint))[model]


def mean_probability(trial: Trial, model: str, max_time: float = -0.50) -> float | None:
    values = [
        row[model]
        for row in trial.steps
        if model in row and row["time"] <= max_time
    ]
    return mean(values) if values else None


def last_n_mean_probability(
    trial: Trial,
    model: str,
    n_last: int = 3,
    max_time: float = -0.50,
) -> float | None:
    values = [
        row[model]
        for row in trial.steps
        if model in row and row["time"] <= max_time
    ]
    values = values[-n_last:]
    return mean(values) if values else None


def weighted_probability(
    trial: Trial,
    model: str,
    max_time: float = -0.50,
) -> float | None:
    values = [
        row[model]
        for row in trial.steps
        if model in row and row["time"] <= max_time
    ]
    if not values:
        return None
    weights = list(range(1, len(values) + 1))
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def majority_decision(
    trial: Trial,
    model: str,
    max_time: float = -0.50,
    p_cut: float = 0.50,
    min_vote_fraction: float = 0.60,
) -> int | None:
    values = [
        row[model]
        for row in trial.steps
        if model in row and row["time"] <= max_time
    ]
    if not values:
        return None
    mi_votes = sum(value >= p_cut for value in values)
    rest_votes = len(values) - mi_votes
    winner_votes = max(mi_votes, rest_votes)
    if winner_votes / len(values) < min_vote_fraction:
        return None
    return 200 if mi_votes > rest_votes else 100


def majority_probability(
    trial: Trial,
    model: str,
    max_time: float = -0.50,
    p_cut: float = 0.50,
    min_vote_fraction: float = 0.70,
) -> float | None:
    values = [
        row[model]
        for row in trial.steps
        if model in row and row["time"] <= max_time
    ]
    if not values:
        return None
    mi_votes = sum(value >= p_cut for value in values)
    rest_votes = len(values) - mi_votes
    winner_votes = max(mi_votes, rest_votes)
    if winner_votes / len(values) < min_vote_fraction:
        return None
    return 1.0 if mi_votes > rest_votes else 0.0


def hybrid_endpoint_then_weighted(
    trial: Trial,
    model: str,
    endpoint: float = -0.50,
    endpoint_mi_threshold: float = 0.70,
    endpoint_rest_threshold: float = 0.30,
    weighted_mi_threshold: float = 0.60,
    weighted_rest_threshold: float = 0.40,
) -> int | None:
    endpoint_p = endpoint_probability(trial, model, endpoint)
    endpoint_decision = classify_p_mi(
        endpoint_p,
        endpoint_mi_threshold,
        endpoint_rest_threshold,
    )
    if endpoint_decision is not None:
        return endpoint_decision

    weighted_p = weighted_probability(trial, model, endpoint)
    return classify_p_mi(
        weighted_p,
        weighted_mi_threshold,
        weighted_rest_threshold,
    )


def hybrid_endpoint_weighted_majority_full(
    trial: Trial,
    model: str,
    endpoint: float = -0.50,
    endpoint_mi_threshold: float = 0.70,
    endpoint_rest_threshold: float = 0.30,
    weighted_mi_threshold: float = 0.70,
    weighted_rest_threshold: float = 0.30,
    majority_vote_fraction: float = 0.70,
    full_mi_threshold: float = 0.70,
    full_rest_threshold: float = 0.30,
) -> int | None:
    endpoint_p = endpoint_probability(trial, model, endpoint)
    endpoint_decision = classify_p_mi(
        endpoint_p,
        endpoint_mi_threshold,
        endpoint_rest_threshold,
    )
    if endpoint_decision is not None:
        return endpoint_decision

    weighted_p = weighted_probability(trial, model, endpoint)
    weighted_decision = classify_p_mi(
        weighted_p,
        weighted_mi_threshold,
        weighted_rest_threshold,
    )
    if weighted_decision is not None:
        return weighted_decision

    majority_p = majority_probability(
        trial,
        model,
        endpoint,
        min_vote_fraction=majority_vote_fraction,
    )
    majority_decision_ = classify_p_mi(majority_p, 0.5, 0.5)
    if majority_decision_ is not None:
        return majority_decision_

    full_p = full_window_probability(trial, model)
    return classify_p_mi(full_p, full_mi_threshold, full_rest_threshold)


def viewer_consensus_decision(
    trial: Trial,
    endpoint: float = -0.50,
    required_votes: int = 3,
    viewer_models: tuple[str, ...] = ("LDA", "LDA3", "LR", "SVM"),
) -> int | None:
    mi_votes = 0
    rest_votes = 0
    for viewer_model in viewer_models:
        p_mi = endpoint_probability(trial, viewer_model, endpoint)
        if p_mi is None:
            continue
        if p_mi >= 0.5:
            mi_votes += 1
        else:
            rest_votes += 1
    if mi_votes >= required_votes:
        return 200
    if rest_votes >= required_votes:
        return 100
    return None


def viewer_temporal_majority_consensus(
    trial: Trial,
    endpoint: float = -0.50,
    required_viewer_votes: int = 3,
    viewer_models: tuple[str, ...] = ("LDA", "LDA3", "LR", "SVM"),
    min_temporal_vote_fraction: float = 0.60,
) -> int | None:
    viewer_mi_votes = 0
    viewer_rest_votes = 0
    for viewer_model in viewer_models:
        decision = majority_decision(
            trial,
            viewer_model,
            max_time=endpoint,
            p_cut=0.50,
            min_vote_fraction=min_temporal_vote_fraction,
        )
        if decision == 200:
            viewer_mi_votes += 1
        elif decision == 100:
            viewer_rest_votes += 1
    if viewer_mi_votes >= required_viewer_votes:
        return 200
    if viewer_rest_votes >= required_viewer_votes:
        return 100
    return None


def hybrid_mdm_then_viewers(
    trial: Trial,
    endpoint: float = -0.50,
    endpoint_mi_threshold: float = 0.70,
    endpoint_rest_threshold: float = 0.30,
    weighted_mi_threshold: float = 0.70,
    weighted_rest_threshold: float = 0.30,
    required_viewer_votes: int = 3,
) -> int | None:
    mdm_endpoint_p = endpoint_probability(trial, "MDM", endpoint)
    mdm_endpoint_decision = classify_p_mi(
        mdm_endpoint_p,
        endpoint_mi_threshold,
        endpoint_rest_threshold,
    )
    if mdm_endpoint_decision is not None:
        return mdm_endpoint_decision

    mdm_weighted_p = weighted_probability(trial, "MDM", endpoint)
    mdm_weighted_decision = classify_p_mi(
        mdm_weighted_p,
        weighted_mi_threshold,
        weighted_rest_threshold,
    )
    if mdm_weighted_decision is not None:
        return mdm_weighted_decision

    return viewer_consensus_decision(
        trial,
        endpoint=endpoint,
        required_votes=required_viewer_votes,
    )


def hybrid_mdm_then_viewer_temporal_majority(
    trial: Trial,
    endpoint: float = -0.50,
    endpoint_mi_threshold: float = 0.70,
    endpoint_rest_threshold: float = 0.30,
    weighted_mi_threshold: float = 0.70,
    weighted_rest_threshold: float = 0.30,
    required_viewer_votes: int = 3,
    min_temporal_vote_fraction: float = 0.60,
) -> int | None:
    mdm_endpoint_p = endpoint_probability(trial, "MDM", endpoint)
    mdm_endpoint_decision = classify_p_mi(
        mdm_endpoint_p,
        endpoint_mi_threshold,
        endpoint_rest_threshold,
    )
    if mdm_endpoint_decision is not None:
        return mdm_endpoint_decision

    mdm_weighted_p = weighted_probability(trial, "MDM", endpoint)
    mdm_weighted_decision = classify_p_mi(
        mdm_weighted_p,
        weighted_mi_threshold,
        weighted_rest_threshold,
    )
    if mdm_weighted_decision is not None:
        return mdm_weighted_decision

    return viewer_temporal_majority_consensus(
        trial,
        endpoint=endpoint,
        required_viewer_votes=required_viewer_votes,
        min_temporal_vote_fraction=min_temporal_vote_fraction,
    )


def full_window_probability(trial: Trial, model: str) -> float | None:
    return trial.full.get(model)


def score_predictions(trials: list[Trial], predictions: list[int | None]) -> dict[str, float | int]:
    total = len(trials)
    correct = 0
    incorrect = 0
    ambiguous = 0
    mi_correct = mi_total = 0
    rest_correct = rest_total = 0
    false_mi = 0
    false_rest = 0

    for trial, prediction in zip(trials, predictions):
        target = trial.target
        if target == 200:
            mi_total += 1
        else:
            rest_total += 1

        if prediction is None:
            ambiguous += 1
            continue

        if prediction == target:
            correct += 1
            if target == 200:
                mi_correct += 1
            else:
                rest_correct += 1
        else:
            incorrect += 1
            if prediction == 200:
                false_mi += 1
            else:
                false_rest += 1

    decided = correct + incorrect
    return {
        "n": total,
        "correct": correct,
        "incorrect": incorrect,
        "ambiguous": ambiguous,
        "coverage": decided / total if total else float("nan"),
        "total_accuracy": correct / total if total else float("nan"),
        "decision_accuracy": correct / decided if decided else float("nan"),
        "mi_recall": mi_correct / mi_total if mi_total else float("nan"),
        "rest_recall": rest_correct / rest_total if rest_total else float("nan"),
        "false_mi": false_mi,
        "false_rest": false_rest,
    }


def evaluate_rules(
    trials: list[Trial],
    mi_threshold: float,
    rest_threshold: float,
    endpoint: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    rules = [
        ("endpoint", lambda trial, model: endpoint_probability(trial, model, endpoint)),
        ("mean_until_endpoint", lambda trial, model: mean_probability(trial, model, endpoint)),
        ("weighted_until_endpoint", lambda trial, model: weighted_probability(trial, model, endpoint)),
        ("last3_mean", lambda trial, model: last_n_mean_probability(trial, model, 3, endpoint)),
        ("full_window", lambda trial, model: full_window_probability(trial, model)),
    ]

    hybrid_specs = [
        ("hybrid_ep70_w60", 0.70, 0.30, 0.60, 0.40),
        ("hybrid_ep70_w65", 0.70, 0.30, 0.65, 0.35),
        ("hybrid_ep70_w70", 0.70, 0.30, 0.70, 0.30),
    ]

    for model in MODEL_NAMES:
        if model == "MDM":
            viewer_hybrid_predictions = [
                hybrid_mdm_then_viewers(trial, endpoint)
                for trial in trials
            ]
            rows.append({
                "model": model,
                "rule": "hybrid_mdm_viewers3",
                **score_predictions(trials, viewer_hybrid_predictions),
            })
            viewer_temporal_hybrid_predictions = [
                hybrid_mdm_then_viewer_temporal_majority(trial, endpoint)
                for trial in trials
            ]
            rows.append({
                "model": model,
                "rule": "hybrid_mdm_viewers3_temporal",
                **score_predictions(trials, viewer_temporal_hybrid_predictions),
            })

        full_hybrid_predictions = [
            hybrid_endpoint_weighted_majority_full(trial, model, endpoint)
            for trial in trials
        ]
        rows.append({
            "model": model,
            "rule": "hybrid_ep_w_maj_full",
            **score_predictions(trials, full_hybrid_predictions),
        })

        for (
            hybrid_name,
            endpoint_mi,
            endpoint_rest,
            weighted_mi,
            weighted_rest,
        ) in hybrid_specs:
            hybrid_predictions = [
                hybrid_endpoint_then_weighted(
                    trial,
                    model,
                    endpoint,
                    endpoint_mi_threshold=endpoint_mi,
                    endpoint_rest_threshold=endpoint_rest,
                    weighted_mi_threshold=weighted_mi,
                    weighted_rest_threshold=weighted_rest,
                )
                for trial in trials
            ]
            rows.append({
                "model": model,
                "rule": hybrid_name,
                **score_predictions(trials, hybrid_predictions),
            })

        for rule_name, probability_fn in rules:
            predictions = [
                classify_p_mi(probability_fn(trial, model), mi_threshold, rest_threshold)
                for trial in trials
            ]
            row = {
                "model": model,
                "rule": rule_name,
                **score_predictions(trials, predictions),
            }
            rows.append(row)

        predictions = [majority_decision(trial, model, endpoint) for trial in trials]
        rows.append({
            "model": model,
            "rule": "majority_0.5",
            **score_predictions(trials, predictions),
        })

    rows.sort(
        key=lambda row: (
            float(row["total_accuracy"]),
            float(row["decision_accuracy"]),
            float(row["coverage"]),
        ),
        reverse=True,
    )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "rule",
        "n",
        "correct",
        "incorrect",
        "ambiguous",
        "coverage",
        "total_accuracy",
        "decision_accuracy",
        "mi_recall",
        "rest_recall",
        "false_mi",
        "false_rest",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rule_comparison(
    rows: list[dict[str, float | int | str]],
    title: str,
    output_path: Path | None = None,
) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row["total_accuracy"]),
            float(row["decision_accuracy"]),
            -float(row["incorrect"]),
        ),
        reverse=True,
    )[:12]
    labels = [f"{row['model']}\n{row['rule']}" for row in top_rows]
    x = np.arange(len(top_rows))
    total_acc = np.asarray([100 * float(row["total_accuracy"]) for row in top_rows])
    decision_acc = np.asarray([100 * float(row["decision_accuracy"]) for row in top_rows])
    coverage = np.asarray([100 * float(row["coverage"]) for row in top_rows])
    errors = np.asarray([int(row["incorrect"]) for row in top_rows])
    ambiguous = np.asarray([int(row["ambiguous"]) for row in top_rows])

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    width = 0.36
    ax.bar(x - width / 2, total_acc, width, label="Total accuracy", color="#4c78a8")
    ax.bar(x + width / 2, decision_acc, width, label="Decision accuracy", color="#f58518")
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.axhline(70, color="gray", linestyle=":", linewidth=1.5, label="Target (70%)")
    ax.set_title("Best rules — accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.scatter(coverage, decision_acc, s=90, c=errors, cmap="Reds", edgecolor="black")
    for row, cx, cy in zip(top_rows, coverage, decision_acc):
        ax.annotate(
            f"{row['model']}:{row['rule']}",
            (cx, cy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.axhline(70, color="gray", linestyle=":", linewidth=1.5)
    ax.axvline(70, color="gray", linestyle=":", linewidth=1.5)
    ax.set_title("Decision accuracy vs coverage")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Decision accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(40, 100)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.bar(x, errors, label="Incorrect", color="#d65f5f")
    ax.bar(x, ambiguous, bottom=errors, label="Ambiguous", color="#bdbdbd")
    ax.set_title("Cost of each rule")
    ax.set_ylabel("Trials")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for model in MODEL_NAMES:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        ax.plot(
            [100 * float(row["coverage"]) for row in model_rows],
            [100 * float(row["total_accuracy"]) for row in model_rows],
            marker="o",
            linewidth=1.8,
            label=model,
        )
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.axhline(70, color="gray", linestyle=":", linewidth=1.5, label="Target (70%)")
    ax.set_title("All rules — total accuracy vs coverage")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Total accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(20, 85)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def format_pct(value: float | int | str) -> str:
    return f"{100 * float(value):5.1f}%"


def print_table(rows: list[dict[str, float | int | str]], limit: int) -> None:
    print("\n=== Pseudo-online decision-rule ranking ===")
    print(
        "Model  Rule                    Total   DecAcc  Cover   "
        "Err  Amb  MIrec  RESTrec  falseMI falseREST"
    )
    print("-" * 92)
    for row in rows[:limit]:
        print(
            f"{row['model']:<6} {row['rule']:<22} "
            f"{format_pct(row['total_accuracy'])} "
            f"{format_pct(row['decision_accuracy'])} "
            f"{format_pct(row['coverage'])} "
            f"{int(row['incorrect']):>4} "
            f"{int(row['ambiguous']):>4} "
            f"{format_pct(row['mi_recall'])} "
            f"{format_pct(row['rest_recall'])} "
            f"{int(row['false_mi']):>7} "
            f"{int(row['false_rest']):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo-online decision rules from online logs."
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--mi-threshold", type=float, default=0.85)
    parser.add_argument("--rest-threshold", type=float, default=0.15)
    parser.add_argument("--endpoint", type=float, default=-0.50)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional PNG output path. If omitted, the figure is displayed only.",
    )
    args = parser.parse_args()

    trials = load_trials(args.log_root)
    rows = evaluate_rules(
        trials,
        mi_threshold=args.mi_threshold,
        rest_threshold=args.rest_threshold,
        endpoint=args.endpoint,
    )
    write_csv(rows, args.output)
    plot_rule_comparison(
        rows,
        title=(
            "Pseudo-online decision rules — "
            f"MI≥{args.mi_threshold:.2f}, REST≤{args.rest_threshold:.2f}, "
            f"endpoint={args.endpoint:+.2f}s"
        ),
        output_path=args.plot,
    )

    print(f"Loaded trials : {len(trials)}")
    print(f"Runs          : {len({trial.run for trial in trials})}")
    print(f"Endpoint      : {args.endpoint:+.2f} s")
    print(
        f"Thresholds    : MI >= {args.mi_threshold:.2f}, "
        f"REST <= {args.rest_threshold:.2f}"
    )
    print_table(rows, args.limit)
    print(f"\nSaved CSV: {args.output}")
    if args.plot is not None:
        print(f"Saved PNG: {args.plot}")
    else:
        print("Displayed figure: not saved")


if __name__ == "__main__":
    main()
