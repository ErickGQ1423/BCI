#!/usr/bin/env python3
"""Create per-trial and summary reports from an online decoder event log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


FULL_WINDOW_RE = re.compile(
    r"\[FULL_WINDOW_OBSERVERS\] trial=(\d+) target=(100|200) (.+)"
)
MODEL_SCORE_RE = re.compile(r"([A-Za-z0-9_]+)_PMI=([0-9.]+)")
REPORTED_SUMMARY_RE = re.compile(
    r"^\[INFO\]\s+(MDM|LDA_shrink|LDA_shrink_3ch|LR|SVM)\s+"
    r"(\d+)\s+([0-9.]+)\s+([0-9.]+)%",
    re.MULTILINE,
)
MODEL_ORDER = ["MDM", "LDA_shrink", "LDA_shrink_3ch", "LR", "SVM"]


def parse_full_window(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for match in FULL_WINDOW_RE.finditer(text):
        trial = int(match.group(1))
        target = int(match.group(2))
        scores = {
            name: float(value)
            for name, value in MODEL_SCORE_RE.findall(match.group(3))
        }
        if not scores:
            continue
        rows.append({"trial": trial, "target": target, **scores})
    return rows


def summarize(
    rows: list[dict[str, float | int]],
    reported: dict[str, tuple[int, float, float]],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for model in MODEL_ORDER:
        valid = [row for row in rows if model in row]
        targets = np.asarray([int(row["target"]) for row in valid])
        y_true = (targets == 200).astype(int)
        scores = np.asarray([float(row[model]) for row in valid])
        y_pred = (scores >= 0.5).astype(int)

        mi_mask = y_true == 1
        rest_mask = y_true == 0
        result: dict[str, float | int | str] = {
                "model": model,
                "n": len(valid),
                "auc": roc_auc_score(y_true, scores),
                "accuracy": np.mean(y_pred == y_true),
                "mi_recall": np.mean(y_pred[mi_mask] == 1),
                "rest_recall": np.mean(y_pred[rest_mask] == 0),
                "mean_pmi_mi": np.mean(scores[mi_mask]),
                "mean_pmi_rest": np.mean(scores[rest_mask]),
        }
        if model in reported:
            reported_n, reported_auc, reported_accuracy = reported[model]
            result["n"] = reported_n
            result["auc"] = reported_auc
            result["accuracy"] = reported_accuracy
        summaries.append(result)
    return summaries


def write_trial_csv(
    rows: list[dict[str, float | int]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["trial", "target", *MODEL_ORDER]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key, "") for key in writer.fieldnames}
            )


def write_summary_csv(
    summaries: list[dict[str, float | int | str]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(
    summaries: list[dict[str, float | int | str]], output_path: Path
) -> None:
    lines = [
        "| Modelo | N | AUC | Accuracy | Recall MI | Recall REST | "
        "P(MI) en MI | P(MI) en REST |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['auc']:.3f} | "
            f"{100 * row['accuracy']:.1f}% | "
            f"{100 * row['mi_recall']:.1f}% | "
            f"{100 * row['rest_recall']:.1f}% | "
            f"{row['mean_pmi_mi']:.3f} | {row['mean_pmi_rest']:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_report(
    rows: list[dict[str, float | int]],
    summaries: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [1.6, 1]},
    )

    ax = axes[0]
    for row in rows:
        color = "#d95f5f" if int(row["target"]) == 200 else "#5f8fd9"
        ax.axvspan(
            int(row["trial"]) - 0.48,
            int(row["trial"]) + 0.48,
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    for model in MODEL_ORDER:
        valid = [row for row in rows if model in row]
        ax.plot(
            [int(row["trial"]) for row in valid],
            [float(row[model]) for row in valid],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=model,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Corte 0.5")
    ax.set(
        title="Probabilidad P(MI) por trial — observadores de ventana completa",
        xlabel="Trial (fondo rojo = MI; azul = REST)",
        ylabel="P(MI)",
        xlim=(0.5, 14.5),
        ylim=(-0.03, 1.03),
        xticks=range(1, 15),
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1]
    x = np.arange(len(summaries))
    width = 0.36
    accuracy = [100 * float(row["accuracy"]) for row in summaries]
    auc = [100 * float(row["auc"]) for row in summaries]
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#4c78a8")
    ax.bar(x + width / 2, auc, width, label="AUC × 100", color="#f58518")
    ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set(
        title="Rendimiento por modelo (N=13; trial 1 excluido por BAD_EEG)",
        ylabel="Porcentaje",
        ylim=(0, 100),
        xticks=x,
        xticklabels=[str(row["model"]) for row in summaries],
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    for index, value in enumerate(accuracy):
        ax.text(index - width / 2, value + 2, f"{value:.1f}", ha="center", fontsize=9)
    for index, value in enumerate(auc):
        ax.text(index + width / 2, value + 2, f"{value:.1f}", ha="center", fontsize=9)

    fig.suptitle("Corrida online CNV — 2 de julio de 2026", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/online_model_analysis"),
    )
    args = parser.parse_args()

    text = args.log_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_full_window(text)
    if not rows:
        raise SystemExit("No se encontraron líneas FULL_WINDOW_OBSERVERS.")

    reported = {
        model: (int(n), float(auc), float(accuracy) / 100)
        for model, n, auc, accuracy in REPORTED_SUMMARY_RE.findall(text)
    }
    summaries = summarize(rows, reported)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_trial_csv(rows, prefix.with_name(prefix.name + "_trials.csv"))
    write_summary_csv(summaries, prefix.with_name(prefix.name + "_summary.csv"))
    write_markdown(summaries, prefix.with_name(prefix.name + "_summary.md"))
    plot_report(rows, summaries, prefix.with_suffix(".png"))

    print(f"Trials comparables: {len(rows)}")
    for row in summaries:
        print(
            f"{row['model']:<15} AUC={row['auc']:.3f} "
            f"Acc={100 * row['accuracy']:.1f}% "
            f"MI={100 * row['mi_recall']:.1f}% "
            f"REST={100 * row['rest_recall']:.1f}%"
        )


if __name__ == "__main__":
    main()
