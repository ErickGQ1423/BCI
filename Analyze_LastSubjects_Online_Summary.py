#!/usr/bin/env python3
"""
Summarize recent online CNV-BCI runs for meeting figures.

Default target:
    CNV_PILOT_SUBJ_025, 026, 027

Outputs:
    - reports/last_subjects_online_summary.csv
    - optional interactive figures with total accuracy, decision accuracy,
      coverage, and MI/REST recall.

This script is read-only with respect to experiment data. It only writes the
summary CSV inside this repository.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


DEFAULT_ROOT = Path("/home/lab-admin/Documents/CNVStudy")
DEFAULT_SUBJECTS = ("025", "026", "027")
DEFAULT_REPORT = Path("reports/last_subjects_online_summary.csv")


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_run_label(logdir: Path) -> str:
    match = re.search(r"run-(\d+)", logdir.name)
    return match.group(1) if match else "NA"


def _extract_time_label(logdir: Path) -> str:
    match = re.search(r"ONLINE_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", logdir.name)
    if not match:
        return logdir.name
    return f"{match.group(1)} {match.group(2).replace('-', ':')}"


def _summarize_trial_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    total = len(rows)
    correct = incorrect = ambiguous = decided = 0
    mi_total = rest_total = mi_correct = rest_correct = mi_amb = rest_amb = 0

    for row in rows:
        true = (row.get("True Label") or row.get("true_label") or "").strip()
        pred = (row.get("Predicted Label") or row.get("predicted_label") or "").strip()

        if true == "200":
            mi_total += 1
        elif true == "100":
            rest_total += 1

        if not pred:
            ambiguous += 1
            if true == "200":
                mi_amb += 1
            elif true == "100":
                rest_amb += 1
            continue

        decided += 1
        if pred == true:
            correct += 1
            if true == "200":
                mi_correct += 1
            elif true == "100":
                rest_correct += 1
        else:
            incorrect += 1

    return {
        "n_trials": total,
        "correct": correct,
        "incorrect": incorrect,
        "ambiguous": ambiguous,
        "decided": decided,
        "total_accuracy": 100 * correct / total if total else np.nan,
        "decision_accuracy": 100 * correct / decided if decided else np.nan,
        "coverage": 100 * decided / total if total else np.nan,
        "mi_recall": 100 * mi_correct / mi_total if mi_total else np.nan,
        "rest_recall": 100 * rest_correct / rest_total if rest_total else np.nan,
        "mi_ambiguous": mi_amb,
        "rest_ambiguous": rest_amb,
    }


def _summarize_event_log(path: Path) -> dict:
    out = {
        "model_loaded": "",
        "recenter_updates": 0,
        "rule_endpoint": 0,
        "rule_weighted": 0,
        "rule_viewer": 0,
        "rule_ambiguous": 0,
        "full_window_mdm_auc": np.nan,
        "full_window_lda_auc": np.nan,
        "full_window_lda3_auc": np.nan,
        "full_window_lr_auc": np.nan,
        "full_window_svm_auc": np.nan,
        "full_window_mdm_acc": np.nan,
        "full_window_lda_acc": np.nan,
        "full_window_lda3_acc": np.nan,
        "full_window_lr_acc": np.nan,
        "full_window_svm_acc": np.nan,
        "endpoint_mdm_auc": np.nan,
        "endpoint_lda_auc": np.nan,
        "endpoint_lda3_auc": np.nan,
        "endpoint_lr_auc": np.nan,
        "endpoint_svm_auc": np.nan,
        "endpoint_mdm_acc": np.nan,
        "endpoint_lda_acc": np.nan,
        "endpoint_lda3_acc": np.nan,
        "endpoint_lr_acc": np.nan,
        "endpoint_svm_acc": np.nan,
        "operational_endpoint_mdm_decision_acc": np.nan,
        "operational_endpoint_lda_decision_acc": np.nan,
        "operational_endpoint_lda3_decision_acc": np.nan,
        "operational_endpoint_lr_decision_acc": np.nan,
        "operational_endpoint_svm_decision_acc": np.nan,
        "operational_endpoint_mdm_total_acc": np.nan,
        "operational_endpoint_lda_total_acc": np.nan,
        "operational_endpoint_lda3_total_acc": np.nan,
        "operational_endpoint_lr_total_acc": np.nan,
        "operational_endpoint_svm_total_acc": np.nan,
        "operational_endpoint_mdm_coverage": np.nan,
        "operational_endpoint_lda_coverage": np.nan,
        "operational_endpoint_lda3_coverage": np.nan,
        "operational_endpoint_lr_coverage": np.nan,
        "operational_endpoint_svm_coverage": np.nan,
        "weighted_mdm_auc": np.nan,
        "weighted_lda_auc": np.nan,
        "weighted_lda3_auc": np.nan,
        "weighted_lr_auc": np.nan,
        "weighted_svm_auc": np.nan,
        "temporal_vote_mdm_auc": np.nan,
        "temporal_vote_lda_auc": np.nan,
        "temporal_vote_lda3_auc": np.nan,
        "temporal_vote_lr_auc": np.nan,
        "temporal_vote_svm_auc": np.nan,
        "observer_lda_acc": np.nan,
        "observer_lda3_acc": np.nan,
        "observer_lr_acc": np.nan,
        "observer_svm_acc": np.nan,
        "shadow_mdm_acc": np.nan,
        "shadow_lda_acc": np.nan,
        "shadow_lda3_acc": np.nan,
        "shadow_lr_acc": np.nan,
        "shadow_svm_acc": np.nan,
        "shadow_mdm_time": np.nan,
        "shadow_lda_time": np.nan,
        "shadow_lda3_time": np.nan,
        "shadow_lr_time": np.nan,
        "shadow_svm_time": np.nan,
        "recenter_update_mi": 0,
        "recenter_update_rest": 0,
        "recenter_reject_ambiguous": 0,
        "recenter_reject_wrong": 0,
        "recenter_reject_low_conf": 0,
        "recenter_reject_bad_eeg": 0,
    }
    if not path.exists():
        return out

    text = path.read_text(errors="ignore")
    fw_pat = re.compile(
        r"\[FULL_WINDOW_OBSERVER_SUMMARY\].*?model=([^ ]+).*?auc=([0-9.]+|NA).*?accuracy=([0-9.]+|NA)",
        re.I,
    )
    fw_trial_pat = re.compile(
        r"\[FULL_WINDOW_OBSERVERS\].*?target=(100|200)\s+"
        r"MDM_PMI=([0-9.]+).*?MDM_pred=(100|200).*?"
        r"LDA_shrink_PMI=([0-9.]+).*?LDA_shrink_pred=(100|200).*?"
        r"(?:LDA_shrink_3ch_PMI=([0-9.]+).*?LDA_shrink_3ch_pred=(100|200).*?)?"
        r"LR_PMI=([0-9.]+).*?LR_pred=(100|200).*?"
        r"SVM_PMI=([0-9.]+).*?SVM_pred=(100|200)",
        re.I,
    )
    observer_pat = re.compile(
        r"\[(LDA|LDA_3CH|LR|SVM)_OBSERVER_DECISION\].*?"
        r"prediction=([^ ]+).*?target=([^ ]+)"
    )
    m2_step_pat = re.compile(
        r"\[M2_step\].*?paso=(\d+)/\d+.*?t=([-0-9.]+)s.*?"
        r"MDM_PMI=([0-9.]+|NA).*?"
        r"LDA=([0-9.]+|NA).*?"
        r"LDA3=([0-9.]+|NA).*?"
        r"LR=([0-9.]+|NA).*?"
        r"SVM=([0-9.]+|NA).*?"
        r"conf_(MI|REST)=",
        re.I,
    )
    shadow_pat = re.compile(
        r"\[SHADOW_STABILITY\].*?model=([^ ]+).*?"
        r"time=([^ ]+).*?target=([^ ]+).*?correct=([^ ]+)"
    )
    observer_counts = defaultdict(lambda: Counter(total=0, correct=0, decided=0))
    shadow_counts = defaultdict(lambda: Counter(total=0, correct=0))
    shadow_times = defaultdict(list)
    fw_targets = []
    fw_probs = defaultdict(list)
    fw_preds = defaultdict(list)
    endpoint_targets = []
    endpoint_probs = defaultdict(list)
    endpoint_preds = defaultdict(list)
    operational_endpoint_preds = defaultdict(list)
    temporal_trials = []
    current_trial = None
    for line in text.splitlines():
        if "Model loaded:" in line:
            out["model_loaded"] = line.split("Model loaded:", 1)[-1].strip()

        if "[M2_recentering] accepted_updates=" in line:
            match = re.search(r"accepted_updates=(\d+)", line)
            if match:
                out["recenter_updates"] = max(out["recenter_updates"], int(match.group(1)))

        if "[RIEMANN_ADAPT_UPDATE]" in line:
            if "class=MI" in line:
                out["recenter_update_mi"] += 1
            elif "class=REST" in line:
                out["recenter_update_rest"] += 1

        if "[RIEMANN_ADAPT_REJECT]" in line:
            if "reason=AMBIGUOUS_DECISION" in line:
                out["recenter_reject_ambiguous"] += 1
            elif "reason=DECISION_DID_NOT_MATCH_TARGET" in line:
                out["recenter_reject_wrong"] += 1
            elif "reason=LOW_CONFIDENCE" in line:
                out["recenter_reject_low_conf"] += 1
            elif "BAD_EEG" in line:
                out["recenter_reject_bad_eeg"] += 1

        if "[PREP_OPERATIONAL_DECISION]" in line:
            if "accepted_mi_viewer_temporal" in line or "accepted_rest_viewer_temporal" in line:
                out["rule_viewer"] += 1
            elif "mdm_weighted" in line:
                out["rule_weighted"] += 1
            elif "control=MDM" in line:
                out["rule_endpoint"] += 1
            elif "ambiguous" in line:
                out["rule_ambiguous"] += 1

        match = m2_step_pat.search(line)
        if match:
            step = int(match.group(1))
            time = _safe_float(match.group(2))
            target = 1 if match.group(8) == "MI" else 0
            if step == 1 or current_trial is None:
                if current_trial is not None:
                    temporal_trials.append(current_trial)
                current_trial = {"target": target, "samples": []}
            elif current_trial["target"] != target:
                temporal_trials.append(current_trial)
                current_trial = {"target": target, "samples": []}

            probs_by_model = {
                "MDM": _safe_float(match.group(3)),
                "LDA": _safe_float(match.group(4)),
                "LDA3": _safe_float(match.group(5)),
                "LR": _safe_float(match.group(6)),
                "SVM": _safe_float(match.group(7)),
            }
            current_trial["samples"].append((time, probs_by_model))

            if time is not None and abs(time - (-0.50)) < 1e-6:
                endpoint_targets.append(target)
                for model in ("MDM", "LDA", "LDA3", "LR", "SVM"):
                    prob = probs_by_model[model]
                    if prob is None:
                        endpoint_probs[model].append(np.nan)
                        endpoint_preds[model].append(np.nan)
                        operational_endpoint_preds[model].append(np.nan)
                    else:
                        endpoint_probs[model].append(prob)
                        endpoint_preds[model].append(1 if prob >= 0.5 else 0)
                        if prob >= 0.70:
                            operational_endpoint_preds[model].append(1)
                        elif prob <= 0.30:
                            operational_endpoint_preds[model].append(0)
                        else:
                            operational_endpoint_preds[model].append(np.nan)

        match = fw_pat.search(line)
        if match:
            model = match.group(1).lower()
            auc = _safe_float(match.group(2))
            acc = _safe_float(match.group(3))
            if model == "mdm":
                out["full_window_mdm_auc"] = auc
                out["full_window_mdm_acc"] = acc
            elif "lda" in model and "lda3" not in model:
                out["full_window_lda_auc"] = auc
                out["full_window_lda_acc"] = acc
            elif "lda3" in model or "3ch" in model:
                out["full_window_lda3_auc"] = auc
                out["full_window_lda3_acc"] = acc
            elif model == "lr":
                out["full_window_lr_auc"] = auc
                out["full_window_lr_acc"] = acc
            elif model == "svm":
                out["full_window_svm_auc"] = auc
                out["full_window_svm_acc"] = acc

        match = fw_trial_pat.search(line)
        if match:
            target = 1 if match.group(1) == "200" else 0
            fw_targets.append(target)
            fw_probs["MDM"].append(float(match.group(2)))
            fw_preds["MDM"].append(1 if match.group(3) == "200" else 0)
            fw_probs["LDA"].append(float(match.group(4)))
            fw_preds["LDA"].append(1 if match.group(5) == "200" else 0)
            if match.group(6) is not None:
                fw_probs["LDA3"].append(float(match.group(6)))
                fw_preds["LDA3"].append(1 if match.group(7) == "200" else 0)
            fw_probs["LR"].append(float(match.group(8)))
            fw_preds["LR"].append(1 if match.group(9) == "200" else 0)
            fw_probs["SVM"].append(float(match.group(10)))
            fw_preds["SVM"].append(1 if match.group(11) == "200" else 0)

        match = observer_pat.search(line)
        if match:
            name = match.group(1).replace("_3CH", "3")
            pred, target = match.group(2), match.group(3)
            observer_counts[name]["total"] += 1
            if pred != "AMBIGUOUS":
                observer_counts[name]["decided"] += 1
                if pred == target:
                    observer_counts[name]["correct"] += 1

        match = shadow_pat.search(line)
        if match:
            name = match.group(1)
            time = _safe_float(match.group(2))
            correct = match.group(4)
            if correct in ("True", "False"):
                shadow_counts[name]["total"] += 1
                if correct == "True":
                    shadow_counts[name]["correct"] += 1
                if time is not None:
                    shadow_times[name].append(time)

    if current_trial is not None:
        temporal_trials.append(current_trial)

    observer_key_map = {
        "LDA": "observer_lda_acc",
        "LDA3": "observer_lda3_acc",
        "LR": "observer_lr_acc",
        "SVM": "observer_svm_acc",
    }
    for model, key in observer_key_map.items():
        decided = observer_counts[model]["decided"]
        if decided:
            out[key] = 100 * observer_counts[model]["correct"] / decided

    for model in ("MDM", "LDA", "LDA3", "LR", "SVM"):
        low = model.lower()
        total = shadow_counts[model]["total"]
        if total:
            out[f"shadow_{low}_acc"] = 100 * shadow_counts[model]["correct"] / total
        if shadow_times[model]:
            out[f"shadow_{low}_time"] = float(np.nanmean(shadow_times[model]))

    full_window_key_map = {
        "MDM": ("full_window_mdm_auc", "full_window_mdm_acc"),
        "LDA": ("full_window_lda_auc", "full_window_lda_acc"),
        "LDA3": ("full_window_lda3_auc", "full_window_lda3_acc"),
        "LR": ("full_window_lr_auc", "full_window_lr_acc"),
        "SVM": ("full_window_svm_auc", "full_window_svm_acc"),
    }
    if len(set(fw_targets)) == 2:
        for model, (auc_key, acc_key) in full_window_key_map.items():
            if len(fw_probs[model]) == len(fw_targets):
                out[auc_key] = roc_auc_score(fw_targets, fw_probs[model])
                out[acc_key] = 100 * accuracy_score(fw_targets, fw_preds[model])

    endpoint_key_map = {
        "MDM": ("endpoint_mdm_auc", "endpoint_mdm_acc"),
        "LDA": ("endpoint_lda_auc", "endpoint_lda_acc"),
        "LDA3": ("endpoint_lda3_auc", "endpoint_lda3_acc"),
        "LR": ("endpoint_lr_auc", "endpoint_lr_acc"),
        "SVM": ("endpoint_svm_auc", "endpoint_svm_acc"),
    }
    if len(set(endpoint_targets)) == 2:
        targets = np.asarray(endpoint_targets, dtype=int)
        for model, (auc_key, acc_key) in endpoint_key_map.items():
            probs = np.asarray(endpoint_probs[model], dtype=float)
            preds = np.asarray(endpoint_preds[model], dtype=float)
            keep = np.isfinite(probs) & np.isfinite(preds)
            if keep.sum() and len(set(targets[keep])) == 2:
                out[auc_key] = roc_auc_score(targets[keep], probs[keep])
                out[acc_key] = 100 * accuracy_score(targets[keep], preds[keep])

            op_preds = np.asarray(operational_endpoint_preds[model], dtype=float)
            op_keep = np.isfinite(op_preds)
            if op_keep.sum():
                out[f"operational_endpoint_{model.lower()}_decision_acc"] = (
                    100 * accuracy_score(targets[op_keep], op_preds[op_keep])
                )
                out[f"operational_endpoint_{model.lower()}_total_acc"] = (
                    100 * np.sum(op_preds[op_keep] == targets[op_keep]) / len(op_preds)
                )
                out[f"operational_endpoint_{model.lower()}_coverage"] = (
                    100 * op_keep.sum() / len(op_preds)
                )

    temporal_targets = []
    weighted_scores = defaultdict(list)
    vote_scores = defaultdict(list)
    for trial in temporal_trials:
        samples = [
            (time, probs)
            for time, probs in trial["samples"]
            if time is not None and -2.50 <= time <= -0.50
        ]
        if not samples:
            continue
        temporal_targets.append(trial["target"])
        weights = np.arange(1, len(samples) + 1, dtype=float)
        weights /= weights.sum()
        for model in ("MDM", "LDA", "LDA3", "LR", "SVM"):
            probs = np.asarray([p[model] for _, p in samples], dtype=float)
            keep = np.isfinite(probs)
            if keep.any():
                model_weights = weights[keep]
                model_weights /= model_weights.sum()
                weighted_scores[model].append(float(np.sum(probs[keep] * model_weights)))
                vote_scores[model].append(float(np.mean(probs[keep] >= 0.5)))
            else:
                weighted_scores[model].append(np.nan)
                vote_scores[model].append(np.nan)

    if len(set(temporal_targets)) == 2:
        targets = np.asarray(temporal_targets, dtype=int)
        for model in ("MDM", "LDA", "LDA3", "LR", "SVM"):
            low = model.lower()
            for prefix, scores_by_model in [
                ("weighted", weighted_scores),
                ("temporal_vote", vote_scores),
            ]:
                scores = np.asarray(scores_by_model[model], dtype=float)
                keep = np.isfinite(scores)
                if keep.sum() and len(set(targets[keep])) == 2:
                    out[f"{prefix}_{low}_auc"] = roc_auc_score(targets[keep], scores[keep])
    return out


def collect_runs(root: Path, subjects: tuple[str, ...]) -> list[dict]:
    all_rows: list[dict] = []
    for sid in subjects:
        subj_dir = root / f"sub-CNV_PILOT_SUBJ_{sid}"
        for session_dir in sorted(subj_dir.glob("ses-S*_ONLINE")):
            logs_dir = session_dir / "logs"
            if not logs_dir.exists():
                continue
            for logdir in sorted(logs_dir.glob("ONLINE_*run-*")):
                trial_csv = logdir / "trial_summary.csv"
                if not trial_csv.exists():
                    continue
                row = {
                    "subject": sid,
                    "session": session_dir.name.replace("ses-", ""),
                    "run": _extract_run_label(logdir),
                    "timestamp": _extract_time_label(logdir),
                    "logdir": str(logdir),
                }
                row.update(_summarize_trial_csv(trial_csv))
                row.update(_summarize_event_log(logdir / "event_log.txt"))
                all_rows.append(row)
    return all_rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\nONLINE SUMMARY BY SUBJECT / SESSION")
    print("subject session runs pooled_acc mean_decision mean_coverage MI_recall REST_recall")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["session"])].append(row)

    for (subject, session), sr in sorted(grouped.items()):
        n = sum(r["n_trials"] for r in sr)
        correct = sum(r["correct"] for r in sr)
        pooled_acc = 100 * correct / n if n else np.nan
        print(
            f"{subject:>7} {session:<12} {len(sr):>4} "
            f"{pooled_acc:>9.1f} "
            f"{np.nanmean([r['decision_accuracy'] for r in sr]):>13.1f} "
            f"{np.nanmean([r['coverage'] for r in sr]):>13.1f} "
            f"{np.nanmean([r['mi_recall'] for r in sr]):>9.1f} "
            f"{np.nanmean([r['rest_recall'] for r in sr]):>11.1f}"
        )

    print("\nRule usage across all runs")
    totals = Counter()
    for row in rows:
        totals["endpoint"] += int(row["rule_endpoint"])
        totals["weighted"] += int(row["rule_weighted"])
        totals["viewer"] += int(row["rule_viewer"])
        totals["ambiguous"] += int(row["rule_ambiguous"])
    for key in ("endpoint", "weighted", "viewer", "ambiguous"):
        print(f"  {key:>9}: {totals[key]}")

    print("\nFull-window observers — mean AUC across runs")
    for model, key in [
        ("MDM", "full_window_mdm_auc"),
        ("LDA", "full_window_lda_auc"),
        ("LDA3", "full_window_lda3_auc"),
        ("LR", "full_window_lr_auc"),
        ("SVM", "full_window_svm_auc"),
    ]:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if vals:
            print(f"  {model:>4}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}  n={len(vals)}")

    print("\nRecenter updates")
    print(f"  total updates: {sum(int(r['recenter_updates']) for r in rows)}")
    print(f"  MI updates   : {sum(int(r['recenter_update_mi']) for r in rows)}")
    print(f"  REST updates : {sum(int(r['recenter_update_rest']) for r in rows)}")


def plot_figures(rows: list[dict]) -> None:
    if not rows:
        print("No rows to plot.")
        return

    model_colors = {
        "MDM": "tab:blue",
        "LDA": "tab:orange",
        "LDA3": "tab:green",
        "LR": "tab:red",
        "SVM": "tab:purple",
    }

    # Figure 1: total accuracy per run.
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharey=True)
    for ax, subject in zip(axes, sorted({r["subject"] for r in rows})):
        sr = [r for r in rows if r["subject"] == subject]
        labels = [f"{r['session'].replace('_ONLINE','')}-r{r['run']}" for r in sr]
        x = np.arange(len(sr))
        ax.plot(x, [r["total_accuracy"] for r in sr], marker="o", label="Total accuracy")
        ax.plot(x, [r["decision_accuracy"] for r in sr], marker="o", label="Decision accuracy")
        ax.plot(x, [r["coverage"] for r in sr], marker="o", linestyle="--", label="Coverage")
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
        ax.axhline(70, color="gray", linestyle=":", linewidth=1, label="Target (70%)")
        ax.set_title(f"Subject {subject} — online runs")
        ax.set_ylabel("%")
        ax.set_ylim(30, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Session/run")
    fig.suptitle("Online performance across recent subjects", fontweight="bold")
    fig.tight_layout()

    # Extra figure: real online decision accuracy vs viewer full-window accuracy per run.
    observer_acc_models = [
        ("Online decision", "decision_accuracy"),
        ("LDA", "full_window_lda_acc"),
        ("LDA3", "full_window_lda3_acc"),
        ("LR", "full_window_lr_acc"),
        ("SVM", "full_window_svm_acc"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharey=True)
    for ax, subject in zip(axes, sorted({r["subject"] for r in rows})):
        sr = [r for r in rows if r["subject"] == subject]
        labels = [f"{r['session'].replace('_ONLINE','')}-r{r['run']}" for r in sr]
        x = np.arange(len(sr))
        for model, key in observer_acc_models:
            alpha = 0.35 if model in {"LR", "SVM"} else 1.0
            ax.plot(
                x,
                [r[key] for r in sr],
                marker="o",
                color=model_colors.get(model, "tab:blue"),
                alpha=alpha,
                label=model,
            )
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
        ax.set_title(f"Subject {subject} — MDM control vs viewer models")
        ax.set_ylabel("Decision accuracy (%)")
        ax.set_ylim(30, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Session/run")
    fig.suptitle(
        "Real online decision accuracy vs viewer-model accuracy across recent subjects\n"
        "Online decision uses the full decision cascade; LDA/LDA3/LR/SVM use full-window observer predictions",
        fontweight="bold",
    )
    fig.tight_layout()

    # Extra figure: real online total accuracy vs viewer full-window accuracy per run.
    observer_total_models = [
        ("Online decision", "total_accuracy"),
        ("LDA", "full_window_lda_acc"),
        ("LDA3", "full_window_lda3_acc"),
        ("LR", "full_window_lr_acc"),
        ("SVM", "full_window_svm_acc"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharey=True)
    for ax, subject in zip(axes, sorted({r["subject"] for r in rows})):
        sr = [r for r in rows if r["subject"] == subject]
        labels = [f"{r['session'].replace('_ONLINE','')}-r{r['run']}" for r in sr]
        x = np.arange(len(sr))
        for model, key in observer_total_models:
            alpha = 0.35 if model in {"LR", "SVM"} else 1.0
            ax.plot(
                x,
                [r[key] for r in sr],
                marker="o",
                color=model_colors.get(model, "tab:blue"),
                alpha=alpha,
                label=model,
            )
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
        ax.set_title(f"Subject {subject} — MDM control vs viewer models")
        ax.set_ylabel("Total accuracy (%)")
        ax.set_ylim(30, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Session/run")
    fig.suptitle(
        "Real online total accuracy vs viewer-model accuracy across recent subjects\n"
        "Online ambiguous trials count as errors; LDA/LDA3/LR/SVM use full-window observer predictions",
        fontweight="bold",
    )
    fig.tight_layout()

    # Figure 2: session means by subject.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["session"])].append(row)

    labels = [f"{s}-{sess.replace('_ONLINE','')}" for s, sess in sorted(grouped)]
    mean_acc = [np.nanmean([r["total_accuracy"] for r in grouped[k]]) for k in sorted(grouped)]
    mean_mi = [np.nanmean([r["mi_recall"] for r in grouped[k]]) for k in sorted(grouped)]
    mean_rest = [np.nanmean([r["rest_recall"] for r in grouped[k]]) for k in sorted(grouped)]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, mean_acc, width, label="Total accuracy")
    ax.bar(x, mean_mi, width, label="MI recall")
    ax.bar(x + width, mean_rest, width, label="REST recall")
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.axhline(70, color="gray", linestyle=":", linewidth=1, label="Target (70%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Session-level means")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    # Figure 3: decision rule usage.
    fig, ax = plt.subplots(figsize=(10, 5))
    rule_counts = {
        "MDM endpoint": sum(int(r["rule_endpoint"]) for r in rows),
        "MDM weighted": sum(int(r["rule_weighted"]) for r in rows),
        "Viewer rescue": sum(int(r["rule_viewer"]) for r in rows),
        "Ambiguous": sum(int(r["rule_ambiguous"]) for r in rows),
    }
    ax.bar(rule_counts.keys(), rule_counts.values())
    ax.set_title("Operational rule usage")
    ax.set_ylabel("Trials")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    # Figure 4: full-window observer AUC by subject.
    observer_models = [
        ("MDM", "full_window_mdm_auc"),
        ("LDA", "full_window_lda_auc"),
        ("LDA3", "full_window_lda3_auc"),
        ("LR", "full_window_lr_auc"),
        ("SVM", "full_window_svm_auc"),
    ]
    subjects = sorted({r["subject"] for r in rows})
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(subjects))
    width = 0.15
    for idx, (model, key) in enumerate(observer_models):
        vals = []
        for subject in subjects:
            sr = [r[key] for r in rows if r["subject"] == subject and not np.isnan(r[key])]
            vals.append(np.nanmean(sr) if sr else np.nan)
        ax.bar(x + (idx - 2) * width, vals, width, label=model)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.axhline(0.7, color="gray", linestyle=":", linewidth=1, label="Target (0.7)")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Full-window observer AUC by subject")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Subject {s}" for s in subjects])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    # Figure 5: recentering behavior by run.
    fig, ax1 = plt.subplots(figsize=(13, 5))
    labels = [f"{r['subject']}-{r['session'].replace('_ONLINE','')}-r{r['run']}" for r in rows]
    x = np.arange(len(rows))
    ax1.bar(x, [r["recenter_update_mi"] for r in rows], label="MI updates", alpha=0.8)
    ax1.bar(
        x,
        [r["recenter_update_rest"] for r in rows],
        bottom=[r["recenter_update_mi"] for r in rows],
        label="REST updates",
        alpha=0.8,
    )
    ax1.set_ylabel("Accepted recenter updates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["total_accuracy"] for r in rows], color="black", marker="o", label="Total accuracy")
    ax2.set_ylim(30, 100)
    ax2.set_ylabel("Total accuracy (%)")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    ax1.set_title("Recenter updates vs online accuracy")
    fig.tight_layout()

    # Figure 6: shadow stabilization time and accuracy.
    shadow_models = [
        ("MDM", "shadow_mdm_acc", "shadow_mdm_time"),
        ("LDA", "shadow_lda_acc", "shadow_lda_time"),
        ("LDA3", "shadow_lda3_acc", "shadow_lda3_time"),
        ("LR", "shadow_lr_acc", "shadow_lr_time"),
        ("SVM", "shadow_svm_acc", "shadow_svm_time"),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, acc_key, time_key in shadow_models:
        acc = np.nanmean([r[acc_key] for r in rows])
        time = np.nanmean([r[time_key] for r in rows])
        if not np.isnan(acc) and not np.isnan(time):
            ax.scatter(time, acc, s=100, label=model)
            ax.text(time + 0.02, acc + 0.5, model)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.axhline(70, color="gray", linestyle=":", linewidth=1, label="Target (70%)")
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=1, label="Endpoint (-0.5 s)")
    ax.set_xlabel("Mean stabilization time (s)")
    ax.set_ylabel("Shadow stability accuracy (%)")
    ax.set_title("Which model stabilizes earlier and correctly?")
    ax.set_xlim(-2.6, 0.1)
    ax.set_ylim(30, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    # Figure 7: endpoint vs full-window observer comparison.
    comparison_specs = [
        ("Endpoint -0.50 s AUC", "auc", "endpoint", (0.3, 1.0)),
        ("Endpoint -0.50 s Accuracy", "acc", "endpoint", (30, 100)),
        ("Full-window AUC", "auc", "full_window", (0.3, 1.0)),
        ("Full-window Accuracy", "acc", "full_window", (30, 100)),
    ]
    model_key_names = ["MDM", "LDA", "LDA3", "LR", "SVM"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (title, metric, prefix, ylim) in zip(axes.ravel(), comparison_specs):
        x = np.arange(len(subjects))
        width = 0.15
        for idx, model in enumerate(model_key_names):
            key = f"{prefix}_{model.lower()}_{metric}"
            vals = []
            for subject in subjects:
                sr = [r[key] for r in rows if r["subject"] == subject and not np.isnan(r[key])]
                vals.append(np.nanmean(sr) if sr else np.nan)
            ax.bar(x + (idx - 2) * width, vals, width, label=model)
        ax.axhline(0.5 if metric == "auc" else 50, color="red", linestyle="--", linewidth=1)
        ax.axhline(0.7 if metric == "auc" else 70, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_ylabel("AUC" if metric == "auc" else "Accuracy (%)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Subject {s}" for s in subjects])
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Observer models at endpoint vs full preparation window", fontweight="bold")
    fig.tight_layout()

    # Figure 8: AUC by decision-rule evidence.
    rule_auc_specs = [
        ("Rule 1 evidence: endpoint at -0.50 s", "endpoint"),
        ("Rule 2 evidence: weighted accumulation", "weighted"),
        ("Rule 3 evidence: temporal-vote score", "temporal_vote"),
        ("Rule 4 evidence: full-window observers", "full_window"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, (title, prefix) in zip(axes.ravel(), rule_auc_specs):
        x = np.arange(len(subjects))
        width = 0.15
        for idx, model in enumerate(model_key_names):
            key = f"{prefix}_{model.lower()}_auc"
            vals = []
            for subject in subjects:
                sr = [r[key] for r in rows if r["subject"] == subject and not np.isnan(r[key])]
                vals.append(np.nanmean(sr) if sr else np.nan)
            ax.bar(x + (idx - 2) * width, vals, width, label=model)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.5)")
        ax.axhline(0.7, color="gray", linestyle=":", linewidth=1, label="Target (0.7)")
        ax.set_title(title)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("AUC")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Subject {s}" for s in subjects])
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "AUC comparison across decision-rule evidence\n"
        "Endpoint/weighted/temporal-vote use data up to -0.50 s; full-window uses -2.50 to 0.00 s",
        fontweight="bold",
    )
    fig.tight_layout()

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--csv", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = tuple(str(s).zfill(3) for s in args.subjects)
    rows = collect_runs(args.root, subjects)
    write_csv(rows, args.csv)
    print(f"Saved summary CSV: {args.csv}")
    print_summary(rows)
    if not args.no_show:
        plot_figures(rows)


if __name__ == "__main__":
    main()
