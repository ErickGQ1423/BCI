#!/usr/bin/env python3
"""
Replay diagnóstico: XDF online → pipeline offline → modelo entrenado.

Este script NO entrena modelos. Carga un paquete M2 ya entrenado (.pkl),
procesa uno o más XDF con el pipeline offline de MotorCap y aplica, trial por
trial, los modelos guardados:

    MDM, LDA_shrink, LDA_shrink_3ch, LR, SVM

Objetivo:
    Separar si el fallo online viene del modelo/datos online o del streaming
    online (baseline, CAR, filtro causal, ventana en vivo).

Uso típico:
    python Replay_Online_XDF_With_Trained_Model.py

Opcional:
    python Replay_Online_XDF_With_Trained_Model.py --session S001_ONLINE
    python Replay_Online_XDF_With_Trained_Model.py --xdf /ruta/run-001_eeg.xdf
    python Replay_Online_XDF_With_Trained_Model.py --latest-labrecorder
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

import bci_runtime_env  # noqa: F401
import matplotlib.pyplot as plt
import mne
import numpy as np
from pyriemann.utils.base import invsqrtm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


# ============================================================
# CONFIGURACIÓN RÁPIDA
# ============================================================
# Edita estas líneas para correr rápido sin argumentos por terminal.
# run = None toma el XDF más reciente; run = 2 fuerza run-002.
# Los argumentos CLI (--subject, --session, --run, --data-dir, --model) siguen
# teniendo prioridad si los pasas explícitamente.
subject = "CNV_PILOT_SUBJ_021"
session = "S002_OFFLINE_FES_GLOVE_WarmUp"
run = 3
model_session = "S001_OFFLINE_FES_GLOVE"
fallback_model_session = "S001_OFFLINE_FES_GLOVE"
base_dir = getattr(
    config,
    "RECORDING_DATA_DIR",
    getattr(config, "DATA_DIR", "/home/lab-admin/Documents/CNVStudy"),
)
model_path = f"{base_dir}/sub-{subject}/models/sub-{subject}_model_motorcap_{model_session}.pkl"
fallback_model_path = (
    f"{base_dir}/sub-{subject}/models/"
    f"sub-{subject}_model_motorcap_{fallback_model_session}.pkl"
)

DEFAULT_LABRECORDER_DIR = Path("/home/lab-admin/Documents/CNVStudy/exp002")
CHANNELS_TO_DROP = ["M1", "M2", "T7", "T8", "Fp1", "Fpz", "Fp2"]
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {
    "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", "AUX9",
    "TRIGGER",
}
TARGET_MARKERS = [100, 200]
REJECT_MAX_PTP_UV = 100.0
REJECT_MIN_PTP_UV = 0.1
EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=4, ftype="butter")
MODEL_ORDER = ["MDM", "LDA_shrink", "LDA_shrink_3ch", "LR", "SVM"]


def parse_marker_value(value) -> int | None:
    try:
        return int(round(float(np.ravel(value)[0])))
    except Exception:
        return None


def discover_xdfs(args: argparse.Namespace) -> list[Path]:
    if args.xdf:
        return [Path(path).expanduser().resolve() for path in args.xdf]

    labrecorder_dir = Path(args.labrecorder_dir).expanduser()
    if args.use_labrecorder_dir:
        return discover_latest_xdf_in_dir(labrecorder_dir)

    data_dir = Path(args.data_dir).expanduser()
    if args.latest_labrecorder:
        return discover_latest_labrecorder_xdf(data_dir)

    xdf_dir = data_dir / f"sub-{args.subject}" / f"ses-{args.session}" / "eeg"
    if not xdf_dir.is_dir():
        print(f"⚠️  BIDS XDF directory does not exist: {xdf_dir}")
        print("   Falling back to latest LabRecorder export: exp*/block*.xdf")
        return discover_latest_labrecorder_xdf(data_dir)

    xdfs = sorted(
        path for path in xdf_dir.glob("*.xdf")
        if "_old" not in path.name and not path.name.endswith("_old.xdf")
    )
    if not xdfs:
        print(f"⚠️  No BIDS XDF files found in: {xdf_dir}")
        print("   Falling back to latest LabRecorder export: exp*/block*.xdf")
        return discover_latest_labrecorder_xdf(data_dir)

    if args.run is not None:
        if args.all_session_xdfs:
            raise ValueError("Use either --run or --all-session-xdfs, not both.")
        run_token = f"run-{int(args.run):03d}"
        run_xdfs = [path for path in xdfs if run_token in path.name]
        if not run_xdfs:
            available = ", ".join(path.name for path in xdfs)
            raise FileNotFoundError(
                f"No XDF found for {run_token} in {xdf_dir}\n"
                f"Available XDFs: {available}"
            )
        selected = max(run_xdfs, key=lambda path: path.stat().st_mtime).resolve()
        print(f"📼 BIDS session directory selected: {xdf_dir}")
        print(f"📼 Requested session XDF selected: {selected}")
        return [selected]

    if not args.all_session_xdfs:
        selected = max(xdfs, key=lambda path: path.stat().st_mtime).resolve()
        print(f"📼 BIDS session directory selected: {xdf_dir}")
        print(f"📼 Latest session XDF selected: {selected}")
        return [selected]

    print(f"📼 BIDS session directory selected: {xdf_dir}")
    print(f"📼 All session XDFs selected: {len(xdfs)} file(s)")
    return xdfs


def discover_latest_xdf_in_dir(xdf_dir: Path) -> list[Path]:
    """Find the newest XDF directly inside a LabRecorder export directory."""
    candidates = sorted(
        xdf_dir.glob("*.xdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [
        path for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"No XDF files found in LabRecorder dir: {xdf_dir}")

    selected = candidates[0].resolve()
    print(f"📼 LabRecorder directory selected: {xdf_dir}")
    print(f"📼 XDF selected: {selected}")
    return [selected]


def discover_latest_labrecorder_xdf(data_dir: Path) -> list[Path]:
    """Find the newest LabRecorder-style XDF under CNVStudy/exp*/block*.xdf."""
    candidates = sorted(
        data_dir.glob("exp*/block*.xdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [
        path for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No LabRecorder XDF found with pattern: {data_dir}/exp*/block*.xdf"
        )

    selected = candidates[0].resolve()
    print(f"📼 Latest LabRecorder XDF selected: {selected}")
    return [selected]


def load_xdfs_as_raw(xdf_paths: list[Path]) -> mne.io.BaseRaw:
    raw_list = []

    for xdf_path in xdf_paths:
        print(f"   └─ Loading XDF: {xdf_path}")
        eeg_s, marker_s = load_xdf(str(xdf_path))

        eeg_data = np.asarray(eeg_s["time_series"], dtype=float).T
        eeg_timestamps = np.asarray(eeg_s["time_stamps"], dtype=float)
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_values_all = [
            parse_marker_value(value)
            for value in marker_s.get("time_series", [])
        ]
        marker_timestamps_all = np.asarray(
            marker_s.get("time_stamps", []),
            dtype=float,
        )
        marker_values = []
        marker_timestamps = []
        for value, timestamp in zip(marker_values_all, marker_timestamps_all):
            if value in TARGET_MARKERS:
                marker_values.append(value)
                marker_timestamps.append(timestamp)

        if not marker_values:
            print("      ⚠️  No target markers 100/200 found; skipping file.")
            continue

        valid_ch = [
            ch for ch in channel_names
            if ch not in NON_EEG_CHANNELS
        ]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info = mne.create_info(
            ch_names=valid_ch,
            sfreq=float(config.FS),
            ch_types="eeg",
        )
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

        renames = {
            old: new for old, new in RENAME_DICT.items()
            if old in raw_tmp.ch_names
        }
        if renames:
            raw_tmp.rename_channels(renames)

        raw_tmp.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            on_missing="warn",
        )

        drop_targets = [
            ch for ch in CHANNELS_TO_DROP
            if ch in raw_tmp.ch_names
        ]
        if drop_targets:
            raw_tmp.drop_channels(drop_targets)

        t0 = float(eeg_timestamps[0])
        raw_tmp.set_annotations(
            mne.Annotations(
                onset=np.asarray(marker_timestamps, dtype=float) - t0,
                duration=np.zeros(len(marker_values)),
                description=[str(value) for value in marker_values],
                orig_time=None,
            )
        )
        raw_list.append(raw_tmp)

    if not raw_list:
        raise RuntimeError("No usable XDF files with target markers were loaded.")

    return mne.concatenate_raws(raw_list)


def preprocess_and_epoch(raw: mne.io.BaseRaw, pkg: dict) -> mne.Epochs:
    print(
        "🎛️  Offline-matched preprocessing: "
        f"CAR → notch60 → {EEG_L_FREQ:.1f}–{EEG_H_FREQ:.1f} Hz Butterworth 4º "
        "→ baseline(-5,-3)"
    )
    raw = raw.copy()
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw.filter(
        l_freq=EEG_L_FREQ,
        h_freq=EEG_H_FREQ,
        method="iir",
        iir_params=EEG_IIR_PARAMS,
        phase="forward",
        picks="eeg",
        verbose=False,
    )

    events_all, event_id_map = mne.events_from_annotations(raw, verbose=False)
    missing = [str(marker) for marker in TARGET_MARKERS if str(marker) not in event_id_map]
    if missing:
        raise RuntimeError(f"Missing target marker(s) in XDF annotations: {missing}")

    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)": event_id_map["200"],
    }
    events = events_all[np.isin(events_all[:, 2], list(event_dict.values()))]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=-5.0,
        tmax=6.0,
        baseline=(-5.0, -3.0),
        reject=None,
        flat=None,
        preload=True,
        detrend=None,
        verbose=False,
    )

    picks = [ch for ch in pkg["picks"] if ch in epochs.ch_names]
    if len(picks) != len(pkg["picks"]):
        missing_picks = sorted(set(pkg["picks"]) - set(picks))
        raise RuntimeError(f"Model channel(s) missing in XDF: {missing_picks}")

    data_uv = epochs.get_data(picks=picks) * 1e6
    ptp = np.ptp(data_uv, axis=2)
    bad = np.where(
        (ptp.max(axis=1) > REJECT_MAX_PTP_UV)
        | (ptp.max(axis=1) < REJECT_MIN_PTP_UV)
    )[0].tolist()
    if bad:
        epochs.drop(bad, reason="LOCAL_PTP_REJECT")
        print(f"🛡️  Dropped {len(bad)} trial(s) by localized PTP gate.")

    return epochs


def match_samples(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    target_samples = int(template.shape[1])
    if signal.shape[1] == target_samples:
        return signal
    if signal.shape[1] < 1 or target_samples < 1:
        raise ValueError("MDM signal/template cannot be empty")
    idx = np.linspace(0, signal.shape[1] - 1, target_samples).round().astype(int)
    return signal[:, idx]


def build_mdm_covariance(
    signal_uv: np.ndarray,
    template_uv: np.ndarray,
    cov_reg: float,
    recenter_ref: np.ndarray | None,
) -> np.ndarray:
    signal_uv = match_samples(signal_uv, template_uv)
    extended = np.concatenate([signal_uv, template_uv], axis=0)
    cov = extended @ extended.T
    trace = np.trace(cov)
    if trace <= 1e-12 or not np.isfinite(trace):
        raise ValueError("Invalid covariance trace")
    cov = cov / trace

    if recenter_ref is not None:
        transform = invsqrtm(recenter_ref)
        cov = transform @ cov @ transform.T
        cov = 0.5 * (cov + cov.T)

    cov = cov + cov_reg * np.eye(cov.shape[0])
    return 0.5 * (cov + cov.T)


def predict_mdm_pmi_at_step(
    signal_uv: np.ndarray,
    pkg: dict,
    step: int,
    cov_reg: float,
    recenter_refs: list,
    mi_id: int,
) -> float:
    """Return MDM P(MI) for one trial at one saved M2 time step."""
    mdm_model = pkg["mdm_models"][step]
    template = pkg["mdm_templates"][step]
    ref = (
        recenter_refs[step]
        if len(recenter_refs) > step
        and pkg.get("mdm_recenter_mode") == "train_riemann_mean"
        else None
    )
    cov = build_mdm_covariance(signal_uv, template, cov_reg, ref)
    proba = mdm_model.predict_proba(np.expand_dims(cov, axis=0))[0]
    mi_col = list(mdm_model.classes_).index(mi_id)
    return float(proba[mi_col])


def predict_trials(epochs: mne.Epochs, pkg: dict, endpoint: float) -> list[dict]:
    picks = list(pkg["picks"])
    t_points = np.asarray(pkg["t_points"], dtype=float)
    endpoint_step = int(np.argmin(np.abs(t_points - endpoint)))
    mi_id = int(pkg["MI_ID"])
    rest_id = int(pkg["REST_ID"])
    cov_reg = float(pkg.get("cov_reg", 1e-4))
    recenter_refs = pkg.get("mdm_recenter_refs", [])

    data_uv = epochs.get_data(picks=picks) * 1e6
    times = epochs.times
    labels = epochs.events[:, -1].astype(int)

    start_idx = int(np.argmin(np.abs(times - float(pkg["t_start"]))))
    all_time_idx = [
        int(np.argmin(np.abs(times - time_point)))
        for time_point in t_points
    ]

    compact_picks = pkg.get("compact_lda_picks", [])
    compact_indices = [picks.index(ch) for ch in compact_picks if ch in picks]

    rows: list[dict] = []
    for trial_idx, label in enumerate(labels, start=1):
        row = {
            "trial": trial_idx,
            "target": int(label),
            "target_name": "MI" if label == mi_id else "REST",
        }
        endpoint_idx = all_time_idx[endpoint_step]
        step_signal = data_uv[trial_idx - 1, :, start_idx:endpoint_idx + 1]

        # MDM at control endpoint.
        try:
            row["MDM"] = predict_mdm_pmi_at_step(
                step_signal, pkg, endpoint_step, cov_reg, recenter_refs, mi_id
            )
        except Exception as exc:
            row["MDM"] = np.nan
            row["MDM_error"] = str(exc)

        # Classical observers at the same endpoint step.
        step_time_idx = all_time_idx[:endpoint_step + 1]
        features = data_uv[trial_idx - 1, :, :][:, step_time_idx].flatten().reshape(1, -1)
        try:
            model = pkg["skl_models"][endpoint_step]
            proba = model.predict_proba(features)[0]
            mi_col = list(model.classes_).index(mi_id)
            row["LDA_shrink"] = float(proba[mi_col])
        except Exception as exc:
            row["LDA_shrink"] = np.nan
            row["LDA_shrink_error"] = str(exc)

        if compact_indices and len(pkg.get("compact_lda_models", [])) > endpoint_step:
            try:
                compact_features = (
                    data_uv[trial_idx - 1, compact_indices, :][:, step_time_idx]
                    .flatten()
                    .reshape(1, -1)
                )
                model = pkg["compact_lda_models"][endpoint_step]
                proba = model.predict_proba(compact_features)[0]
                mi_col = list(model.classes_).index(mi_id)
                row["LDA_shrink_3ch"] = float(proba[mi_col])
            except Exception as exc:
                row["LDA_shrink_3ch"] = np.nan
                row["LDA_shrink_3ch_error"] = str(exc)

        observer_models = pkg.get("observer_skl_models", {})
        for name in ("LR", "SVM"):
            models = observer_models.get(name, [])
            if len(models) <= endpoint_step:
                continue
            try:
                model = models[endpoint_step]
                proba = model.predict_proba(features)[0]
                mi_col = list(model.classes_).index(mi_id)
                row[name] = float(proba[mi_col])
            except Exception as exc:
                row[name] = np.nan
                row[f"{name}_error"] = str(exc)

        rows.append(row)

    return rows


def summarize(rows: list[dict], model_names: list[str]) -> list[dict]:
    y_true = np.asarray([1 if row["target_name"] == "MI" else 0 for row in rows])
    summaries = []
    for model in model_names:
        scores = np.asarray([row.get(model, np.nan) for row in rows], dtype=float)
        valid = np.isfinite(scores)
        if valid.sum() == 0:
            continue
        y = y_true[valid]
        s = scores[valid]
        pred = (s >= 0.5).astype(int)
        mi_mask = y == 1
        rest_mask = y == 0
        auc = (
            roc_auc_score(y, s)
            if len(np.unique(y)) == 2
            else float("nan")
        )
        summaries.append({
            "model": model,
            "n": int(valid.sum()),
            "auc": float(auc),
            "accuracy": float(accuracy_score(y, pred)),
            "mi_recall": float(np.mean(pred[mi_mask] == 1)) if mi_mask.any() else np.nan,
            "rest_recall": float(np.mean(pred[rest_mask] == 0)) if rest_mask.any() else np.nan,
            "mean_pmi_mi": float(np.mean(s[mi_mask])) if mi_mask.any() else np.nan,
            "mean_pmi_rest": float(np.mean(s[rest_mask])) if rest_mask.any() else np.nan,
        })
    return summaries


def predict_observer_pmi_at_step(
    data_uv_trial: np.ndarray,
    pkg: dict,
    step: int,
    step_time_idx: list[int],
    model_name: str,
    compact_indices: list[int],
    mi_id: int,
) -> float:
    """Return observer P(MI) for one trial at one saved M2 time step."""
    if model_name == "LDA_shrink":
        model = pkg["skl_models"][step]
        features = data_uv_trial[:, step_time_idx].flatten().reshape(1, -1)
    elif model_name == "LDA_shrink_3ch":
        if not compact_indices or len(pkg.get("compact_lda_models", [])) <= step:
            return np.nan
        model = pkg["compact_lda_models"][step]
        features = data_uv_trial[compact_indices, :][:, step_time_idx].flatten().reshape(1, -1)
    else:
        models = pkg.get("observer_skl_models", {}).get(model_name, [])
        if len(models) <= step:
            return np.nan
        model = models[step]
        features = data_uv_trial[:, step_time_idx].flatten().reshape(1, -1)

    proba = model.predict_proba(features)[0]
    mi_col = list(model.classes_).index(mi_id)
    return float(proba[mi_col])


def timepoint_diagnostics(
    epochs: mne.Epochs,
    pkg: dict,
    model_names: list[str],
    threshold: float,
) -> list[dict]:
    """Passive per-timepoint accuracy/recall for saved M2 models."""
    picks = list(pkg["picks"])
    t_points = np.asarray(pkg["t_points"], dtype=float)
    mi_id = int(pkg["MI_ID"])
    cov_reg = float(pkg.get("cov_reg", 1e-4))
    recenter_refs = pkg.get("mdm_recenter_refs", [])

    data_uv = epochs.get_data(picks=picks) * 1e6
    times = epochs.times
    labels = epochs.events[:, -1].astype(int)
    y_true = np.asarray([1 if int(label) == mi_id else 0 for label in labels])

    start_idx = int(np.argmin(np.abs(times - float(pkg["t_start"]))))
    all_time_idx = [
        int(np.argmin(np.abs(times - time_point)))
        for time_point in t_points
    ]
    compact_picks = pkg.get("compact_lda_picks", [])
    compact_indices = [picks.index(ch) for ch in compact_picks if ch in picks]

    rows: list[dict] = []
    for step, time_point in enumerate(t_points):
        endpoint_idx = all_time_idx[step]
        step_time_idx = all_time_idx[:step + 1]
        for model_name in model_names:
            scores = []
            for trial_idx in range(len(labels)):
                try:
                    if model_name == "MDM":
                        step_signal = data_uv[trial_idx, :, start_idx:endpoint_idx + 1]
                        p_mi = predict_mdm_pmi_at_step(
                            step_signal, pkg, step, cov_reg, recenter_refs, mi_id
                        )
                    else:
                        p_mi = predict_observer_pmi_at_step(
                            data_uv[trial_idx],
                            pkg,
                            step,
                            step_time_idx,
                            model_name,
                            compact_indices,
                            mi_id,
                        )
                except Exception:
                    p_mi = np.nan
                scores.append(p_mi)

            scores = np.asarray(scores, dtype=float)
            valid = np.isfinite(scores)
            if valid.sum() == 0:
                continue
            y = y_true[valid]
            pred = (scores[valid] >= threshold).astype(int)
            mi_mask = y == 1
            rest_mask = y == 0
            rows.append({
                "model": model_name,
                "step": int(step),
                "time": float(time_point),
                "n": int(valid.sum()),
                "auc": (
                    float(roc_auc_score(y, scores[valid]))
                    if len(np.unique(y)) == 2 else np.nan
                ),
                "accuracy": float(accuracy_score(y, pred)),
                "mi_recall": (
                    float(np.mean(pred[mi_mask] == 1)) if mi_mask.any() else np.nan
                ),
                "rest_recall": (
                    float(np.mean(pred[rest_mask] == 0)) if rest_mask.any() else np.nan
                ),
            })
    return rows


def print_timepoint_diagnostics(rows: list[dict], focus_model: str = "MDM") -> None:
    focus_rows = [row for row in rows if row["model"] == focus_model]
    if not focus_rows:
        return
    print("\n" + "=" * 78)
    print(f"⏱️  TIMEPOINT DIAGNOSTIC — {focus_model} por instante")
    print("=" * 78)
    print(" step   time      AUC    Acc%   MI recall   REST recall")
    print(" " + "-" * 58)
    for row in focus_rows:
        print(
            f" {row['step']:>4d}  "
            f"{row['time']:>+6.2f}s  "
            f"{row['auc']:>6.3f}  "
            f"{100*row['accuracy']:>6.1f}  "
            f"{100*row['mi_recall']:>9.1f}%  "
            f"{100*row['rest_recall']:>11.1f}%"
        )


def plot_timepoint_diagnostics(
    rows: list[dict],
    subject_name: str,
    session_name: str,
    threshold: float,
    save_path: Path,
    show_plot: bool,
) -> None:
    if not rows:
        return

    models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    colors = {
        "MDM": "#5e2a8a",
        "LDA_shrink": "#7b6fdc",
        "LDA_shrink_3ch": "#80b918",
        "LR": "#f4a582",
        "SVM": "#d6604d",
    }
    panels = [
        ("accuracy", "Accuracy (%)"),
        ("mi_recall", "MI recall (%)"),
        ("rest_recall", "REST recall (%)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f"Replay timepoint diagnostic — {subject_name} | {session_name}\n"
        f"Predicciones correctas por timepoint, threshold P(MI)={threshold:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (metric, ylabel) in zip(axes, panels):
        for model in models:
            model_rows = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: row["time"],
            )
            times = [row["time"] for row in model_rows]
            values = [100.0 * row[metric] for row in model_rows]
            ax.plot(
                times,
                values,
                marker="o",
                linestyle="--" if model == "MDM" else "-",
                linewidth=3.0 if model == "MDM" else 2.0,
                color=colors.get(model),
                label=model,
            )
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Azar (50%)")
        ax.axhline(70, color="gray", linestyle=":", linewidth=1, label="Objetivo (70%)")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Tiempo disponible (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"\n🖼️  Figura timepoint guardada: {save_path}")

    if show_plot:
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"⚠️  No se pudo abrir ventana matplotlib: {exc}")
    else:
        plt.close(fig)


def mdm_vote_replay(
    epochs: mne.Epochs,
    pkg: dict,
    window_start: float,
    window_end: float,
    required_votes: int,
    threshold: float,
) -> list[dict]:
    """
    Passive diagnostic rule: accumulate MDM votes across saved M2 time points.

    This does not change the online decoder. It only answers:
    "What would have happened if MDM decided by temporal majority/votes?"
    """
    picks = list(pkg["picks"])
    t_points = np.asarray(pkg["t_points"], dtype=float)
    mi_id = int(pkg["MI_ID"])
    cov_reg = float(pkg.get("cov_reg", 1e-4))
    recenter_refs = pkg.get("mdm_recenter_refs", [])

    lo = min(float(window_start), float(window_end))
    hi = max(float(window_start), float(window_end))
    vote_steps = [
        step for step, time_point in enumerate(t_points)
        if lo - 1e-9 <= float(time_point) <= hi + 1e-9
    ]
    if not vote_steps:
        raise ValueError(
            f"No M2 time points inside vote window [{window_start}, {window_end}]"
        )
    if required_votes < 1:
        raise ValueError("required_votes must be >= 1")

    data_uv = epochs.get_data(picks=picks) * 1e6
    times = epochs.times
    labels = epochs.events[:, -1].astype(int)
    start_idx = int(np.argmin(np.abs(times - float(pkg["t_start"]))))
    all_time_idx = [
        int(np.argmin(np.abs(times - time_point)))
        for time_point in t_points
    ]

    vote_rows: list[dict] = []
    for trial_idx, label in enumerate(labels, start=1):
        mi_votes = 0
        rest_votes = 0
        winner = "AMBIGUOUS"
        stop_step = None
        stop_time = np.nan
        stop_confidence = np.nan
        valid_steps = 0

        for step in vote_steps:
            endpoint_idx = all_time_idx[step]
            step_signal = data_uv[trial_idx - 1, :, start_idx:endpoint_idx + 1]
            try:
                p_mi = predict_mdm_pmi_at_step(
                    step_signal, pkg, step, cov_reg, recenter_refs, mi_id
                )
            except Exception:
                continue

            if not np.isfinite(p_mi):
                continue

            valid_steps += 1
            if p_mi >= threshold:
                mi_votes += 1
                current_class = "MI"
                current_confidence = p_mi
            else:
                rest_votes += 1
                current_class = "REST"
                current_confidence = 1.0 - p_mi

            if mi_votes >= required_votes or rest_votes >= required_votes:
                winner = current_class
                stop_step = int(step)
                stop_time = float(t_points[step])
                stop_confidence = float(current_confidence)
                break

        target_name = "MI" if int(label) == mi_id else "REST"
        vote_rows.append({
            "trial": trial_idx,
            "target": int(label),
            "target_name": target_name,
            "decision": winner,
            "correct": bool(winner == target_name) if winner != "AMBIGUOUS" else False,
            "mi_votes": int(mi_votes),
            "rest_votes": int(rest_votes),
            "valid_steps": int(valid_steps),
            "stop_step": stop_step,
            "stop_time": stop_time,
            "confidence": stop_confidence,
        })

    return vote_rows


def summarize_vote_rows(vote_rows: list[dict]) -> dict:
    n_trials = len(vote_rows)
    decided = [row for row in vote_rows if row["decision"] != "AMBIGUOUS"]
    n_decided = len(decided)
    n_ambiguous = n_trials - n_decided
    correct_all = sum(row["correct"] for row in vote_rows)
    correct_decided = sum(row["correct"] for row in decided)

    mi_trials = [row for row in vote_rows if row["target_name"] == "MI"]
    rest_trials = [row for row in vote_rows if row["target_name"] == "REST"]
    mi_recall = (
        np.mean([row["decision"] == "MI" for row in mi_trials])
        if mi_trials else np.nan
    )
    rest_recall = (
        np.mean([row["decision"] == "REST" for row in rest_trials])
        if rest_trials else np.nan
    )
    stop_times = [
        row["stop_time"] for row in decided
        if np.isfinite(row.get("stop_time", np.nan))
    ]
    return {
        "n_trials": n_trials,
        "n_decided": n_decided,
        "n_ambiguous": n_ambiguous,
        "accuracy_all": correct_all / n_trials if n_trials else np.nan,
        "accuracy_decided": correct_decided / n_decided if n_decided else np.nan,
        "mi_recall": float(mi_recall),
        "rest_recall": float(rest_recall),
        "mean_stop_time": float(np.mean(stop_times)) if stop_times else np.nan,
    }


def print_vote_summary(
    vote_rows: list[dict],
    window_start: float,
    window_end: float,
    required_votes: int,
    threshold: float,
) -> None:
    summary = summarize_vote_rows(vote_rows)
    print("\n" + "=" * 78)
    print("🗳️  MDM TEMPORAL VOTE REPLAY — diagnóstico, NO controla la BCI")
    print("=" * 78)
    print(
        f"Ventana: {window_start:+.2f} → {window_end:+.2f} s | "
        f"votos para ganar: {required_votes} | threshold P(MI): {threshold:.2f}"
    )
    print(
        "Trials  Decididos  Ambiguous  Acc total  Acc decidida  "
        "MI recall  REST recall  Stop medio"
    )
    print("-" * 91)
    print(
        f"{summary['n_trials']:>6d}  "
        f"{summary['n_decided']:>9d}  "
        f"{summary['n_ambiguous']:>9d}  "
        f"{100*summary['accuracy_all']:>8.1f}%  "
        f"{100*summary['accuracy_decided']:>11.1f}%  "
        f"{100*summary['mi_recall']:>8.1f}%  "
        f"{100*summary['rest_recall']:>10.1f}%  "
        f"{summary['mean_stop_time']:>9.2f}s"
    )


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_threshold_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {value}")
        values.append(value)
    if not values:
        raise ValueError("Threshold grid cannot be empty.")
    return values


def mdm_threshold_sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    scores = np.asarray([row.get("MDM", np.nan) for row in rows], dtype=float)
    y_true = np.asarray([1 if row["target_name"] == "MI" else 0 for row in rows])
    valid = np.isfinite(scores)
    y = y_true[valid]
    s = scores[valid]
    mi_mask = y == 1
    rest_mask = y == 0

    table = []
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        mi_recall = float(np.mean(pred[mi_mask] == 1)) if mi_mask.any() else np.nan
        rest_recall = float(np.mean(pred[rest_mask] == 0)) if rest_mask.any() else np.nan
        balanced_acc = float(np.nanmean([mi_recall, rest_recall]))
        table.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": balanced_acc,
            "mi_recall": mi_recall,
            "rest_recall": rest_recall,
            "mi_pred": int(np.sum(pred == 1)),
            "rest_pred": int(np.sum(pred == 0)),
        })
    return table


def print_mdm_threshold_sweep(rows: list[dict], thresholds: list[float]) -> None:
    if not any("MDM" in row and np.isfinite(row.get("MDM", np.nan)) for row in rows):
        return

    table = mdm_threshold_sweep(rows, thresholds)
    best = max(
        table,
        key=lambda row: (
            row["balanced_accuracy"],
            row["accuracy"],
            -abs(row["mi_recall"] - row["rest_recall"]),
        ),
    )

    print("\nBarrido de umbral MDM endpoint:")
    print("  thr    Acc%   BalAcc%   MI recall   REST recall   pred_MI  pred_REST")
    print("  " + "-" * 73)
    for row in table:
        marker = "  <-- mejor balance" if row is best else ""
        print(
            f"  {row['threshold']:.2f}  "
            f"{100*row['accuracy']:>6.1f}  "
            f"{100*row['balanced_accuracy']:>8.1f}  "
            f"{100*row['mi_recall']:>9.1f}%  "
            f"{100*row['rest_recall']:>11.1f}%  "
            f"{row['mi_pred']:>7d}  "
            f"{row['rest_pred']:>9d}"
            f"{marker}"
        )


def print_summary(rows: list[dict], summaries: list[dict], mdm_thresholds: list[float]) -> None:
    print("\n" + "=" * 78)
    print("🔁  REPLAY ONLINE XDF — MODELO ENTRENADO, PIPELINE OFFLINE")
    print("=" * 78)
    n_mi = sum(row["target_name"] == "MI" for row in rows)
    n_rest = sum(row["target_name"] == "REST" for row in rows)
    print(f"Trials usados: {len(rows)} | MI={n_mi} | REST={n_rest}")
    print("\nModelo             N     AUC    Acc%   MI recall   REST recall   P(MI)|MI  P(MI)|REST")
    print("-" * 88)
    for row in summaries:
        print(
            f"{row['model']:<16} {row['n']:>3d}  "
            f"{row['auc']:>6.3f}  {100*row['accuracy']:>6.1f}  "
            f"{100*row['mi_recall']:>9.1f}%  {100*row['rest_recall']:>11.1f}%  "
            f"{row['mean_pmi_mi']:>8.3f}  {row['mean_pmi_rest']:>10.3f}"
        )

    if "MDM" in [row["model"] for row in summaries]:
        mdm_scores = np.asarray([row.get("MDM", np.nan) for row in rows], dtype=float)
        y = np.asarray([1 if row["target_name"] == "MI" else 0 for row in rows])
        valid = np.isfinite(mdm_scores)
        if valid.any():
            pred = (mdm_scores[valid] >= 0.5).astype(int)
            cm = confusion_matrix(y[valid], pred, labels=[1, 0])
            print("\nMatriz MDM endpoint (filas reales MI/REST, columnas pred MI/REST):")
            print(f"  MI   -> MI={cm[0,0]} | REST={cm[0,1]}")
            print(f"  REST -> MI={cm[1,0]} | REST={cm[1,1]}")
            print_mdm_threshold_sweep(rows, mdm_thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay XDF online with an already-trained M2 model."
    )
    parser.add_argument(
        "--model",
        default=model_path,
        help="Path to trained M2 .pkl model.",
    )
    parser.add_argument(
        "--data-dir",
        default=base_dir,
        help="Base study directory containing sub-*/ses-*/eeg.",
    )
    parser.add_argument(
        "--subject",
        default=subject,
        help="Subject ID, e.g. CNV_PILOT_SUBJ_014.",
    )
    parser.add_argument(
        "--session",
        default=session,
        help="Session name to replay when --xdf is not provided.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=run,
        help=(
            "Specific BIDS run number to replay, e.g. --run 2 selects run-002. "
            "Default None selects the newest XDF unless --all-session-xdfs is used."
        ),
    )
    parser.add_argument(
        "--xdf",
        action="append",
        help="Specific XDF path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-session-xdfs",
        action="store_true",
        help=(
            "Replay all valid .xdf files in the selected BIDS session. "
            "Default is to replay only the newest .xdf in that session."
        ),
    )
    parser.add_argument(
        "--labrecorder-dir",
        default=str(DEFAULT_LABRECORDER_DIR),
        help=(
            "LabRecorder directory containing block*.xdf. "
            f"Default: {DEFAULT_LABRECORDER_DIR}"
        ),
    )
    parser.add_argument(
        "--use-labrecorder-dir",
        action="store_true",
        help="Force replay from --labrecorder-dir.",
    )
    parser.add_argument(
        "--latest-labrecorder",
        action="store_true",
        help="Replay the newest LabRecorder-style XDF: DATA_DIR/exp*/block*.xdf.",
    )
    parser.add_argument(
        "--endpoint",
        type=float,
        default=float(getattr(config, "PREP_CONTROL_ENDPOINT", -1.00)),
        help="Control endpoint to evaluate, usually -0.75.",
    )
    parser.add_argument(
        "--mdm-thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated MDM thresholds to evaluate in replay.",
    )
    parser.add_argument(
        "--vote-window-start",
        type=float,
        default=-1.25,
        help="Start time for passive MDM temporal-vote replay.",
    )
    parser.add_argument(
        "--vote-window-end",
        type=float,
        default=-0.75,
        help="End time for passive MDM temporal-vote replay.",
    )
    parser.add_argument(
        "--vote-required-votes",
        type=int,
        default=2,
        help="Minimum MDM votes needed to make a temporal-vote decision.",
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.85,
        help="P(MI) threshold for MDM vote replay. >= threshold votes MI.",
    )
    parser.add_argument(
        "--timepoint-threshold",
        type=float,
        default=0.50,
        help="P(MI) threshold for the per-timepoint diagnostic.",
    )
    parser.add_argument(
        "--no-timepoint-plot",
        action="store_true",
        help="Skip opening the per-timepoint diagnostic figure.",
    )
    parser.add_argument(
        "--fig-dir",
        default="figuras",
        help="Directory where diagnostic figures are saved.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write trial/summary CSV files. By default, only prints results.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/replay_online_xdf",
        help="Output prefix used only when --save-csv is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        default_model_path = Path(model_path).expanduser().resolve()
        fallback_path = Path(fallback_model_path).expanduser().resolve()
        if str(args.model) == str(globals()["model_path"]) and fallback_path.is_file():
            print(f"⚠️  Default warmup model not found yet: {default_model_path}")
            print(f"   Falling back to previous available model: {fallback_path}")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

    with model_path.open("rb") as file:
        pkg = pickle.load(file)
    if pkg.get("model_type") != "M2_LDA_shrink_MDM":
        raise RuntimeError(f"Unsupported model_type: {pkg.get('model_type')}")

    print(f"✅ Modelo: {model_path}")
    print(f"   Canales: {pkg['picks']}")
    print(f"   Endpoint replay: {args.endpoint:+.2f} s")

    xdfs = discover_xdfs(args)
    raw = load_xdfs_as_raw(xdfs)
    epochs = preprocess_and_epoch(raw, pkg)
    rows = predict_trials(epochs, pkg, args.endpoint)
    summaries = summarize(rows, [name for name in MODEL_ORDER if any(name in row for row in rows)])
    print_summary(rows, summaries, parse_threshold_grid(args.mdm_thresholds))

    vote_rows = mdm_vote_replay(
        epochs=epochs,
        pkg=pkg,
        window_start=args.vote_window_start,
        window_end=args.vote_window_end,
        required_votes=args.vote_required_votes,
        threshold=args.vote_threshold,
    )
    print_vote_summary(
        vote_rows,
        window_start=args.vote_window_start,
        window_end=args.vote_window_end,
        required_votes=args.vote_required_votes,
        threshold=args.vote_threshold,
    )

    timepoint_rows = timepoint_diagnostics(
        epochs=epochs,
        pkg=pkg,
        model_names=[name for name in MODEL_ORDER if name == "MDM" or any(name in row for row in rows)],
        threshold=args.timepoint_threshold,
    )
    print_timepoint_diagnostics(timepoint_rows, focus_model="MDM")
    fig_name = (
        f"{args.subject}_{args.session}_timepoint_diagnostic_"
        f"thr{args.timepoint_threshold:.2f}.png"
    )
    plot_timepoint_diagnostics(
        timepoint_rows,
        subject_name=args.subject,
        session_name=args.session,
        threshold=args.timepoint_threshold,
        save_path=Path(args.fig_dir) / fig_name,
        show_plot=not args.no_timepoint_plot,
    )

    if args.save_csv:
        prefix = Path(args.output_prefix)
        trial_fields = ["trial", "target", "target_name", *MODEL_ORDER]
        write_csv(rows, prefix.with_name(prefix.name + "_trials.csv"), trial_fields)
        vote_fields = [
            "trial", "target", "target_name", "decision", "correct",
            "mi_votes", "rest_votes", "valid_steps",
            "stop_step", "stop_time", "confidence",
        ]
        write_csv(vote_rows, prefix.with_name(prefix.name + "_mdm_vote_trials.csv"), vote_fields)
        timepoint_fields = [
            "model", "step", "time", "n", "auc",
            "accuracy", "mi_recall", "rest_recall",
        ]
        write_csv(
            timepoint_rows,
            prefix.with_name(prefix.name + "_timepoint_summary.csv"),
            timepoint_fields,
        )
        write_csv(
            summaries,
            prefix.with_name(prefix.name + "_summary.csv"),
            [
                "model", "n", "auc", "accuracy", "mi_recall", "rest_recall",
                "mean_pmi_mi", "mean_pmi_rest",
            ],
        )
        print(f"\n💾 CSV trials : {prefix.with_name(prefix.name + '_trials.csv')}")
        print(f"💾 CSV summary: {prefix.with_name(prefix.name + '_summary.csv')}")
        print(f"💾 CSV votes  : {prefix.with_name(prefix.name + '_mdm_vote_trials.csv')}")
        print(f"💾 CSV TP     : {prefix.with_name(prefix.name + '_timepoint_summary.csv')}")


if __name__ == "__main__":
    main()
