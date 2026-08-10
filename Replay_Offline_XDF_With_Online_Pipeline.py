#!/usr/bin/env python3
"""
Replay diagnóstico inverso: XDF offline → pipeline estilo online → modelo entrenado.

Este script NO entrena modelos y NO modifica el decoder online.

Objetivo:
    Tomar registros OFFLINE, que sabemos que pueden verse bien con el pipeline
    offline de entrenamiento, y pasarlos por una aproximación del pipeline
    online streaming. Esto ayuda a identificar si la caída online viene de la
    señal/modelo o de diferencias de preprocesamiento/ventana/baseline.

Uso típico:
    python Replay_Offline_XDF_With_Online_Pipeline.py

Modos CAR:
    legacy_filter_then_car  = comportamiento online previo al ajuste de CAR
                              (notch/bandpass → CAR)
    current_car_then_filter = comportamiento online actual
                              (CAR → notch/bandpass)
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import bci_runtime_env  # noqa: F401
import numpy as np
from pyriemann.utils.base import invsqrtm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import config
from Utils.preprocessing import (
    apply_streaming_filters,
    get_valid_channel_mask_and_metadata,
    initialize_filter_bank,
)
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


TARGET_MARKERS = [100, 200]
MODEL_ORDER = ["MDM", "LDA_shrink", "LDA_shrink_3ch", "LR", "SVM"]


def parse_marker_value(value) -> int | None:
    try:
        return int(round(float(np.ravel(value)[0])))
    except Exception:
        return None


def discover_xdfs(args: argparse.Namespace) -> list[Path]:
    if args.xdf:
        return [Path(path).expanduser().resolve() for path in args.xdf]

    xdf_dir = (
        Path(args.data_dir).expanduser()
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / "eeg"
    )
    if not xdf_dir.is_dir():
        raise FileNotFoundError(f"XDF directory does not exist: {xdf_dir}")

    xdfs = sorted(
        path for path in xdf_dir.glob("*.xdf")
        if "_old" not in path.name and not path.name.endswith("_old.xdf")
    )
    if not xdfs:
        raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")
    return xdfs


def marker_events(marker_s: dict, eeg_timestamps: np.ndarray) -> list[dict]:
    values_all = [parse_marker_value(value) for value in marker_s.get("time_series", [])]
    timestamps_all = np.asarray(marker_s.get("time_stamps", []), dtype=float)

    events = []
    for value, timestamp in zip(values_all, timestamps_all):
        if value not in TARGET_MARKERS:
            continue
        sample = int(np.argmin(np.abs(eeg_timestamps - float(timestamp))))
        events.append({
            "target": int(value),
            "target_name": "MI" if int(value) == 200 else "REST",
            "sample": sample,
            "timestamp": float(timestamp),
        })
    return events


def process_like_online(
    eeg_data_uv: np.ndarray,
    channel_names: list[str],
    pkg: dict,
    car_order: str,
    car_reference: str,
) -> tuple[np.ndarray, list[str]]:
    """Return filtered data in µV, channels x samples, in model-pick space."""
    valid_names, _valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
        eeg_data_uv,
        channel_names,
        fs=float(config.FS),
        drop_mastoids=True,
    )
    valid_data = eeg_data_uv[valid_indices, :]

    picks = list(pkg["picks"])
    missing = [ch for ch in picks if ch not in valid_names]
    if missing:
        raise RuntimeError(f"Model channel(s) missing in XDF: {missing}")

    if car_reference == "all_valid_eeg":
        reference_data = valid_data
        reference_names = list(valid_names)
    elif car_reference == "selected":
        pick_idx = [valid_names.index(ch) for ch in picks]
        reference_data = valid_data[pick_idx, :]
        reference_names = list(picks)
    else:
        raise ValueError(f"Unsupported car_reference: {car_reference}")

    filter_bank = initialize_filter_bank(
        fs=float(config.FS),
        lowcut=float(config.LOWCUT),
        highcut=float(config.HIGHCUT),
        notch_freqs=[60],
        notch_q=30,
    )

    if car_order == "legacy_filter_then_car":
        filtered, _state = apply_streaming_filters(reference_data, filter_bank, {})
        filtered = filtered - filtered.mean(axis=0, keepdims=True)
    elif car_order == "current_car_then_filter":
        referenced = reference_data - reference_data.mean(axis=0, keepdims=True)
        filtered, _state = apply_streaming_filters(referenced, filter_bank, {})
    else:
        raise ValueError(f"Unsupported car_order: {car_order}")

    pick_idx = [reference_names.index(ch) for ch in picks]
    return filtered[pick_idx, :], picks


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


def predict_event_online_style(
    filtered_picks_uv: np.ndarray,
    event_sample: int,
    trigger: int,
    pkg: dict,
    endpoint: float,
) -> dict | None:
    fs = float(config.FS)
    picks = list(pkg["picks"])
    t_points = np.asarray(pkg["t_points"], dtype=float)
    endpoint_step = int(np.argmin(np.abs(t_points - endpoint)))
    mi_id = int(pkg["MI_ID"])
    rest_id = int(pkg["REST_ID"])
    cov_reg = float(pkg.get("cov_reg", 1e-4))
    recenter_refs = pkg.get("mdm_recenter_refs", [])
    target_id = mi_id if int(trigger) == 200 else rest_id

    baseline_duration = float(
        getattr(config, "ONLINE_BASELINE_DURATION", getattr(config, "BASELINE_DURATION", 1.0))
    )
    baseline_end_offset = float(getattr(config, "ONLINE_BASELINE_END_OFFSET", 0.0))
    window_samples = int(round(float(config.CLASSIFY_WINDOW) / 1000.0 * fs))

    # Online trial geometry:
    # prep starts at event + t_start. At endpoint, the live 2.5 s window ends
    # at event + endpoint. Baseline approximates the online pre-prep baseline:
    # [event + t_start - baseline_end_offset - duration,
    #  event + t_start - baseline_end_offset].
    t_start = float(pkg["t_start"])
    prep_start = event_sample + int(round(t_start * fs))
    endpoint_sample = event_sample + int(round(endpoint * fs))
    baseline_end = prep_start - int(round(baseline_end_offset * fs))
    baseline_start = baseline_end - int(round(baseline_duration * fs))
    window_start = endpoint_sample - window_samples + 1
    window_end = endpoint_sample + 1

    if baseline_start < 0 or window_start < 0 or window_end > filtered_picks_uv.shape[1]:
        return None

    baseline = filtered_picks_uv[:, baseline_start:baseline_end].mean(axis=1, keepdims=True)
    epoch_ch = filtered_picks_uv[:, window_start:window_end] - baseline

    n_steps = int(pkg["n_timepoints"])
    all_t_idx = np.linspace(0, epoch_ch.shape[1] - 1, n_steps).astype(int)
    t_end = all_t_idx[endpoint_step] + 1
    raw_step = epoch_ch[:, -t_end:]

    row = {
        "trigger": int(trigger),
        "target": int(target_id),
        "target_name": "MI" if int(trigger) == 200 else "REST",
        "endpoint_step": int(endpoint_step + 1),
        "endpoint_time": float(t_points[endpoint_step]),
    }

    try:
        mdm_model = pkg["mdm_models"][endpoint_step]
        template = pkg["mdm_templates"][endpoint_step]
        ref = (
            recenter_refs[endpoint_step]
            if len(recenter_refs) > endpoint_step
            and pkg.get("mdm_recenter_mode") == "train_riemann_mean"
            else None
        )
        cov = build_mdm_covariance(raw_step, template, cov_reg, ref)
        proba = mdm_model.predict_proba(np.expand_dims(cov, axis=0))[0]
        mi_col = list(mdm_model.classes_).index(mi_id)
        row["MDM"] = float(proba[mi_col])
    except Exception as exc:
        row["MDM"] = np.nan
        row["MDM_error"] = str(exc)

    lda_t_idx = np.linspace(0, raw_step.shape[1] - 1, endpoint_step + 1).astype(int)
    features = raw_step[:, lda_t_idx].flatten().reshape(1, -1)

    try:
        model = pkg["skl_models"][endpoint_step]
        proba = model.predict_proba(features)[0]
        mi_col = list(model.classes_).index(mi_id)
        row["LDA_shrink"] = float(proba[mi_col])
    except Exception as exc:
        row["LDA_shrink"] = np.nan
        row["LDA_shrink_error"] = str(exc)

    compact_picks = pkg.get("compact_lda_picks", [])
    compact_indices = [picks.index(ch) for ch in compact_picks if ch in picks]
    if compact_indices and len(pkg.get("compact_lda_models", [])) > endpoint_step:
        try:
            compact_features = raw_step[compact_indices, :][:, lda_t_idx].flatten().reshape(1, -1)
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

    return row


def replay_xdf_file(xdf_path: Path, pkg: dict, args: argparse.Namespace, trial_offset: int) -> list[dict]:
    print(f"   └─ Loading XDF: {xdf_path}")
    eeg_s, marker_s = load_xdf(str(xdf_path))
    eeg_data_uv = np.asarray(eeg_s["time_series"], dtype=float).T
    eeg_timestamps = np.asarray(eeg_s["time_stamps"], dtype=float)
    channel_names = get_channel_names_from_xdf(eeg_s)
    events = marker_events(marker_s, eeg_timestamps)

    if not events:
        print("      ⚠️  No target markers 100/200 found; skipping file.")
        return []

    filtered_picks_uv, _picks = process_like_online(
        eeg_data_uv,
        channel_names,
        pkg,
        car_order=args.car_order,
        car_reference=args.car_reference,
    )

    rows = []
    skipped = 0
    for local_trial, event in enumerate(events, start=1):
        row = predict_event_online_style(
            filtered_picks_uv,
            event_sample=int(event["sample"]),
            trigger=int(event["target"]),
            pkg=pkg,
            endpoint=float(args.endpoint),
        )
        if row is None:
            skipped += 1
            continue
        row["trial"] = trial_offset + local_trial
        row["xdf"] = str(xdf_path)
        rows.append(row)

    if skipped:
        print(f"      ⚠️  Skipped {skipped} trial(s) without enough baseline/window.")
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
        auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
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


def print_summary(rows: list[dict], summaries: list[dict], args: argparse.Namespace) -> None:
    print("\n" + "=" * 82)
    print("🔁  REPLAY OFFLINE XDF — PIPELINE ESTILO ONLINE")
    print("=" * 82)
    print(f"Pipeline: {args.car_order} | CAR reference: {args.car_reference}")
    print(f"Endpoint replay: {float(args.endpoint):+.2f} s")
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

    if any("MDM" in row and np.isfinite(row.get("MDM", np.nan)) for row in rows):
        mdm_scores = np.asarray([row.get("MDM", np.nan) for row in rows], dtype=float)
        y = np.asarray([1 if row["target_name"] == "MI" else 0 for row in rows])
        valid = np.isfinite(mdm_scores)
        pred = (mdm_scores[valid] >= 0.5).astype(int)
        cm = confusion_matrix(y[valid], pred, labels=[1, 0])
        print("\nMatriz MDM endpoint @0.50 (filas reales MI/REST, columnas pred MI/REST):")
        print(f"  MI   -> MI={cm[0,0]} | REST={cm[0,1]}")
        print(f"  REST -> MI={cm[1,0]} | REST={cm[1,1]}")


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay offline XDF with online-style streaming preprocessing."
    )
    parser.add_argument(
        "--model",
        default=getattr(config, "ONLINE_MODEL_PATH", ""),
        help="Path to trained M2 .pkl model.",
    )
    parser.add_argument(
        "--data-dir",
        default=getattr(config, "DATA_DIR", ""),
        help="Base study directory containing sub-*/ses-*/eeg.",
    )
    parser.add_argument(
        "--subject",
        default=getattr(config, "TRAINING_SUBJECT", ""),
        help="Subject ID, e.g. CNV_PILOT_SUBJ_014.",
    )
    parser.add_argument(
        "--session",
        default="S001_OFFLINE",
        help="Offline session name to replay when --xdf is not provided.",
    )
    parser.add_argument(
        "--xdf",
        action="append",
        help="Specific offline XDF path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--endpoint",
        type=float,
        default=float(getattr(config, "PREP_CONTROL_ENDPOINT", -0.75)),
        help="Control endpoint to evaluate, usually -0.75.",
    )
    parser.add_argument(
        "--car-order",
        choices=["legacy_filter_then_car", "current_car_then_filter"],
        default="legacy_filter_then_car",
        help=(
            "legacy_filter_then_car reproduces the online order before yesterday's "
            "CAR adjustment; current_car_then_filter uses the new order."
        ),
    )
    parser.add_argument(
        "--car-reference",
        choices=["all_valid_eeg", "selected"],
        default=getattr(config, "ONLINE_CAR_REFERENCE", "all_valid_eeg"),
        help="Channels used to compute CAR before selecting model picks.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write trial/summary CSV files. By default, only prints results.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/replay_offline_online_pipeline",
        help="Output prefix used only when --save-csv is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    with model_path.open("rb") as file:
        pkg = pickle.load(file)
    if pkg.get("model_type") != "M2_LDA_shrink_MDM":
        raise RuntimeError(f"Unsupported model_type: {pkg.get('model_type')}")

    print(f"✅ Modelo: {model_path}")
    print(f"   Canales: {pkg['picks']}")
    print(f"   Modo online-style: {args.car_order}")

    xdfs = discover_xdfs(args)
    rows: list[dict] = []
    for xdf in xdfs:
        rows.extend(replay_xdf_file(xdf, pkg, args, trial_offset=len(rows)))

    if not rows:
        raise RuntimeError("No usable target trials were replayed.")

    summaries = summarize(rows, [name for name in MODEL_ORDER if any(name in row for row in rows)])
    print_summary(rows, summaries, args)

    if args.save_csv:
        prefix = Path(args.output_prefix)
        trial_fields = [
            "trial", "xdf", "trigger", "target", "target_name", "endpoint_step",
            "endpoint_time", *MODEL_ORDER,
        ]
        write_csv(rows, prefix.with_name(prefix.name + "_trials.csv"), trial_fields)
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


if __name__ == "__main__":
    main()
