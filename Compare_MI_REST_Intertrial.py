"""
Quick exploratory comparison: MI vs REST vs INTERTRIAL.

This script is intentionally diagnostic only. It does not save online models and
does not modify the decoder. It reuses the same broad offline preprocessing used
for CNV inspection:

    CAR -> notch 60 Hz -> 0.1-1 Hz -> epoch baseline

Default comparison window is -2.5 to 0.0 s for visualization. Use
`--tmax -0.75` to reproduce the current online prep endpoint.
For INTERTRIAL, trigger 600 is treated as the beginning of the neutral interval;
the 3 s intertrial is mapped to the visual axis (-2.5, 0.5), like the gray line
in Generate_Decoder_MotorCap_TimePoints.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bci_runtime_env  # noqa: F401
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


SUBJECT = "CNV_PILOT_SUBJ_016"
SESSION = "S001_OFFLINE"
BASE_DIR = getattr(config, "DATA_DIR", "/home/lab-admin/Documents/CNVStudy")

PICKS_CNV = [
    "FC3", "FC1", "FCz",
    "C3", "C1", "Cz",
    "CP3", "CP1", "CPz",
]

CHANNELS_TO_DROP = ["M1", "M2", "T7", "T8", "Fp1", "Fpz", "Fp2"]
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}

TARGET_MARKERS = [100, 200]
INTERTRIAL_MARKERS = [600, 620]
INTERTRIAL_PLOT_WINDOW = (-2.5, 0.5)
REJECT_THRESHOLD = dict(eeg=250e-6)
FLAT_THRESHOLD = dict(eeg=0.1e-6)
TIMEPOINT_STEP = 0.25
RIEMANN_COV_REG = 1e-4
RIEMANN_MAX_FS = 32.0


def _load_raw_and_events(subject: str, session: str, base_dir: str):
    xdf_dir = Path(base_dir) / f"sub-{subject}" / f"ses-{session}" / "eeg"
    if not xdf_dir.is_dir():
        raise FileNotFoundError(f"XDF directory does not exist: {xdf_dir}")

    xdf_files = sorted(p for p in xdf_dir.iterdir() if p.suffix == ".xdf")
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")

    print(f"📂 Processing {len(xdf_files)} XDF file(s)")
    print(f"   Subject: {subject} | Session: {session}")

    raw_list = []
    event_run_labels_all = []

    for run_idx, xdf_file in enumerate(xdf_files, start=1):
        print(f"   └─ Loading: {xdf_file.name}")
        eeg_s, marker_s = load_xdf(str(xdf_file))

        eeg_data = np.asarray(eeg_s["time_series"]).T
        eeg_timestamps = np.asarray(eeg_s["time_stamps"])
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_data_all = np.asarray([
            int(round(float(np.ravel(v)[0])))
            for v in marker_s["time_series"]
        ])
        marker_timestamps_all = np.asarray(marker_s["time_stamps"])

        keep = np.isin(marker_data_all, TARGET_MARKERS + INTERTRIAL_MARKERS)
        marker_data = marker_data_all[keep]
        marker_timestamps = marker_timestamps_all[keep]
        event_run_labels_all.extend([run_idx] * len(marker_data))

        valid_ch = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

        existing_renames = {k: v for k, v in RENAME_DICT.items() if k in raw_tmp.ch_names}
        if existing_renames:
            raw_tmp.rename_channels(existing_renames)

        raw_tmp.set_montage(mne.channels.make_standard_montage("standard_1020"))

        drop_targets = [ch for ch in CHANNELS_TO_DROP if ch in raw_tmp.ch_names]
        if drop_targets:
            raw_tmp.drop_channels(drop_targets)

        t0 = eeg_timestamps[0]
        raw_tmp.set_annotations(mne.Annotations(
            onset=marker_timestamps - t0,
            duration=np.zeros(len(marker_data)),
            description=[str(m) for m in marker_data],
            orig_time=None,
        ))
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    events_all, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_run_labels_all = np.asarray(event_run_labels_all, dtype=int)
    if len(event_run_labels_all) != len(events_all):
        raise RuntimeError(
            "Could not align events with run labels: "
            f"{len(event_run_labels_all)} labels for {len(events_all)} events."
        )
    return raw, events_all, event_id_map, event_run_labels_all


def _preprocess(raw, use_car=False):
    raw = raw.copy()
    if use_car:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw.filter(l_freq=0.1, h_freq=1.0, method="iir", phase="forward",
               picks="eeg", verbose=False)
    return raw


def _drop_bad_epochs(epochs):
    pick_idx = [epochs.ch_names.index(ch) for ch in PICKS_CNV if ch in epochs.ch_names]
    data = epochs.get_data()[:, pick_idx, :]
    pp = data.max(axis=2) - data.min(axis=2)
    drop_mask = (
        (pp.max(axis=1) > REJECT_THRESHOLD["eeg"])
        | (pp.max(axis=1) < FLAT_THRESHOLD["eeg"])
    )
    epochs = epochs.copy()
    epochs.drop(np.flatnonzero(drop_mask).tolist(), reason="LOCAL_REJECT")
    return epochs


def _make_epochs(raw, events_all, event_id_map, event_run_labels_all):
    needed = {"100", "200"}
    missing = needed - set(event_id_map)
    if missing:
        raise RuntimeError(f"Missing required target markers: {sorted(missing)}")

    target_ids = {"REST": event_id_map["100"], "MI": event_id_map["200"]}
    target_mask = np.isin(events_all[:, 2], list(target_ids.values()))
    target_events = events_all[target_mask]
    target_groups = event_run_labels_all[target_mask]

    epochs_targets = mne.Epochs(
        raw, target_events, event_id=target_ids,
        tmin=-5.0, tmax=0.5, baseline=(-5.0, -3.0),
        reject=None, flat=None, preload=True, detrend=None, verbose=False,
    )
    epochs_targets = _drop_bad_epochs(epochs_targets)
    target_groups = target_groups[[i for i, d in enumerate(epochs_targets.drop_log) if len(d) == 0]]

    if "600" not in event_id_map:
        raise RuntimeError("No marker 600 found. Need INTERTRIAL_BEGIN triggers.")

    sfreq = raw.info["sfreq"]
    intertrial_begin_events = events_all[events_all[:, 2] == event_id_map["600"]]
    intertrial_groups = event_run_labels_all[events_all[:, 2] == event_id_map["600"]]

    neutral_event_samples = (
        intertrial_begin_events[:, 0]
        + int(round(-INTERTRIAL_PLOT_WINDOW[0] * sfreq))
    )
    valid = (
        (neutral_event_samples + int(round(INTERTRIAL_PLOT_WINDOW[0] * sfreq)) >= raw.first_samp)
        & (neutral_event_samples + int(round(INTERTRIAL_PLOT_WINDOW[1] * sfreq)) < raw.first_samp + raw.n_times)
    )
    neutral_events = np.column_stack([
        neutral_event_samples[valid].astype(int),
        np.zeros(np.sum(valid), dtype=int),
        np.full(np.sum(valid), 999, dtype=int),
    ])
    intertrial_groups = intertrial_groups[valid]

    epochs_intertrial = mne.Epochs(
        raw, neutral_events, event_id={"INTERTRIAL": 999},
        tmin=INTERTRIAL_PLOT_WINDOW[0], tmax=INTERTRIAL_PLOT_WINDOW[1],
        baseline=(INTERTRIAL_PLOT_WINDOW[0], -0.5),
        reject=None, flat=None, preload=True, detrend=None, verbose=False,
    )
    epochs_intertrial = _drop_bad_epochs(epochs_intertrial)
    intertrial_groups = intertrial_groups[[i for i, d in enumerate(epochs_intertrial.drop_log) if len(d) == 0]]
    return epochs_targets, target_groups, epochs_intertrial, intertrial_groups


def _features(epochs, crop_window, picks):
    ep = epochs.copy().pick(picks).crop(tmin=crop_window[0], tmax=crop_window[1])
    data = ep.get_data() * 1e6
    # Compact but transparent features: mean amplitude in 250 ms bins per channel.
    sfreq = ep.info["sfreq"]
    bin_size = max(1, int(round(0.250 * sfreq)))
    n_bins = data.shape[2] // bin_size
    data = data[:, :, : n_bins * bin_size]
    data = data.reshape(data.shape[0], data.shape[1], n_bins, bin_size).mean(axis=3)
    return data.reshape(data.shape[0], -1)


def _point_features(epochs, picks, t_points, step=None):
    pts = t_points[:step] if step is not None else t_points
    ep = epochs.copy().pick(picks)
    times = ep.times
    t_idx = [int(np.argmin(np.abs(times - t))) for t in pts]
    data = ep.get_data()[:, :, t_idx] * 1e6
    return data.reshape(data.shape[0], -1)


def _epoch_trials(epochs, picks, tmin, tmax):
    ep = epochs.copy().pick(picks).crop(tmin=tmin, tmax=tmax)
    data = ep.get_data()
    stride = max(1, int(round(ep.info["sfreq"] / RIEMANN_MAX_FS)))
    return data[:, :, ::stride]


def _template_covariances(trials, template):
    repeated_template = np.repeat(template[np.newaxis, :, :], trials.shape[0], axis=0)
    extended = np.concatenate([trials, repeated_template], axis=1)
    covariances = np.empty((len(trials), extended.shape[1], extended.shape[1]), dtype=float)
    eye = np.eye(extended.shape[1])
    for idx, trial in enumerate(extended):
        cov = trial @ trial.T
        trace = np.trace(cov)
        if trace > 0:
            cov /= trace
        cov += RIEMANN_COV_REG * eye
        covariances[idx] = 0.5 * (cov + cov.T)
    return covariances


def _evaluate_mdm_logo(trials0, trials1, g0, g1):
    try:
        from pyriemann.classification import MDM
    except ImportError:
        return np.nan, np.nan, np.nan, np.nan

    n = min(len(trials0), len(trials1))
    X_trials = np.concatenate([trials0[:n], trials1[:n]], axis=0)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    groups = np.r_[g0[:n], g1[:n]]

    if len(np.unique(groups)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    scores = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1, dtype=int)
    fold_auc = []
    fold_acc = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X_trials, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        train_trials = X_trials[train_idx]
        y_train = y[train_idx]
        template = train_trials[y_train == 1].mean(axis=0)
        cov_train = _template_covariances(train_trials, template)
        cov_test = _template_covariances(X_trials[test_idx], template)
        mdm = MDM(metric="riemann")
        mdm.fit(cov_train, y_train)
        pos_idx = int(np.where(mdm.classes_ == 1)[0][0])
        fold_scores = mdm.predict_proba(cov_test)[:, pos_idx]
        fold_predictions = mdm.predict(cov_test)
        scores[test_idx] = fold_scores
        predictions[test_idx] = fold_predictions
        fold_auc.append(roc_auc_score(y[test_idx], fold_scores))
        fold_acc.append(accuracy_score(y[test_idx], fold_predictions) * 100.0)

    valid = np.isfinite(scores) & (predictions >= 0)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return np.nan, np.nan, np.nan, np.nan
    return (
        roc_auc_score(y[valid], scores[valid]),
        accuracy_score(y[valid], predictions[valid]) * 100.0,
        float(np.nanstd(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
        float(np.nanstd(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0,
    )


def _classical_models():
    return {
        "LDA_shrink": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced")),
        ]),
    }


def _make_splits(X, y, groups):
    if len(np.unique(groups)) >= 2:
        splits = list(LeaveOneGroupOut().split(X, y, groups))
        return splits, "Leave-One-Run-Out"
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    return splits, "StratifiedKFold"


def _evaluate_classical(X0, X1, g0, g1):
    n = min(len(X0), len(X1))
    X = np.vstack([X0[:n], X1[:n]])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    groups = np.r_[g0[:n], g1[:n]]
    splits, cv_name = _make_splits(X, y, groups)
    out = {}
    for model_name, model in _classical_models().items():
        y_true, y_score, y_pred = [], [], []
        fold_auc = []
        fold_acc = []
        for train_idx, test_idx in splits:
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            score = model.predict_proba(X[test_idx])[:, 1]
            y_true.extend(y[test_idx])
            y_pred.extend(pred)
            y_score.extend(score)
            if len(np.unique(y[test_idx])) >= 2:
                fold_auc.append(roc_auc_score(y[test_idx], score))
            fold_acc.append(accuracy_score(y[test_idx], pred) * 100.0)
        out[model_name] = {
            "auc": roc_auc_score(y_true, y_score),
            "acc": accuracy_score(y_true, y_pred) * 100.0,
            "auc_std": float(np.nanstd(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
            "acc_std": float(np.nanstd(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0,
        }
    return out, cv_name


def _evaluate_pair(name, X0, X1, g0, g1):
    n = min(len(X0), len(X1))
    results, cv_name = _evaluate_classical(X0, X1, g0, g1)

    print("\n" + "=" * 78)
    print(f"{name} | n_per_class={n} | CV={cv_name}")
    print("-" * 78)
    print(f"{'Model':<12} {'AUC':>7} {'±std':>7} {'Acc%':>8} {'±std':>7}")
    print("-" * 78)
    for model_name, metrics in results.items():
        auc = metrics["auc"]
        acc = metrics["acc"]
        print(
            f"{model_name:<12} {auc:7.3f} {metrics['auc_std']:7.3f} "
            f"{acc:8.1f} {metrics['acc_std']:7.1f}"
        )
    return results


def _evaluate_cumulative(pair_name, epochs0, epochs1, g0, g1, picks, t_points):
    print("\n" + "=" * 78)
    print(f"⏱️  M2 acumulativo — {pair_name}")
    print("=" * 78)
    seq = {
        name: {"auc": [], "acc": [], "auc_std": [], "acc_std": []}
        for name in ["LDA_shrink", "LR", "SVM", "MDM"]
    }

    for step, t_end in enumerate(t_points, start=1):
        X0 = _point_features(epochs0, picks, t_points, step=step)
        X1 = _point_features(epochs1, picks, t_points, step=step)
        classical, _ = _evaluate_classical(X0, X1, g0, g1)
        for name in ["LDA_shrink", "LR", "SVM"]:
            seq[name]["auc"].append(classical[name]["auc"])
            seq[name]["acc"].append(classical[name]["acc"])
            seq[name]["auc_std"].append(classical[name]["auc_std"])
            seq[name]["acc_std"].append(classical[name]["acc_std"])

        trials0 = _epoch_trials(epochs0, picks, t_points[0], t_end)
        trials1 = _epoch_trials(epochs1, picks, t_points[0], t_end)
        mdm_auc, mdm_acc, mdm_auc_std, mdm_acc_std = _evaluate_mdm_logo(
            trials0, trials1, g0, g1
        )
        seq["MDM"]["auc"].append(mdm_auc)
        seq["MDM"]["acc"].append(mdm_acc)
        seq["MDM"]["auc_std"].append(mdm_auc_std)
        seq["MDM"]["acc_std"].append(mdm_acc_std)
        print(
            f"   t={t_end:>5.2f}s | "
            f"LDA AUC={classical['LDA_shrink']['auc']:.3f} "
            f"LR AUC={classical['LR']['auc']:.3f} "
            f"SVM AUC={classical['SVM']['auc']:.3f} "
            f"MDM AUC={mdm_auc:.3f}"
        )
    return seq


def _maybe_save(fig, path, save_figures):
    if save_figures and path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        return path
    return None


def _plot_classification_summary(pair_key, pair_title, static_results, t_points, seq,
                                 out_dir, save_figures, reference_label):
    colors = {
        "LDA_shrink": "#756bb1",
        "LR": "#fdae6b",
        "SVM": "#e34a33",
        "MDM": "#542788",
    }
    models = ["LDA_shrink", "LR", "SVM", "MDM"]
    static_auc = [
        static_results[m]["auc"] if m in static_results else seq[m]["auc"][-1]
        for m in models
    ]
    static_auc_std = [
        static_results[m]["auc_std"] if m in static_results else seq[m]["auc_std"][-1]
        for m in models
    ]
    static_acc = [
        static_results[m]["acc"] if m in static_results else seq[m]["acc"][-1]
        for m in models
    ]
    static_acc_std = [
        static_results[m]["acc_std"] if m in static_results else seq[m]["acc_std"][-1]
        for m in models
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    fig.suptitle(
        f"CNV Intertrial Diagnostic — {pair_title} | ref={reference_label}",
        fontsize=15,
        fontweight="bold",
    )

    ax_auc_bar = axes[0, 0]
    x = np.arange(len(models))
    bars = ax_auc_bar.bar(
        x,
        static_auc,
        yerr=static_auc_std,
        capsize=5,
        ecolor="black",
        color=[colors[m] for m in models],
    )
    ax_auc_bar.axhline(0.5, color="red", ls="--", lw=1.1, label="Azar (0.5)")
    ax_auc_bar.axhline(0.7, color="gray", ls=":", lw=1.1, label="Objetivo (0.7)")
    ax_auc_bar.set_title("Comparación full-window\nAUC", fontweight="bold")
    ax_auc_bar.set_ylabel("AUC")
    ax_auc_bar.set_ylim(0.3, 1.02)
    ax_auc_bar.set_xticks(x)
    ax_auc_bar.set_xticklabels(models, rotation=25, ha="right")
    ax_auc_bar.grid(True, axis="y", ls=":", alpha=0.35)
    ax_auc_bar.legend(fontsize=8)
    for b, val in zip(bars, static_auc):
        ax_auc_bar.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax_acc_bar = axes[1, 0]
    bars = ax_acc_bar.bar(
        x,
        static_acc,
        yerr=static_acc_std,
        capsize=5,
        ecolor="black",
        color=[colors[m] for m in models],
    )
    ax_acc_bar.axhline(50.0, color="red", ls="--", lw=1.1, label="Azar (50%)")
    ax_acc_bar.axhline(70.0, color="gray", ls=":", lw=1.1, label="Objetivo (70%)")
    ax_acc_bar.set_title("Comparación full-window\nAccuracy", fontweight="bold")
    ax_acc_bar.set_ylabel("Accuracy (%)")
    ax_acc_bar.set_ylim(30, 100)
    ax_acc_bar.set_xticks(x)
    ax_acc_bar.set_xticklabels(models, rotation=25, ha="right")
    ax_acc_bar.grid(True, axis="y", ls=":", alpha=0.35)
    ax_acc_bar.legend(fontsize=8)
    for b, val in zip(bars, static_acc):
        ax_acc_bar.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8)

    ax_auc_seq = axes[0, 1]
    ax_acc_seq = axes[1, 1]
    for name, vals in seq.items():
        auc = np.asarray(vals["auc"], dtype=float)
        acc = np.asarray(vals["acc"], dtype=float)
        auc_std = np.asarray(vals["auc_std"], dtype=float)
        acc_std = np.asarray(vals["acc_std"], dtype=float)
        ax_auc_seq.plot(t_points, auc, "o-", lw=2.2,
                        color=colors[name], label=name)
        ax_auc_seq.fill_between(
            t_points,
            np.clip(auc - auc_std, 0.0, 1.0),
            np.clip(auc + auc_std, 0.0, 1.0),
            color=colors[name],
            alpha=0.12,
            linewidth=0,
        )
        ax_acc_seq.plot(t_points, acc, "o-", lw=2.2,
                        color=colors[name], label=name)
        ax_acc_seq.fill_between(
            t_points,
            np.clip(acc - acc_std, 0.0, 100.0),
            np.clip(acc + acc_std, 0.0, 100.0),
            color=colors[name],
            alpha=0.12,
            linewidth=0,
        )
    ax_auc_seq.axhline(0.5, color="red", ls="--", lw=1.1, label="Azar (0.5)")
    ax_auc_seq.axhline(0.7, color="gray", ls=":", lw=1.1, label="Objetivo (0.7)")
    ax_auc_seq.axvline(0.0, color="black", ls="--", lw=1.2, label="Trigger (0 s)")
    ax_auc_seq.set_title("Comparación acumulativa\nAUC vs instante temporal",
                         fontweight="bold")
    ax_auc_seq.set_ylabel("AUC")
    ax_auc_seq.set_ylim(0.3, 1.02)
    ax_auc_seq.grid(True, ls=":", alpha=0.35)
    ax_auc_seq.invert_xaxis()
    ax_auc_seq.legend(fontsize=8)

    ax_acc_seq.axhline(50.0, color="red", ls="--", lw=1.1, label="Azar (50%)")
    ax_acc_seq.axhline(70.0, color="gray", ls=":", lw=1.1, label="Objetivo (70%)")
    ax_acc_seq.axvline(0.0, color="black", ls="--", lw=1.2, label="Trigger (0 s)")
    ax_acc_seq.set_title("Comparación acumulativa\nAccuracy vs instante temporal",
                         fontweight="bold")
    ax_acc_seq.set_xlabel("Tiempo disponible (s)")
    ax_acc_seq.set_ylabel("Accuracy (%)")
    ax_acc_seq.set_ylim(30, 100)
    ax_acc_seq.grid(True, ls=":", alpha=0.35)
    ax_acc_seq.invert_xaxis()
    ax_acc_seq.legend(fontsize=8)

    fig.tight_layout()
    out_path = _maybe_save(
        fig,
        out_dir / f"intertrial_{pair_key}_classification_summary.png",
        save_figures,
    )
    return out_path


def _plot_topomap_grid(pair_key, pair_title, class0_name, class1_name,
                       epochs0, epochs1, picks, out_dir, save_figures,
                       reference_label):
    # Topoplots are intentionally visualized with all EEG channels that have a
    # valid montage position. The classifiers above still use `picks`
    # (PICKS_CNV, 9 channels) so the model comparison stays unchanged.
    eeg0 = epochs0.copy().pick_types(eeg=True)
    eeg1 = epochs1.copy().pick_types(eeg=True)
    eeg1_names = set(eeg1.ch_names)
    topomap_picks = []
    for ch in eeg0.ch_names:
        if ch not in eeg1_names:
            continue
        loc = eeg0.info["chs"][eeg0.ch_names.index(ch)]["loc"][:3]
        if np.allclose(loc, 0):
            continue
        topomap_picks.append(ch)
    if not topomap_picks:
        raise RuntimeError("No valid EEG channels with montage positions for topomap.")

    info = eeg0.copy().pick(topomap_picks).info
    windows = [
        (-2.5, -2.25),
        (-2.25, -2.0),
        (-2.0, -1.75),
        (-1.75, -1.5),
        (-1.5, -1.25),
        (-1.25, -1.0),
        (-1.0, -0.75),
        (-0.75, -0.5),
    ]
    data_grid = []
    for ep in (epochs0, epochs1):
        row = []
        for t0, t1 in windows:
            row.append(
                ep.copy()
                .pick(topomap_picks)
                .crop(t0, t1)
                .get_data()
                .mean(axis=2)
                .mean(axis=0)
                * 1e6
            )
        data_grid.append(row)
    vlim_abs = max(1.0, float(np.nanmax(np.abs(data_grid))))

    fig, axes = plt.subplots(2, len(windows), figsize=(18, 7.5))
    fig.suptitle(
        f"CNV Topographic Maps — {pair_title} | ref={reference_label}\n"
        f"Topoplot: {len(topomap_picks)} EEG channels | Models: {len(picks)} CNV channels",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    for row_idx, label in enumerate([class0_name, class1_name]):
        axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight="bold")
        for col_idx, (t0, t1) in enumerate(windows):
            ax = axes[row_idx, col_idx]
            im, _ = mne.viz.plot_topomap(
                data_grid[row_idx][col_idx],
                info,
                axes=ax,
                show=False,
                contours=4,
                cmap="RdBu_r",
                vlim=(-vlim_abs, vlim_abs),
            )
            ax.set_title(f"{t0:.2f} – {t1:.2f} s", fontsize=10)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.50])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"Amplitude (µV) [vlim ±{vlim_abs:.1f}]")
    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.08,
                        wspace=0.18, hspace=0.45)
    out_path = _maybe_save(
        fig,
        out_dir / f"intertrial_{pair_key}_topomap_grid.png",
        save_figures,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--tmin", type=float, default=-2.5)
    parser.add_argument("--tmax", type=float, default=0.0)
    parser.add_argument("--fig-dir", default="figuras/intertrial_diagnostic")
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save figures to --fig-dir. By default figures are only shown.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(). Useful for batch/headless runs.",
    )
    parser.add_argument(
        "--no-car",
        action="store_true",
        help="Skip average reference/CAR. Diagnostic only.",
    )
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    raw, events_all, event_id_map, event_run_labels_all = _load_raw_and_events(
        args.subject, args.session, args.base_dir
    )
    raw = _preprocess(raw, use_car=not args.no_car)
    epochs_targets, target_groups, epochs_intertrial, intertrial_groups = _make_epochs(
        raw, events_all, event_id_map, event_run_labels_all
    )

    picks = [ch for ch in PICKS_CNV if ch in epochs_targets.ch_names and ch in epochs_intertrial.ch_names]
    crop_window = (args.tmin, args.tmax)
    out_dir = Path(args.fig_dir)
    if args.save_figures:
        out_dir.mkdir(parents=True, exist_ok=True)
    t_points = np.arange(args.tmin, args.tmax + TIMEPOINT_STEP / 2.0, TIMEPOINT_STEP)
    print("\n🎯 Quick diagnostic comparison")
    print(f"   Channels: {picks}")
    reference_label = "sin CAR" if args.no_car else "CAR"
    print(f"   Reference: {reference_label}")
    print(f"   Window  : {crop_window[0]:.2f} to {crop_window[1]:.2f} s")
    print(f"   M2 steps: {np.round(t_points, 3).tolist()}")
    print(
        f"   Figures : {'show only' if not args.save_figures else out_dir}"
    )
    print(
        f"   Kept epochs: REST={len(epochs_targets['REST'])} | "
        f"MI={len(epochs_targets['MI'])} | INTERTRIAL={len(epochs_intertrial)}"
    )

    X_rest = _features(epochs_targets["REST"], crop_window, picks)
    X_mi = _features(epochs_targets["MI"], crop_window, picks)
    X_inter = _features(epochs_intertrial, crop_window, picks)

    g_rest = target_groups[epochs_targets.events[:, 2] == epochs_targets.event_id["REST"]]
    g_mi = target_groups[epochs_targets.events[:, 2] == epochs_targets.event_id["MI"]]
    g_inter = intertrial_groups

    static_mi_rest = _evaluate_pair(
        "MI vs REST        (label 1 = MI)", X_rest, X_mi, g_rest, g_mi
    )
    static_mi_inter = _evaluate_pair(
        "MI vs INTERTRIAL  (label 1 = MI)", X_inter, X_mi, g_inter, g_mi
    )
    static_rest_inter = _evaluate_pair(
        "REST vs INTERTRIAL(label 1 = REST)", X_inter, X_rest, g_inter, g_rest
    )

    pairs = [
        (
            "mi_vs_rest",
            "MI vs REST (1=MI)",
            "REST (100)", "MI (200)",
            epochs_targets["REST"], epochs_targets["MI"],
            g_rest, g_mi,
            static_mi_rest,
        ),
        (
            "mi_vs_intertrial",
            "MI vs INTERTRIAL (1=MI)",
            "INTERTRIAL (600)", "MI (200)",
            epochs_intertrial, epochs_targets["MI"],
            g_inter, g_mi,
            static_mi_inter,
        ),
        (
            "rest_vs_intertrial",
            "REST vs INTERTRIAL (1=REST)",
            "INTERTRIAL (600)", "REST (100)",
            epochs_intertrial, epochs_targets["REST"],
            g_inter, g_rest,
            static_rest_inter,
        ),
    ]

    saved = []
    for pair_key, pair_title, class0_name, class1_name, ep0, ep1, gg0, gg1, static_results in pairs:
        seq = _evaluate_cumulative(pair_title, ep0, ep1, gg0, gg1, picks, t_points)
        saved.append(
            _plot_classification_summary(
                pair_key, pair_title, static_results, t_points, seq,
                out_dir, args.save_figures, reference_label,
            )
        )
        saved.append(
            _plot_topomap_grid(
                pair_key, pair_title, class0_name, class1_name, ep0, ep1,
                picks, out_dir, args.save_figures, reference_label,
            )
        )

    saved = [p for p in saved if p is not None]
    if saved:
        print("\n💾 Figuras guardadas:")
        for path in saved:
            print(f"   {path}")
    else:
        print("\n🖼️  Figuras listas para visualización (no guardadas).")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
