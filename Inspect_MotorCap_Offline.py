#!/usr/bin/env python3
"""
Offline inspection for the high-density motor cap.

Goal: decide whether the central FC/C/CP electrode layout shows cleaner
motor-related structure before changing the full decoder pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bci_runtime_env  # noqa: F401
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


TARGET_MARKERS = (100, 200)
EVENT_LABELS = {"100": "Rest", "200": "MI"}

RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}

NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}

MOTOR_GRID = [
    ["FC3", "FC1", "FCz"],
    ["C3", "C1", "Cz"],
    ["CP3", "CP1", "CPz"],
]

EXTRA_MOTOR_CHANNELS = ["FC5", "C5", "CP5", "Fz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect offline EEG from the motor-focused 32-channel cap."
    )
    parser.add_argument("--subject", default=config.RECORDING_SUBJECT)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--base-dir",
        default=getattr(config, "RECORDING_DATA_DIR", config.DATA_DIR),
        help="Study directory containing sub-*/ses-*/eeg folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="offline_qc_reports",
        help="Relative or absolute directory where figures are saved.",
    )
    parser.add_argument("--no-csd", action="store_true", help="Skip CSD/Laplacian.")
    return parser.parse_args()


def marker_values_and_times(marker_stream: dict) -> tuple[np.ndarray, np.ndarray]:
    marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float)
    rows = np.asarray(marker_stream["time_series"], dtype=object)

    values = np.asarray([int(float(row[0])) for row in rows])

    embedded_times = []
    if rows.ndim == 2 and rows.shape[1] > 1:
        for row in rows:
            try:
                embedded_times.append(float(row[1]))
            except (TypeError, ValueError):
                embedded_times.append(np.nan)

    if embedded_times:
        embedded_times = np.asarray(embedded_times, dtype=float)
        if np.isfinite(embedded_times).all() and np.ptp(embedded_times) > 0:
            return values, embedded_times

    return values, marker_ts


def load_session(xdf_files: list[Path]) -> mne.io.BaseRaw:
    raw_list = []

    for xdf_file in xdf_files:
        print(f"   loading {xdf_file.name}")
        eeg_s, marker_s = load_xdf(str(xdf_file))

        eeg_data = np.asarray(eeg_s["time_series"], dtype=float).T / 1e6
        eeg_timestamps = np.asarray(eeg_s["time_stamps"], dtype=float)
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_data, marker_timestamps = marker_values_and_times(marker_s)
        keep = np.isin(marker_data, TARGET_MARKERS)
        marker_data = marker_data[keep]
        marker_timestamps = marker_timestamps[keep]

        valid_ch = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data = eeg_data[valid_idx]

        ch_types = ["emg" if ch == "AUX7" else "eeg" for ch in valid_ch]
        info = mne.create_info(valid_ch, sfreq=config.FS, ch_types=ch_types)
        raw_tmp = mne.io.RawArray(eeg_data, info, verbose=False)

        renames = {k: v for k, v in RENAME_DICT.items() if k in raw_tmp.ch_names}
        if renames:
            raw_tmp.rename_channels(renames)

        raw_tmp.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            match_case=True,
            on_missing="warn",
        )

        t0 = eeg_timestamps[0]
        raw_tmp.set_annotations(
            mne.Annotations(
                onset=marker_timestamps - t0,
                duration=np.zeros(len(marker_data)),
                description=[str(m) for m in marker_data],
                orig_time=None,
            )
        )
        raw_list.append(raw_tmp)

    return mne.concatenate_raws(raw_list)


def present_motor_channels(raw: mne.io.BaseRaw) -> list[str]:
    desired = [ch for row in MOTOR_GRID for ch in row] + EXTRA_MOTOR_CHANNELS
    present = [ch for ch in desired if ch in raw.ch_names]
    missing = [ch for ch in desired if ch not in raw.ch_names]
    print(f"\nMotor channels present ({len(present)}): {present}")
    if missing:
        print(f"Motor channels missing: {missing}")
    return present


def make_epochs(raw: mne.io.BaseRaw, use_csd: bool) -> tuple[mne.Epochs, dict[str, int]]:
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_id = {"Rest": event_id_map["100"], "MI": event_id_map["200"]}
    print(
        f"Events: Rest={np.sum(events[:, 2] == event_id['Rest'])}, "
        f"MI={np.sum(events[:, 2] == event_id['MI'])}"
    )

    raw_eeg = raw.copy().pick_types(eeg=True, emg=False)
    raw_eeg.set_eeg_reference("average", projection=False, verbose=False)
    raw_eeg.notch_filter(freqs=[60.0], method="iir", verbose=False)

    raw_cnv = raw_eeg.copy().filter(
        l_freq=0.1, h_freq=3.0, method="iir", phase="forward", verbose=False
    )

    if use_csd:
        raw_cnv = mne.preprocessing.compute_current_source_density(raw_cnv)
        print("Spatial filter: CSD/Laplacian")
    else:
        print("Spatial filter: average reference only")

    epochs = mne.Epochs(
        raw_cnv,
        events,
        event_id=event_id,
        tmin=-3.0,
        tmax=5.0,
        baseline=(-3.0, -2.0),
        preload=True,
        reject=None,
        verbose=False,
    )
    return epochs, event_id


def plot_cnv(epochs: mne.Epochs, channels: list[str], out_file: Path) -> None:
    grid_channels = [ch for row in MOTOR_GRID for ch in row]
    plot_channels = [ch for ch in grid_channels if ch in channels]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    times = epochs.times
    data_rest = epochs["Rest"].get_data()
    data_mi = epochs["MI"].get_data()
    ch_names = epochs.ch_names

    all_mean = []
    for ch in plot_channels:
        idx = ch_names.index(ch)
        all_mean.extend([data_rest[:, idx].mean(axis=0), data_mi[:, idx].mean(axis=0)])
    ymax = max(3.0, np.percentile(np.abs(np.asarray(all_mean)), 99) * 1e6 * 1.25)

    for r, row in enumerate(MOTOR_GRID):
        for c, ch in enumerate(row):
            ax = axes[r, c]
            if ch in ch_names:
                idx = ch_names.index(ch)
                rest = data_rest[:, idx] * 1e6
                mi = data_mi[:, idx] * 1e6
                rest_mean = rest.mean(axis=0)
                mi_mean = mi.mean(axis=0)
                rest_sem = rest.std(axis=0) / np.sqrt(rest.shape[0])
                mi_sem = mi.std(axis=0) / np.sqrt(mi.shape[0])

                ax.plot(times, rest_mean, color="#2166ac", lw=2, label="Rest")
                ax.fill_between(times, rest_mean - rest_sem, rest_mean + rest_sem,
                                color="#2166ac", alpha=0.15)
                ax.plot(times, mi_mean, color="#b2182b", lw=2, label="MI")
                ax.fill_between(times, mi_mean - mi_sem, mi_mean + mi_sem,
                                color="#b2182b", alpha=0.15)
                ax.axvspan(-2.5, 0.0, color="#f4d35e", alpha=0.14)
                ax.set_ylim(-ymax, ymax)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="0.45")

            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_title(ch, fontweight="bold")
            ax.grid(True, ls=":", alpha=0.35)

    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.supxlabel("Time from cue/trigger (s)")
    fig.supylabel("Amplitude (uV, CSD-scaled if enabled)")
    fig.suptitle("Offline motor-cap CNV/MRCP inspection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_band_erd(raw: mne.io.BaseRaw, event_id: dict[str, int],
                  channels: list[str], out_file: Path) -> None:
    events, _ = mne.events_from_annotations(raw, verbose=False)
    raw_mu = raw.copy().pick_types(eeg=True, emg=False)
    raw_mu.set_eeg_reference("average", projection=False, verbose=False)
    raw_mu.notch_filter(freqs=[60.0], method="iir", verbose=False)

    bands = {"mu 8-13 Hz": (8.0, 13.0), "beta 13-30 Hz": (13.0, 30.0)}
    plot_channels = [ch for row in MOTOR_GRID for ch in row if ch in channels]

    fig, axes = plt.subplots(len(bands), 1, figsize=(13, 8), sharex=True)
    if len(bands) == 1:
        axes = [axes]

    for ax, (band_name, (lo, hi)) in zip(axes, bands.items()):
        filtered = raw_mu.copy().filter(lo, hi, fir_design="firwin", verbose=False)
        envelope = filtered.apply_hilbert(envelope=True, verbose=False)
        epochs = mne.Epochs(
            envelope,
            events,
            event_id=event_id,
            tmin=-3.0,
            tmax=5.0,
            baseline=None,
            preload=True,
            reject=None,
            verbose=False,
        )

        times = epochs.times
        baseline = (times >= -3.0) & (times <= -2.0)

        rest = epochs["Rest"].get_data(picks=plot_channels) ** 2
        mi = epochs["MI"].get_data(picks=plot_channels) ** 2
        rest_base = rest[:, :, baseline].mean(axis=2, keepdims=True)
        mi_base = mi[:, :, baseline].mean(axis=2, keepdims=True)
        rest_db = 10 * np.log10(rest / rest_base)
        mi_db = 10 * np.log10(mi / mi_base)

        rest_mean = rest_db.mean(axis=(0, 1))
        mi_mean = mi_db.mean(axis=(0, 1))
        ax.plot(times, rest_mean, color="#2166ac", lw=2, label="Rest")
        ax.plot(times, mi_mean, color="#b2182b", lw=2, label="MI")
        ax.axhline(0, color="0.35", lw=1)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.axvspan(0.0, 3.0, color="#8ecae6", alpha=0.12)
        ax.set_ylabel("Power change (dB)")
        ax.set_title(f"{band_name} average across {plot_channels}", fontweight="bold")
        ax.grid(True, ls=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)

    fig.supxlabel("Time from cue/trigger (s)")
    fig.suptitle("Offline motor-cap ERD/ERS inspection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def print_cnv_stats(epochs: mne.Epochs, channels: list[str]) -> None:
    times = epochs.times
    mask = (times >= -2.5) & (times <= 0.0)
    print("\nCNV/MRCP mean amplitude in [-2.5, 0.0] s")
    print(f"{'channel':<8} {'Rest':>10} {'MI':>10} {'MI-Rest':>10} {'p':>10}")
    print("-" * 54)
    for ch in [ch for row in MOTOR_GRID for ch in row if ch in channels]:
        rest = epochs["Rest"].get_data(picks=[ch])[:, 0, :][:, mask]
        mi = epochs["MI"].get_data(picks=[ch])[:, 0, :][:, mask]
        rest_amp = rest.mean(axis=1) * 1e6
        mi_amp = mi.mean(axis=1) * 1e6
        _, p_val = stats.ttest_ind(rest_amp, mi_amp, equal_var=False)
        print(
            f"{ch:<8} {rest_amp.mean():>10.3f} {mi_amp.mean():>10.3f} "
            f"{(mi_amp.mean() - rest_amp.mean()):>10.3f} {p_val:>10.4f}"
        )


def main() -> None:
    args = parse_args()
    xdf_dir = Path(args.base_dir) / f"sub-{args.subject}" / f"ses-{args.session}" / "eeg"
    if not xdf_dir.is_dir():
        raise FileNotFoundError(f"EEG directory does not exist: {xdf_dir}")

    xdf_files = sorted(xdf_dir.glob("*.xdf"))
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")

    out_dir = Path(args.output_dir) / f"sub-{args.subject}_ses-{args.session}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting {len(xdf_files)} XDF file(s)")
    print(f"Subject={args.subject} | Session={args.session}")
    raw = load_session(xdf_files)
    channels = present_motor_channels(raw)

    if len([ch for row in MOTOR_GRID for ch in row if ch in channels]) < 4:
        raise RuntimeError("Too few FC/C/CP motor-grid channels were found.")

    epochs, event_id = make_epochs(raw, use_csd=not args.no_csd)
    print(f"Usable epochs: Rest={len(epochs['Rest'])}, MI={len(epochs['MI'])}")

    print_cnv_stats(epochs, channels)

    cnv_path = out_dir / "motor_cap_cnv_grid.png"
    erd_path = out_dir / "motor_cap_mu_beta_erd.png"
    plot_cnv(epochs, channels, cnv_path)
    plot_band_erd(raw, event_id, channels, erd_path)

    print("\nSaved:")
    print(f"  {cnv_path}")
    print(f"  {erd_path}")
    print("\nDecision guide:")
    print("  promising = repeated MI-Rest separation around C3/C1/Cz or FC/CP neighbors")
    print("              plus visible mu/beta suppression during MI.")
    print("  weak      = no localization, unstable baselines, or separation only in one noisy channel.")


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()
