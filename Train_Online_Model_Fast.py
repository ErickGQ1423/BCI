"""
================================================================================
FAST ONLINE MODEL TRAINER — CNV MotorCap M2
================================================================================

Entrena únicamente el paquete que consume ExperimentDriver_Online.py:

  - MDM Riemanniano acumulativo
  - LDA_shrink 9ch
  - LDA_shrink 3ch
  - LR
  - SVM calibrado

No genera figuras, no hace LOGO completo, no hace channel selection.
Uso operativo para calibración rápida antes de online.

Ejemplo:
  python Train_Online_Model_Fast.py

Opcional:
  python Train_Online_Model_Fast.py \
    --subject CNV_PILOT_SUBJ_020 \
    --session S003_OFFLINE_FES_WARMUP
================================================================================
"""

from __future__ import annotations

import argparse
import os
import pickle

import bci_runtime_env  # noqa: F401
import mne
import numpy as np
from pyriemann.classification import MDM
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


# ---------------------------------------------------------------------
# Defaults: mantener sincronizados con Generate_Decoder_MotorCap_TimePoints.py
# ---------------------------------------------------------------------
DEFAULT_SUBJECT = "CNV_PILOT_SUBJ_021"
DEFAULT_SESSION = "S001_OFFLINE_FES"

CHANNELS_TO_DROP = ["M1", "M2", "T7", "T8", "Fp1", "Fpz", "Fp2"]
CHANNELS_TO_INTERPOLATE: list[str] = []

PICKS_CNV = [
    "FC3", "FC1", "FCz",
    "C3",  "C1",  "Cz",
    "CP3", "CP1", "CPz",
]
COMPACT_LDA_PICKS = ["FCz", "C3", "CP3"]

EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=4, ftype="butter")

REJECT_THRESHOLD = dict(eeg=100e-6)
FLAT_THRESHOLD = dict(eeg=0.1e-6)

TARGET_MARKERS = [100, 200]
INTERTRIAL_MARKERS = [600, 620]
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}
# Auxiliary channels are not part of the EEG montage. Keep them out before
# applying standard_1020 so the fast trainer matches the online motor-cap model.
NON_EEG_CHANNELS = {
    "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", "AUX9",
    "TRIGGER",
}

T_START = -2.5
T_END = 0.0
TIMEPOINT_STEP = 0.25
T_POINTS = np.arange(T_START, T_END + TIMEPOINT_STEP / 2.0, TIMEPOINT_STEP)
N_TIMEPOINTS = len(T_POINTS)
N_CHANNELS = len(PICKS_CNV)
N_FEATURES = N_TIMEPOINTS * N_CHANNELS

RIEMANN_COV_REG = 1e-4
RIEMANN_MAX_FS = 32.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast trainer for the CNV online MotorCap model package."
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument(
        "--base-dir",
        default=getattr(config, "DATA_DIR", "/home/lab-admin/Documents/CNVStudy"),
    )
    parser.add_argument("--output", default=None, help="Optional explicit .pkl path.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Train only with the first N XDF files in the session. Useful for "
            "training on warmup/calibration runs before acquiring a holdout run."
        ),
    )
    parser.add_argument(
        "--include-runs",
        default=None,
        help=(
            "Comma-separated 1-based run indices to include, e.g. 1,2,3,4,5,6. "
            "Applied after sorting XDF files by name."
        ),
    )
    parser.add_argument(
        "--exclude-runs",
        default=None,
        help=(
            "Comma-separated 1-based run indices to exclude, e.g. 7. "
            "Applied after --max-runs/--include-runs."
        ),
    )
    return parser.parse_args()


def parse_run_list(text: str | None) -> set[int] | None:
    if text is None:
        return None
    runs: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        run = int(item)
        if run < 1:
            raise ValueError(f"Run indices are 1-based; invalid value: {run}")
        runs.add(run)
    return runs


def make_clf(name):
    if name == "LDA_shrink":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
    if name == "SVM":
        base = SVC(kernel="linear", C=1.0, probability=False, random_state=42)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
        ])
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )),
        ])
    raise ValueError(f"Unknown classifier: {name}")


def select_xdf_files(xdf_files, max_runs=None, include_runs=None, exclude_runs=None):
    include = parse_run_list(include_runs)
    exclude = parse_run_list(exclude_runs) or set()

    selected = []
    for run_idx, path in enumerate(xdf_files, start=1):
        if max_runs is not None and run_idx > max_runs:
            continue
        if include is not None and run_idx not in include:
            continue
        if run_idx in exclude:
            continue
        selected.append((run_idx, path))

    if not selected:
        raise RuntimeError(
            "No XDF files selected for training. Check --max-runs, "
            "--include-runs, and --exclude-runs."
        )
    return selected


def load_raw_and_events(
    subject,
    session,
    base_dir,
    max_runs=None,
    include_runs=None,
    exclude_runs=None,
):
    xdf_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}", "eeg")
    if not os.path.isdir(xdf_dir):
        raise FileNotFoundError(f"XDF directory does not exist: {xdf_dir}")

    xdf_files = sorted(
        os.path.join(xdf_dir, name)
        for name in os.listdir(xdf_dir)
        if name.endswith(".xdf")
    )
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")

    selected_xdfs = select_xdf_files(
        xdf_files,
        max_runs=max_runs,
        include_runs=include_runs,
        exclude_runs=exclude_runs,
    )

    print(f"📂 Fast training from {len(selected_xdfs)} / {len(xdf_files)} XDF file(s)")
    print(f"   Subject: {subject}")
    print(f"   Session: {session}")
    print(f"   Dir    : {xdf_dir}")
    print("   Runs   : " + ", ".join(str(run_idx) for run_idx, _ in selected_xdfs))

    raw_list = []
    event_run_labels = []

    for run_idx, xdf_file in selected_xdfs:
        print(f"   └─ Loading run {run_idx}: {os.path.basename(xdf_file)}")
        eeg_s, marker_s = load_xdf(xdf_file)

        eeg_data = np.array(eeg_s["time_series"]).T
        eeg_timestamps = np.array(eeg_s["time_stamps"])
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_data_all = np.array([
            int(round(float(np.ravel(value)[0])))
            for value in marker_s["time_series"]
        ])
        marker_timestamps_all = np.array(marker_s["time_stamps"])

        keep = np.isin(marker_data_all, TARGET_MARKERS + INTERTRIAL_MARKERS)
        marker_data = marker_data_all[keep]
        marker_timestamps = marker_timestamps_all[keep]
        event_run_labels.extend([run_idx] * len(marker_data))

        valid_ch = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        existing_renames = {
            old: new for old, new in RENAME_DICT.items()
            if old in raw_tmp.ch_names
        }
        if existing_renames:
            raw_tmp.rename_channels(existing_renames)

        raw_tmp.set_montage(mne.channels.make_standard_montage("standard_1020"))

        drop_targets = [ch for ch in CHANNELS_TO_DROP if ch in raw_tmp.ch_names]
        if drop_targets:
            raw_tmp.drop_channels(drop_targets)

        if CHANNELS_TO_INTERPOLATE:
            raw_tmp.info["bads"] = [
                ch for ch in CHANNELS_TO_INTERPOLATE if ch in raw_tmp.ch_names
            ]
            raw_tmp.interpolate_bads(reset_bads=True, verbose=False)

        t0 = eeg_timestamps[0]
        raw_tmp.set_annotations(mne.Annotations(
            onset=marker_timestamps - t0,
            duration=np.zeros(len(marker_data)),
            description=[str(marker) for marker in marker_data],
            orig_time=None,
        ))
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    events_all, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_run_labels = np.asarray(event_run_labels, dtype=int)
    if len(event_run_labels) != len(events_all):
        raise RuntimeError(
            "Event/run alignment failed: "
            f"{len(event_run_labels)} labels for {len(events_all)} events."
        )

    if "100" not in event_id_map or "200" not in event_id_map:
        raise RuntimeError(f"Missing target markers 100/200. Found: {event_id_map}")

    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)": event_id_map["200"],
    }
    target_mask = np.isin(events_all[:, 2], list(event_dict.values()))
    events = events_all[target_mask]
    event_run_labels = event_run_labels[target_mask]

    return raw, events, event_dict, event_id_map, event_run_labels, [path for _, path in selected_xdfs]


def preprocess_and_epoch(raw, events, event_dict):
    print("🎛️  Pipeline: CAR → notch60 → BPF 0.1–2.0 Hz Butterworth 4º")
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

    epochs_all = mne.Epochs(
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

    pick_idx = [
        epochs_all.ch_names.index(ch)
        for ch in PICKS_CNV
        if ch in epochs_all.ch_names
    ]
    if len(pick_idx) != len(PICKS_CNV):
        missing = [ch for ch in PICKS_CNV if ch not in epochs_all.ch_names]
        raise RuntimeError(f"Missing model channels after preprocessing: {missing}")

    data_cnv = epochs_all.get_data()[:, pick_idx, :]
    pp = data_cnv.max(axis=2) - data_cnv.min(axis=2)
    reject_val = REJECT_THRESHOLD["eeg"]
    flat_val = FLAT_THRESHOLD["eeg"]
    drop_mask = (pp.max(axis=1) > reject_val) | (pp.max(axis=1) < flat_val)
    drop_indices = np.flatnonzero(drop_mask).tolist()

    epochs = epochs_all.copy()
    epochs.drop(drop_indices, reason="MANUAL_REJECT")

    n_rest = len(epochs["Rest (100)"])
    n_mi = len(epochs["MI (200)"])
    n_total = len(drop_indices) + n_rest + n_mi
    print("🛡️  Rechazo localizado en canales CNV:")
    print(f"   Rechazados : {len(drop_indices)} / {n_total} ({100*len(drop_indices)/n_total:.1f}%)")
    print(f"   Rest       : {n_rest}")
    print(f"   MI         : {n_mi}")
    print(f"   P2P95      : {np.percentile(pp.max(axis=1) * 1e6, 95):.1f} µV")

    if n_rest < 3 or n_mi < 3:
        raise RuntimeError("Need at least 3 accepted trials per class for SVM calibration.")

    return epochs, drop_indices


def extract_features(epochs_obj, picks, t_points, step=None):
    times = epochs_obj.times
    pts = t_points[:step] if step is not None else t_points
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
    ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    data = epochs_obj.get_data(picks="eeg")
    x = np.hstack([data[:, ci, :][:, t_idx] for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return x, y


def template_covariances_riemann(trials, template):
    repeated_template = np.repeat(template[np.newaxis, :, :], trials.shape[0], axis=0)
    extended = np.concatenate([trials, repeated_template], axis=1)
    covariances = np.empty(
        (len(trials), extended.shape[1], extended.shape[1]),
        dtype=float,
    )
    for trial_idx, trial in enumerate(extended):
        covariance = trial @ trial.T
        trace = np.trace(covariance)
        if trace > 0:
            covariance /= trace
        covariance += RIEMANN_COV_REG * np.eye(covariance.shape[0])
        covariances[trial_idx] = 0.5 * (covariance + covariance.T)
    return covariances


def recenter_covariances(covariances, reference):
    transform = invsqrtm(reference)
    recentered = np.empty_like(covariances)
    for idx, covariance in enumerate(covariances):
        cov = transform @ covariance @ transform.T
        cov = 0.5 * (cov + cov.T)
        cov += RIEMANN_COV_REG * np.eye(cov.shape[0])
        recentered[idx] = cov
    return recentered


def train_online_m2_package(epochs, subject, session, event_dict, mi_id):
    labels = epochs.events[:, -1].copy()
    online_data_uv = epochs.get_data(picks=PICKS_CNV) * 1e6
    compact_data_uv = epochs.get_data(picks=COMPACT_LDA_PICKS) * 1e6

    riemann_start_idx = int(np.argmin(np.abs(epochs.times - T_START)))

    mdm_models = []
    mdm_templates = []
    mdm_centers = []
    mdm_recenter_refs = []
    skl_models = []
    compact_lda_models = []
    observer_skl_models = {"LR": [], "SVM": []}

    for step, endpoint in enumerate(T_POINTS):
        endpoint_idx = int(np.argmin(np.abs(epochs.times - endpoint)))
        step_trials = online_data_uv[:, :, riemann_start_idx:endpoint_idx + 1]
        step_template = step_trials[labels == mi_id].mean(axis=0)
        step_covariances = template_covariances_riemann(step_trials, step_template)

        step_recenter_ref = mean_riemann(step_covariances)
        step_covariances_train = recenter_covariances(
            step_covariances,
            step_recenter_ref,
        )

        mdm = MDM(metric="riemann")
        mdm.fit(step_covariances_train, labels)
        mdm_models.append(mdm)
        mdm_templates.append(step_template)
        mdm_recenter_refs.append(step_recenter_ref)
        mdm_centers.append({
            label: mdm.covmeans_[idx].copy()
            for idx, label in enumerate(mdm.classes_)
        })

        x_step, _ = extract_features(epochs, PICKS_CNV, T_POINTS, step=step + 1)
        x_step = x_step * 1e6

        lda = make_clf("LDA_shrink")
        lda.fit(x_step, labels)
        skl_models.append(lda)

        step_time_indices = [
            int(np.argmin(np.abs(epochs.times - time_point)))
            for time_point in T_POINTS[:step + 1]
        ]
        compact_features = compact_data_uv[:, :, step_time_indices].reshape(
            len(labels),
            -1,
        )
        compact_lda = make_clf("LDA_shrink")
        compact_lda.fit(compact_features, labels)
        compact_lda_models.append(compact_lda)

        for observer_name, observer_models in observer_skl_models.items():
            observer = make_clf(observer_name)
            observer.fit(x_step, labels)
            observer_models.append(observer)

    return {
        "model_type": "M2_LDA_shrink_MDM",
        "is_maestro": False,
        "picks": PICKS_CNV,
        "t_points": T_POINTS.copy(),
        "t_start": T_START,
        "t_end": T_END,
        "n_timepoints": N_TIMEPOINTS,
        "n_samples": int(round((T_END - T_START) * config.FS)) + 1,
        "REST_ID": event_dict["Rest (100)"],
        "MI_ID": mi_id,
        "subjects_train": [subject],
        "subject_calib": subject,
        "session_calib": session,
        "n_total": int(len(labels)),
        "mdm_models": mdm_models,
        "mdm_templates": mdm_templates,
        "mdm_recenter_refs": mdm_recenter_refs,
        "mdm_recenter_mode": "train_riemann_mean",
        "mdm_centers": mdm_centers,
        "mdm_available": True,
        "cov_reg": RIEMANN_COV_REG,
        "skl_models": skl_models,
        "skl_control_name": "LDA_shrink",
        "compact_lda_models": compact_lda_models,
        "compact_lda_picks": COMPACT_LDA_PICKS,
        "compact_lda_name": "LDA_shrink_3ch",
        "observer_skl_models": observer_skl_models,
        "observer_skl_names": list(observer_skl_models.keys()),
        "full_window_observer_names": [
            "MDM",
            "LDA_shrink",
            "LDA_shrink_3ch",
            "LR",
            "SVM",
        ],
        "full_feature_count": N_FEATURES,
        "compact_full_feature_count": len(COMPACT_LDA_PICKS) * N_TIMEPOINTS,
        "training_pipeline": (
            f"CAR + notch60 + {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz "
            "Butterworth 4º, sin CSD"
        ),
        "training_scale": "uV",
        "online_note": (
            f"Fast MotorCap {session}, 9 canales completos; MDM/LR/SVM observers"
        ),
    }


def main():
    args = parse_args()
    mne.set_log_level("WARNING")

    raw, events, event_dict, event_id_map, _, _ = load_raw_and_events(
        args.subject,
        args.session,
        args.base_dir,
        max_runs=args.max_runs,
        include_runs=args.include_runs,
        exclude_runs=args.exclude_runs,
    )
    print(
        f"📌 Eventos — Rest: {np.sum(events[:, 2] == event_dict['Rest (100)'])} | "
        f"MI: {np.sum(events[:, 2] == event_dict['MI (200)'])}"
    )

    epochs, _ = preprocess_and_epoch(raw, events, event_dict)
    mi_id = event_id_map["200"]

    print("🚀 Entrenando paquete online rápido...")
    pkg = train_online_m2_package(epochs, args.subject, args.session, event_dict, mi_id)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            args.base_dir,
            f"sub-{args.subject}",
            "models",
            f"sub-{args.subject}_model_motorcap_{args.session}.pkl",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as model_file:
        pickle.dump(pkg, model_file)

    print("\n" + "=" * 72)
    print("💾 FAST ONLINE MOTORCAP MODEL READY")
    print("=" * 72)
    print(f"   Ruta       : {output_path}")
    print(f"   Trials     : {pkg['n_total']}")
    print(f"   Canales    : {pkg['picks']}")
    print(f"   Pasos M2   : {pkg['n_timepoints']} ({pkg['t_start']} → {pkg['t_end']} s)")
    print(f"   Pipeline   : {pkg['training_pipeline']}")
    print("   Modelos    : MDM + LDA_shrink + LDA3 + LR + SVM")
    print("=" * 72)


if __name__ == "__main__":
    main()
