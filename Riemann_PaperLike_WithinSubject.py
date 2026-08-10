"""
================================================================================
RIEMANN PAPER-LIKE — WITHIN-SUBJECT CNV CLASSIFICATION
================================================================================

Objetivo
--------
Comprobar si el clasificador Riemanniano MDM se comporta como esperaríamos al
seguir más de cerca la metodología del paper:

  "Riemannian geometry-based detection of slow cortical potentials during
   movement preparation"

Esta prueba NO hace transferencia entre sujetos. Evalúa cada sujeto/sesión por
separado usando Leave-One-Run-Out:

  entrenar runs del mismo sujeto - 1 → probar el run restante del mismo sujeto

Decisiones paper-like aplicadas aquí
------------------------------------
  - CAR
  - Filtro 0.1–2 Hz
  - Butterworth de 2º orden
  - Zero-phase offline
  - Rechazo por amplitud absoluta ±100 µV
  - Template-extended covariance
  - Normalización por traza
  - MDM con métrica Riemanniana
  - Recenter/whitening calculado solo con el entrenamiento de cada fold

Decisión intencional del proyecto
---------------------------------
Mantenemos la ventana de clasificación en -2.5 a 0.0 s, no -3 a 0 s.

Este script no guarda modelos online y no modifica datos.
================================================================================
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import bci_runtime_env  # noqa: F401

os.environ["HOME"] = os.environ.get("BCI_ANALYSIS_HOME", "/tmp/bci-riemann-home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("MNE_CONFIG_DIR", "/tmp/mne-codex")
os.makedirs(os.environ["MNE_CONFIG_DIR"], exist_ok=True)

import matplotlib.pyplot as plt
import mne
import numpy as np

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

try:
    from pyriemann.classification import MDM
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Este análisis requiere pyriemann en el entorno activo."
    ) from exc


# ============================================================
# CONFIGURACIÓN
# ============================================================
DEFAULT_ANALYSES = [
    ("CNV_PILOT_SUBJ_021", "S001_OFFLINE_FES_GLOVE"),
    ("CNV_PILOT_SUBJ_022", "S001_OFFLINE"),
    ("CNV_PILOT_SUBJ_023", "S001_OFFLINE"),
]

PICKS_CNV = [
    "FC3", "FC1", "FCz",
    "C3",  "C1",  "Cz",
    "CP3", "CP1", "CPz",
]

CHANNELS_TO_DROP = ["M1", "M2", "T7", "T8", "Fp1", "Fpz", "Fp2"]
CHANNELS_TO_INTERPOLATE = []
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}

REST_MARKER = 100
MI_MARKER = 200
TARGET_MARKERS = [REST_MARKER, MI_MARKER]

EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=2, ftype="butter")

EPOCH_TMIN = -3.0
EPOCH_TMAX = 2.0
BASELINE = (-3.0, -2.5)
ABS_REJECT_UV = 100.0
FLAT_THRESHOLD_UV = 0.1

T_START = -2.5
T_END = 0.0
TIMEPOINT_STEP = 0.25
T_POINTS = np.arange(T_START, T_END + TIMEPOINT_STEP / 2.0, TIMEPOINT_STEP)

RIEMANN_COV_REG = 1e-4
RIEMANN_MAX_FS = 32.0
RIEMANN_MEAN_TOL = 1e-6
RIEMANN_MEAN_MAXITER = 20


@dataclass
class SessionData:
    subject: str
    session: str
    xdf_files: list[str]
    epochs: mne.Epochs
    labels: np.ndarray
    groups: np.ndarray
    event_id: dict[str, int]
    mi_id: int
    rest_id: int
    n_events_total: int


def _parse_subject_session(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Usa el formato SUBJECT:SESSION, por ejemplo "
            "CNV_PILOT_SUBJ_021:S001_OFFLINE_FES_GLOVE"
        )
    subject, session = value.split(":", 1)
    return subject.strip(), session.strip()


def _session_xdf_files(base_dir: str, subject: str, session: str) -> list[str]:
    xdf_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}", "eeg")
    if not os.path.isdir(xdf_dir):
        raise FileNotFoundError(f"No existe la carpeta XDF: {xdf_dir}")

    xdf_files = sorted(
        os.path.join(xdf_dir, name)
        for name in os.listdir(xdf_dir)
        if name.endswith(".xdf")
    )
    if not xdf_files:
        raise FileNotFoundError(f"No encontré archivos .xdf en: {xdf_dir}")
    return xdf_files


def load_paperlike_session(base_dir: str, subject: str, session: str) -> SessionData:
    xdf_files = _session_xdf_files(base_dir, subject, session)
    print(f"\n📂  {subject} | {session}: {len(xdf_files)} run(s)")

    raw_list = []
    event_run_labels = []

    for run_idx, xdf_file in enumerate(xdf_files, start=1):
        print(f"   └─ Run {run_idx}: {os.path.basename(xdf_file)}")
        eeg_s, marker_s = load_xdf(xdf_file)

        eeg_data = np.asarray(eeg_s["time_series"]).T
        eeg_timestamps = np.asarray(eeg_s["time_stamps"])
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_data_all = np.asarray([
            int(round(float(np.ravel(value)[0])))
            for value in marker_s["time_series"]
        ])
        marker_timestamps_all = np.asarray(marker_s["time_stamps"])

        keep = np.isin(marker_data_all, TARGET_MARKERS)
        marker_data = marker_data_all[keep]
        marker_timestamps = marker_timestamps_all[keep]
        event_run_labels.extend([run_idx] * len(marker_data))

        valid_ch = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info = mne.create_info(valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

        existing_renames = {
            old: new
            for old, new in RENAME_DICT.items()
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
                ch for ch in CHANNELS_TO_INTERPOLATE
                if ch in raw_tmp.ch_names
            ]
            raw_tmp.interpolate_bads(reset_bads=True, verbose=False)

        missing = [ch for ch in PICKS_CNV if ch not in raw_tmp.ch_names]
        if missing:
            raise RuntimeError(
                f"{subject} {session} no contiene canales requeridos: {missing}"
            )

        annotations = mne.Annotations(
            onset=marker_timestamps - eeg_timestamps[0],
            duration=np.zeros(len(marker_data)),
            description=[str(marker) for marker in marker_data],
            orig_time=None,
        )
        raw_tmp.set_annotations(annotations)
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)

    # Paper-like preprocessing: 0.1–2 Hz, Butterworth 2nd order, zero-phase,
    # then common average reference. The paper describes both; for linear
    # filtering/CAR this order is acceptable and keeps MNE happy.
    raw.filter(
        l_freq=EEG_L_FREQ,
        h_freq=EEG_H_FREQ,
        method="iir",
        iir_params=EEG_IIR_PARAMS,
        phase="zero",
        picks="eeg",
        verbose=False,
    )
    raw.set_eeg_reference("average", projection=False, verbose=False)

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_run_labels = np.asarray(event_run_labels, dtype=int)
    if len(event_run_labels) != len(events):
        raise RuntimeError(
            f"No se alinearon eventos/runs: {len(event_run_labels)} etiquetas "
            f"para {len(events)} eventos."
        )

    event_id = {
        "Rest": event_id_map[str(REST_MARKER)],
        "MI": event_id_map[str(MI_MARKER)],
    }
    epochs_all = mne.Epochs(
        raw,
        events,
        event_id={"Rest": event_id["Rest"], "MI": event_id["MI"]},
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        reject=None,
        flat=None,
        preload=True,
        detrend=None,
        verbose=False,
    )

    data_uv = epochs_all.get_data(picks=PICKS_CNV) * 1e6
    abs_bad = np.max(np.abs(data_uv), axis=(1, 2)) > ABS_REJECT_UV
    flat_bad = (
        data_uv.max(axis=2) - data_uv.min(axis=2)
    ).max(axis=1) < FLAT_THRESHOLD_UV
    drop_idx = np.flatnonzero(abs_bad | flat_bad).tolist()
    if drop_idx:
        epochs_all.drop(drop_idx, reason="PAPERLIKE_ABS_REJECT")

    labels = epochs_all.events[:, -1]
    groups = event_run_labels[epochs_all.selection]
    n_rest = int(np.sum(labels == event_id["Rest"]))
    n_mi = int(np.sum(labels == event_id["MI"]))
    print(
        f"   ✅ Epochs aceptados: {len(labels)} / {len(events)} "
        f"(Rest={n_rest}, MI={n_mi}, runs={np.unique(groups).tolist()})"
    )

    return SessionData(
        subject=subject,
        session=session,
        xdf_files=xdf_files,
        epochs=epochs_all,
        labels=labels,
        groups=groups,
        event_id=event_id,
        mi_id=event_id["MI"],
        rest_id=event_id["Rest"],
        n_events_total=len(events),
    )


def template_covariances(trials: np.ndarray, template: np.ndarray) -> np.ndarray:
    repeated_template = np.repeat(template[np.newaxis, :, :], len(trials), axis=0)
    extended = np.concatenate([trials, repeated_template], axis=1)
    covariances = np.empty(
        (len(trials), extended.shape[1], extended.shape[1]),
        dtype=float,
    )

    for idx, trial in enumerate(extended):
        covariance = trial @ trial.T
        trace = np.trace(covariance)
        if trace > 0:
            covariance /= trace
        covariance += RIEMANN_COV_REG * np.eye(covariance.shape[0])
        covariances[idx] = covariance
    return covariances


def recenter_covariances(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    transform = invsqrtm(reference)
    out = np.empty_like(covariances)
    for idx, covariance in enumerate(covariances):
        cov = transform @ covariance @ transform.T
        cov = 0.5 * (cov + cov.T)
        cov += RIEMANN_COV_REG * np.eye(cov.shape[0])
        out[idx] = cov
    return out


def trials_for_step(epochs: mne.Epochs, step: int) -> np.ndarray:
    data = epochs.get_data(picks=PICKS_CNV) * 1e6
    start_idx = int(np.argmin(np.abs(epochs.times - T_START)))
    endpoint_idx = int(np.argmin(np.abs(epochs.times - T_POINTS[step - 1])))
    stride = max(1, int(round(epochs.info["sfreq"] / RIEMANN_MAX_FS)))
    return data[:, :, start_idx:endpoint_idx + 1:stride]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def evaluate_within_subject(session_data: SessionData, recenter: bool) -> dict:
    model_name = "MDM+recenter" if recenter else "MDM"
    groups = session_data.groups
    labels = session_data.labels
    logo = LeaveOneGroupOut()
    by_step = []

    for step in range(1, len(T_POINTS) + 1):
        print(
            f"   {model_name}: endpoint {step:02d}/{len(T_POINTS)} "
            f"(t={T_POINTS[step - 1]:.2f} s)",
            flush=True,
        )
        trials = trials_for_step(session_data.epochs, step)
        scores = np.full(len(labels), np.nan)
        predictions = np.full(len(labels), -1, dtype=int)
        fold_aucs = []
        fold_accs = []

        for train_idx, test_idx in logo.split(trials, labels, groups):
            y_train = labels[train_idx]
            template = trials[train_idx][y_train == session_data.mi_id].mean(axis=0)
            cov_train = template_covariances(trials[train_idx], template)
            cov_test = template_covariances(trials[test_idx], template)

            if recenter:
                # R comes only from training covariances, regardless of class.
                reference = mean_riemann(
                    cov_train,
                    tol=RIEMANN_MEAN_TOL,
                    maxiter=RIEMANN_MEAN_MAXITER,
                )
                cov_train = recenter_covariances(cov_train, reference)
                cov_test = recenter_covariances(cov_test, reference)

            model = MDM(metric="riemann")
            model.fit(cov_train, y_train)
            positive_idx = int(np.where(model.classes_ == session_data.mi_id)[0][0])
            scores[test_idx] = model.predict_proba(cov_test)[:, positive_idx]
            predictions[test_idx] = model.predict(cov_test)
            fold_aucs.append(safe_auc(labels[test_idx], scores[test_idx]))
            fold_accs.append(
                float(accuracy_score(labels[test_idx], predictions[test_idx]) * 100.0)
            )

        by_step.append({
            "model": model_name,
            "step": step,
            "time": float(T_POINTS[step - 1]),
            "n_samples": int(trials.shape[2]),
            "auc_oof": safe_auc(labels, scores),
            "acc_oof": float(accuracy_score(labels, predictions) * 100.0),
            "auc_fold_mean": float(np.nanmean(fold_aucs)),
            "auc_fold_std": float(np.nanstd(fold_aucs)),
            "acc_fold_mean": float(np.nanmean(fold_accs)),
            "acc_fold_std": float(np.nanstd(fold_accs)),
            "scores": scores,
            "predictions": predictions,
        })

    return {
        "model": model_name,
        "by_step": by_step,
    }


def print_results(session_data: SessionData, results: list[dict]) -> None:
    print("\n" + "=" * 82)
    print(
        f"🧭  PAPER-LIKE RIEMANN WITHIN-SUBJECT — "
        f"{session_data.subject} | {session_data.session}"
    )
    print("=" * 82)
    print(
        f"   Pipeline: CAR + {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz "
        "Butterworth 2nd order, zero-phase | abs reject ±100 µV"
    )
    print(
        f"   Window: {T_START:.1f} to {T_END:.1f} s | "
        f"Covariance Fs ≤{RIEMANN_MAX_FS:.0f} Hz | "
        f"Runs: {np.unique(session_data.groups).tolist()}"
    )

    for result in results:
        print("\n" + result["model"])
        print(
            f"   {'Endpoint':>9} {'Samples':>8} {'AUC OOF':>8} "
            f"{'AUC folds':>10} {'±std':>7} {'Acc OOF':>8}"
        )
        print("   " + "-" * 58)
        for row in result["by_step"]:
            print(
                f"   {row['time']:>9.3f} {row['n_samples']:>8} "
                f"{row['auc_oof']:>8.3f} "
                f"{row['auc_fold_mean']:>10.3f} "
                f"{row['auc_fold_std']:>7.3f} "
                f"{row['acc_oof']:>7.1f}%"
            )

        best = max(result["by_step"], key=lambda item: item["auc_oof"])
        final = result["by_step"][-1]
        print(
            f"   Full window: AUC={final['auc_oof']:.3f}, "
            f"Acc={final['acc_oof']:.1f}%"
        )
        print(
            f"   Best endpoint: t={best['time']:.3f} s, "
            f"AUC={best['auc_oof']:.3f}"
        )

    if len(results) == 2:
        final_delta_auc = (
            results[1]["by_step"][-1]["auc_oof"]
            - results[0]["by_step"][-1]["auc_oof"]
        )
        final_delta_acc = (
            results[1]["by_step"][-1]["acc_oof"]
            - results[0]["by_step"][-1]["acc_oof"]
        )
        print(
            f"\n   Recenter effect at full window: "
            f"ΔAUC={final_delta_auc:+.3f}, ΔAcc={final_delta_acc:+.1f}%"
        )
    print("=" * 82)


def plot_results(session_data: SessionData, results: list[dict], save_dir: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    colors = {"MDM": "#542788", "MDM+recenter": "#b85c00"}

    for result in results:
        name = result["model"]
        times = np.asarray([row["time"] for row in result["by_step"]])
        aucs = np.asarray([row["auc_oof"] for row in result["by_step"]])
        auc_stds = np.asarray([row["auc_fold_std"] for row in result["by_step"]])
        accs = np.asarray([row["acc_oof"] for row in result["by_step"]])
        acc_stds = np.asarray([row["acc_fold_std"] for row in result["by_step"]])

        axes[0].plot(times, aucs, marker="o", linewidth=2.2,
                     color=colors[name], label=name)
        axes[0].fill_between(times, aucs - auc_stds, aucs + auc_stds,
                             color=colors[name], alpha=0.15)
        axes[1].plot(times, accs, marker="o", linewidth=2.2,
                     color=colors[name], label=name)
        axes[1].fill_between(times, accs - acc_stds, accs + acc_stds,
                             color=colors[name], alpha=0.15)

    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                    label="Chance (0.5)")
    axes[0].axhline(0.7, color="gray", linestyle=":", linewidth=1.1,
                    label="Target (0.7)")
    axes[0].set_ylim(0.3, 0.9)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC vs endpoint", fontweight="bold")

    axes[1].axhline(50, color="red", linestyle="--", linewidth=1.2,
                    label="Chance (50%)")
    axes[1].axhline(70, color="gray", linestyle=":", linewidth=1.1,
                    label="Target (70%)")
    axes[1].set_ylim(30, 90)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy vs endpoint", fontweight="bold")

    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1,
                   label="Trigger (0 s)")
        ax.set_xlim(T_END + 0.1, T_START - 0.1)
        ax.set_xlabel("Available time (s)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Paper-like Riemannian MDM — Within-subject LOGO\n"
        f"{session_data.subject} | {session_data.session}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir,
            f"paperlike_riemann_{session_data.subject}_{session_data.session}.png",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"   🖼️  Figure saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-like Riemannian MDM within-subject LOGO."
    )
    parser.add_argument("--base-dir", default=getattr(config, "DATA_DIR", "."))
    parser.add_argument(
        "--analysis",
        action="append",
        type=_parse_subject_session,
        help=(
            "Subject/session to analyze, format SUBJECT:SESSION. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--cov-fs",
        type=float,
        default=RIEMANN_MAX_FS,
        help="Max sampling rate used for covariance matrices. Default: 128 Hz.",
    )
    parser.add_argument(
        "--save-fig-dir",
        default=None,
        help="If provided, saves figures to this directory instead of showing them.",
    )
    return parser.parse_args()


def main() -> None:
    global RIEMANN_MAX_FS

    args = parse_args()
    RIEMANN_MAX_FS = float(args.cov_fs)
    analyses = args.analysis if args.analysis else DEFAULT_ANALYSES
    mne.set_log_level("WARNING")

    print("\n" + "=" * 82)
    print("🧪  RIEMANN PAPER-LIKE — WITHIN-SUBJECT LOGO")
    print("=" * 82)
    print(
        f"   Analyses: {analyses}\n"
        f"   Channels: {PICKS_CNV}\n"
        f"   Filter  : {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz, "
        "Butterworth 2nd order, zero-phase\n"
        f"   Window  : {T_START:.1f} to {T_END:.1f} s "
        f"({len(T_POINTS)} endpoints)\n"
        f"   Cov Fs  : ≤{RIEMANN_MAX_FS:.0f} Hz"
    )

    for subject, session in analyses:
        session_data = load_paperlike_session(args.base_dir, subject, session)
        results = [
            evaluate_within_subject(session_data, recenter=False),
            evaluate_within_subject(session_data, recenter=True),
        ]
        print_results(session_data, results)
        plot_results(session_data, results, args.save_fig_dir)


if __name__ == "__main__":
    main()
