"""
================================================================================
EXPERTO + CALIBRACIÓN PROGRESIVA DEL SUJETO NUEVO
================================================================================

Pregunta experimental
---------------------
Si el modelo experto no transfiere bien de forma pura, ¿cuántos runs de
calibración del sujeto nuevo hacen falta para mejorar?

Diseño
------
Experto = todos sus runs offline.
Nuevo sujeto = runs offline ordenados cronológicamente.

K = 0:
  train = 6 runs experto
  test  = todos los runs del nuevo sujeto

K = 1:
  train = 6 runs experto + run 1 del nuevo sujeto
  test  = runs 2-6 del nuevo sujeto

K = 2:
  train = 6 runs experto + runs 1-2 del nuevo sujeto
  test  = runs 3-6 del nuevo sujeto

...

K = N-1:
  train = 6 runs experto + runs 1..N-1 del nuevo sujeto
  test  = último run del nuevo sujeto

Esto simula una calibración breve antes del uso online, sin usar nunca los runs
de validación del sujeto nuevo para entrenar.

Este script NO guarda modelos online y NO modifica datos.
================================================================================
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import bci_runtime_env  # noqa: F401

os.environ["HOME"] = os.environ.get("BCI_ANALYSIS_HOME", "/tmp/bci-adapt-home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("MNE_CONFIG_DIR", "/tmp/mne-codex")
os.makedirs(os.environ["MNE_CONFIG_DIR"], exist_ok=True)

import matplotlib.pyplot as plt
import mne
import numpy as np

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from pyriemann.classification import MDM
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Este análisis requiere pyriemann en el entorno activo."
    ) from exc


# ============================================================
# CONFIGURACIÓN EDITABLE
# ============================================================
DEFAULT_EXPERT_SUBJECT = "CNV_PILOT_SUBJ_021"
DEFAULT_EXPERT_SESSION = "S001_OFFLINE_FES_GLOVE"
DEFAULT_NEW_SUBJECT = "CNV_PILOT_SUBJ_025"
DEFAULT_NEW_SESSION = "S001_OFFLINE"

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

# Pipeline paper-like práctico: este fue el que mostró que MDM sí funciona
# dentro de sujeto, manteniendo la ventana del proyecto (-2.5 a 0 s).
EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=2, ftype="butter")
FILTER_PHASE = "zero"

EPOCH_TMIN = -3.0
EPOCH_TMAX = 2.0
BASELINE = (-3.0, -2.5)
ABS_REJECT_UV = 100.0
FLAT_THRESHOLD_UV = 0.1

T_START = -2.5
T_END = 0.0
TIMEPOINT_STEP = 0.25
T_POINTS = np.arange(T_START, T_END + TIMEPOINT_STEP / 2.0, TIMEPOINT_STEP)

CLASSIFIERS = ["LDA", "LDA_shrink", "LR", "SVM"]
MODEL_ORDER = ["LDA", "LDA_shrink", "LR", "SVM", "MDM", "MDM+recenter"]
MODEL_COLORS = {
    "LDA": "#2166ac",
    "LDA_shrink": "#7F77DD",
    "LR": "#f4a582",
    "SVM": "#d6604d",
    "MDM": "#542788",
    "MDM+recenter": "#b85c00",
}

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


def load_session(base_dir: str, subject: str, session: str) -> SessionData:
    xdf_files = _session_xdf_files(base_dir, subject, session)
    print(f"\n📂  {subject} | {session}: {len(xdf_files)} run(s)", flush=True)

    raw_list = []
    event_run_labels = []

    for run_idx, xdf_file in enumerate(xdf_files, start=1):
        print(f"   └─ Run {run_idx}: {os.path.basename(xdf_file)}", flush=True)
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
                ch for ch in CHANNELS_TO_INTERPOLATE
                if ch in raw_tmp.ch_names
            ]
            raw_tmp.interpolate_bads(reset_bads=True, verbose=False)

        missing = [ch for ch in PICKS_CNV if ch not in raw_tmp.ch_names]
        if missing:
            raise RuntimeError(
                f"{subject} {session} no contiene canales requeridos: {missing}"
            )

        raw_tmp.set_annotations(mne.Annotations(
            onset=marker_timestamps - eeg_timestamps[0],
            duration=np.zeros(len(marker_data)),
            description=[str(marker) for marker in marker_data],
            orig_time=None,
        ))
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    raw.filter(
        l_freq=EEG_L_FREQ,
        h_freq=EEG_H_FREQ,
        method="iir",
        iir_params=EEG_IIR_PARAMS,
        phase=FILTER_PHASE,
        picks="eeg",
        verbose=False,
    )
    raw.set_eeg_reference("average", projection=False, verbose=False)

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_run_labels = np.asarray(event_run_labels, dtype=int)
    if len(event_run_labels) != len(events):
        raise RuntimeError(
            f"No se alinearon eventos/runs en {subject} {session}: "
            f"{len(event_run_labels)} etiquetas para {len(events)} eventos."
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
        epochs_all.drop(drop_idx, reason="ABS_REJECT")

    labels = epochs_all.events[:, -1]
    groups = event_run_labels[epochs_all.selection]
    n_rest = int(np.sum(labels == event_id["Rest"]))
    n_mi = int(np.sum(labels == event_id["MI"]))
    print(
        f"   ✅ Epochs aceptados: {len(labels)} / {len(events)} "
        f"(Rest={n_rest}, MI={n_mi}, runs={np.unique(groups).tolist()})",
        flush=True,
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
    )


def make_clf(name: str):
    if name == "LDA":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ])
    if name == "LDA_shrink":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
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
    if name == "SVM":
        base = SVC(kernel="linear", C=1.0, probability=False, random_state=42)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
        ])
    raise ValueError(f"Clasificador desconocido: {name}")


def features_for_step(epochs: mne.Epochs, step: int) -> np.ndarray:
    data = epochs.get_data(picks=PICKS_CNV) * 1e6
    time_indices = [
        int(np.argmin(np.abs(epochs.times - time_point)))
        for time_point in T_POINTS[:step]
    ]
    return np.hstack([
        data[:, channel_idx, :][:, time_indices]
        for channel_idx in range(len(PICKS_CNV))
    ])


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def positive_probability(model, x_test: np.ndarray, mi_id: int) -> np.ndarray:
    return model.predict_proba(x_test)[:, list(model.classes_).index(mi_id)]


def concat_train_test_by_k(
    expert: SessionData,
    new: SessionData,
    k_new_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_runs = np.unique(new.groups)
    train_new_runs = new_runs[:k_new_train]
    test_new_runs = new_runs[k_new_train:]
    train_new_mask = np.isin(new.groups, train_new_runs)
    test_new_mask = np.isin(new.groups, test_new_runs)
    return train_new_mask, test_new_mask, train_new_runs, test_new_runs


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


def riemann_trials_for_step(epochs: mne.Epochs, step: int) -> np.ndarray:
    data = epochs.get_data(picks=PICKS_CNV) * 1e6
    start_idx = int(np.argmin(np.abs(epochs.times - T_START)))
    endpoint_idx = int(np.argmin(np.abs(epochs.times - T_POINTS[step - 1])))
    stride = max(1, int(round(epochs.info["sfreq"] / RIEMANN_MAX_FS)))
    return data[:, :, start_idx:endpoint_idx + 1:stride]


def init_results(n_new_runs: int) -> dict:
    max_k = n_new_runs - 1
    return {
        model_name: {
            k: [
                {"auc": np.nan, "acc": np.nan, "n_train": 0, "n_test": 0}
                for _ in T_POINTS
            ]
            for k in range(max_k + 1)
        }
        for model_name in MODEL_ORDER
    }


def evaluate_adaptation(expert: SessionData, new: SessionData) -> dict:
    new_runs = np.unique(new.groups)
    max_k = len(new_runs) - 1
    results = init_results(len(new_runs))

    print("\n" + "=" * 90)
    print("🧪  PROGRESSIVE NEW-SUBJECT CALIBRATION")
    print("=" * 90)
    print(
        "   K means first K chronological runs from the new subject are added "
        "to all expert runs."
    )

    for k in range(max_k + 1):
        train_new_mask, test_new_mask, train_new_runs, test_new_runs = (
            concat_train_test_by_k(expert, new, k)
        )
        print(
            f"\n   K={k}: train = expert all + new runs "
            f"{train_new_runs.tolist()} | test new runs {test_new_runs.tolist()}",
            flush=True,
        )

        y_train = np.concatenate([expert.labels, new.labels[train_new_mask]])
        y_test = new.labels[test_new_mask]

        for step in range(1, len(T_POINTS) + 1):
            print(
                f"      endpoint {step:02d}/{len(T_POINTS)} "
                f"(t={T_POINTS[step - 1]:.2f} s)",
                flush=True,
            )

            x_exp = features_for_step(expert.epochs, step)
            x_new = features_for_step(new.epochs, step)
            x_train = np.vstack([x_exp, x_new[train_new_mask]])
            x_test = x_new[test_new_mask]

            for clf_name in CLASSIFIERS:
                clf = make_clf(clf_name)
                clf.fit(x_train, y_train)
                scores = positive_probability(clf, x_test, new.mi_id)
                pred = clf.predict(x_test)
                results[clf_name][k][step - 1] = {
                    "auc": safe_auc(y_test, scores),
                    "acc": float(accuracy_score(y_test, pred) * 100.0),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }

            trials_exp = riemann_trials_for_step(expert.epochs, step)
            trials_new = riemann_trials_for_step(new.epochs, step)
            trials_train = np.concatenate(
                [trials_exp, trials_new[train_new_mask]],
                axis=0,
            )
            trials_test = trials_new[test_new_mask]
            template = trials_train[y_train == expert.mi_id].mean(axis=0)
            cov_train = template_covariances(trials_train, template)
            cov_test = template_covariances(trials_test, template)

            for recenter, model_name in [
                (False, "MDM"),
                (True, "MDM+recenter"),
            ]:
                cov_train_model = cov_train
                cov_test_model = cov_test
                if recenter:
                    reference = mean_riemann(
                        cov_train,
                        tol=RIEMANN_MEAN_TOL,
                        maxiter=RIEMANN_MEAN_MAXITER,
                    )
                    cov_train_model = recenter_covariances(cov_train, reference)
                    cov_test_model = recenter_covariances(cov_test, reference)

                model = MDM(metric="riemann")
                model.fit(cov_train_model, y_train)
                positive_idx = int(np.where(model.classes_ == new.mi_id)[0][0])
                scores = model.predict_proba(cov_test_model)[:, positive_idx]
                pred = model.predict(cov_test_model)
                results[model_name][k][step - 1] = {
                    "auc": safe_auc(y_test, scores),
                    "acc": float(accuracy_score(y_test, pred) * 100.0),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }

    return results


def print_tables(results: dict, new: SessionData) -> None:
    new_runs = np.unique(new.groups)
    print("\n" + "=" * 90)
    print("📊  SUMMARY — FULL WINDOW AND BEST ENDPOINT")
    print("=" * 90)
    print(
        f"   {'Model':<14} {'K':>2} {'TrainN':>7} {'Test runs':<16} "
        f"{'AUC end':>8} {'Acc end':>8} {'Best AUC':>18}"
    )
    print("   " + "-" * 82)
    for model_name in MODEL_ORDER:
        for k, by_step in results[model_name].items():
            final = by_step[-1]
            aucs = np.asarray([row["auc"] for row in by_step])
            best_idx = int(np.nanargmax(aucs))
            test_runs = new_runs[k:].tolist()
            print(
                f"   {model_name:<14} {k:>2} {final['n_train']:>7} "
                f"{str(test_runs):<16} "
                f"{final['auc']:>8.3f} {final['acc']:>7.1f}% "
                f"{aucs[best_idx]:>8.3f} @ t={T_POINTS[best_idx]:>5.2f}"
            )
    print("=" * 90)


def plot_adaptation_summary(
    results: dict,
    expert: SessionData,
    new: SessionData,
    save_path: str | None,
    show: bool,
) -> None:
    k_values = sorted(next(iter(results.values())).keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name in MODEL_ORDER:
        final_auc = [results[model_name][k][-1]["auc"] for k in k_values]
        final_acc = [results[model_name][k][-1]["acc"] for k in k_values]
        color = MODEL_COLORS.get(model_name, "gray")
        axes[0].plot(
            k_values,
            final_auc,
            marker="o",
            linewidth=2.0,
            color=color,
            label=model_name,
        )
        axes[1].plot(
            k_values,
            final_acc,
            marker="o",
            linewidth=2.0,
            color=color,
            label=model_name,
        )

    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                    label="Chance (0.5)")
    axes[0].axhline(0.7, color="gray", linestyle=":", linewidth=1.1,
                    label="Target (0.7)")
    axes[0].set_ylim(0.3, 0.9)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Full-window AUC vs calibration runs", fontweight="bold")

    axes[1].axhline(50, color="red", linestyle="--", linewidth=1.2,
                    label="Chance (50%)")
    axes[1].axhline(70, color="gray", linestyle=":", linewidth=1.1,
                    label="Target (70%)")
    axes[1].set_ylim(30, 90)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Full-window Accuracy vs calibration runs", fontweight="bold")

    for ax in axes:
        ax.set_xlabel("New-subject calibration runs added (K)")
        ax.set_xticks(k_values)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Expert + Progressive New-Subject Calibration\n"
        f"Expert: {expert.subject} | {expert.session} → "
        f"New subject: {new.subject} | {new.session}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    if save_path:
        if save_path == "auto":
            out_dir = os.path.join(os.getcwd(), "adaptation_figures")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(
                out_dir,
                (
                    f"adapt_{expert.subject}_{expert.session}_to_"
                    f"{new.subject}_{new.session}.png"
                ),
            )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n🖼️   Figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expert + progressive new-subject calibration."
    )
    parser.add_argument("--base-dir", default=getattr(config, "DATA_DIR", "."))
    parser.add_argument("--expert-subject", default=DEFAULT_EXPERT_SUBJECT)
    parser.add_argument("--expert-session", default=DEFAULT_EXPERT_SESSION)
    parser.add_argument("--new-subject", default=DEFAULT_NEW_SUBJECT)
    parser.add_argument("--new-session", default=DEFAULT_NEW_SESSION)
    parser.add_argument(
        "--cov-fs",
        type=float,
        default=RIEMANN_MAX_FS,
        help="Max sampling rate for Riemann covariance. Default: 32 Hz.",
    )
    parser.add_argument(
        "--save-fig",
        nargs="?",
        const="auto",
        default=None,
        help="Save summary figure. Without path uses adaptation_figures/*.png.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RIEMANN_MAX_FS

    args = parse_args()
    RIEMANN_MAX_FS = float(args.cov_fs)
    mne.set_log_level("WARNING")

    print("\n" + "=" * 90)
    print("🧪  EXPERT + PROGRESSIVE NEW-SUBJECT CALIBRATION")
    print("=" * 90)
    print(
        f"   Expert      : {args.expert_subject} | {args.expert_session}\n"
        f"   New subject : {args.new_subject} | {args.new_session}\n"
        f"   Channels    : {PICKS_CNV}\n"
        f"   Filter      : {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz, "
        f"Butterworth {EEG_IIR_PARAMS['order']}nd order, {FILTER_PHASE}\n"
        f"   Window      : {T_START:.1f} to {T_END:.1f} s "
        f"({len(T_POINTS)} endpoints)\n"
        f"   Cov Fs      : ≤{RIEMANN_MAX_FS:.0f} Hz"
    )

    expert = load_session(args.base_dir, args.expert_subject, args.expert_session)
    new = load_session(args.base_dir, args.new_subject, args.new_session)

    if len(np.unique(new.groups)) < 2:
        raise RuntimeError("Se requieren al menos 2 runs del sujeto nuevo.")

    results = evaluate_adaptation(expert, new)
    print_tables(results, new)
    plot_adaptation_summary(
        results,
        expert,
        new,
        save_path=args.save_fig,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
