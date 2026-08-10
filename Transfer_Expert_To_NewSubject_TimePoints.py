"""
================================================================================
TRANSFERENCIA EXPERTO → NUEVO SUJETO — MODELOS ACUMULATIVOS
================================================================================

Pregunta experimental
---------------------
Entrenar únicamente con los runs offline del sujeto experto y probar en todos
los runs offline de un sujeto nuevo, sin usar datos del sujeto nuevo para:

  - entrenamiento,
  - selección de canales,
  - whitening/recentering,
  - ajuste de ventana temporal,
  - ajuste de umbral.

Diseño
------
Si el experto tiene 6 runs:

  Fold 1: entrenar experto runs 2-6 → probar nuevo runs 1-6
  Fold 2: entrenar experto runs 1,3-6 → probar nuevo runs 1-6
  ...
  Fold 6: entrenar experto runs 1-5 → probar nuevo runs 1-6

Para cada clasificador clásico se entrenan 6 × 11 modelos acumulativos.
Para Riemann se evalúan MDM y MDM+recenter/whitening, también por endpoint.

Este script NO guarda modelos online y NO modifica ningún archivo de datos.
================================================================================
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import bci_runtime_env  # noqa: F401  # prepara entorno local antes de importar MNE

os.environ["HOME"] = os.environ.get("BCI_ANALYSIS_HOME", "/tmp/bci-transfer-home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("MNE_CONFIG_DIR", "/tmp/mne-codex")
os.makedirs(os.environ["MNE_CONFIG_DIR"], exist_ok=True)

import mne
import matplotlib.pyplot as plt
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
except ImportError as exc:  # pragma: no cover - depende del entorno local
    raise RuntimeError(
        "Este análisis requiere pyriemann en el entorno activo."
    ) from exc


# ============================================================
# CONFIGURACIÓN EDITABLE
# ============================================================
DEFAULT_EXPERT_SUBJECT = "CNV_PILOT_SUBJ_021"
DEFAULT_EXPERT_SESSION = "S001_OFFLINE_FES_GLOVE"
DEFAULT_NEW_SUBJECT = "CNV_PILOT_SUBJ_022"
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

EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=4, ftype="butter")

EPOCH_TMIN = -3.0
EPOCH_TMAX = 2.0
# Para transferencia solo usamos features de -2.5 a 0.0 s. Esta línea base
# compacta es compatible con sesiones que no tienen suficiente tiempo pre-evento
# para usar el baseline largo [-5, -3] del generador visual.
BASELINE = (-3.0, -2.5)
REJECT_THRESHOLD = dict(eeg=100e-6)
FLAT_THRESHOLD = dict(eeg=0.1e-6)

T_START = -2.5
T_END = 0.0
TIMEPOINT_STEP = 0.25
T_POINTS = np.arange(T_START, T_END + TIMEPOINT_STEP / 2.0, TIMEPOINT_STEP)

CLASSIFIERS = ["LDA", "LDA_shrink", "LR", "SVM"]

PLOT_MODEL_ORDER = [
    "LDA",
    "LDA_shrink",
    "LR",
    "SVM",
    "MDM",
    "MDM+recenter",
]
PLOT_COLORS = {
    "LDA": "#2166ac",
    "LDA_shrink": "#7F77DD",
    "LR": "#f4a582",
    "SVM": "#d6604d",
    "MDM": "#542788",
    "MDM+recenter": "#b85c00",
}

RIEMANN_COV_REG = 1e-4
RIEMANN_MAX_FS = 32.0


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


def load_preprocessed_session(base_dir: str, subject: str, session: str) -> SessionData:
    """Carga XDF, aplica CAR/notch/banda lenta y devuelve epochs aceptados."""
    xdf_files = _session_xdf_files(base_dir, subject, session)
    print(
        f"\n📂  {subject} | {session}: cargando {len(xdf_files)} run(s) offline"
    )

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

        missing_picks = [ch for ch in PICKS_CNV if ch not in raw_tmp.ch_names]
        if missing_picks:
            raise RuntimeError(
                f"{subject} {session} no contiene canales requeridos: "
                f"{missing_picks}"
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

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_run_labels = np.asarray(event_run_labels, dtype=int)
    if len(event_run_labels) != len(events):
        raise RuntimeError(
            f"No se alinearon eventos/runs en {subject} {session}: "
            f"{len(event_run_labels)} etiquetas para {len(events)} eventos"
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

    pick_idx = [epochs_all.ch_names.index(ch) for ch in PICKS_CNV]
    data_cnv = epochs_all.get_data()[:, pick_idx, :]
    peak_to_peak = data_cnv.max(axis=2) - data_cnv.min(axis=2)

    reject_mask = peak_to_peak.max(axis=1) > REJECT_THRESHOLD["eeg"]
    flat_mask = peak_to_peak.max(axis=1) < FLAT_THRESHOLD["eeg"]
    drop_idx = np.flatnonzero(reject_mask | flat_mask).tolist()
    if drop_idx:
        epochs_all.drop(drop_idx, reason="LOCAL_CNV_REJECT")

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
            ("clf", LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )),
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
        base = SVC(
            kernel="linear",
            C=1.0,
            probability=False,
            random_state=42,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
        ])
    raise ValueError(f"Clasificador desconocido: {name}")


def features_for_step(epochs_obj: mne.Epochs, step: int) -> np.ndarray:
    """Features acumulativas: canales completos × primeros `step` timepoints."""
    data = epochs_obj.get_data(picks=PICKS_CNV) * 1e6
    times = epochs_obj.times
    time_indices = [
        int(np.argmin(np.abs(times - time_point)))
        for time_point in T_POINTS[:step]
    ]
    return np.hstack([
        data[:, channel_idx, :][:, time_indices]
        for channel_idx in range(len(PICKS_CNV))
    ])


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def _positive_probability(model, x_test: np.ndarray, mi_id: int) -> np.ndarray:
    classes = list(model.classes_)
    positive_idx = classes.index(mi_id)
    return model.predict_proba(x_test)[:, positive_idx]


def evaluate_classical_transfer(
    expert: SessionData,
    new: SessionData,
    classifier_names: list[str],
) -> tuple[dict, dict]:
    """Entrena 5 runs experto → prueba todos los runs del sujeto nuevo."""
    expert_runs = np.unique(expert.groups)
    new_runs = np.unique(new.groups)
    results = {
        clf_name: [
            {
                "fold_auc": [],
                "fold_acc": [],
                "new_run_auc_matrix": [],
            }
            for _ in T_POINTS
        ]
        for clf_name in classifier_names
    }

    for step in range(1, len(T_POINTS) + 1):
        x_exp = features_for_step(expert.epochs, step)
        x_new = features_for_step(new.epochs, step)
        y_exp = expert.labels
        y_new = new.labels

        for held_run in expert_runs:
            train_idx = np.flatnonzero(expert.groups != held_run)

            for clf_name in classifier_names:
                clf = make_clf(clf_name)
                clf.fit(x_exp[train_idx], y_exp[train_idx])
                scores = _positive_probability(clf, x_new, new.mi_id)
                pred = clf.predict(x_new)

                step_result = results[clf_name][step - 1]
                step_result["fold_auc"].append(_safe_auc(y_new, scores))
                step_result["fold_acc"].append(
                    float(accuracy_score(y_new, pred) * 100.0)
                )

                run_aucs = []
                for new_run in new_runs:
                    run_mask = new.groups == new_run
                    run_aucs.append(_safe_auc(y_new[run_mask], scores[run_mask]))
                step_result["new_run_auc_matrix"].append(run_aucs)

    summary = summarize_timepoint_results(results)
    return results, summary


def template_covariances_riemann(trials: np.ndarray, template: np.ndarray) -> np.ndarray:
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
    recentered = np.empty_like(covariances)
    for idx, covariance in enumerate(covariances):
        cov = transform @ covariance @ transform.T
        cov = 0.5 * (cov + cov.T)
        cov += RIEMANN_COV_REG * np.eye(cov.shape[0])
        recentered[idx] = cov
    return recentered


def riemann_trials_for_step(epochs_obj: mne.Epochs, step: int) -> np.ndarray:
    data = epochs_obj.get_data(picks=PICKS_CNV) * 1e6
    start_idx = int(np.argmin(np.abs(epochs_obj.times - T_START)))
    endpoint_idx = int(np.argmin(np.abs(epochs_obj.times - T_POINTS[step - 1])))
    stride = max(1, int(round(epochs_obj.info["sfreq"] / RIEMANN_MAX_FS)))
    return data[:, :, start_idx:endpoint_idx + 1:stride]


def evaluate_riemann_transfer(
    expert: SessionData,
    new: SessionData,
    recenter: bool = False,
) -> tuple[dict, dict]:
    model_name = "MDM+recenter" if recenter else "MDM"
    expert_runs = np.unique(expert.groups)
    new_runs = np.unique(new.groups)
    results = {
        model_name: [
            {
                "fold_auc": [],
                "fold_acc": [],
                "new_run_auc_matrix": [],
                "n_samples": np.nan,
            }
            for _ in T_POINTS
        ]
    }

    for step in range(1, len(T_POINTS) + 1):
        trials_exp = riemann_trials_for_step(expert.epochs, step)
        trials_new = riemann_trials_for_step(new.epochs, step)
        y_exp = expert.labels
        y_new = new.labels
        results[model_name][step - 1]["n_samples"] = trials_exp.shape[2]

        for held_run in expert_runs:
            train_idx = np.flatnonzero(expert.groups != held_run)
            y_train = y_exp[train_idx]
            template = trials_exp[train_idx][y_train == expert.mi_id].mean(axis=0)

            cov_train = template_covariances_riemann(
                trials_exp[train_idx],
                template,
            )
            cov_new = template_covariances_riemann(trials_new, template)

            if recenter:
                # Whitening/recentering calculado únicamente con runs expertos
                # de entrenamiento. El nuevo sujeto permanece invisible.
                reference = mean_riemann(cov_train)
                cov_train = recenter_covariances(cov_train, reference)
                cov_new = recenter_covariances(cov_new, reference)

            model = MDM(metric="riemann")
            model.fit(cov_train, y_train)
            positive_idx = int(np.where(model.classes_ == expert.mi_id)[0][0])
            scores = model.predict_proba(cov_new)[:, positive_idx]
            pred = model.predict(cov_new)

            step_result = results[model_name][step - 1]
            step_result["fold_auc"].append(_safe_auc(y_new, scores))
            step_result["fold_acc"].append(
                float(accuracy_score(y_new, pred) * 100.0)
            )

            run_aucs = []
            for new_run in new_runs:
                run_mask = new.groups == new_run
                run_aucs.append(_safe_auc(y_new[run_mask], scores[run_mask]))
            step_result["new_run_auc_matrix"].append(run_aucs)

    summary = summarize_timepoint_results(results)
    return results, summary


def summarize_timepoint_results(results: dict) -> dict:
    summary = {}
    for model_name, by_step in results.items():
        auc_mean = np.asarray([
            np.nanmean(step_result["fold_auc"])
            for step_result in by_step
        ])
        auc_std = np.asarray([
            np.nanstd(step_result["fold_auc"])
            for step_result in by_step
        ])
        acc_mean = np.asarray([
            np.nanmean(step_result["fold_acc"])
            for step_result in by_step
        ])
        acc_std = np.asarray([
            np.nanstd(step_result["fold_acc"])
            for step_result in by_step
        ])
        best_idx = int(np.nanargmax(auc_mean))
        summary[model_name] = {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "auc_start": float(auc_mean[0]),
            "auc_end": float(auc_mean[-1]),
            "acc_start": float(acc_mean[0]),
            "acc_end": float(acc_mean[-1]),
            "best_idx": best_idx,
            "best_time": float(T_POINTS[best_idx]),
            "best_auc": float(auc_mean[best_idx]),
        }
    return summary


def print_summary_table(summary: dict, title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"   {'Modelo':<14} {'AUC inicio':>10} {'AUC fin':>9} "
        f"{'Acc inicio':>11} {'Acc fin':>9} {'Mejor AUC':>22}"
    )
    print("   " + "-" * 72)
    for model_name, item in summary.items():
        print(
            f"   {model_name:<14} "
            f"{item['auc_start']:>10.3f} {item['auc_end']:>9.3f} "
            f"{item['acc_start']:>10.1f}% {item['acc_end']:>8.1f}% "
            f"{item['best_auc']:>9.3f} @ t={item['best_time']:>5.2f} s"
        )


def print_endpoint_matrix(
    results: dict,
    model_name: str,
    endpoint_idx: int,
    expert_runs: np.ndarray,
    new_runs: np.ndarray,
) -> None:
    matrix = np.asarray(results[model_name][endpoint_idx]["new_run_auc_matrix"])
    print("\n" + "-" * 78)
    print(
        f"Matriz AUC — {model_name}, endpoint t={T_POINTS[endpoint_idx]:.2f} s"
    )
    print("   Filas: run experto excluido | Columnas: run sujeto nuevo")
    print("   " + " ".join([f"N{int(run):>6}" for run in new_runs]))
    for row_idx, held_run in enumerate(expert_runs):
        row = " ".join(
            "   nan" if np.isnan(value) else f"{value:6.3f}"
            for value in matrix[row_idx]
        )
        print(f"   E-{int(held_run):<3} {row}")


def plot_transfer_classification_summary(
    all_summary: dict,
    expert: SessionData,
    new: SessionData,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Figura tipo Generate_Decoder: barras full-window + curvas acumulativas."""
    model_names = [name for name in PLOT_MODEL_ORDER if name in all_summary]
    if not model_names:
        print("⚠️   No hay modelos disponibles para graficar.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Expert-to-New Subject Transfer — Cumulative Classification\n"
        f"Expert: {expert.subject} | {expert.session}  →  "
        f"New subject: {new.subject} | {new.session}\n"
        f"Channels: {PICKS_CNV}",
        fontsize=13,
        fontweight="bold",
    )

    colors = [PLOT_COLORS.get(name, "gray") for name in model_names]

    # ── Barras full-window AUC ────────────────────────────────
    ax_auc_bar = axes[0, 0]
    auc_end = [all_summary[name]["auc_end"] for name in model_names]
    auc_std_end = [all_summary[name]["auc_std"][-1] for name in model_names]
    bars = ax_auc_bar.bar(
        model_names,
        auc_end,
        yerr=auc_std_end,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(elinewidth=1.4, capsize=5),
    )
    ax_auc_bar.axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                       label="Chance (0.5)")
    ax_auc_bar.axhline(0.7, color="gray", linestyle=":", linewidth=1.1,
                       label="Target (0.7)")
    ax_auc_bar.set_ylim(0.3, 0.8)
    ax_auc_bar.set_ylabel("AUC")
    ax_auc_bar.set_title("Full-window comparison\nAUC", fontweight="bold")
    ax_auc_bar.tick_params(axis="x", rotation=20)
    ax_auc_bar.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_auc_bar.legend(fontsize=9)
    for bar, value in zip(bars, auc_end):
        ax_auc_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ── Curvas acumulativas AUC ───────────────────────────────
    ax_auc_seq = axes[0, 1]
    for name in model_names:
        item = all_summary[name]
        color = PLOT_COLORS.get(name, "gray")
        ax_auc_seq.plot(
            T_POINTS,
            item["auc_mean"],
            marker="o",
            linewidth=2,
            color=color,
            label=name,
        )
        ax_auc_seq.fill_between(
            T_POINTS,
            item["auc_mean"] - item["auc_std"],
            item["auc_mean"] + item["auc_std"],
            color=color,
            alpha=0.12,
        )
    ax_auc_seq.axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                       label="Chance (0.5)")
    ax_auc_seq.axhline(0.7, color="gray", linestyle=":", linewidth=1.1,
                       label="Target (0.7)")
    ax_auc_seq.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
                       label="Trigger (0 s)")
    ax_auc_seq.set_ylim(0.3, 0.8)
    ax_auc_seq.set_xlim(T_END + 0.1, T_START - 0.1)
    ax_auc_seq.set_xlabel("Available time (s)")
    ax_auc_seq.set_ylabel("AUC")
    ax_auc_seq.set_title(
        "Cumulative comparison\nAUC vs time point",
        fontweight="bold",
    )
    ax_auc_seq.grid(True, linestyle=":", alpha=0.4)
    ax_auc_seq.legend(fontsize=8, loc="upper right")

    # ── Barras full-window Accuracy ───────────────────────────
    ax_acc_bar = axes[1, 0]
    acc_end = [all_summary[name]["acc_end"] for name in model_names]
    acc_std_end = [all_summary[name]["acc_std"][-1] for name in model_names]
    bars = ax_acc_bar.bar(
        model_names,
        acc_end,
        yerr=acc_std_end,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(elinewidth=1.4, capsize=5),
    )
    ax_acc_bar.axhline(50.0, color="red", linestyle="--", linewidth=1.2,
                       label="Chance (50%)")
    ax_acc_bar.axhline(70.0, color="gray", linestyle=":", linewidth=1.1,
                       label="Target (70%)")
    ax_acc_bar.set_ylim(30, 80)
    ax_acc_bar.set_ylabel("Accuracy (%)")
    ax_acc_bar.set_title("Full-window comparison\nAccuracy", fontweight="bold")
    ax_acc_bar.tick_params(axis="x", rotation=20)
    ax_acc_bar.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_acc_bar.legend(fontsize=9)
    for bar, value in zip(bars, acc_end):
        ax_acc_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ── Curvas acumulativas Accuracy ──────────────────────────
    ax_acc_seq = axes[1, 1]
    for name in model_names:
        item = all_summary[name]
        color = PLOT_COLORS.get(name, "gray")
        ax_acc_seq.plot(
            T_POINTS,
            item["acc_mean"],
            marker="o",
            linewidth=2,
            color=color,
            label=name,
        )
        ax_acc_seq.fill_between(
            T_POINTS,
            item["acc_mean"] - item["acc_std"],
            item["acc_mean"] + item["acc_std"],
            color=color,
            alpha=0.12,
        )
    ax_acc_seq.axhline(50.0, color="red", linestyle="--", linewidth=1.2,
                       label="Chance (50%)")
    ax_acc_seq.axhline(70.0, color="gray", linestyle=":", linewidth=1.1,
                       label="Target (70%)")
    ax_acc_seq.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
                       label="Trigger (0 s)")
    ax_acc_seq.set_ylim(30, 80)
    ax_acc_seq.set_xlim(T_END + 0.1, T_START - 0.1)
    ax_acc_seq.set_xlabel("Available time (s)")
    ax_acc_seq.set_ylabel("Accuracy (%)")
    ax_acc_seq.set_title(
        "Cumulative comparison\nAccuracy vs time point",
        fontweight="bold",
    )
    ax_acc_seq.grid(True, linestyle=":", alpha=0.4)
    ax_acc_seq.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=(0, 0, 1, 0.90))

    if save_path:
        if save_path == "auto":
            out_dir = os.path.join(os.getcwd(), "transfer_figures")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(
                out_dir,
                (
                    f"transfer_{expert.subject}_{expert.session}_to_"
                    f"{new.subject}_{new.session}.png"
                ),
            )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n🖼️   Figura guardada: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transferencia pura experto→nuevo sujeto con modelos acumulativos."
        )
    )
    parser.add_argument("--base-dir", default=getattr(config, "DATA_DIR", "."))
    parser.add_argument("--expert-subject", default=DEFAULT_EXPERT_SUBJECT)
    parser.add_argument("--expert-session", default=DEFAULT_EXPERT_SESSION)
    parser.add_argument("--new-subject", default=DEFAULT_NEW_SUBJECT)
    parser.add_argument("--new-session", default=DEFAULT_NEW_SESSION)
    parser.add_argument(
        "--filter-high",
        type=float,
        default=EEG_H_FREQ,
        help="Frecuencia alta EEG; por defecto 1.0 Hz.",
    )
    parser.add_argument(
        "--save-fig",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Guarda la figura resumen. Sin ruta usa transfer_figures/*.png."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abre ventana de Matplotlib; útil para ejecución remota.",
    )
    return parser.parse_args()


def main() -> None:
    global EEG_H_FREQ

    args = parse_args()
    EEG_H_FREQ = float(args.filter_high)
    mne.set_log_level("WARNING")

    print("\n" + "=" * 78)
    print("🧪  TRANSFERENCIA PURA EXPERTO → NUEVO SUJETO")
    print("=" * 78)
    print(
        f"   Experto      : {args.expert_subject} | {args.expert_session}\n"
        f"   Nuevo sujeto : {args.new_subject} | {args.new_session}\n"
        f"   Canales      : {PICKS_CNV}\n"
        f"   Ventana      : {T_START:.2f} a {T_END:.2f} s, "
        f"{len(T_POINTS)} puntos cada {TIMEPOINT_STEP:.2f} s\n"
        f"   Filtro EEG   : CAR + notch60 + {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz\n"
        "   Regla        : entrenar 5 runs del experto → probar todos "
        "los runs del sujeto nuevo"
    )

    expert = load_preprocessed_session(
        args.base_dir,
        args.expert_subject,
        args.expert_session,
    )
    new = load_preprocessed_session(
        args.base_dir,
        args.new_subject,
        args.new_session,
    )

    expert_runs = np.unique(expert.groups)
    new_runs = np.unique(new.groups)
    if len(expert_runs) < 2:
        raise RuntimeError("Se requieren al menos 2 runs del experto.")
    if len(new_runs) < 1:
        raise RuntimeError("Se requiere al menos 1 run del sujeto nuevo.")
    if len(expert_runs) != 6:
        print(
            f"⚠️   El experto tiene {len(expert_runs)} runs, no 6. "
            "El análisis continuará con los runs disponibles."
        )
    if len(new_runs) != 6:
        print(
            f"⚠️   El sujeto nuevo tiene {len(new_runs)} runs, no 6. "
            "El análisis continuará con los runs disponibles."
        )

    classical_results, classical_summary = evaluate_classical_transfer(
        expert,
        new,
        CLASSIFIERS,
    )
    print_summary_table(
        classical_summary,
        "⏱️  MODELOS CLÁSICOS — TRANSFERENCIA ACUMULATIVA",
    )

    riemann_results, riemann_summary = evaluate_riemann_transfer(
        expert,
        new,
        recenter=False,
    )
    riemann_recenter_results, riemann_recenter_summary = evaluate_riemann_transfer(
        expert,
        new,
        recenter=True,
    )
    riemann_combined_summary = {
        **riemann_summary,
        **riemann_recenter_summary,
    }
    print_summary_table(
        riemann_combined_summary,
        "🧭  RIEMANN — MDM Y MDM+RECENTER/WHITENING",
    )

    all_results = {
        **classical_results,
        **riemann_results,
        **riemann_recenter_results,
    }
    all_summary = {
        **classical_summary,
        **riemann_combined_summary,
    }
    best_model = max(
        all_summary,
        key=lambda name: all_summary[name]["best_auc"],
    )
    best_idx = all_summary[best_model]["best_idx"]

    print("\n" + "=" * 78)
    print("🏁  RESUMEN GLOBAL")
    print("=" * 78)
    print(
        f"   Mejor transferencia: {best_model} | "
        f"AUC={all_summary[best_model]['best_auc']:.3f} "
        f"@ t={all_summary[best_model]['best_time']:.2f} s"
    )
    print(
        "   Nota: cada AUC es promedio de los modelos entrenados dejando "
        "un run experto fuera; el sujeto nuevo nunca entra al entrenamiento."
    )

    print_endpoint_matrix(
        all_results,
        best_model,
        best_idx,
        expert_runs,
        new_runs,
    )

    plot_transfer_classification_summary(
        all_summary,
        expert,
        new,
        save_path=args.save_fig,
        show=not args.no_show,
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
