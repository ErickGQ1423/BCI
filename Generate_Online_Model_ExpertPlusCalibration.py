"""
================================================================================
GENERADOR ONLINE — EXPERTO + CALIBRACIÓN DEL SUJETO NUEVO
================================================================================

Entrena un paquete online compatible con ExperimentDriver_Online.py usando:

  todos los runs del experto
  +
  los primeros K runs offline del sujeto nuevo

Uso previsto para mañana:

  1. Registrar 4 runs offline del sujeto nuevo.
  2. Ejecutar este script con --new-runs 4.
  3. Apuntar config.ONLINE_MODEL_PATH al .pkl generado.
  4. Correr online sin recalibración adicional.

Este script guarda UN modelo .pkl y no modifica config.py automáticamente.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import pickle

import bci_runtime_env  # noqa: F401

os.environ["HOME"] = os.environ.get("BCI_ANALYSIS_HOME", "/tmp/bci-gen-adapt-home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("MNE_CONFIG_DIR", "/tmp/mne-codex")
os.makedirs(os.environ["MNE_CONFIG_DIR"], exist_ok=True)

import mne
import numpy as np

import config
import Adapt_Expert_With_NewSubject_Runs as adapt

from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann


# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================
DEFAULT_EXPERT_SUBJECT = "CNV_PILOT_SUBJ_021"
DEFAULT_EXPERT_SESSION = "S001_OFFLINE_FES_GLOVE"
DEFAULT_NEW_SUBJECT = "CNV_PILOT_SUBJ_029"
DEFAULT_NEW_SESSION = "S002_OFFLINE"
DEFAULT_NEW_RUNS = 4

COMPACT_LDA_PICKS = ["FCz", "C3", "CP3"]


def _validate_compact_picks() -> list[str]:
    missing = [ch for ch in COMPACT_LDA_PICKS if ch not in adapt.PICKS_CNV]
    if missing:
        raise RuntimeError(
            f"COMPACT_LDA_PICKS contiene canales fuera de PICKS_CNV: {missing}"
        )
    return COMPACT_LDA_PICKS


def _select_first_new_runs(new: adapt.SessionData, k_runs: int) -> np.ndarray:
    new_runs = np.unique(new.groups)
    if k_runs < 0:
        raise ValueError("--new-runs debe ser >= 0")
    if k_runs > len(new_runs):
        raise ValueError(
            f"--new-runs={k_runs}, pero el sujeto nuevo solo tiene "
            f"{len(new_runs)} run(s)."
        )
    selected_runs = new_runs[:k_runs]
    return np.isin(new.groups, selected_runs)


def _combine_training_data(
    expert: adapt.SessionData,
    new: adapt.SessionData,
    new_train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.concatenate([expert.labels, new.labels[new_train_mask]])
    sources = np.concatenate([
        np.full(len(expert.labels), "expert", dtype=object),
        np.full(int(np.sum(new_train_mask)), "new_calib", dtype=object),
    ])
    groups = np.concatenate([
        expert.groups,
        new.groups[new_train_mask],
    ])
    return labels, sources, groups


def _features_for_training(
    expert: adapt.SessionData,
    new: adapt.SessionData,
    new_train_mask: np.ndarray,
    step: int,
    picks: list[str] | None = None,
) -> np.ndarray:
    picks = picks or adapt.PICKS_CNV
    if picks == adapt.PICKS_CNV:
        x_exp = adapt.features_for_step(expert.epochs, step)
        x_new = adapt.features_for_step(new.epochs, step)
    else:
        x_exp = _features_for_custom_picks(expert.epochs, step, picks)
        x_new = _features_for_custom_picks(new.epochs, step, picks)
    return np.vstack([x_exp, x_new[new_train_mask]])


def _features_for_custom_picks(
    epochs: mne.Epochs,
    step: int,
    picks: list[str],
) -> np.ndarray:
    data = epochs.get_data(picks=picks) * 1e6
    time_indices = [
        int(np.argmin(np.abs(epochs.times - time_point)))
        for time_point in adapt.T_POINTS[:step]
    ]
    return np.hstack([
        data[:, channel_idx, :][:, time_indices]
        for channel_idx in range(len(picks))
    ])


def _riemann_trials_for_training(
    expert: adapt.SessionData,
    new: adapt.SessionData,
    new_train_mask: np.ndarray,
    step: int,
) -> np.ndarray:
    trials_exp = adapt.riemann_trials_for_step(expert.epochs, step)
    trials_new = adapt.riemann_trials_for_step(new.epochs, step)
    return np.concatenate([trials_exp, trials_new[new_train_mask]], axis=0)


def train_package(
    expert: adapt.SessionData,
    new: adapt.SessionData,
    new_train_mask: np.ndarray,
    control_model: str,
    output_subject: str,
    output_session: str,
) -> dict:
    if expert.rest_id != new.rest_id or expert.mi_id != new.mi_id:
        raise RuntimeError(
            "Los event_id internos de experto y sujeto nuevo no coinciden: "
            f"expert REST/MI=({expert.rest_id}, {expert.mi_id}) vs "
            f"new REST/MI=({new.rest_id}, {new.mi_id}). "
            "No se entrenará para evitar etiquetas mezcladas."
        )

    labels, sources, train_groups = _combine_training_data(
        expert,
        new,
        new_train_mask,
    )
    compact_picks = _validate_compact_picks()

    mdm_models = []
    mdm_templates = []
    mdm_recenter_refs = []
    mdm_centers = []
    skl_models = []
    compact_lda_models = []
    observer_skl_models = {
        "LR": [],
        "SVM": [],
    }

    for step, endpoint in enumerate(adapt.T_POINTS, start=1):
        print(
            f"   Training endpoint {step:02d}/{len(adapt.T_POINTS)} "
            f"(t={endpoint:.2f} s)",
            flush=True,
        )

        step_trials = _riemann_trials_for_training(
            expert,
            new,
            new_train_mask,
            step,
        )
        step_template = step_trials[labels == expert.mi_id].mean(axis=0)
        covariances = adapt.template_covariances(step_trials, step_template)
        recenter_ref = mean_riemann(
            covariances,
            tol=adapt.RIEMANN_MEAN_TOL,
            maxiter=adapt.RIEMANN_MEAN_MAXITER,
        )
        covariances_recentered = adapt.recenter_covariances(
            covariances,
            recenter_ref,
        )
        mdm = MDM(metric="riemann")
        mdm.fit(covariances_recentered, labels)
        mdm_models.append(mdm)
        mdm_templates.append(step_template)
        mdm_recenter_refs.append(recenter_ref)
        mdm_centers.append({
            label: mdm.covmeans_[idx].copy()
            for idx, label in enumerate(mdm.classes_)
        })

        x_step = _features_for_training(
            expert,
            new,
            new_train_mask,
            step,
            picks=adapt.PICKS_CNV,
        )
        lda = adapt.make_clf("LDA_shrink")
        lda.fit(x_step, labels)
        skl_models.append(lda)

        x_compact = _features_for_training(
            expert,
            new,
            new_train_mask,
            step,
            picks=compact_picks,
        )
        compact_lda = adapt.make_clf("LDA_shrink")
        compact_lda.fit(x_compact, labels)
        compact_lda_models.append(compact_lda)

        for observer_name, observer_models in observer_skl_models.items():
            observer = adapt.make_clf(observer_name)
            observer.fit(x_step, labels)
            observer_models.append(observer)

    n_new_calib = int(np.sum(new_train_mask))
    selected_new_runs = np.unique(new.groups[new_train_mask]).tolist()
    package = {
        "model_type": "M2_LDA_shrink_MDM",
        "is_maestro": False,
        "picks": adapt.PICKS_CNV,
        "t_points": adapt.T_POINTS.copy(),
        "t_start": adapt.T_START,
        "t_end": adapt.T_END,
        "n_timepoints": len(adapt.T_POINTS),
        "n_samples": int(round((adapt.T_END - adapt.T_START) * config.FS)) + 1,
        "REST_ID": expert.rest_id,
        "MI_ID": expert.mi_id,
        "subjects_train": [expert.subject, new.subject],
        "subject_calib": output_subject,
        "session_calib": output_session,
        "expert_subject": expert.subject,
        "expert_session": expert.session,
        "new_subject": new.subject,
        "new_session": new.session,
        "new_calibration_runs": selected_new_runs,
        "n_expert": int(len(expert.labels)),
        "n_new_calibration": n_new_calib,
        "n_total": int(len(labels)),
        "training_sources": sources.tolist(),
        "training_run_labels": train_groups.tolist(),
        "mdm_models": mdm_models,
        "mdm_templates": mdm_templates,
        "mdm_recenter_refs": mdm_recenter_refs,
        "mdm_recenter_mode": "train_riemann_mean",
        "mdm_centers": mdm_centers,
        "mdm_available": True,
        "cov_reg": adapt.RIEMANN_COV_REG,
        "skl_models": skl_models,
        "skl_control_name": "LDA_shrink",
        "compact_lda_models": compact_lda_models,
        "compact_lda_picks": compact_picks,
        "compact_lda_name": "LDA_shrink_3ch",
        "observer_skl_models": observer_skl_models,
        "observer_skl_names": list(observer_skl_models.keys()),
        "recommended_control_model": control_model,
        "full_window_observer_names": [
            "MDM",
            "LDA_shrink",
            "LDA_shrink_3ch",
            "LR",
            "SVM",
        ],
        "full_feature_count": len(adapt.PICKS_CNV) * len(adapt.T_POINTS),
        "compact_full_feature_count": len(compact_picks) * len(adapt.T_POINTS),
        "training_pipeline": (
            f"CAR + {adapt.EEG_L_FREQ:.1f}-{adapt.EEG_H_FREQ:.1f} Hz "
            f"Butterworth {adapt.EEG_IIR_PARAMS['order']}nd order, "
            f"{adapt.FILTER_PHASE}; abs reject ±{adapt.ABS_REJECT_UV:.0f} µV; "
            "no CSD"
        ),
        "training_scale": "uV",
        "online_note": (
            f"Expert {expert.subject}/{expert.session} + "
            f"{len(selected_new_runs)} run(s) {new.subject}/{new.session}; "
            f"recommended control={control_model}"
        ),
    }
    return package


def default_output_path(
    base_dir: str,
    new_subject: str,
    expert_subject: str,
    new_session: str,
    k_runs: int,
) -> str:
    model_dir = os.path.join(base_dir, f"sub-{new_subject}", "models")
    expert_short = expert_subject.replace("CNV_PILOT_SUBJ_", "SUBJ")
    filename = (
        f"sub-{new_subject}_model_expert-{expert_short}_"
        f"plus-{k_runs}runs_{new_session}.pkl"
    )
    return os.path.join(model_dir, filename)


def inspect_package(path: str) -> None:
    with open(path, "rb") as fh:
        pkg = pickle.load(fh)
    print("\n🔎  Verificación del paquete guardado")
    print(f"   model_type     : {pkg.get('model_type')}")
    print(f"   subject_calib  : {pkg.get('subject_calib')}")
    print(f"   session_calib  : {pkg.get('session_calib')}")
    print(f"   expert         : {pkg.get('expert_subject')} | {pkg.get('expert_session')}")
    print(f"   new subject    : {pkg.get('new_subject')} | {pkg.get('new_session')}")
    print(f"   new runs       : {pkg.get('new_calibration_runs')}")
    print(f"   n_total        : {pkg.get('n_total')}")
    print(f"   n_timepoints   : {pkg.get('n_timepoints')}")
    print(f"   picks          : {pkg.get('picks')}")
    print(f"   observers      : {pkg.get('observer_skl_names')}")
    print(f"   recommended    : {pkg.get('recommended_control_model')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate online model from expert + K new-subject runs."
    )
    parser.add_argument("--base-dir", default=getattr(config, "DATA_DIR", "."))
    parser.add_argument("--expert-subject", default=DEFAULT_EXPERT_SUBJECT)
    parser.add_argument("--expert-session", default=DEFAULT_EXPERT_SESSION)
    parser.add_argument("--new-subject", default=DEFAULT_NEW_SUBJECT)
    parser.add_argument("--new-session", default=DEFAULT_NEW_SESSION)
    parser.add_argument(
        "--new-runs",
        type=int,
        default=DEFAULT_NEW_RUNS,
        help="Number of first chronological new-subject runs used for calibration.",
    )
    parser.add_argument(
        "--control-model",
        default="MDM",
        choices=["MDM", "LDA_shrink", "LDA3", "LR", "SVM"],
        help="Recommended online control model saved in metadata.",
    )
    parser.add_argument(
        "--cov-fs",
        type=float,
        default=adapt.RIEMANN_MAX_FS,
        help="Max sampling rate for Riemann covariance. Default: 32 Hz.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .pkl output path. Defaults to new subject models folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/preprocess data but do not train or save package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapt.RIEMANN_MAX_FS = float(args.cov_fs)
    mne.set_log_level("WARNING")

    print("\n" + "=" * 90)
    print("💾  GENERATE ONLINE MODEL — EXPERT + NEW-SUBJECT CALIBRATION")
    print("=" * 90)
    print(
        f"   Expert      : {args.expert_subject} | {args.expert_session}\n"
        f"   New subject : {args.new_subject} | {args.new_session}\n"
        f"   New runs    : first {args.new_runs} run(s)\n"
        f"   Control rec.: {args.control_model}\n"
        f"   Cov Fs      : ≤{adapt.RIEMANN_MAX_FS:.0f} Hz\n"
        f"   Channels    : {adapt.PICKS_CNV}"
    )

    expert = adapt.load_session(
        args.base_dir,
        args.expert_subject,
        args.expert_session,
    )
    new = adapt.load_session(
        args.base_dir,
        args.new_subject,
        args.new_session,
    )
    new_train_mask = _select_first_new_runs(new, args.new_runs)
    selected_runs = np.unique(new.groups[new_train_mask]).tolist()

    print("\n📌  Training composition")
    print(f"   Expert epochs       : {len(expert.labels)}")
    print(f"   New calibration runs: {selected_runs}")
    print(f"   New calibration ep. : {int(np.sum(new_train_mask))}")
    print(f"   Total train epochs  : {len(expert.labels) + int(np.sum(new_train_mask))}")

    if args.dry_run:
        print("\n   Dry run complete — model not trained/saved.")
        return

    package = train_package(
        expert,
        new,
        new_train_mask,
        control_model=args.control_model,
        output_subject=args.new_subject,
        output_session=f"{args.new_session}_expert_plus_{args.new_runs}runs",
    )

    output_path = args.output or default_output_path(
        args.base_dir,
        args.new_subject,
        args.expert_subject,
        args.new_session,
        args.new_runs,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(package, fh)

    print("\n" + "=" * 90)
    print("✅  ONLINE MODEL SAVED")
    print("=" * 90)
    print(f"   Path       : {output_path}")
    print(f"   Trials     : {package['n_total']}")
    print(f"   Timepoints : {package['n_timepoints']}")
    print(f"   Control    : set config.PREP_CONTROL_MODEL = {args.control_model!r}")
    print("   Recenter   : recommended config.RECENTERING = 0 for first online test")
    print("=" * 90)
    inspect_package(output_path)


if __name__ == "__main__":
    main()
