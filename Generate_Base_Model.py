"""
================================================================================
Generate_Base_Model.py
================================================================================
Genera modelos CNV BCI en dos modos:

  MODO MAESTRO  (CALIB_SUBJECT = None)
    Entrenamiento : SUBJ_001/003/004/005/006 + SUBJ_012 S001OFFLINE
    Salida        : sub-CNV_MAESTRO/models/maestro_model.pkl
    Uso           : base permanente, nunca se modifica

  MODO PER-SUJETO  (CALIB_SUBJECT = (base_dir, sujeto, sesión))
    Entrenamiento : mismos sujetos del maestro + datos de calibración del día
    Salida        : sub-{calib_subj}/models/sub-{calib_subj}_model.pkl
    Uso           : modelo operativo para la sesión online del día

Clasificadores guardados:
  MDM Riemanniano M2 : 11 modelos por paso temporal → decisión online
  LDA_shrink M2      : 11 modelos por paso temporal → validación paralela

Canales   : FC1, Cz, CP1
Ventana   : -2.5 → 0.0 s, 11 pasos
Pipeline  : avg-ref → notch 60 Hz → BP 0.1–1.0 Hz
================================================================================
"""

import os
import pickle
import numpy as np
import bci_runtime_env
import mne

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

try:
    from pyriemann.classification import MDM
    PYRIEMANN_OK = True
except ImportError:
    PYRIEMANN_OK = False
    print("⚠️   pyriemann no instalado — MDM desactivado")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN
# ============================================================

XDF_CNV  = "/home/lab-admin/Documents/CNVStudy"
XDF_CURR = "/home/lab-admin/Documents/CurrentStudy"

# Cada entrada: (base_dir, subject, session)
TRAIN_SUBJECTS = [
    (XDF_CNV,  "CNV_PILOT_SUBJ_001", "S001OFFLINE_GLOVE"),
    (XDF_CNV,  "CNV_PILOT_SUBJ_003", "S001OFFLINE_GLOVE"),
    (XDF_CNV,  "CNV_PILOT_SUBJ_004", "S001OFFLINE_GLOVE"),
    (XDF_CNV,  "CNV_PILOT_SUBJ_005", "S001OFFLINE_GLOVE"),
    (XDF_CNV,  "CNV_PILOT_SUBJ_006", "S001OFFLINE_GLOVE"),
    (XDF_CURR, "CNV_PILOT_SUBJ_012", "S001OFFLINE"),
]

# None        → modo MAESTRO   (genera maestro_model.pkl)
# (dir, subj, sess) → modo PER-SUJETO (genera sub-{subj}_model.pkl)
CALIB_SUBJECT = (XDF_CURR, "CNV_PILOT_SUBJ_012", "S002OFFLINE")  # modo PER-SUJETO

MAESTRO_DIR = os.path.join(XDF_CURR, "sub-CNV_MAESTRO", "models")

if CALIB_SUBJECT is None:
    MODEL_SAVE_PATH = os.path.join(MAESTRO_DIR, "maestro_model.pkl")
else:
    _, _cs, _css = CALIB_SUBJECT
    MODEL_SAVE_PATH = os.path.join(
        XDF_CURR, f"sub-{_cs}", "models", f"sub-{_cs}_model.pkl"
    )

PICKS        = ['FC1', 'Cz', 'CP1']
T_START      = -2.5
T_END        =  0.0
N_TIMEPOINTS =  11
T_POINTS     = np.linspace(T_START, T_END, N_TIMEPOINTS)

BP_LOW, BP_HIGH = 0.1, 1.0
REJECT_UV       = 150.0
COV_REG         = 1e-4

CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
    "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def load_and_preprocess(base_dir, subject, session):
    xdf_dir   = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}", "eeg/")
    xdf_files = sorted([os.path.join(xdf_dir, f)
                        for f in os.listdir(xdf_dir)
                        if f.endswith(".xdf") and "_old" not in f])
    if not xdf_files:
        raise FileNotFoundError(f"No XDF en: {xdf_dir}")

    raw_list = []
    for xdf_file in xdf_files:
        eeg_s, marker_s   = load_xdf(xdf_file)
        eeg_data          = np.array(eeg_s["time_series"]).T
        eeg_timestamps    = np.array(eeg_s["time_stamps"])
        channel_names     = get_channel_names_from_xdf(eeg_s)

        marker_data       = np.array([int(v[0]) for v in marker_s["time_series"]])
        marker_timestamps = np.array(marker_s["time_stamps"])
        keep              = np.isin(marker_data, TARGET_MARKERS)
        marker_data       = marker_data[keep]
        marker_timestamps = marker_timestamps[keep]

        valid_ch        = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx       = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info    = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

        renames = {k: v for k, v in RENAME_DICT.items() if k in raw_tmp.ch_names}
        if renames:
            raw_tmp.rename_channels(renames)

        raw_tmp.set_montage(mne.channels.make_standard_montage("standard_1020"),
                            on_missing="warn")
        for ch in CHANNELS_TO_DROP:
            if ch in raw_tmp.ch_names:
                raw_tmp.drop_channels([ch])

        t0    = eeg_timestamps[0]
        annot = mne.Annotations(
            onset       = marker_timestamps - t0,
            duration    = np.zeros(len(marker_data)),
            description = [str(m) for m in marker_data],
            orig_time   = None,
        )
        raw_tmp.set_annotations(annot)
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw.filter(
        l_freq=BP_LOW, h_freq=BP_HIGH,
        method="iir", iir_params=dict(order=2, ftype="butter"),
        phase="forward", picks="eeg", verbose=False,
    )

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)":   event_id_map["200"],
    }

    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=5.0,
        baseline=(-3.0, -2.0),
        reject=None, flat=None,
        preload=True, detrend=None, verbose=False,
    )

    ch_names_eeg = epochs.copy().pick_types(eeg=True).ch_names
    picks_avail  = [ch for ch in PICKS if ch in ch_names_eeg]
    drop_idx = []
    if picks_avail:
        pick_idx  = [ch_names_eeg.index(ch) for ch in picks_avail]
        data_cnv  = epochs.get_data()[:, pick_idx, :] * 1e6
        pp        = data_cnv.max(axis=2) - data_cnv.min(axis=2)
        drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
        drop_idx  = np.where(drop_mask)[0].tolist()
        epochs.drop(drop_idx, reason="MANUAL_REJECT")

    print(f"   {subject.split('_')[-1]} / {session}: "
          f"Rest={len(epochs['Rest (100)'])}  "
          f"MI={len(epochs['MI (200)'])}  "
          f"rechazados={len(drop_idx)}")
    return epochs, event_dict


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def get_eeg_data(epochs_obj, picks):
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0
    except Exception:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    return data, ch_idx, scale


def extract_features_step(epochs_obj, picks, t_points, step):
    """Amplitud en los primeros `step` puntos temporales (M2 acumulativo)."""
    times = epochs_obj.times
    t_idx = [np.argmin(np.abs(times - t)) for t in t_points[:step]]
    data, ch_idx, scale = get_eeg_data(epochs_obj, picks)
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def extract_raw_step(epochs_obj, picks, t_start, t_end_step):
    """Señal raw desde t_start hasta t_end_step para covarianzas MDM."""
    times  = epochs_obj.times
    t_mask = (times >= t_start) & (times <= t_end_step)
    data, ch_idx, scale = get_eeg_data(epochs_obj, picks)
    y = epochs_obj.events[:, -1]
    return data[:, ch_idx, :][:, :, t_mask] * scale, y


# ============================================================
# COVARIANZAS RIEMANNIANAS
# ============================================================

def compute_cov_trace_norm(data_3d):
    n, n_ch, n_t = data_3d.shape
    covs = np.zeros((n, n_ch, n_ch))
    for i in range(n):
        X  = data_3d[i].T
        C  = X.T @ X
        tr = np.trace(C)
        C  = C / tr if tr > 0 else C
        C += COV_REG * np.eye(n_ch)
        covs[i] = C
    return covs


def build_template_covs(data_3d, template):
    tmpl_rep = np.tile(template[np.newaxis], (data_3d.shape[0], 1, 1))
    extended = np.concatenate([data_3d, tmpl_rep], axis=1)
    return compute_cov_trace_norm(extended)


def make_lda():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


# ============================================================
# MAIN
# ============================================================

_mode_label = "MAESTRO" if CALIB_SUBJECT is None else "PER-SUJETO"
_train_ids  = [s.split('_')[-1] for _, s, _ in TRAIN_SUBJECTS]
_calib_id   = CALIB_SUBJECT[1].split('_')[-1] if CALIB_SUBJECT else "—"
_calib_sess = CALIB_SUBJECT[2] if CALIB_SUBJECT else "—"

print(f"\n{'='*60}")
print(f"🧠  GENERATE BASE MODEL — CNV BCI  [{_mode_label}]")
print(f"{'='*60}")
print(f"   Canales   : {PICKS}")
print(f"   Ventana   : [{T_START}, {T_END}]s  ({N_TIMEPOINTS} pasos)")
print(f"   Entren.   : {_train_ids}")
print(f"   Calibr.   : {_calib_id} / {_calib_sess}")
print(f"   Salida    : {MODEL_SAVE_PATH}")
mne.set_log_level("WARNING")

# ── Construir lista completa de sujetos a cargar ─────────────
all_entries = list(TRAIN_SUBJECTS)
if CALIB_SUBJECT is not None:
    all_entries.append(CALIB_SUBJECT)

# Clave única por (sujeto, sesión) para evitar colisiones cuando
# el mismo sujeto aparece en train y calib con sesiones distintas
all_epochs  = {}
calib_key   = None
event_dict  = None

# ── Carga sujetos de entrenamiento ──────────────────────────
print(f"\n{'─'*45}")
print("📂  Cargando sujetos de entrenamiento ...")
for entry in TRAIN_SUBJECTS:
    base_dir, subj, sess = entry
    epochs, event_dict = load_and_preprocess(base_dir, subj, sess)
    all_epochs[f"{subj}_{sess}"] = epochs

# ── Carga calibración (si aplica) ───────────────────────────
if CALIB_SUBJECT is not None:
    base_dir, subj, sess = CALIB_SUBJECT
    print(f"\n{'─'*45}")
    print(f"📂  Cargando calibración ({subj.split('_')[-1]} / {sess}) ...")
    epochs, event_dict = load_and_preprocess(base_dir, subj, sess)
    calib_key = f"{subj}_{sess}"
    all_epochs[calib_key] = epochs

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]

# ── Preparar features por paso M2 ───────────────────────────
print(f"\n{'─'*45}")
print("🔧  Preparando features M2 ...")

X_steps   = [[] for _ in range(N_TIMEPOINTS)]
raw_steps = [[] for _ in range(N_TIMEPOINTS)]
y_list    = []

for ep in all_epochs.values():
    _, y_subj = extract_features_step(ep, PICKS, T_POINTS, N_TIMEPOINTS)
    y_list.append(y_subj)
    for step in range(N_TIMEPOINTS):
        Xs, _ = extract_features_step(ep, PICKS, T_POINTS, step + 1)
        X_steps[step].append(Xs)
        rs, _ = extract_raw_step(ep, PICKS, T_START, T_POINTS[step])
        raw_steps[step].append(rs)

y_all     = np.concatenate(y_list)
X_steps   = [np.vstack(X_steps[s])   for s in range(N_TIMEPOINTS)]
raw_steps = [np.vstack(raw_steps[s]) for s in range(N_TIMEPOINTS)]

n_total  = len(y_all)
n_rest   = int(np.sum(y_all == REST_ID))
n_mi     = int(np.sum(y_all == MI_ID))
n_calib  = len(all_epochs[calib_key].events) if calib_key else 0
n_base   = n_total - n_calib

print(f"   Total: {n_total} trials  (REST={n_rest}, MI={n_mi})")
print(f"   Desglose: maestro={n_base}  calibración={n_calib}")

# ── Entrenar MDM M2 (11 modelos) — decisión online ──────────
mdm_models    = []
mdm_templates = []
mdm_centers   = []

if PYRIEMANN_OK:
    print(f"\n{'─'*45}")
    print("🔴  Entrenando MDM Riemanniano M2 (11 pasos) ...")
    for step in range(N_TIMEPOINTS):
        template = raw_steps[step].mean(axis=0)
        covs     = build_template_covs(raw_steps[step], template)
        mdm      = MDM(metric="riemann")
        try:
            mdm.fit(covs, y_all)
            centers = {
                label: mdm.covmeans_[i].copy()
                for i, label in enumerate(mdm.classes_)
            }
            status = "OK"
        except Exception as e:
            centers = {}
            status  = f"WARN: {e}"
        mdm_models.append(mdm)
        mdm_templates.append(template)
        mdm_centers.append(centers)
        print(f"   paso {step+1:2d} | t={T_POINTS[step]:+.2f}s | {status}")
else:
    print("\n   MDM desactivado (pyriemann no disponible)")

# ── Entrenar LDA_shrink M2 (11 modelos) — validación ────────
print(f"\n{'─'*45}")
print("🔵  Entrenando LDA_shrink M2 (11 pasos) ...")

skl_models = []
for step in range(N_TIMEPOINTS):
    clf = make_lda()
    clf.fit(X_steps[step], y_all)
    acc = np.mean(clf.predict(X_steps[step]) == y_all) * 100
    print(f"   paso {step+1:2d} | t={T_POINTS[step]:+.2f}s | "
          f"{X_steps[step].shape[1]:3d} features | train acc={acc:.1f}%")
    skl_models.append(clf)

# ── Guardar modelo ───────────────────────────────────────────
print(f"\n{'─'*45}")
print("💾  Guardando modelo ...")

save_obj = {
    "model_type"     : "M2_LDA_shrink_MDM",
    "is_maestro"     : CALIB_SUBJECT is None,
    "picks"          : PICKS,
    "t_points"       : T_POINTS,
    "t_start"        : T_START,
    "t_end"          : T_END,
    "n_timepoints"   : N_TIMEPOINTS,
    "n_samples"      : int(round((T_END - T_START) * config.FS)) + 1,
    "REST_ID"        : REST_ID,
    "MI_ID"          : MI_ID,
    "subjects_train" : [s for _, s, _ in TRAIN_SUBJECTS],
    "subject_calib"  : CALIB_SUBJECT[1] if CALIB_SUBJECT else None,
    "session_calib"  : CALIB_SUBJECT[2] if CALIB_SUBJECT else None,
    "n_total"        : n_total,
    # MDM M2 — 11 modelos + templates + centroides (decisión online)
    "mdm_models"     : mdm_models    if PYRIEMANN_OK else [],
    "mdm_templates"  : mdm_templates if PYRIEMANN_OK else [],
    "mdm_centers"    : mdm_centers   if PYRIEMANN_OK else [],
    "mdm_available"  : PYRIEMANN_OK,
    "cov_reg"        : COV_REG,
    # LDA_shrink M2 — 11 modelos (validación paralela)
    "skl_models"     : skl_models,
}

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\n✅  Modelo guardado: {MODEL_SAVE_PATH}")
print(f"{'='*60}")
print(f"   Modo          : {_mode_label}")
print(f"   MDM M2        : {len(mdm_models)} modelos  (decisión online)")
print(f"   LDA_shrink M2 : {len(skl_models)} modelos  (validación)")
print(f"   Canales       : {PICKS}")
print(f"   Trials totales: {n_total}  (REST={n_rest}, MI={n_mi})")
print(f"   Desglose      : maestro={n_base}  calibración={n_calib}")
print(f"{'='*60}")
