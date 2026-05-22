"""
================================================================================
CNV BCI — ENTRENAMIENTO MODELO ONLINE
================================================================================
Entrena MDM Riemanniano para CNV_PILOT_SUBJ_010.
Guarda el modelo + template para uso en ExperimentDriver_Online.py

Sujeto  : CNV_PILOT_SUBJ_010
Sesión  : S001
Canales : FC5, FC1, Cz, CP1, Fz
Modelo  : MDM Riemanniano con template matching (Racz 2023)
Ventana : [-2.5s, 0.0s] — M1 estático
================================================================================
"""

import os
import pickle
import numpy as np
import mne

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf
from pyriemann.classification import MDM

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUBJECTS = [
    ("CNV_PILOT_SUBJ_011", "S001_OFF"),
]

XDF_BASE  = "/home/lab-admin/Documents/CNVStudy"
MODEL_DIR = "/home/lab-admin/Documents/CurrentStudy/sub-CNV_PILOT_SUBJ_011/models"
os.makedirs(MODEL_DIR, exist_ok=True)

PICKS         = ['FC5', 'C3', 'Cz', 'CP1', 'Fz']
T_START       = -2.5
T_END         =  0.0
COV_REG       =  1e-4
BP_LOW        =  0.1
BP_HIGH       =  1.0
REJECT_UV     =  150.0

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

def load_and_preprocess(subject, session):
    xdf_dir   = os.path.join(XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/")
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

        existing_renames = {k: v for k, v in RENAME_DICT.items()
                            if k in raw_tmp.ch_names}
        if existing_renames:
            raw_tmp.rename_channels(existing_renames)

        raw_tmp.set_montage(mne.channels.make_standard_montage("standard_1020"),
                            on_missing="warn")

        drop_targets = [ch for ch in CHANNELS_TO_DROP if ch in raw_tmp.ch_names]
        if drop_targets:
            raw_tmp.drop_channels(drop_targets)

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
        phase="forward", picks="eeg", verbose=False
    )

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)":   event_id_map["200"],
    }

    epochs_all = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=5.0,
        baseline=(-3.0, -2.0),
        reject=None, flat=None,
        preload=True, detrend=None, verbose=False,
    )

    # Rechazo manual
    ch_names_eeg = epochs_all.copy().pick_types(eeg=True).ch_names
    picks_avail  = [ch for ch in PICKS if ch in ch_names_eeg]
    drop_idx = []
    if picks_avail:
        pick_idx  = [ch_names_eeg.index(ch) for ch in picks_avail]
        data_cnv  = epochs_all.get_data()[:, pick_idx, :] * 1e6
        pp        = data_cnv.max(axis=2) - data_cnv.min(axis=2)
        drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
        drop_idx  = np.where(drop_mask)[0].tolist()
        epochs_all.drop(drop_idx, reason="MANUAL_REJECT")

    print(f"   {subject.split('_')[-1]}: "
          f"Rest={len(epochs_all['Rest (100)'])} "
          f"MI={len(epochs_all['MI (200)'])} "
          f"rechazados={len(drop_idx)}")

    return epochs_all, event_dict


# ============================================================
# EXTRACCIÓN DE DATOS CRUDOS
# ============================================================

def extract_raw_data(epochs_obj, picks, tmin, tmax):
    times  = epochs_obj.times
    t_mask = (times >= tmin) & (times <= tmax)
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0
    except ValueError:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    y      = epochs_obj.events[:, -1]
    return data[:, ch_idx, :][:, :, t_mask] * scale, y


# ============================================================
# COVARIANZA CON TEMPLATE MATCHING (Racz 2023)
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
    """Covarianza extendida: concatena cada trial con el template promedio."""
    tmpl_rep = np.tile(template[np.newaxis], (data_3d.shape[0], 1, 1))
    extended = np.concatenate([data_3d, tmpl_rep], axis=1)
    return compute_cov_trace_norm(extended)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print(f"\n{'='*60}")
print("🔄  CNV BCI — ENTRENAMIENTO MODELO ONLINE (M1 estático)")
print(f"{'='*60}")
print(f"   Sujetos : {[s[0].split('_')[-1] for s in SUBJECTS]}")
print(f"   Canales : {PICKS}")
print(f"   Ventana : [{T_START}, {T_END}]s")
print(f"   Modelo  : MDM Riemanniano + template matching (Racz 2023)")

mne.set_log_level("WARNING")

print(f"\n{'─'*40}")
print("📂  Cargando sujetos ...")
print(f"{'─'*40}")

raw_all    = []
y_all      = []
event_dict = None

for subj, sess in SUBJECTS:
    epochs, event_dict = load_and_preprocess(subj, sess)
    raw_data, y        = extract_raw_data(epochs, PICKS, T_START, T_END)
    raw_all.append(raw_data)
    y_all.append(y)

raw_tr = np.vstack(raw_all)
y_tr   = np.concatenate(y_all)

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]

print(f"\n   Total: Rest={int(np.sum(y_tr==REST_ID))} "
      f"MI={int(np.sum(y_tr==MI_ID))} trials")
print(f"   Shape datos crudos: {raw_tr.shape}")

# Template — promedio de todos los trials
print(f"\n{'─'*40}")
print("⚙️   Calculando template y covarianzas ...")
print(f"{'─'*40}")

template = raw_tr.mean(axis=0)
print(f"   Template shape: {template.shape}")

covs_tr  = build_template_covs(raw_tr, template)
print(f"   Covarianzas shape: {covs_tr.shape}")

# Entrenar MDM
print(f"\n{'─'*40}")
print("🧠  Entrenando MDM Riemanniano ...")
print(f"{'─'*40}")

model = MDM(metric="riemann")
model.fit(covs_tr, y_tr)

print(f"   Clases    : {model.classes_}")
print(f"   Centroides: {model.covmeans_.shape}")

# Guardar modelo + template + metadata
model_path = os.path.join(
    MODEL_DIR, f"sub-{SUBJECTS[0][0]}_model.pkl")

save_obj = {
    "model"    : model,
    "template" : template,
    "picks"    : PICKS,
    "t_start"  : T_START,
    "t_end"    : T_END,
    "cov_reg"  : COV_REG,
    "subjects" : [s[0] for s in SUBJECTS],
    "n_samples": raw_tr.shape[2],
}

with open(model_path, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\n✅  Modelo guardado: {model_path}")
print(f"   Canales   : {PICKS}")
print(f"   Trials    : {len(y_tr)} total")
print(f"   Muestras  : {raw_tr.shape[2]} por trial ({T_START} a {T_END}s a 512Hz)")
print("="*60)