"""
CNV BCI — ENTRENAMIENTO MODELO ONLINE (LDA_shrink con xDAWN)
xDAWN : filtro espacial supervisado que maximiza la SNR de la señal CNV
        Se ajusta sólo sobre los epochs MI para capturar la CNV
Pipeline: xDAWN (sobre PICKS) → amplitudes en T_POINTS → LDA Ledoit-Wolf
"""

import os
import pickle
import numpy as np
import mne
from mne.preprocessing import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUBJECTS = [
    ("CNV_PILOT_SUBJ_012", "S001OFFLINE"),
]

XDF_BASE  = "/home/lab-admin/Documents/CurrentStudy"
MODEL_DIR = "/home/lab-admin/Documents/CurrentStudy/sub-CNV_PILOT_SUBJ_012/models"
os.makedirs(MODEL_DIR, exist_ok=True)

PICKS              = ['FC5', 'C3', 'Cz', 'CP1', 'Fz']
T_START            = -2.5
T_END              =  0.0
N_TIMEPOINTS       =  11
T_POINTS           = np.linspace(T_START, T_END, N_TIMEPOINTS)
BP_LOW             =  0.1
BP_HIGH            =  1.0
REJECT_UV          =  150.0
N_XDAWN_COMPONENTS =  4   # componentes xDAWN (máx = len(PICKS) = 5)

CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
    "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]


# ============================================================
# PREPROCESAMIENTO  (idéntico al pipeline online)
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
# EXTRACCIÓN DE FEATURES — amplitud en T_POINTS
# ============================================================

def extract_features(epochs_obj, picks, t_points):
    """Features de canales raw (baseline para comparar)."""
    times    = epochs_obj.times
    t_idx    = [np.argmin(np.abs(times - t)) for t in t_points]
    ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
    ch_idx   = [ch_names.index(ch) for ch in picks if ch in ch_names]
    data     = epochs_obj.get_data() * 1e6   # µV
    X = np.hstack([data[:, ci, :][:, t_idx] for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def apply_xdawn_and_extract(epochs_obj, picks, t_points, xdawn_filters):
    """
    Aplica el filtro xDAWN y extrae features de los componentes resultantes.
    xdawn_filters: array (n_picks, n_components) — columnas = filtros espaciales
    """
    times    = epochs_obj.times
    t_idx    = [np.argmin(np.abs(times - t)) for t in t_points]
    ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
    ch_idx   = [ch_names.index(ch) for ch in picks if ch in ch_names]

    data = epochs_obj.get_data()[:, ch_idx, :] * 1e6  # (n_epochs, n_picks, n_times)

    # data_xd[epoch, component, time] = sum_ch W[ch, comp] * data[epoch, ch, time]
    data_xd = np.einsum('ect,cn->ent', data, xdawn_filters)  # (n_epochs, n_components, n_times)

    n_components = xdawn_filters.shape[1]
    X = np.hstack([data_xd[:, ci, :][:, t_idx] for ci in range(n_components)])
    y = epochs_obj.events[:, -1]
    return X, y


# ============================================================
# ENTRENAMIENTO
# ============================================================

print(f"\n{'='*60}")
print("🔄  CNV BCI — ENTRENAMIENTO MODELO ONLINE (LDA_shrink + xDAWN)")
print(f"{'='*60}")
print(f"   Sujetos    : {[s[0].split('_')[-1] for s in SUBJECTS]}")
print(f"   Canales    : {PICKS}")
print(f"   Ventana    : [{T_START}, {T_END}]s  ({N_TIMEPOINTS} puntos)")
print(f"   xDAWN      : {N_XDAWN_COMPONENTS} componentes sobre {len(PICKS)} canales")
print(f"   Modelo     : LDA con shrinkage automático (Ledoit-Wolf)")

mne.set_log_level("WARNING")

print(f"\n{'─'*40}")
print("📂  Cargando sujetos ...")
print(f"{'─'*40}")

X_all      = []
y_all      = []
epochs_all_list = []
event_dict = None

for subj, sess in SUBJECTS:
    epochs, event_dict = load_and_preprocess(subj, sess)
    epochs_all_list.append(epochs)

# Concatenar todos los epochs
epochs_combined = mne.concatenate_epochs(epochs_all_list)

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]

print(f"\n   Total: Rest={len(epochs_combined['Rest (100)'])} "
      f"MI={len(epochs_combined['MI (200)'])} trials")


# ============================================================
# BASELINE — features de canales raw (para comparar)
# ============================================================

print(f"\n{'─'*40}")
print("📊  Evaluando baseline (canales raw, sin xDAWN) ...")
print(f"{'─'*40}")

X_raw, y_raw = extract_features(epochs_combined, PICKS, T_POINTS)

model_baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_baseline = cross_val_score(model_baseline, X_raw, y_raw, cv=cv, scoring="accuracy")

print(f"   Features   : {X_raw.shape[1]}  ({len(PICKS)} ch × {N_TIMEPOINTS} pts)")
print(f"   CV accuracy: {scores_baseline.mean()*100:.1f}% ± {scores_baseline.std()*100:.1f}%")
print(f"   Por fold   : {[f'{s*100:.1f}%' for s in scores_baseline]}")


# ============================================================
# AJUSTAR xDAWN sobre epochs MI
# ============================================================

print(f"\n{'─'*40}")
print("🔧  Ajustando xDAWN sobre epochs MI ...")
print(f"{'─'*40}")

# Seleccionar sólo los canales PICKS para xDAWN
epochs_picks = epochs_combined.copy().pick_channels(
    [ch for ch in PICKS if ch in epochs_combined.ch_names]
)

# Ajustar xDAWN sobre epochs MI únicamente
# (maximiza la SNR de la respuesta evocada CNV)
epochs_mi_only = epochs_picks["MI (200)"].copy()

xdawn = Xdawn(n_components=N_XDAWN_COMPONENTS, correct_overlap=False)
xdawn.fit(epochs_mi_only)

# filters_ es un dict {nombre_evento: array (n_picks, n_components)}
event_key = list(xdawn.filters_.keys())[0]
W = xdawn.filters_[event_key][:, :N_XDAWN_COMPONENTS]
print(f"   Evento ajuste : {event_key}")
print(f"   Filtro xDAWN  : {W.shape[0]} canales → {W.shape[1]} componentes")
print(f"   Canales usados: {[ch for ch in PICKS if ch in epochs_combined.ch_names]}")


# ============================================================
# xDAWN + LDA — evaluación y entrenamiento
# ============================================================

print(f"\n{'─'*40}")
print("📊  Evaluando xDAWN + LDA ...")
print(f"{'─'*40}")

X_xd, y_xd = apply_xdawn_and_extract(epochs_picks, PICKS, T_POINTS, W)

model_xdawn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])

scores_xdawn = cross_val_score(model_xdawn, X_xd, y_xd, cv=cv, scoring="accuracy")

print(f"   Features   : {X_xd.shape[1]}  ({N_XDAWN_COMPONENTS} comp × {N_TIMEPOINTS} pts)")
print(f"   CV accuracy: {scores_xdawn.mean()*100:.1f}% ± {scores_xdawn.std()*100:.1f}%")
print(f"   Por fold   : {[f'{s*100:.1f}%' for s in scores_xdawn]}")

delta = (scores_xdawn.mean() - scores_baseline.mean()) * 100
print(f"\n   Δ xDAWN vs baseline: {delta:+.1f}%")


# ============================================================
# ENTRENAR MODELO FINAL EN TODOS LOS DATOS
# ============================================================

print(f"\n{'─'*40}")
print("🧠  Entrenando modelo final (todos los datos) ...")
print(f"{'─'*40}")

model_xdawn.fit(X_xd, y_xd)
train_acc = np.mean(model_xdawn.predict(X_xd) == y_xd) * 100
print(f"   Clases    : {np.unique(y_xd)}")
print(f"   Train acc : {train_acc:.1f}%  (referencia — usar CV para evaluar)")

model_path = os.path.join(
    MODEL_DIR, f"sub-{SUBJECTS[0][0]}_model_LDA_xDAWN.pkl")

picks_used = [ch for ch in PICKS if ch in epochs_combined.ch_names]

save_obj = {
    "model"             : model_xdawn,
    "model_type"        : "LDA_shrink_xDAWN",
    "picks"             : picks_used,
    "xdawn_filter"      : W,           # (n_picks, n_components) — aplicar online
    "xdawn_n_components": N_XDAWN_COMPONENTS,
    "t_points"          : T_POINTS,
    "t_start"           : T_START,
    "t_end"             : T_END,
    "n_samples"         : int(round((T_END - T_START) * config.FS)) + 1,
    "subjects"          : [s[0] for s in SUBJECTS],
}

with open(model_path, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\n✅  Modelo guardado: {model_path}")
print(f"{'='*60}")
print("RESUMEN COMPARATIVO")
print(f"{'─'*40}")
print(f"   Baseline (raw)  : {scores_baseline.mean()*100:.1f}% ± {scores_baseline.std()*100:.1f}%")
print(f"   xDAWN + LDA     : {scores_xdawn.mean()*100:.1f}% ± {scores_xdawn.std()*100:.1f}%")
print(f"   Mejora          : {delta:+.1f}%")
print(f"{'='*60}")
