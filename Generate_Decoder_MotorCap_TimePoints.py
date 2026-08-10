"""
================================================================================
CNV BCI PIPELINE — Central Motor Cap Time-Point Decoder
================================================================================
Subject:    CNV_PILOT_SUBJ_012
Session:    S008_OFFLINE
Classes:    Rest (100) vs Motor Imagery / Move (200)

MEJORAS vs versión anterior:
  SEÑAL:
  [S1] Re-referencia promedio (CAR)
  [S2] Notch 60 Hz en EEG (elimina ruido de línea de potencia)
  [S3] Análisis cuantitativo de amplitud CNV en ventana [-1.5, 0] s
  [S4] Test t pareado por canal para significancia estadística Rest vs MI

  VISUALIZACIONES:
  [V1] Drop log — ver qué trials se rechazaron y por canal
  [V2] Butterfly plot EEG pre-rechazo — detectar outliers visualmente
  [V3] Plot diferencia MI − Rest con banda de confianza al 95%
  [V4] Anotación de amplitud pico CNV en cada subplot
  [V5] Onset EMG: raster de latencias individuales + histograma
  [V6] vlim de topomaps calculado de los datos reales (no fijo)
================================================================================
"""

import os
import pickle
import numpy as np
import bci_runtime_env
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import config  # Debe definir config.FS (ej. 512)
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# 1. IDENTIDAD Y CONFIGURACIÓN
# ============================================================
subject = "CNV_PILOT_SUBJ_022"
session  = "S001_OFFLINE"
base_dir = getattr(config, "DATA_DIR", "/home/lab-admin/Documents/CNVStudy")

# Modo exploratorio: analiza y muestra resultados sin escribir un modelo.
# Actívalo solo cuando la sesión completa esté lista para entrenamiento.
SAVE_ONLINE_MODEL = False 

CHANNELS_TO_DROP        = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
CHANNELS_TO_INTERPOLATE = []

PICKS_CNV = [
    'FC3', 'FC1', 'FCz',
    'C3',  'C1',  'Cz',
    'CP3', 'CP1', 'CPz', 
]

# Selección compacta congelada después de la validación anidada de S012.
# Se guarda únicamente como observador online; no controla el feedback.
COMPACT_LDA_PICKS = ['FCz', 'C3', 'CP3']

# Ventana de análisis CNV (negatividad pre-movimiento)
CNV_WINDOW = (-2.0, 0.0)   # segundos
ERP_XLIM = (-3.0, 2.0)     # segundos, solo para visualización ERP
ERP_YLIM = (-5.0, 5.0)   # µV, solo para visualización
# Solo afecta Figure 2/3. Modelos/entrenamiento siguen usando CAR + notch + BPF.
# Opciones: "CAR", "NO_CAR", "CAR_LAPLACIAN"
VISUAL_EEG_REFERENCE_MODE = "CAR"

# Banda lenta para MRCP/CNV.
EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0
EEG_IIR_PARAMS = dict(order=4, ftype="butter")

# El artefacto de FES es pulsátil y de banda ancha: no se puede eliminar con
# un simple notch en la frecuencia de estimulación. Si el RMS pre-trigger
# sigue siendo excesivo después del filtrado EMG, la latencia se considera
# no estimable y no se dibuja un onset por defecto.
EMG_L_FREQ = 20.0
EMG_H_FREQ = 200.0
EMG_BASELINE_WINDOW = (-2.0, -0.1)
EMG_BASELINE_RMS_MAX_UV = 150.0
MIN_EMG_ONSET_DETECTIONS = 5  # no dibujar onset EMG si hay muy pocos trials detectados

# Umbrales iniciales para EEG después de CAR y filtro lento.
REJECT_THRESHOLD = dict(eeg=100e-6)
FLAT_THRESHOLD   = dict(eeg=0.1e-6)

RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ":  "Pz",  "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]
INTERTRIAL_MARKERS = [600, 620]
INTERTRIAL_DURATION = 3.0
INTERTRIAL_PLOT_WINDOW = (-2.5, 0.5)
# S008/S009 were recorded with 2.0 s of preparation and without 600/620.
LEGACY_PRETRIAL_DURATION = 2.0

xdf_dir = os.path.join(
    base_dir,
    f"sub-{subject}", f"ses-{session}", "eeg/"
)
if not os.path.isdir(xdf_dir):
    raise FileNotFoundError(f"XDF directory does not exist: {xdf_dir}")

xdf_files = sorted(
    [os.path.join(xdf_dir, f) for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
)
if not xdf_files:
    raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")

print(f"📂  Processing {len(xdf_files)} XDF file(s) — subject: {subject} | session: {session}")
mne.set_log_level("WARNING")


# ============================================================
# 2. CARGA Y PREPROCESAMIENTO POR ARCHIVO
# ============================================================
raw_list = []
event_run_labels = []

for run_idx, xdf_file in enumerate(xdf_files, start=1):
    print(f"   └─ Loading: {os.path.basename(xdf_file)}")
    eeg_s, marker_s = load_xdf(xdf_file)

    eeg_data       = np.array(eeg_s["time_series"]).T
    eeg_timestamps = np.array(eeg_s["time_stamps"])
    channel_names  = get_channel_names_from_xdf(eeg_s)

    marker_data_all = np.array([
        int(round(float(np.ravel(v)[0])))
        for v in marker_s["time_series"]
    ])
    marker_timestamps_all = np.array(marker_s["time_stamps"])

    keep = np.isin(
        marker_data_all,
        TARGET_MARKERS + INTERTRIAL_MARKERS,
    )
    marker_data = marker_data_all[keep]
    marker_timestamps = marker_timestamps_all[keep]
    event_run_labels.extend([run_idx] * len(marker_data))

    valid_ch        = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
    valid_idx       = [channel_names.index(ch) for ch in valid_ch]
    eeg_data_subset = eeg_data[valid_idx, :] / 1e6

    info    = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
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

    if CHANNELS_TO_INTERPOLATE:
        raw_tmp.info["bads"] = [ch for ch in CHANNELS_TO_INTERPOLATE if ch in raw_tmp.ch_names]
        raw_tmp.interpolate_bads(reset_bads=True, verbose=False)

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
print(f"✅  Raw concatenado — {raw.n_times / raw.info['sfreq']:.1f} s totales")


# ============================================================
# 3. DETECCIÓN DE EVENTOS
# ============================================================
events_all, event_id_map = mne.events_from_annotations(raw, verbose=False)
event_run_labels = np.asarray(event_run_labels, dtype=int)
if len(event_run_labels) != len(events_all):
    raise RuntimeError(
        "No se pudo alinear cada evento con su XDF de origen: "
        f"{len(event_run_labels)} etiquetas para {len(events_all)} eventos."
    )
event_dict = {
    "Rest (100)": event_id_map["100"],
    "MI (200)":   event_id_map["200"],
}
target_event_mask = np.isin(
    events_all[:, 2],
    list(event_dict.values()),
)
events = events_all[target_event_mask]
event_run_labels = event_run_labels[target_event_mask]
mi_id = event_id_map["200"]
print(f"📌  Eventos — Rest: {np.sum(events[:,2]==event_dict['Rest (100)'])}  |"
      f"  MI: {np.sum(events[:,2]==mi_id)}")


# ============================================================
# 4. DETECCIÓN DE ONSET EMG (AUX7)
# ============================================================
print("\n💪  Calculando latencia de onset EMG desde AUX7 ...")

avg_emg_onset = np.nan
std_emg_onset = np.nan
n_detected    = 0
all_onsets    = []
emg_fes_contaminated = False
emg_baseline_rms_uv = np.nan

if "AUX7" in raw.ch_names:
    raw_emg = raw.copy().pick(["emg"])
    raw_emg.filter(l_freq=EMG_L_FREQ, h_freq=EMG_H_FREQ, picks="all",
                   method="iir", phase="forward", verbose=False)
    raw_emg.notch_filter(freqs=[60.0], picks="all", verbose=False)

    # Guardamos la señal filtrada sin rectificar para el butterfly plot
    raw_emg_filt = raw_emg.copy()

    raw_env       = raw_emg.copy()
    raw_env._data = np.abs(raw_emg.get_data())
    raw_env.filter(l_freq=None, h_freq=10.0, picks="all",
                   method="iir", phase="forward", verbose=False)

    epochs_emg = mne.Epochs(
        raw_env, events, event_id={"MI": mi_id},
        tmin=-2.0, tmax=5.0, baseline=None, preload=True, verbose=False
    )
    epochs_emg_filt = mne.Epochs(
        raw_emg_filt, events, event_id={"MI": mi_id},
        tmin=-2.0, tmax=5.0, baseline=None, preload=True, verbose=False
    )

    emg_times      = epochs_emg.times
    emg_data       = epochs_emg.get_data()[:, 0, :] * 1e6
    emg_filt_data  = epochs_emg_filt.get_data()[:, 0, :] * 1e6

    baseline_mask_emg = (
        (emg_times >= EMG_BASELINE_WINDOW[0])
        & (emg_times <= EMG_BASELINE_WINDOW[1])
    )
    baseline_rms_by_trial = np.sqrt(
        np.mean(emg_filt_data[:, baseline_mask_emg] ** 2, axis=1)
    )
    emg_baseline_rms_uv = float(np.median(baseline_rms_by_trial))
    emg_fes_contaminated = emg_baseline_rms_uv > EMG_BASELINE_RMS_MAX_UV

    if emg_fes_contaminated:
        print(
            "⚠️   EMG no interpretable: actividad pulsátil compatible con FES "
            f"(RMS pre-trigger mediano={emg_baseline_rms_uv:.1f} µV; "
            f"límite={EMG_BASELINE_RMS_MAX_UV:.1f} µV)."
        )
        print(
            "     No se estimará ni mostrará latencia muscular. Un notch en la "
            "frecuencia de FES no elimina sus pulsos de banda ancha."
        )
        print(
            "     Advertencia: como FES ocurre en MI y no en REST, también puede "
            "actuar como confusor de la clasificación EEG."
        )
    else:
        for trial in emg_data:
            idx_zero  = np.argmin(np.abs(emg_times))
            threshold = np.mean(trial[:idx_zero]) + 5 * np.std(trial[:idx_zero])
            post      = trial[idx_zero:]
            if np.any(post > threshold):
                all_onsets.append(
                    emg_times[idx_zero + np.argmax(post > threshold)]
                )

    if all_onsets:
        avg_emg_onset = float(np.mean(all_onsets))
        std_emg_onset = float(np.std(all_onsets))
        n_detected    = len(all_onsets)

        if n_detected < MIN_EMG_ONSET_DETECTIONS:
            print(
                "⚠️   Onset EMG detectado en muy pocos trials "
                f"({n_detected}/{len(emg_data)}); no se mostrará en ERP."
            )
            avg_emg_onset = np.nan
            std_emg_onset = np.nan
        else:
            print(
                f"⏱️   EMG Onset: {avg_emg_onset:.3f} s ± "
                f"{std_emg_onset:.3f} s ({n_detected}/{len(emg_data)} trials)"
            )
    elif not emg_fes_contaminated:
        print(
            "⚠️   No se detectó un onset EMG confiable; "
            "no se mostrará una latencia por defecto."
        )

    # ── [V5] Plot EMG mejorado: butterfly + raster de latencias ──
    fig_emg = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig_emg, hspace=0.45, wspace=0.35)

    # Butterfly plot señal filtrada
    ax_bf = fig_emg.add_subplot(gs[0, :])
    ax_bf.plot(emg_times, emg_filt_data.T, color="gray", alpha=0.25, linewidth=0.6)
    ax_bf.plot(emg_times, np.mean(np.abs(emg_filt_data), axis=0),
               color="crimson", linewidth=2.2, label="Promedio |EMG|")
    ax_bf.axvline(0, color="black", ls="--", linewidth=1.5, label="Trigger (0 s)")
    if all_onsets:
        ax_bf.axvline(avg_emg_onset, color="tab:green", ls="-",
                      linewidth=2.0, label=f"Onset µ = {avg_emg_onset:.3f} s")
        ax_bf.axvspan(avg_emg_onset - std_emg_onset,
                      avg_emg_onset + std_emg_onset,
                      color="tab:green", alpha=0.15)
    ax_bf.set_xlabel("Tiempo (s)")
    ax_bf.set_ylabel("Amplitud (µV)")
    ax_bf.set_title(
        f"Butterfly plot EMG — todos los trials "
        f"({EMG_L_FREQ:.0f}–{EMG_H_FREQ:.0f} Hz)",
        fontweight="bold",
    )
    if emg_fes_contaminated:
        ax_bf.text(
            0.5, 0.96,
            "EMG contaminado por FES — latencia muscular no estimable",
            transform=ax_bf.transAxes,
            ha="center", va="top", color="darkred", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="darkred"),
        )
    ax_bf.legend(fontsize=9)
    ax_bf.grid(True, ls=":", alpha=0.4)

    # Raster de latencias individuales
    ax_raster = fig_emg.add_subplot(gs[1, 0])
    if all_onsets:
        for i, t in enumerate(all_onsets):
            ax_raster.plot([t, t], [i - 0.4, i + 0.4], color="tab:green",
                           linewidth=1.5, solid_capstyle="round")
        ax_raster.axvline(avg_emg_onset, color="crimson", ls="--",
                          linewidth=1.5, label=f"µ = {avg_emg_onset:.3f} s")
        ax_raster.set_xlabel("Latencia de onset (s)")
        ax_raster.set_ylabel("Trial #")
        ax_raster.set_title("Raster de latencias EMG", fontweight="bold")
        ax_raster.legend(fontsize=9)
        ax_raster.grid(True, ls=":", alpha=0.4)
    else:
        ax_raster.set_axis_off()
        ax_raster.text(
            0.5, 0.5,
            "Sin latencias EMG válidas",
            transform=ax_raster.transAxes,
            ha="center", va="center", fontsize=12, color="dimgray",
        )

    # Histograma de latencias
    ax_hist = fig_emg.add_subplot(gs[1, 1])
    if all_onsets:
        ax_hist.hist(all_onsets, bins=12, color="tab:green",
                     edgecolor="white", alpha=0.8)
        ax_hist.axvline(avg_emg_onset, color="crimson", ls="--",
                        linewidth=1.5, label=f"µ = {avg_emg_onset:.3f} s")
        ax_hist.set_xlabel("Latencia de onset (s)")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.set_title("Distribución de latencias EMG", fontweight="bold")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, ls=":", alpha=0.4)
    else:
        ax_hist.set_axis_off()
        ax_hist.text(
            0.5, 0.5,
            "FES presente: no interpretar como onset voluntario"
            if emg_fes_contaminated else
            "No se detectó un onset confiable",
            transform=ax_hist.transAxes,
            ha="center", va="center", fontsize=12, color="dimgray",
        )

    fig_emg.suptitle(f"Análisis EMG — {subject} | {session}",
                     fontsize=13, fontweight="bold")
    #plt.show()

else:
    print("⚠️   AUX7 no encontrado — no se mostrará latencia EMG")


# ============================================================
# 5. PREPROCESAMIENTO EEG
# ============================================================
print("\n🎛️   Preprocesando EEG ...")

visual_mode = str(VISUAL_EEG_REFERENCE_MODE).upper()
valid_visual_modes = {"CAR", "NO_CAR", "CAR_LAPLACIAN"}
if visual_mode not in valid_visual_modes:
    raise ValueError(
        f"VISUAL_EEG_REFERENCE_MODE must be one of {sorted(valid_visual_modes)}, "
        f"got {VISUAL_EEG_REFERENCE_MODE!r}"
    )
raw_visual = raw.copy()

# [S1] Re-referencia promedio (CAR), igual que la configuración visual final.
raw.set_eeg_reference("average", projection=False, verbose=False)
print("   ✓ Re-referencia a promedio aplicada")

# [S2] Notch 60 Hz — elimina ruido de línea de potencia
raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
print("   ✓ Notch 60 Hz aplicado")

# [S3] Filtro de paso de banda para CNV (forward-phase)
raw.filter(
    l_freq=EEG_L_FREQ, h_freq=EEG_H_FREQ,
    method="iir", iir_params=EEG_IIR_PARAMS, phase="forward",
    picks="eeg", verbose=False
)
print(
    f"   ✓ Filtro {EEG_L_FREQ:.1f}–{EEG_H_FREQ:.1f} Hz "
    "(Butterworth 4º orden, forward-phase) aplicado"
)

if visual_mode == "CAR":
    raw_visual = raw.copy()
    erp_reference_label = "CAR"
elif visual_mode == "NO_CAR":
    raw_visual.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw_visual.filter(
        l_freq=EEG_L_FREQ, h_freq=EEG_H_FREQ,
        method="iir", iir_params=EEG_IIR_PARAMS, phase="forward",
        picks="eeg", verbose=False,
    )
    erp_reference_label = "sin CAR"
elif visual_mode == "CAR_LAPLACIAN":
    raw_visual.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw_visual.filter(
        l_freq=EEG_L_FREQ, h_freq=EEG_H_FREQ,
        method="iir", iir_params=EEG_IIR_PARAMS, phase="forward",
        picks="eeg", verbose=False,
    )
    raw_visual.set_eeg_reference("average", projection=False, verbose=False)
    raw_visual = mne.preprocessing.compute_current_source_density(raw_visual)
    # Después de CSD/Laplacian, MNE cambia el tipo de canal de "eeg" a "csd".
    # Este modo queda disponible solo para exploración visual.
    erp_reference_label = "CAR+Laplacian"
print(f"   ✓ Copia visual preparada: {erp_reference_label}")


# ============================================================
# 6. SEÑAL CAR FINAL
# ============================================================
eeg_ch_names = [ch for ch in raw.ch_names
                if raw.get_channel_types(picks=ch)[0] == "eeg"]
missing_pos  = [
    ch for ch in eeg_ch_names
    if np.allclose(raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3], 0)
]
if missing_pos:
    print(f"⚠️   Canales sin posición 3D (marcados como bad): {missing_pos}")
    raw.info["bads"] += missing_pos

print("✅  Señal CAR conservada sin aplicar CSD")


# ============================================================
# 7. EPOCHING CON RECHAZO LOCALIZADO EN PICKS_CNV
# ============================================================
epochs_all = mne.Epochs(
    raw, events,
    event_id = event_dict,
    tmin     = -5.0,
    tmax     =  6.0,
    baseline = (-5.0, -3.0),
    reject   = None,
    flat     = None,
    preload  = True,
    detrend  = None,
    verbose  = False,
)

# Peak-to-peak solo en PICKS_CNV
pick_idx = [epochs_all.ch_names.index(ch) for ch in PICKS_CNV if ch in epochs_all.ch_names]
data_cnv = epochs_all.get_data()[:, pick_idx, :]
pp       = data_cnv.max(axis=2) - data_cnv.min(axis=2)

reject_val = REJECT_THRESHOLD["eeg"]
flat_val   = FLAT_THRESHOLD["eeg"]

reject_mask = pp.max(axis=1) > reject_val
flat_mask   = pp.max(axis=1) < flat_val
drop_mask   = reject_mask | flat_mask

# Diagnóstico de amplitudes
pp_uv = pp.max(axis=1) * 1e6
print("\n📊  Distribución de amplitudes en canales CNV (peak-to-peak):")
for p in [50, 75, 90, 95, 99]:
    print(f"   {p:>3}th percentil : {np.percentile(pp_uv, p):.1f} µV equiv.")
print(f"   Máximo         : {pp_uv.max():.1f} µV equiv.")
print(f"   Threshold usado: {reject_val * 1e6:.1f} µV")

# [V1] Drop log — visualizar qué trials se rechazan por canal
drop_counts = {}
for ch_i, ch in enumerate(PICKS_CNV):
    if ch in epochs_all.ch_names:
        idx = epochs_all.ch_names.index(ch)
        n_bad = np.sum(
            (epochs_all.get_data()[:, idx, :].max(axis=1) -
             epochs_all.get_data()[:, idx, :].min(axis=1)) > reject_val
        )
        drop_counts[ch] = n_bad

print("\n📋  Epochs rechazados por canal:")
for ch, n in drop_counts.items():
    bar = "█" * n + "░" * (max(drop_counts.values()) - n)
    print(f"   {ch:>5} : {bar} {n}")

# [V2] Butterfly plot pre-rechazo
# fig_bf, ax_bf2 = plt.subplots(figsize=(14, 4))
# data_butterfly = epochs_all.get_data()[:, pick_idx, :]
# times_all      = epochs_all.times
# for i, ch_i in enumerate(pick_idx):
#     ch_name = epochs_all.ch_names[ch_i]
#     ax_bf2.plot(times_all,
#                 epochs_all.get_data()[:, ch_i, :].T * 1e6,
#                 alpha=0.12, linewidth=0.5,
#                 color=plt.cm.tab10(i / len(pick_idx)))
# ax_bf2.axhline( reject_val * 1e6, color="red", ls="--",
#                linewidth=1.2, label=f"Threshold +{reject_val*1e6:.0f} µV")
# ax_bf2.axhline(-reject_val * 1e6, color="red", ls="--", linewidth=1.2)
# ax_bf2.axvline(0, color="black", ls="--", linewidth=1.2)
# ax_bf2.set_xlabel("Tiempo (s)")
# ax_bf2.set_ylabel("Amplitud (µV)")
# ax_bf2.set_title(
#     f"Butterfly plot EEG pre-rechazo — {subject} | canales CNV\n"
#     "Líneas rojas = threshold de rechazo",
#     fontweight="bold"
# )
# ax_bf2.legend(fontsize=9)
# ax_bf2.grid(True, ls=":", alpha=0.4)
# plt.tight_layout()
# plt.show()

# Aplicar drop
drop_indices = np.where(drop_mask)[0].tolist()
epochs = epochs_all.copy()
epochs.drop(drop_indices, reason="MANUAL_REJECT")
epochs_visual = epochs

n_rest    = len(epochs["Rest (100)"])
n_mi      = len(epochs["MI (200)"])
n_dropped = len(drop_indices)
n_total   = n_dropped + n_rest + n_mi

print(f"\n🛡️   Rechazo localizado en {PICKS_CNV}:")
print(f"   Rechazados  : {n_dropped} / {n_total} ({100*n_dropped/n_total:.1f}%)")
print(f"   Rest trials : {n_rest}")
print(f"   MI trials   : {n_mi}")

if n_rest == 0 or n_mi == 0:
    raise RuntimeError(
        "❌  Todos los epochs fueron rechazados.\n"
        f"   Threshold actual : {reject_val * 1e6:.1f} µV equiv.\n"
        f"   Máximo observado : {pp_uv.max():.1f} µV equiv.\n"
        "   Aumenta REJECT_THRESHOLD, ej: dict(eeg=300e-6)"
    )


epochs_visual_all = mne.Epochs(
    raw_visual, events,
    event_id = event_dict,
    tmin     = -5.0,
    tmax     =  6.0,
    baseline = (-5.0, -3.0),
    reject   = None,
    flat     = None,
    preload  = True,
    detrend  = None,
    verbose  = False,
)
epochs_visual = epochs_visual_all.copy()
epochs_visual.drop(drop_indices, reason="MATCH_CAR_REJECT")
print(
    f"👁️   Figuras ERP/topoplot usarán {erp_reference_label} "
    "(mismos epochs rechazados que el pipeline CAR de modelos)."
)


# ============================================================
# 7B. GRAND AVERAGE DEL INTERVALO NEUTRAL ENTRE TRIALS
# ============================================================
# El eje visual [-2.5, 0.5] representa los 3 s completos del intertrial.
# No comparte el cero temporal de MI/Rest y no participa en entrenamiento.
sfreq = raw.info["sfreq"]
neutral_event_code = 999

if "600" in event_id_map:
    intertrial_begin_events = events_all[
        events_all[:, 2] == event_id_map["600"]
    ]
    neutral_event_samples = (
        intertrial_begin_events[:, 0]
        + int(round(-INTERTRIAL_PLOT_WINDOW[0] * sfreq))
    )
    intertrial_source = "triggers 600/620"
else:
    legacy_event_indices = []
    for run_idx in np.unique(event_run_labels):
        run_indices = np.flatnonzero(event_run_labels == run_idx)
        legacy_event_indices.extend(run_indices[1:])

    legacy_event_indices = np.asarray(legacy_event_indices, dtype=int)
    legacy_shift = LEGACY_PRETRIAL_DURATION + INTERTRIAL_PLOT_WINDOW[1]
    neutral_event_samples = (
        events[legacy_event_indices, 0]
        - int(round(legacy_shift * sfreq))
    )
    intertrial_source = (
        f"reconstrucción legacy ({LEGACY_PRETRIAL_DURATION:.1f} s prep)"
    )

neutral_event_samples = np.asarray(neutral_event_samples, dtype=int)
neutral_event_samples = neutral_event_samples[
    (neutral_event_samples
     + int(round(INTERTRIAL_PLOT_WINDOW[0] * sfreq)) >= raw.first_samp)
    & (neutral_event_samples
       + int(round(INTERTRIAL_PLOT_WINDOW[1] * sfreq))
       < raw.first_samp + raw.n_times)
]

neutral_events = np.column_stack([
    neutral_event_samples,
    np.zeros(len(neutral_event_samples), dtype=int),
    np.full(len(neutral_event_samples), neutral_event_code, dtype=int),
])

epochs_intertrial = mne.Epochs(
    raw,
    neutral_events,
    event_id={"Intertrial": neutral_event_code},
    tmin=INTERTRIAL_PLOT_WINDOW[0],
    tmax=INTERTRIAL_PLOT_WINDOW[1],
    baseline=(INTERTRIAL_PLOT_WINDOW[0], -0.5),
    reject=None,
    flat=None,
    preload=True,
    detrend=None,
    verbose=False,
)

neutral_pick_idx = [
    epochs_intertrial.ch_names.index(ch)
    for ch in PICKS_CNV
    if ch in epochs_intertrial.ch_names
]
neutral_data_cnv = epochs_intertrial.get_data()[:, neutral_pick_idx, :]
neutral_pp = neutral_data_cnv.max(axis=2) - neutral_data_cnv.min(axis=2)
neutral_drop_mask = (
    (neutral_pp.max(axis=1) > reject_val)
    | (neutral_pp.max(axis=1) < flat_val)
)
epochs_intertrial.drop(
    np.flatnonzero(neutral_drop_mask).tolist(),
    reason="INTERTRIAL_REJECT",
)
epochs_intertrial_visual = mne.Epochs(
    raw_visual,
    neutral_events,
    event_id={"Intertrial": neutral_event_code},
    tmin=INTERTRIAL_PLOT_WINDOW[0],
    tmax=INTERTRIAL_PLOT_WINDOW[1],
    baseline=(INTERTRIAL_PLOT_WINDOW[0], -0.5),
    reject=None,
    flat=None,
    preload=True,
    detrend=None,
    verbose=False,
)
epochs_intertrial_visual.drop(
    np.flatnonzero(neutral_drop_mask).tolist(),
    reason="MATCH_CAR_INTERTRIAL_REJECT",
)

print("\n⚪  Referencia intertrial para visualización:")
print(f"   Fuente      : {intertrial_source}")
print(
    f"   Ventana     : {INTERTRIAL_PLOT_WINDOW[0]:.1f} a "
    f"{INTERTRIAL_PLOT_WINDOW[1]:.1f} s (eje visual)"
)
print(
    f"   Conservados : {len(epochs_intertrial)} / "
    f"{len(neutral_events)} intervalos"
)


# ============================================================
# 8. ANÁLISIS CUANTITATIVO CNV  [S3, S4]
# ============================================================
print("\n📐  Análisis cuantitativo CNV ...")

t_cnv  = epochs.times
t_mask = (t_cnv >= CNV_WINDOW[0]) & (t_cnv <= CNV_WINDOW[1])

ch_names_eeg = epochs.copy().pick_types(eeg=True).ch_names

print(f"\n   Ventana CNV: {CNV_WINDOW[0]} → {CNV_WINDOW[1]} s")
print(f"   {'Canal':<8} {'Rest µV':>10} {'MI µV':>10} {'Δ µV':>10} {'p-valor':>10} {'sig':>5}")
print("   " + "-" * 57)

cnv_stats = {}
for ch in PICKS_CNV:
    if ch not in ch_names_eeg:
        continue
    idx = ch_names_eeg.index(ch)

    data_rest = (
        epochs["Rest (100)"].get_data(picks="eeg")[:, idx, :] * 1e6
    )
    data_mi = (
        epochs["MI (200)"].get_data(picks="eeg")[:, idx, :] * 1e6
    )

    # Amplitud media en ventana CNV por trial
    amp_rest = data_rest[:, t_mask].mean(axis=1)
    amp_mi   = data_mi[:, t_mask].mean(axis=1)

    mean_rest = amp_rest.mean()
    mean_mi   = amp_mi.mean()
    delta     = mean_mi - mean_rest

    # [S4] Test t de Welch (no asume varianzas iguales)
    t_stat, p_val = stats.ttest_ind(amp_rest, amp_mi, equal_var=False)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

    cnv_stats[ch] = dict(mean_rest=mean_rest, mean_mi=mean_mi,
                         delta=delta, p=p_val, sig=sig)
    print(f"   {ch:<8} {mean_rest:>10.2f} {mean_mi:>10.2f} {delta:>10.2f} {p_val:>10.4f} {sig:>5}")


# ============================================================
# 9. VISUALIZACIÓN ERP CON OVERLAY EMG  [V3, V4]
# ============================================================
print("\n🖥️   Generando plots ERP con overlay EMG ...")

erp_epochs = epochs_visual
erp_intertrial_epochs = epochs_intertrial_visual
times = erp_epochs.times
visual_data_picks = "data" if visual_mode == "CAR_LAPLACIAN" else "eeg"
visual_y_label = "CSD (scaled)" if visual_mode == "CAR_LAPLACIAN" else "Amplitude (µV)"

def get_mean_sem(epochs_obj, condition):
    data = epochs_obj[condition].get_data(picks=visual_data_picks)
    mean = np.mean(data, axis=0) * 1e6
    sem  = np.std(data,  axis=0) / np.sqrt(data.shape[0]) * 1e6
    return mean, sem

m_100, s_100 = get_mean_sem(erp_epochs, "Rest (100)")
m_200, s_200 = get_mean_sem(erp_epochs, "MI (200)")
intertrial_times = erp_intertrial_epochs.times
intertrial_data = erp_intertrial_epochs.get_data(picks=visual_data_picks)
intertrial_mean = np.mean(intertrial_data, axis=0) * 1e6
intertrial_sem = (
    np.std(intertrial_data, axis=0)
    / np.sqrt(intertrial_data.shape[0])
    * 1e6
)
intertrial_ch_names = erp_intertrial_epochs.copy().pick(visual_data_picks).ch_names
erp_ch_names_eeg = erp_epochs.copy().pick(visual_data_picks).ch_names

# Mantener la visualización ERP sincronizada con los canales reales del modelo.
# Antes esta grilla estaba hard-coded a 12 canales; ahora depende de PICKS_CNV.
channel_grid = [
    PICKS_CNV[0:3],
    PICKS_CNV[3:6],
    PICKS_CNV[6:9],
]
fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)

for row in range(3):
    for col in range(3):
        ch = channel_grid[row][col]
        ax = axes[row, col]

        if ch in erp_ch_names_eeg:
            idx = erp_ch_names_eeg.index(ch)

            # Curvas principales
            ax.plot(times, m_100[idx], color="#2166ac",
                    label="Rest (100)", linewidth=2.0)
            ax.fill_between(times,
                            m_100[idx] - s_100[idx],
                            m_100[idx] + s_100[idx],
                            color="#2166ac", alpha=0.15)
            ax.plot(times, m_200[idx], color="#d6604d",
                    label="MI (200)", linewidth=2.5)
            ax.fill_between(times,
                            m_200[idx] - s_200[idx],
                            m_200[idx] + s_200[idx],
                            color="#d6604d", alpha=0.20)

            if ch in intertrial_ch_names:
                neutral_idx = intertrial_ch_names.index(ch)
                ax.plot(
                    intertrial_times,
                    intertrial_mean[neutral_idx],
                    color="#4d4d4d",
                    label=f"Intertrial (n={len(epochs_intertrial)})",
                    linewidth=2.0,
                    linestyle="--",
                )
                ax.fill_between(
                    intertrial_times,
                    intertrial_mean[neutral_idx] - intertrial_sem[neutral_idx],
                    intertrial_mean[neutral_idx] + intertrial_sem[neutral_idx],
                    color="#737373",
                    alpha=0.12,
                )

            # [V3] Diferencia MI − Rest con CI al 95%
            #diff  = m_200[idx] - m_100[idx]
            # SEM de la diferencia (propagación de error)
            #sem_diff = np.sqrt(s_100[idx]**2 + s_200[idx]**2)
            #ci95     = 1.96 * sem_diff
            #ax.plot(times, diff, color="#4d9221", linewidth=1.5,
            #        ls="--", alpha=0.85, label="MI − Rest")
            #ax.fill_between(times, diff - ci95, diff + ci95,
            #                color="#4d9221", alpha=0.10)

            # EMG onset window
            if np.isfinite(avg_emg_onset):
                ax.axvspan(avg_emg_onset - std_emg_onset,
                           avg_emg_onset + std_emg_onset,
                           color="limegreen", alpha=0.18, label="EMG window")
                ax.axvline(avg_emg_onset, color="darkgreen",
                           linestyle="-", linewidth=2.0,
                           label=f"EMG µ = {avg_emg_onset:.2f} s")

            # [V4] Anotación de amplitud CNV en ventana de análisis
            ax.axvspan(CNV_WINDOW[0], CNV_WINDOW[1],
                       color="gold", alpha=0.10, zorder=0)
            # if ch in cnv_stats:
            #     s = cnv_stats[ch]
            #     annot_str = (f"Rest: {s['mean_rest']:.1f} µV\n"
            #                  f"MI:   {s['mean_mi']:.1f} µV\n"
            #                  f"Δ:    {s['delta']:.1f} µV  {s['sig']}")
            #     ax.text(0.02, 0.97, annot_str,
            #             transform=ax.transAxes, fontsize=7.5,
            #             verticalalignment="top",
            #             bbox=dict(boxstyle="round,pad=0.3",
            #                       facecolor="white", alpha=0.75, linewidth=0))

        ax.axvline(0,    color="black", ls="--", linewidth=1.5, label="Onset (0 s)")
        ax.axvline(-2.0, color="black", ls=":",  linewidth=1.2, label="Prep (−2 s)")
        ax.set_title(f"Ch: {ch}", fontweight="bold")
        ax.set_xlim(*ERP_XLIM)
        ax.set_ylim(*ERP_YLIM)
        ax.grid(True, ls=":", alpha=0.4)
        if col == 0: ax.set_ylabel(visual_y_label)
        if row == 2: ax.set_xlabel("Time (s)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right",
           bbox_to_anchor=(0.99, 0.97), fontsize=9)
plt.suptitle(
    f"CNV Validation — EEG + Muscle Latency + Intertrial Reference\n"
    f"{subject}  |  {session}  |  n_rest={n_rest}, n_mi={n_mi}  "
    f"|  ERP ref = {erp_reference_label}  "
    f"|  zona amarilla = ventana CNV {CNV_WINDOW}\n"
    "Intertrial gris: 3 s trasladados al eje visual "
    f"{INTERTRIAL_PLOT_WINDOW}; no comparte el onset de MI/Rest",
    fontsize=13, fontweight="bold"
)
plt.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.08,
                    hspace=0.38, wspace=0.15)
#plt.show()


# ============================================================
# 10. TOPOGRAPHIC MAPS  [V6]
# ============================================================
print("\n🗺️   Generando paneles topográficos ...")

topo_times  = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
# El topoplot necesita toda la cobertura espacial de la gorra para evitar
# extrapolaciones artificiales. Los modelos siguen usando solo PICKS_CNV.
evoked_rest = erp_epochs["Rest (100)"].average(picks=visual_data_picks)
evoked_mi   = erp_epochs["MI (200)"].average(picks=visual_data_picks)

# [V6] vlim calculado de los datos reales (no fijo en ±15)
topo_data = np.concatenate([
    evoked_rest.data * 1e6,
    evoked_mi.data * 1e6
], axis=1)
# vlim_abs = float(np.percentile(np.abs(topo_data), 98))
vlim_abs = 3
vlim_abs = max(vlim_abs, 2.0)
print(f"   vlim topomaps (98th pct): ±{vlim_abs:.1f} µV")

fig_topo, axes_topo = plt.subplots(
    2, len(topo_times), figsize=(18, 10), constrained_layout=True
)

# Fila 0: Rest
evoked_rest.plot_topomap(
    times=topo_times, axes=axes_topo[0, :],
    average=0.2, cmap="RdBu_r", vlim=(-vlim_abs, vlim_abs),
    show=False, colorbar=False
)
# Fila 1: MI
evoked_mi.plot_topomap(
    times=topo_times, axes=axes_topo[1, :],
    average=0.2, cmap="RdBu_r", vlim=(-vlim_abs, vlim_abs),
    show=False, colorbar=False
)
# Fila 2: Diferencia MI − Rest
# evoked_diff = evoked_mi.copy()
# evoked_diff.data -= evoked_rest.data
# evoked_diff.plot_topomap(
#     times=topo_times, axes=axes_topo[2, :],
#     average=0.2, cmap="RdBu_r", vlim=(-50, 50),
#     show=False, colorbar=False
#)

axes_topo[0, 0].set_ylabel("REST (100)", fontsize=12, fontweight="bold")
axes_topo[1, 0].set_ylabel("MI  (200)",  fontsize=12, fontweight="bold")
#axes_topo[2, 0].set_ylabel("MI − REST",  fontsize=12, fontweight="bold")

im   = axes_topo[1, -1].images[0]
cbar = fig_topo.colorbar(im, ax=axes_topo.ravel().tolist(),
                         shrink=0.5, orientation="vertical", pad=0.02)
cbar.set_label(f"{visual_y_label}  [vlim ±{vlim_abs:.1f}]", fontsize=11)
plt.suptitle(
    f"CNV Topographic Maps — {subject} | {session} | ref={erp_reference_label}\n",
    #f"Fila inferior = diferencia MI − Rest",
    fontsize=13, fontweight="bold"
)
#plt.show()


# ============================================================
# 11. RESUMEN FINAL
# ============================================================
X = epochs.get_data(picks=PICKS_CNV)
y = epochs.events[:, -1]

print("\n" + "="*60)
print("🚀  RESUMEN FINAL")
print("="*60)
print(f"   Feature matrix X : {X.shape}  (epochs × canales × muestras)")
print(f"   Labels vector  y : {y.shape}")
print(f"   Clases           : Rest={np.sum(y==event_dict['Rest (100)'])}"
      f"  |  MI={np.sum(y==event_dict['MI (200)'])}")
if np.isfinite(avg_emg_onset):
    print(
        f"   EMG onset        : {avg_emg_onset:.3f} s ± "
        f"{std_emg_onset:.3f} s ({n_detected} trials)"
    )
elif emg_fes_contaminated:
    print(
        "   EMG onset        : no estimable (contaminación FES; "
        f"RMS pre-trigger={emg_baseline_rms_uv:.1f} µV)"
    )
else:
    print("   EMG onset        : no disponible")
print(f"   Fs               : {config.FS} Hz")
print(f"   Epoch window     : {epochs.tmin:.1f} → {epochs.tmax:.1f} s")
print(f"   Rejection        : {n_dropped}/{n_total} ({100*n_dropped/n_total:.1f}%)")
print(
    f"   Pipeline         : CAR → notch60 → "
    f"BP({EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f}Hz, Butterworth 4º, forward)"
)
print("="*60)

print("\n📐  CNV amplitude summary (ventana CNV):")
print(f"   {'Canal':<8} {'Rest µV':>10} {'MI µV':>10} {'Δ µV':>10} {'sig':>6}")
for ch, s in cnv_stats.items():
    print(f"   {ch:<8} {s['mean_rest']:>10.2f} {s['mean_mi']:>10.2f}"
          f" {s['delta']:>10.2f} {s['sig']:>6}")

"""
================================================================================
CNV BCI — MODELO DE CLASIFICACIÓN
================================================================================
Este script se ejecuta DESPUÉS del pipeline de preprocesamiento.
Asume que ya existen en memoria:
    epochs      : mne.Epochs (notch + BP + CAR, con baseline corregida)
    event_dict  : dict con 'Rest (100)' y 'MI (200)'
    config.FS   : frecuencia de muestreo

Implementa:
    MODELO 1 — Estático:    99 features (11 puntos × 9 canales), 1 predicción
    MODELO 2 — Acumulativo: 11 clasificadores independientes, 1 por instante
================================================================================
"""

# ============================================================
# CONFIGURACIÓN DE FEATURES
# ============================================================
T_START      = -2.5    # s — inicio de la ventana de features
T_END        =  0.0    # s — fin (instante de activación del guante)
TIMEPOINT_STEP = 0.25

# Puntos temporales: -2.50, -2.25, -2.00, ..., 0.00 s
T_POINTS = np.arange(
    T_START,
    T_END + TIMEPOINT_STEP / 2.0,
    TIMEPOINT_STEP,
)
N_TIMEPOINTS = len(T_POINTS)
 
N_CHANNELS   = len(PICKS_CNV)
N_FEATURES   = N_TIMEPOINTS * N_CHANNELS
 
 
# ============================================================
# FUNCIÓN: EXTRACCIÓN DE FEATURES
# ============================================================
def extract_features(epochs_obj, picks, t_points, step=None):
    """
    Extrae amplitud en t_points equidistantes para cada canal en picks.
 
    Parámetros
    ----------
    epochs_obj : mne.Epochs
    picks      : list[str] — nombres de canales
    t_points   : array — puntos temporales a usar (subconjunto para modelo acumulativo)
    step       : int o None — si se indica, usa solo los primeros `step` puntos
 
    Retorna
    -------
    X : np.ndarray (n_trials, n_features)  — amplitud EEG (V)
    y : np.ndarray (n_trials,)
    """
    times = epochs_obj.times
    pts   = t_points[:step] if step is not None else t_points
 
    # Índices del array de tiempo más cercanos a cada t_point
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
 
    ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
    ch_idx   = [ch_names.index(ch) for ch in picks if ch in ch_names]
 
    data = epochs_obj.get_data(picks="eeg")   # (n_trials, n_ch, n_times)
 
    # Para cada trial: amplitudes temporales de cada canal, concatenadas.
    X = np.hstack([
        data[:, ci, :][:, t_idx]   # (n_trials, n_timepoints_used)
        for ci in ch_idx
    ])
    y = epochs_obj.events[:, -1]
    return X, y
 
 
# ============================================================
# FUNCIÓN: CONSTRUIR PIPELINE DE CLASIFICADOR
# ============================================================
def make_clf(name):
    """Retorna un pipeline sklearn listo para entrenar."""
    if name == "LDA":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearDiscriminantAnalysis()),
        ])
    elif name == "LDA_shrink":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )),
        ])
    elif name == "SVM":
        base = SVC(kernel="linear", C=1.0,
                   probability=False, random_state=42)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(base, cv=3, method="sigmoid")),
        ])
    elif name == "LR":
        # Regresión logística con regularización L2
        # max_iter alto porque con pocas muestras puede tardar en converger
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=42,
            )),
        ])
    elif name == "RF":
        # Random Forest — no necesita scaler (basado en árboles)
        # n_estimators bajo porque tenemos pocos trials
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100, max_depth=4,
                min_samples_leaf=3, random_state=42,
            )),
        ])
    elif name == "DT":
        # Árbol de decisión — limitado en profundidad para evitar overfit
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=5,
                random_state=42,
            )),
        ])
    elif name == "KNN":
        # Tres vecinos permiten validar también sesiones piloto pequeñas.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=3)),
        ])
    elif name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=False,
            )),
        ])
    else:
        raise ValueError(f"Clasificador desconocido: {name}")
 
 
# ============================================================
# FUNCIÓN: VALIDACIÓN CRUZADA CON AUC + ACCURACY
# ============================================================
def cross_val_metrics(clf_pipeline, X, y, n_splits=None):
    """
    Stratified K-Fold CV que retorna AUC y Accuracy por fold.
    n_splits se adapta automáticamente al mínimo de trials por clase.
 
    Retorna
    -------
    aucs : np.ndarray — AUC por fold
    accs : np.ndarray — Accuracy (%) por fold
    k    : int        — número de folds usados
    """
    classes, counts = np.unique(y, return_counts=True)
    min_class = counts.min()
    k = min(10, min_class) if n_splits is None else min(n_splits, min_class)
 
    if k < 2:
        print(f"   ⚠️  Solo {min_class} trial(s) en la clase más pequeña — CV no posible")
        return np.array([0.5]), np.array([50.0]), 0
 
    cv   = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        clf_pipeline.fit(X[train_idx], y[train_idx])
 
        # AUC — requiere probabilidades
        proba = clf_pipeline.predict_proba(X[test_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[test_idx], proba))
        except ValueError:
            aucs.append(0.5)
 
        # Accuracy — usa la clase predicha directamente
        y_pred = clf_pipeline.predict(X[test_idx])
        accs.append(np.mean(y_pred == y[test_idx]) * 100.0)
 
    return np.array(aucs), np.array(accs), k
 
 
# ============================================================
# EXTRACCIÓN DE DATOS
# ============================================================
print("\n" + "="*65)
print("🧠  EXTRACCIÓN DE FEATURES")
print("="*65)



# Matriz base compartida por los modelos estático y acumulativo.
X_full, y = extract_features(epochs, PICKS_CNV, T_POINTS)

n_rest = np.sum(y == event_dict["Rest (100)"])
n_mi   = np.sum(y == event_dict["MI (200)"])
 
print(f"   Canales        : {PICKS_CNV}")
print(f"   Puntos temp.   : {np.round(T_POINTS, 3)} s")
print(f"   Features total : {X_full.shape[1]}  ({N_CHANNELS} ch × {N_TIMEPOINTS} pts)")
print(f"   Trials         : Rest={n_rest}  |  MI={n_mi}")
print(f"   Shape X        : {X_full.shape}")
 
 
# ============================================================
# MODELO 1 — ESTÁTICO (una predicción por trial)
# ============================================================
print("\n" + "="*65)
print(f"📊  MODELO 1 — ESTÁTICO ({N_FEATURES} features)")
print("="*65)
 
CLASSIFIERS = ["LDA", "LDA_shrink", "SVM", "LR", "RF", "DT", "KNN", "MLP"]
results_static = {}
 
print(f"   {'Modelo':<6}  {'AUC':>8}  {'±std':>6}  {'Acc%':>7}  {'±std':>6}  Folds")
print("   " + "-"*52)
 
for clf_name in CLASSIFIERS:
    clf              = make_clf(clf_name)
    aucs, accs, k    = cross_val_metrics(clf, X_full, y)
    mean_auc, std_auc = aucs.mean(), aucs.std()
    mean_acc, std_acc = accs.mean(), accs.std()
    results_static[clf_name] = dict(
        auc_mean=mean_auc, auc_std=std_auc,
        acc_mean=mean_acc, acc_std=std_acc, k=k
    )
    print(f"   {clf_name:<6}  {mean_auc:>8.3f}  {std_auc:>6.3f}  "
          f"{mean_acc:>6.1f}%  {std_acc:>6.1f}%  ({k}-fold)")
 
best_static = max(results_static, key=lambda c: results_static[c]["auc_mean"])
print(f"\n   Mejor modelo estático: {best_static} "
      f"(AUC={results_static[best_static]['auc_mean']:.3f}, "
      f"Acc={results_static[best_static]['acc_mean']:.1f}%)")
 
 
# ============================================================
# MODELO 2 — ACUMULATIVO (11 clasificadores independientes)
# ============================================================
print("\n" + "="*65)
print(
    f"⏱️   MODELO 2 — ACUMULATIVO "
    f"({N_TIMEPOINTS} instantes × {len(CLASSIFIERS)} modelos)"
)
print("="*65)
print(f"   Cada CLF_k entrenado con k×{N_CHANNELS} features (k=1..{N_TIMEPOINTS})")
print()
 
results_seq = {clf_name: [] for clf_name in CLASSIFIERS}
 
header = f"   {'Tiempo':>8}  {'Feat':>5}  " + "  ".join(
    f"{'AUC_'+c+'/Acc_'+c:<22}" for c in CLASSIFIERS)
print(f"   {'Tiempo':>8}  {'Feat':>5}  " +
      "  ".join(f"{'--- '+c+' ---':<22}" for c in CLASSIFIERS))
print(f"   {'':>8}  {'':>5}  " +
      "  ".join(f"{'AUC    /  Acc%':<22}" for c in CLASSIFIERS))
print("   " + "-" * (8 + 5 + 26 * len(CLASSIFIERS)))
 
for step in range(1, N_TIMEPOINTS + 1):
    t_current = T_POINTS[step - 1]
    n_feat    = step * N_CHANNELS
    X_step, _ = extract_features(epochs, PICKS_CNV, T_POINTS, step=step)
 
    row = f"   {t_current:>8.3f}  {n_feat:>5}  "
    for clf_name in CLASSIFIERS:
        clf              = make_clf(clf_name)
        aucs, accs, k    = cross_val_metrics(clf, X_step, y)
        mean_auc, std_auc = aucs.mean(), aucs.std()
        mean_acc, std_acc = accs.mean(), accs.std()
        results_seq[clf_name].append(dict(
            t=t_current, n_feat=n_feat,
            auc_mean=mean_auc, auc_std=std_auc,
            acc_mean=mean_acc, acc_std=std_acc, k=k
        ))
        row += f"  {mean_auc:.3f} / {mean_acc:5.1f}%       "
    print(row)
 
 
# ============================================================
# VISUALIZACIÓN — 2 filas: AUC arriba, Accuracy abajo
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
colors_bar  = {
    "LDA":       "#2166ac",
    "LDA_shrink":"#7F77DD",
    "SVM":       "#d6604d",
    "LR":        "#f4a582",
    "RF":        "#1a9641",
    "DT":        "#a6d96a",
    "KNN":       "#d9ef8b",
    "MLP":       "#4d9221",
    "MDM":       "#542788",
    "MDM+recenter": "#b35806",
}
colors_line = colors_bar.copy()
 
# ── Fila 1: AUC ─────────────────────────────────────────────
ax_auc_bar = axes[0, 0]
means_auc = [results_static[c]["auc_mean"] for c in CLASSIFIERS]
stds_auc  = [results_static[c]["auc_std"]  for c in CLASSIFIERS]
bars = ax_auc_bar.bar(CLASSIFIERS, means_auc, yerr=stds_auc,
                      color=[colors_bar[c] for c in CLASSIFIERS],
                      edgecolor="white", linewidth=0.8,
                      error_kw=dict(elinewidth=1.5, capsize=5))
ax_auc_bar.axhline(0.5, color="red",  ls="--", lw=1.2, label="Azar (0.5)")
ax_auc_bar.axhline(0.7, color="gray", ls=":",  lw=1.0, label="Objetivo (0.7)")
ax_auc_bar.set_ylim(0.3, 1.0)
ax_auc_bar.set_ylabel("AUC")
ax_auc_bar.set_title(
    f"Comparación full-window\nAUC",
    fontweight="bold",
)
ax_auc_bar.legend(fontsize=9)
ax_auc_bar.grid(True, ls=":", alpha=0.4, axis="y")
for bar, val in zip(bars, means_auc):
    ax_auc_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
 
ax_auc_seq = axes[0, 1]
for clf_name in CLASSIFIERS:
    t_vals    = [r["t"]        for r in results_seq[clf_name]]
    auc_means = [r["auc_mean"] for r in results_seq[clf_name]]
    auc_stds  = [r["auc_std"]  for r in results_seq[clf_name]]
    ax_auc_seq.plot(t_vals, auc_means, "o-", color=colors_line[clf_name],
                    linewidth=2.0, markersize=6, label=clf_name)
    ax_auc_seq.fill_between(t_vals,
                            np.array(auc_means) - np.array(auc_stds),
                            np.array(auc_means) + np.array(auc_stds),
                            color=colors_line[clf_name], alpha=0.12)
ax_auc_seq.axhline(0.5, color="red",  ls="--", lw=1.2, label="Azar (0.5)")
ax_auc_seq.axhline(0.7, color="gray", ls=":",  lw=1.0, label="Objetivo (0.7)")
ax_auc_seq.axvline(0.0, color="black", ls="--", lw=1.5, label="Trigger (0 s)")
ax_auc_seq.set_xlim(T_START - 0.1, T_END + 0.1)
ax_auc_seq.set_ylim(0.3, 1.0)
ax_auc_seq.set_xlabel("Tiempo disponible (s)")
ax_auc_seq.set_ylabel("AUC")
ax_auc_seq.set_title("Comparación acumulativa\nAUC vs instante temporal", fontweight="bold")
ax_auc_seq.legend(fontsize=9)
ax_auc_seq.grid(True, ls=":", alpha=0.4)
ax_auc_seq.invert_xaxis()
 
# ── Fila 2: Accuracy ─────────────────────────────────────────
ax_acc_bar = axes[1, 0]
means_acc = [results_static[c]["acc_mean"] for c in CLASSIFIERS]
stds_acc  = [results_static[c]["acc_std"]  for c in CLASSIFIERS]
bars2 = ax_acc_bar.bar(CLASSIFIERS, means_acc, yerr=stds_acc,
                       color=[colors_bar[c] for c in CLASSIFIERS],
                       edgecolor="white", linewidth=0.8,
                       error_kw=dict(elinewidth=1.5, capsize=5))
ax_acc_bar.axhline(50.0, color="red",  ls="--", lw=1.2, label="Azar (50%)")
ax_acc_bar.axhline(70.0, color="gray", ls=":",  lw=1.0, label="Objetivo (70%)")
ax_acc_bar.set_ylim(30, 100)
ax_acc_bar.set_ylabel("Accuracy (%)")
ax_acc_bar.set_title(
    f"Comparación full-window\nAccuracy",
    fontweight="bold",
)
ax_acc_bar.legend(fontsize=9)
ax_acc_bar.grid(True, ls=":", alpha=0.4, axis="y")
for bar, val in zip(bars2, means_acc):
    ax_acc_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
 
ax_acc_seq = axes[1, 1]
for clf_name in CLASSIFIERS:
    t_vals    = [r["t"]        for r in results_seq[clf_name]]
    acc_means = [r["acc_mean"] for r in results_seq[clf_name]]
    acc_stds  = [r["acc_std"]  for r in results_seq[clf_name]]
    ax_acc_seq.plot(t_vals, acc_means, "o-", color=colors_line[clf_name],
                    linewidth=2.0, markersize=6, label=clf_name)
    ax_acc_seq.fill_between(t_vals,
                            np.array(acc_means) - np.array(acc_stds),
                            np.array(acc_means) + np.array(acc_stds),
                            color=colors_line[clf_name], alpha=0.12)
ax_acc_seq.axhline(50.0, color="red",  ls="--", lw=1.2, label="Azar (50%)")
ax_acc_seq.axhline(70.0, color="gray", ls=":",  lw=1.0, label="Objetivo (70%)")
ax_acc_seq.axvline(0.0,  color="black", ls="--", lw=1.5, label="Trigger (0 s)")
ax_acc_seq.set_xlim(T_START - 0.1, T_END + 0.1)
ax_acc_seq.set_ylim(30, 100)
ax_acc_seq.set_xlabel("Tiempo disponible (s)")
ax_acc_seq.set_ylabel("Accuracy (%)")
ax_acc_seq.set_title("Comparación acumulativa\nAccuracy vs instante temporal",
                     fontweight="bold")
ax_acc_seq.legend(fontsize=9)
ax_acc_seq.grid(True, ls=":", alpha=0.4)
ax_acc_seq.invert_xaxis()
 
plt.suptitle(
    f"CNV BCI — Clasificación  |  {subject} | {session}\n"
    f"Canales: {PICKS_CNV}",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()


def add_riemann_to_classification_figure(
    riemann_results=None,
    riemann_recenter_results=None,
):
    """Añade MDM/MDM+recenter a la figura clásica sin cambiar métricas."""
    extra_static = []
    if riemann_results:
        extra_static.append(("MDM", riemann_results[-1]))
    if riemann_recenter_results:
        extra_static.append(("MDM+recenter", riemann_recenter_results[-1]))

    if not extra_static:
        return

    # Barras full-window. MDM no usa las mismas features clásicas, por eso
    # se agrega como comparación visual, no como reemplazo del Modelo 1.
    for name, result in extra_static:
        auc_val = result["auc_oof"]
        auc_err = result.get("auc_fold_std", 0.0)
        acc_val = result["accuracy"]

        ax_auc_bar.bar(
            [name],
            [auc_val],
            yerr=[auc_err],
            color=colors_bar[name],
            edgecolor="white",
            linewidth=0.8,
            error_kw=dict(elinewidth=1.5, capsize=5),
        )
        ax_auc_bar.text(
            name,
            auc_val + 0.01,
            f"{auc_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        ax_acc_bar.bar(
            [name],
            [acc_val],
            color=colors_bar[name],
            edgecolor="white",
            linewidth=0.8,
        )
        ax_acc_bar.text(
            name,
            acc_val + 0.5,
            f"{acc_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Líneas acumulativas por endpoint temporal.
    for name, result_list in (
        ("MDM", riemann_results),
        ("MDM+recenter", riemann_recenter_results),
    ):
        if not result_list:
            continue
        t_vals = [r["endpoint"] for r in result_list]
        auc_vals = [r["auc_oof"] for r in result_list]
        acc_vals = [r["accuracy"] for r in result_list]
        auc_stds = [r.get("auc_fold_std", 0.0) for r in result_list]

        ax_auc_seq.plot(
            t_vals,
            auc_vals,
            "D--",
            color=colors_line[name],
            linewidth=2.4,
            markersize=6,
            label=name,
        )
        ax_auc_seq.fill_between(
            t_vals,
            np.array(auc_vals) - np.array(auc_stds),
            np.array(auc_vals) + np.array(auc_stds),
            color=colors_line[name],
            alpha=0.10,
        )
        ax_acc_seq.plot(
            t_vals,
            acc_vals,
            "D--",
            color=colors_line[name],
            linewidth=2.4,
            markersize=6,
            label=name,
        )

    ax_auc_seq.legend(fontsize=9)
    ax_acc_seq.legend(fontsize=9)
    for ax in (ax_auc_bar, ax_acc_bar):
        ax.tick_params(axis="x", rotation=20)
    fig.canvas.draw_idle()
 
 
# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*65)
print("🚀  RESUMEN DEL MODELO")
print("="*65)
 
print(f"\n   Modelo 1 — Estático ({N_FEATURES} features):")
print(f"   {'Modelo':<6}  {'AUC':>8}  {'±std':>6}  {'Acc%':>7}  {'±std':>6}")
print("   " + "-"*42)
for clf_name in CLASSIFIERS:
    r = results_static[clf_name]
    print(f"   {clf_name:<6}  {r['auc_mean']:>8.3f}  {r['auc_std']:>6.3f}  "
          f"{r['acc_mean']:>6.1f}%  {r['acc_std']:>6.1f}%")
 
print(f"\n   Modelo 2 — Acumulativo (t={T_START:.1f} s → t={T_END:.1f} s):")
print(f"   {'Modelo':<6}  {'AUC inicio':>10}  {'AUC fin':>8}  "
      f"{'Acc inicio':>10}  {'Acc fin':>8}  {'Mejor AUC':>10}")
print("   " + "-"*65)
for clf_name in CLASSIFIERS:
    first = results_seq[clf_name][0]
    last  = results_seq[clf_name][-1]
    best  = max(results_seq[clf_name], key=lambda r: r["auc_mean"])
    print(f"   {clf_name:<6}  {first['auc_mean']:>10.3f}  {last['auc_mean']:>8.3f}  "
          f"{first['acc_mean']:>9.1f}%  {last['acc_mean']:>7.1f}%  "
          f"{best['auc_mean']:>8.3f} @ t={best['t']:.2f} s")
 
print(f"\n   Anticipación detectable (AUC > 0.65  /  Acc > 60%):")
for clf_name in CLASSIFIERS:
    for r in results_seq[clf_name]:
        if r["auc_mean"] > 0.65:
            print(f"   {clf_name:<6}  AUC → detectable desde t={r['t']:.3f} s "
                  f"(AUC={r['auc_mean']:.3f}, Acc={r['acc_mean']:.1f}%)")
            break
    else:
        print(f"   {clf_name:<6}  AUC nunca supera 0.65")
 
    for r in results_seq[clf_name]:
        if r["acc_mean"] > 60.0:
            print(f"   {clf_name:<6}  Acc → detectable desde t={r['t']:.3f} s "
                  f"(Acc={r['acc_mean']:.1f}%, AUC={r['auc_mean']:.3f})")
            break
    else:
        print(f"   {clf_name:<6}  Acc nunca supera 60%")
 
print("="*65)


# ============================================================
# MODELO 3 — RIEMANN MDM SOBRE CAR (LEAVE-ONE-RUN-OUT)
# ============================================================
# Las covarianzas incluyen una plantilla MI calculada solo con los runs
# de entrenamiento. Esto conserva información de polaridad/amplitud.

try:
    from pyriemann.classification import MDM
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
except ImportError as exc:
    raise RuntimeError(
        "El modelo Riemann requiere pyriemann en el entorno activo."
    ) from exc


RIEMANN_COV_REG = 1e-4
RIEMANN_MAX_FS = 32.0
groups_riemann = event_run_labels[epochs.selection]
unique_runs_riemann = np.unique(groups_riemann)

if len(unique_runs_riemann) < 2:
    print("\n" + "="*68)
    print("🧭  MODELO 3/4 — omitidos por ahora")
    print("="*68)
    print(
        "   Riemann y channel selection con Leave-One-Run-Out requieren "
        "al menos 2 runs/XDF."
    )
    print(
        f"   Runs disponibles en {session}: {len(unique_runs_riemann)}. "
        "Cuando agregues la siguiente corrida, estos modelos se activan solos."
    )
    print("="*68)
    plt.show()
    raise SystemExit(0)

data_riemann = epochs.get_data(picks=PICKS_CNV)
riemann_start_idx = int(np.argmin(np.abs(epochs.times - T_START)))
riemann_stride = max(
    1,
    int(round(epochs.info["sfreq"] / RIEMANN_MAX_FS)),
)
logo_riemann = LeaveOneGroupOut()


def template_covariances_riemann(trials, template):
    """Covarianzas SPD de [trial; plantilla MI], normalizadas por traza."""
    repeated_template = np.repeat(
        template[np.newaxis, :, :],
        trials.shape[0],
        axis=0,
    )
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
        covariances[trial_idx] = covariance

    return covariances


def _recenter_covariances(covariances, reference):
    """Blanquea covarianzas usando la media Riemanniana de entrenamiento."""
    transform = invsqrtm(reference)
    recentered = np.empty_like(covariances)

    for idx, covariance in enumerate(covariances):
        cov = transform @ covariance @ transform.T
        cov = 0.5 * (cov + cov.T)
        cov += RIEMANN_COV_REG * np.eye(cov.shape[0])
        recentered[idx] = cov

    return recentered


def evaluate_riemann_logo(trials, labels, recenter=False):
    """Genera predicciones fuera de muestra dejando un XDF fuera."""
    scores = np.full(len(labels), np.nan)
    predictions = np.full(len(labels), -1, dtype=int)
    fold_aucs = []

    for train_idx, test_idx in logo_riemann.split(
        trials,
        labels,
        groups_riemann,
    ):
        y_train = labels[train_idx]
        template = trials[train_idx][y_train == mi_id].mean(axis=0)
        cov_train = template_covariances_riemann(
            trials[train_idx],
            template,
        )
        cov_test = template_covariances_riemann(
            trials[test_idx],
            template,
        )
        if recenter:
            reference = mean_riemann(cov_train)
            cov_train = _recenter_covariances(cov_train, reference)
            cov_test = _recenter_covariances(cov_test, reference)

        model = MDM(metric="riemann")
        model.fit(cov_train, y_train)
        positive_idx = int(np.where(model.classes_ == mi_id)[0][0])
        scores[test_idx] = model.predict_proba(cov_test)[:, positive_idx]
        predictions[test_idx] = model.predict(cov_test)
        fold_aucs.append(
            roc_auc_score(labels[test_idx], scores[test_idx])
        )

    return {
        "auc_oof": roc_auc_score(labels, scores),
        "auc_fold_mean": float(np.mean(fold_aucs)),
        "auc_fold_std": float(np.std(fold_aucs)),
        "accuracy": accuracy_score(labels, predictions) * 100.0,
        "scores": scores,
        "predictions": predictions,
    }


print("\n" + "="*68)
print("🧭  MODELO 3 — RIEMANN MDM SOBRE CAR (LEAVE-ONE-RUN-OUT)")
print("="*68)
print(
    f"   Runs: {unique_runs_riemann.tolist()} | "
    f"Canales: {len(PICKS_CNV)} | "
    f"Fs covarianza: ≤{RIEMANN_MAX_FS:.0f} Hz"
)

riemann_results = []
print(
    f"   {'Endpoint':>9} {'Muestras':>9} {'AUC OOF':>9} "
    f"{'AUC folds':>10} {'±std':>7} {'Acc':>8}"
)
print("   " + "-"*59)

for endpoint in T_POINTS:
    endpoint_idx = int(np.argmin(np.abs(epochs.times - endpoint)))
    trials_endpoint = data_riemann[
        :,
        :,
        riemann_start_idx:endpoint_idx + 1:riemann_stride,
    ]
    result = evaluate_riemann_logo(trials_endpoint, y)
    result["endpoint"] = float(endpoint)
    result["n_samples"] = trials_endpoint.shape[2]
    riemann_results.append(result)
    print(
        f"   {endpoint:>9.3f} {result['n_samples']:>9} "
        f"{result['auc_oof']:>9.3f} "
        f"{result['auc_fold_mean']:>10.3f} "
        f"{result['auc_fold_std']:>7.3f} "
        f"{result['accuracy']:>7.1f}%"
    )

best_riemann = max(riemann_results, key=lambda item: item["auc_oof"])
final_riemann = riemann_results[-1]
print(
    f"\n   Ventana completa: AUC={final_riemann['auc_oof']:.3f}, "
    f"Acc={final_riemann['accuracy']:.1f}%"
)
print(
    f"   Mejor endpoint: t={best_riemann['endpoint']:.3f} s, "
    f"AUC={best_riemann['auc_oof']:.3f}"
)
print("="*68)


# ============================================================
# DIAGNÓSTICO POR RUN — CALIDAD + GENERALIZACIÓN EXTERNA
# ============================================================
# Este bloque es únicamente informativo. No elimina epochs, no selecciona
# runs y no modifica las métricas globales ni los modelos entrenados.
groups_before_reject = event_run_labels[epochs_all.selection]
cz_data_uv = epochs.get_data(picks=["Cz"])[:, 0, :] * 1e6
cnv_mask_run = (
    (epochs.times >= CNV_WINDOW[0])
    & (epochs.times <= CNV_WINDOW[1])
)
cz_cnv_by_trial = cz_data_uv[:, cnv_mask_run].mean(axis=1)

print("\n" + "="*100)
print("🔎  DIAGNÓSTICO POR RUN (INFORMATIVO; SIN EXCLUSIÓN DE DATOS)")
print("="*100)
print(
    f"   {'Run':>3} {'Archivo':<20} {'Epoch':>5} {'OK':>4} {'Drop%':>6} "
    f"{'R/M':>7} {'P2P50':>7} {'P2P95':>7} {'ΔCz':>7} "
    f"{'AUC ext':>8} {'Acc ext':>8}"
)
print("   " + "-"*96)

for run_id in unique_runs_riemann:
    before_mask = groups_before_reject == run_id
    accepted_mask = groups_riemann == run_id
    n_before = int(np.sum(before_mask))
    n_accepted = int(np.sum(accepted_mask))
    n_rejected = n_before - n_accepted
    reject_pct = 100.0 * n_rejected / n_before if n_before else np.nan

    y_run = y[accepted_mask]
    n_rest_run = int(np.sum(y_run == event_dict["Rest (100)"]))
    n_mi_run = int(np.sum(y_run == mi_id))

    pp_run = pp_uv[before_mask]
    pp50 = float(np.percentile(pp_run, 50)) if len(pp_run) else np.nan
    pp95 = float(np.percentile(pp_run, 95)) if len(pp_run) else np.nan

    cz_run = cz_cnv_by_trial[accepted_mask]
    cz_rest = cz_run[y_run == event_dict["Rest (100)"]]
    cz_mi = cz_run[y_run == mi_id]
    delta_cz = (
        float(np.mean(cz_mi) - np.mean(cz_rest))
        if len(cz_rest) and len(cz_mi)
        else np.nan
    )

    run_scores = final_riemann["scores"][accepted_mask]
    run_predictions = final_riemann["predictions"][accepted_mask]
    run_auc = (
        float(roc_auc_score(y_run, run_scores))
        if len(np.unique(y_run)) == 2
        else np.nan
    )
    run_accuracy = (
        float(accuracy_score(y_run, run_predictions) * 100.0)
        if len(y_run)
        else np.nan
    )
    filename = os.path.basename(xdf_files[int(run_id) - 1])
    short_filename = (
        filename
        if len(filename) <= 20
        else f"...{filename[-17:]}"
    )

    print(
        f"   {run_id:>3} {short_filename:<20} {n_before:>5} "
        f"{n_accepted:>4} {reject_pct:>5.1f}% "
        f"{n_rest_run:>3}/{n_mi_run:<3} "
        f"{pp50:>7.1f} {pp95:>7.1f} {delta_cz:>+7.2f} "
        f"{run_auc:>8.3f} {run_accuracy:>7.1f}%"
    )

print(
    "\n   P2P50/P2P95: amplitud peak-to-peak en µV antes del rechazo. "
    "ΔCz: MI - Rest en la ventana CNV."
)
print(
    "   AUC/Acc ext: cada run se evalúa con un MDM entrenado únicamente "
    "con los otros runs."
)
print("="*100)


print("\n" + "="*68)
print("🧭  MODELO 3B — RIEMANN MDM + RECENTER OFFLINE")
print("="*68)
print(
    "   Recenter: media Riemanniana del fold de entrenamiento "
    "aplicada a train/test"
)

riemann_recenter_results = []
print(
    f"   {'Endpoint':>9} {'Muestras':>9} {'AUC OOF':>9} "
    f"{'AUC folds':>10} {'±std':>7} {'Acc':>8}"
)
print("   " + "-"*59)

for endpoint in T_POINTS:
    endpoint_idx = int(np.argmin(np.abs(epochs.times - endpoint)))
    trials_endpoint = data_riemann[
        :,
        :,
        riemann_start_idx:endpoint_idx + 1:riemann_stride,
    ]
    result = evaluate_riemann_logo(trials_endpoint, y, recenter=True)
    result["endpoint"] = float(endpoint)
    result["n_samples"] = trials_endpoint.shape[2]
    riemann_recenter_results.append(result)
    print(
        f"   {endpoint:>9.3f} {result['n_samples']:>9} "
        f"{result['auc_oof']:>9.3f} "
        f"{result['auc_fold_mean']:>10.3f} "
        f"{result['auc_fold_std']:>7.3f} "
        f"{result['accuracy']:>7.1f}%"
    )

best_riemann_recenter = max(
    riemann_recenter_results,
    key=lambda item: item["auc_oof"],
)
final_riemann_recenter = riemann_recenter_results[-1]
print(
    f"\n   Ventana completa: AUC={final_riemann_recenter['auc_oof']:.3f}, "
    f"Acc={final_riemann_recenter['accuracy']:.1f}%"
)
print(
    f"   Mejor endpoint: t={best_riemann_recenter['endpoint']:.3f} s, "
    f"AUC={best_riemann_recenter['auc_oof']:.3f}"
)
print(
    f"   Comparación final vs Riemann sin recenter: "
    f"ΔAUC={final_riemann_recenter['auc_oof'] - final_riemann['auc_oof']:+.3f}, "
    f"ΔAcc={final_riemann_recenter['accuracy'] - final_riemann['accuracy']:+.1f}%"
)
print("="*68)

# Ahora que ya existen las métricas MDM, completamos y mostramos la figura
# comparativa. Esto no cambia ningún entrenamiento ni resultado.
add_riemann_to_classification_figure(
    riemann_results=riemann_results,
    riemann_recenter_results=riemann_recenter_results,
)
plt.show()


# ============================================================
# GUARDAR MODELO ONLINE — MOTORCAP
# ============================================================
# Paquete compatible con ExperimentDriver_Online.py / runtime_common.py.
# Se guarda antes de la selección de canales porque el modelo operativo
# recomendado para esta prueba usa los 9 canales completos.

def train_online_m2_package():
    """Entrena MDM + modelos clásicos acumulativos en escala compatible online."""
    online_data_uv = epochs.get_data(picks=PICKS_CNV) * 1e6
    compact_data_uv = epochs.get_data(picks=COMPACT_LDA_PICKS) * 1e6
    labels = y.copy()
    mdm_models = []
    mdm_templates = []
    mdm_centers = []
    mdm_recenter_refs = []
    skl_models = []
    compact_lda_models = []
    observer_skl_models = {
        "LR": [],
        "SVM": [],
    }

    for step, endpoint in enumerate(T_POINTS):
        endpoint_idx = int(np.argmin(np.abs(epochs.times - endpoint)))
        step_trials = online_data_uv[
            :,
            :,
            riemann_start_idx:endpoint_idx + 1,
        ]
        step_template = step_trials[labels == mi_id].mean(axis=0)
        step_covariances = template_covariances_riemann(
            step_trials,
            step_template,
        )
        # Transfer-learning / whitening reference:
        # train and online prediction must use the same Riemannian recentering.
        # This lets an expert model provide a geometrical starting point, while
        # online adaptive recentering can still update it trial by trial.
        step_recenter_ref = mean_riemann(step_covariances)
        step_covariances_train = _recenter_covariances(
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

        X_step, _ = extract_features(epochs, PICKS_CNV, T_POINTS, step=step + 1)
        X_step = X_step * 1e6
        lda = make_clf("LDA_shrink")
        lda.fit(X_step, labels)
        skl_models.append(lda)

        step_time_indices = [
            int(np.argmin(np.abs(epochs.times - time_point)))
            for time_point in T_POINTS[:step + 1]
        ]
        compact_features = compact_data_uv[
            :,
            :,
            step_time_indices,
        ].reshape(len(labels), -1)
        compact_lda = make_clf("LDA_shrink")
        compact_lda.fit(compact_features, labels)
        compact_lda_models.append(compact_lda)

        for observer_name, observer_models in observer_skl_models.items():
            observer = make_clf(observer_name)
            observer.fit(X_step, labels)
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
        "compact_full_feature_count": (
            len(COMPACT_LDA_PICKS) * N_TIMEPOINTS
        ),
        "training_pipeline": (
            f"CAR + notch60 + {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz "
            "Butterworth 4º, sin CSD"
        ),
        "training_scale": "uV",
        "online_note": (
            f"MotorCap {session}, 9 canales completos; "
            "MDM/LR/SVM observers"
        ),
    }


online_model_path = os.path.join(
    base_dir,
    f"sub-{subject}",
    "models",
    f"sub-{subject}_model_motorcap_{session}.pkl",
)
print("\n" + "="*68)
print("💾  MODELO ONLINE MOTORCAP")
print("="*68)
if SAVE_ONLINE_MODEL:
    online_pkg = train_online_m2_package()
    os.makedirs(os.path.dirname(online_model_path), exist_ok=True)
    with open(online_model_path, "wb") as model_file:
        pickle.dump(online_pkg, model_file)
    print(f"   Ruta       : {online_model_path}")
    print(f"   Canales    : {online_pkg['picks']}")
    print(f"   Pasos      : {online_pkg['n_timepoints']}")
    print(f"   Trials     : {online_pkg['n_total']}")
    print("   Control    : configurable (MDM recomendado)")
    print(
        "   Observador : LDA_shrink 9ch + LDA_shrink 3ch "
        f"{online_pkg['compact_lda_picks']} + LR + SVM"
    )
else:
    print("   Guardado omitido: SAVE_ONLINE_MODEL = False")
    print("   Las métricas y figuras sí fueron generadas.")
print("="*68)


# ============================================================
# MODELO 4 — SELECCIÓN ANIDADA DE CANALES
# ============================================================
# La selección usa CAR sin CSD y LDA shrinkage dentro de los runs de
# entrenamiento. El run externo nunca participa en la elección ni de
# los canales ni de k.

epochs_car = epochs
car_data_selection = epochs.get_data(picks=PICKS_CNV) * 1e6
car_time_indices = [
    int(np.argmin(np.abs(epochs_car.times - time_point)))
    for time_point in T_POINTS
]
X_points_by_channel = car_data_selection[:, :, car_time_indices]
all_channel_indices = tuple(range(N_CHANNELS))


def point_features_for_channels(epoch_indices, channel_indices):
    return X_points_by_channel[
        epoch_indices,
    ][:, channel_indices, :].reshape(len(epoch_indices), -1)


def lda_logo_auc_for_channels(
    epoch_indices,
    channel_indices,
):
    """AUC OOF interna para un subconjunto fijo de canales."""
    labels = y[epoch_indices]
    inner_groups = groups_riemann[epoch_indices]
    scores = np.full(len(epoch_indices), np.nan)
    inner_logo = LeaveOneGroupOut()

    for inner_train, inner_test in inner_logo.split(
        epoch_indices,
        labels,
        inner_groups,
    ):
        X_train = point_features_for_channels(
            epoch_indices[inner_train],
            channel_indices,
        )
        X_test = point_features_for_channels(
            epoch_indices[inner_test],
            channel_indices,
        )
        model = make_clf("LDA_shrink")
        model.fit(X_train, labels[inner_train])
        classes = model.named_steps["clf"].classes_
        positive_idx = int(np.where(classes == mi_id)[0][0])
        scores[inner_test] = model.predict_proba(X_test)[:, positive_idx]

    return roc_auc_score(labels, scores)


def greedy_channel_path(epoch_indices):
    """Construye subconjuntos anidados de 1..N canales sin usar test."""
    selected = []
    remaining = list(all_channel_indices)
    path = []

    while remaining:
        best_candidate = None
        best_auc = -np.inf

        for candidate in remaining:
            subset = tuple(selected + [candidate])
            candidate_auc = lda_logo_auc_for_channels(
                epoch_indices,
                subset,
            )
            if candidate_auc > best_auc + 1e-12:
                best_auc = candidate_auc
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        path.append({
            "channels": tuple(selected),
            "auc": float(best_auc),
        })

    return path


def best_subset_from_path(path):
    """Elige el mayor AUC; en empate conserva el subconjunto menor."""
    best_position = max(
        range(len(path)),
        key=lambda idx: (path[idx]["auc"], -len(path[idx]["channels"])),
    )
    return path[best_position]


def fit_predict_lda_channels(train_idx, test_idx, channel_indices):
    model = make_clf("LDA_shrink")
    X_train = point_features_for_channels(train_idx, channel_indices)
    X_test = point_features_for_channels(test_idx, channel_indices)
    model.fit(X_train, y[train_idx])
    classes = model.named_steps["clf"].classes_
    positive_idx = int(np.where(classes == mi_id)[0][0])
    return (
        model.predict_proba(X_test)[:, positive_idx],
        model.predict(X_test),
    )


def fit_predict_riemann_channels(
    train_idx,
    test_idx,
    channel_indices,
):
    start_idx = int(np.argmin(np.abs(epochs_car.times - T_START)))
    endpoint_idx = int(np.argmin(np.abs(epochs_car.times - T_END)))
    trials = car_data_selection[
        :,
        channel_indices,
        start_idx:endpoint_idx + 1:riemann_stride,
    ]
    template = trials[train_idx][y[train_idx] == mi_id].mean(axis=0)
    cov_train = template_covariances_riemann(
        trials[train_idx],
        template,
    )
    cov_test = template_covariances_riemann(
        trials[test_idx],
        template,
    )
    model = MDM(metric="riemann")
    model.fit(cov_train, y[train_idx])
    positive_idx = int(np.where(model.classes_ == mi_id)[0][0])
    return (
        model.predict_proba(cov_test)[:, positive_idx],
        model.predict(cov_test),
    )


lda_full_channel_name = f"LDA-CAR-{N_CHANNELS}ch"
lda_selected_name = "LDA-CAR-selected"
riemann_full_channel_name = f"Riemann-CAR-{N_CHANNELS}ch"
riemann_selected_name = "Riemann-CAR-selected"
channel_selection_models = (
    lda_full_channel_name,
    lda_selected_name,
    riemann_full_channel_name,
    riemann_selected_name,
)
selection_scores = {
    name: np.full(len(y), np.nan)
    for name in channel_selection_models
}
selection_predictions = {
    name: np.full(len(y), -1, dtype=int)
    for name in channel_selection_models
}
selected_channels_by_fold = []

print("\n" + "="*76)
print("🎯  MODELO 4 — CHANNEL SELECTION ANIDADO (LEAVE-ONE-RUN-OUT)")
print("="*76)
print(
    f"   Señal: CAR + notch60 + {EEG_L_FREQ:.1f}-{EEG_H_FREQ:.1f} Hz "
    "Butterworth 4º, sin CSD\n"
    f"   Selección interna: greedy 1..{N_CHANNELS} canales "
    "con LDA shrinkage\n"
    "   Evaluación externa: LDA shrinkage y Riemann MDM sobre CAR"
)

if len(unique_runs_riemann) < 3:
    print(
        "\n   Omitido por ahora: la selección anidada necesita al menos "
        "3 runs/XDF."
    )
    print(
        "   Con 2 runs, cada fold externo deja solo 1 run para la selección "
        "interna, y Leave-One-Run-Out interno no es válido."
    )
    print(
        "   Cuando agregues el tercer run, este bloque se activa automáticamente."
    )
    print("="*76)
    raise SystemExit(0)

for outer_fold, (train_idx, test_idx) in enumerate(
    logo_riemann.split(X_points_by_channel, y, groups_riemann),
    start=1,
):
    held_out_run = int(np.unique(groups_riemann[test_idx])[0])
    path = greedy_channel_path(train_idx)
    selected_result = best_subset_from_path(path)
    selected_indices = selected_result["channels"]
    selected_names = tuple(
        PICKS_CNV[channel_idx]
        for channel_idx in selected_indices
    )
    selected_channels_by_fold.append(selected_names)

    lda_full_scores, lda_full_predictions = fit_predict_lda_channels(
        train_idx,
        test_idx,
        all_channel_indices,
    )
    lda_selected_scores, lda_selected_predictions = (
        fit_predict_lda_channels(
            train_idx,
            test_idx,
            selected_indices,
        )
    )
    riemann_full_scores, riemann_full_predictions = (
        fit_predict_riemann_channels(
            train_idx,
            test_idx,
            all_channel_indices,
        )
    )
    riemann_selected_scores, riemann_selected_predictions = (
        fit_predict_riemann_channels(
            train_idx,
            test_idx,
            selected_indices,
        )
    )

    fold_outputs = {
        lda_full_channel_name: (lda_full_scores, lda_full_predictions),
        lda_selected_name: (
            lda_selected_scores,
            lda_selected_predictions,
        ),
        riemann_full_channel_name: (
            riemann_full_scores,
            riemann_full_predictions,
        ),
        riemann_selected_name: (
            riemann_selected_scores,
            riemann_selected_predictions,
        ),
    }
    for name, (scores, predictions) in fold_outputs.items():
        selection_scores[name][test_idx] = scores
        selection_predictions[name][test_idx] = predictions

    print(
        f"   Fold {outer_fold} | test run {held_out_run} | "
        f"k={len(selected_indices)} | "
        f"AUC interna={selected_result['auc']:.3f} | "
        f"{list(selected_names)}"
    )


print("\n   Rendimiento externo combinado")
print(f"   {'Modelo':<20} {'AUC OOF':>9} {'Accuracy':>10}")
print("   " + "-"*41)
channel_selection_results = {}
for name in channel_selection_models:
    auc = roc_auc_score(y, selection_scores[name])
    accuracy = accuracy_score(y, selection_predictions[name]) * 100.0
    channel_selection_results[name] = {
        "auc": float(auc),
        "accuracy": float(accuracy),
    }
    print(f"   {name:<20} {auc:>9.3f} {accuracy:>9.1f}%")


selection_counts = {
    channel: sum(
        channel in selected_fold
        for selected_fold in selected_channels_by_fold
    )
    for channel in PICKS_CNV
}
print("\n   Estabilidad entre folds externos")
for channel in sorted(
    PICKS_CNV,
    key=lambda name: (-selection_counts[name], PICKS_CNV.index(name)),
):
    count = selection_counts[channel]
    print(
        f"   {channel:<4}: {count}/{len(selected_channels_by_fold)} "
        f"folds ({100.0 * count / len(selected_channels_by_fold):.0f}%)"
    )


# Recomendación para despliegue: selección usando los cuatro runs.
final_channel_path = greedy_channel_path(np.arange(len(y)))
final_channel_result = best_subset_from_path(final_channel_path)
recommended_channels = [
    PICKS_CNV[channel_idx]
    for channel_idx in final_channel_result["channels"]
]

print(
    f"\n   Canales recomendados para el siguiente prototipo online "
    f"(k={len(recommended_channels)}):"
)
print(f"   {recommended_channels}")
print(
    f"   AUC interna LOGO de selección: "
    f"{final_channel_result['auc']:.3f}"
)
print(
    "   Nota: la métrica válida para comparar modelos es la AUC OOF "
    "externa; esta última selección define el conjunto de despliegue."
)
print("="*76)
