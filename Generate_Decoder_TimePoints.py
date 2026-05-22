"""
================================================================================
CNV BCI PIPELINE — Exoskeleton Glove Rehabilitation
================================================================================
Subject:    CNV_PILOT_SUBJ_003
Session:    S00XOFFLINE_GLOVE  /  S00XOFFLINE_NOGLOVE
Classes:    Rest (100) vs Motor Imagery / Move (200)

MEJORAS vs versión anterior:
  SEÑAL:
  [S1] Re-referencia promedio antes del filtro (mejora SNR pre-CSD)
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
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import config  # Debe definir config.FS (ej. 512)
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf
from mne.preprocessing import compute_current_source_density

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# 1. IDENTIDAD Y CONFIGURACIÓN
# ============================================================
subject = "CNV_PILOT_SUBJ_011"
session  = "S001_OFF"     # ← cambiar a S002OFFLINE_NOGLOVE según sesión

CHANNELS_TO_DROP        = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
CHANNELS_TO_INTERPOLATE = []

PICKS_CNV = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1']

# Ventana de análisis CNV (negatividad pre-movimiento)
CNV_WINDOW = (-2.0, 0.0)   # segundos

# Threshold rechazo post-CSD (ajustar si se rechazan demasiados trials)
REJECT_THRESHOLD = dict(csd=150e-3)
FLAT_THRESHOLD   = dict(csd=1e-3)

RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]

xdf_dir = os.path.join(
    "/home/lab-admin/Documents/CNVStudy",
    f"sub-{subject}", f"ses-{session}", "eeg/"
)
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

for xdf_file in xdf_files:
    print(f"   └─ Loading: {os.path.basename(xdf_file)}")
    eeg_s, marker_s = load_xdf(xdf_file)

    eeg_data       = np.array(eeg_s["time_series"]).T
    eeg_timestamps = np.array(eeg_s["time_stamps"])
    channel_names  = get_channel_names_from_xdf(eeg_s)

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
events, event_id_map = mne.events_from_annotations(raw, verbose=False)
event_dict = {
    "Rest (100)": event_id_map["100"],
    "MI (200)":   event_id_map["200"],
}
mi_id = event_id_map["200"]
print(f"📌  Eventos — Rest: {np.sum(events[:,2]==event_dict['Rest (100)'])}  |"
      f"  MI: {np.sum(events[:,2]==mi_id)}")


# ============================================================
# 4. DETECCIÓN DE ONSET EMG (AUX7)
# ============================================================
print("\n💪  Calculando latencia de onset EMG desde AUX7 ...")

avg_emg_onset = 0.5
std_emg_onset = 0.1
n_detected    = 0
all_onsets    = []

if "AUX7" in raw.ch_names:
    raw_emg = raw.copy().pick(["emg"])
    raw_emg.filter(l_freq=10.0, h_freq=250.0, picks="all",
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

    for trial in emg_data:
        idx_zero  = np.argmin(np.abs(emg_times))
        threshold = np.mean(trial[:idx_zero]) + 5 * np.std(trial[:idx_zero])
        post      = trial[idx_zero:]
        if np.any(post > threshold):
            all_onsets.append(emg_times[idx_zero + np.argmax(post > threshold)])

    if all_onsets:
        avg_emg_onset = float(np.mean(all_onsets))
        std_emg_onset = float(np.std(all_onsets))
        n_detected    = len(all_onsets)

    print(f"⏱️   EMG Onset: {avg_emg_onset:.3f} s ± {std_emg_onset:.3f} s"
          f"  ({n_detected}/{len(emg_data)} trials)")

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
    ax_bf.set_title("Butterfly plot EMG — todos los trials (20–200 Hz)", fontweight="bold")
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

    fig_emg.suptitle(f"Análisis EMG — {subject} | {session}",
                     fontsize=13, fontweight="bold")
    #plt.show()

else:
    print("⚠️   AUX7 no encontrado — onset EMG por defecto: 0.5 s")


# ============================================================
# 5. PREPROCESAMIENTO EEG
# ============================================================
print("\n🎛️   Preprocesando EEG ...")

# [S1] Re-referencia a promedio — reduce artefactos comunes antes del filtro
# Mejora el SNR previo al CSD porque CSD es sensible a la referencia
raw.set_eeg_reference("average", projection=False, verbose=False)
print("   ✓ Re-referencia a promedio aplicada")

# [S2] Notch 60 Hz — elimina ruido de línea de potencia
raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
print("   ✓ Notch 60 Hz aplicado")

# Filtro de paso de banda para CNV (0.1–1.0 Hz, zero-phase offline)
raw.filter(
    l_freq=0.1, h_freq=3.0,
    method="iir", phase="forward",
    picks="eeg", verbose=False
)
print("   ✓ Filtro 0.1–3.0 Hz (forward-phase) aplicado")


# ============================================================
# 6. CURRENT SOURCE DENSITY (LAPLACIANO)
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

raw = compute_current_source_density(raw)
print("✅  CSD (Laplaciano) aplicado")


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

reject_val = REJECT_THRESHOLD["csd"]
flat_val   = FLAT_THRESHOLD["csd"]

reject_mask = pp.max(axis=1) > reject_val
flat_mask   = pp.max(axis=1) < flat_val
drop_mask   = reject_mask | flat_mask

# Diagnóstico de amplitudes
pp_uv = pp.max(axis=1) * 1e6
print("\n📊  Distribución de amplitudes en canales CNV (peak-to-peak):")
for p in [50, 75, 90, 95, 99]:
    print(f"   {p:>3}th percentil : {np.percentile(pp_uv, p):.1f} µV equiv.")
print(f"   Máximo         : {pp_uv.max():.1f} µV equiv.")
print(f"   Threshold usado: {reject_val * 1e6:.1f} µV equiv.  ({reject_val:.4f} escala CSD)")

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
        "   Aumenta REJECT_THRESHOLD, ej: dict(csd=300e-3)"
    )


# ============================================================
# 8. ANÁLISIS CUANTITATIVO CNV  [S3, S4]
# ============================================================
print("\n📐  Análisis cuantitativo CNV ...")

t_cnv  = epochs.times
t_mask = (t_cnv >= CNV_WINDOW[0]) & (t_cnv <= CNV_WINDOW[1])

ch_names_csd = epochs.copy().pick_types(csd=True).ch_names

print(f"\n   Ventana CNV: {CNV_WINDOW[0]} → {CNV_WINDOW[1]} s")
print(f"   {'Canal':<8} {'Rest µV':>10} {'MI µV':>10} {'Δ µV':>10} {'p-valor':>10} {'sig':>5}")
print("   " + "-" * 57)

cnv_stats = {}
for ch in PICKS_CNV:
    if ch not in ch_names_csd:
        continue
    idx = ch_names_csd.index(ch)

    data_rest = epochs["Rest (100)"].get_data(picks="csd")[:, idx, :]
    #data_rest = epochs["Rest (100)"].get_data(picks="csd")[:, idx, :] * 1e6

    data_mi   = epochs["MI (200)"].get_data(picks="csd")[:, idx, :]
    #data_mi   = epochs["MI (200)"].get_data(picks="csd")[:, idx, :] * 1e6

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

times    = epochs.times

def get_mean_sem(epochs_obj, condition):
    data = epochs_obj[condition].get_data(picks="csd")
    mean = np.mean(data, axis=0) * 1e6
    sem  = np.std(data,  axis=0) / np.sqrt(data.shape[0]) * 1e6
    return mean, sem

m_100, s_100 = get_mean_sem(epochs, "Rest (100)")
m_200, s_200 = get_mean_sem(epochs, "MI (200)")

all_signals = np.concatenate([m_100, m_200], axis=1)
ymax = np.ceil(np.percentile(np.abs(all_signals), 99) * 1.3)
ymax = max(ymax, 5.0)

channel_grid = [["FC5", "FC1"], ["C3", "Cz"], ["CP5", "CP1"]]
fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)

for row in range(3):
    for col in range(2):
        ch = channel_grid[row][col]
        ax = axes[row, col]

        if ch in ch_names_csd:
            idx = ch_names_csd.index(ch)

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
        ax.set_ylim(-ymax, ymax)
        ax.grid(True, ls=":", alpha=0.4)
        if col == 0: ax.set_ylabel("Amplitude (µV)")
        if row == 2: ax.set_xlabel("Time (s)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right",
           bbox_to_anchor=(0.99, 0.97), fontsize=9)
plt.suptitle(
    f"CNV Validation — EEG + Muscle Latency Overlay\n"
    f"{subject}  |  {session}  |  n_rest={n_rest}, n_mi={n_mi}  "
    f"|  zona amarilla = ventana CNV {CNV_WINDOW}",
    fontsize=13, fontweight="bold"
)
plt.subplots_adjust(left=0.08, right=0.95, top=0.87, bottom=0.08,
                    hspace=0.38, wspace=0.15)
#plt.show()


# ============================================================
# 10. TOPOGRAPHIC MAPS  [V6]
# ============================================================
print("\n🗺️   Generando paneles topográficos ...")

topo_times  = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
evoked_rest = epochs["Rest (100)"].average(picks="csd")
evoked_mi   = epochs["MI (200)"].average(picks="csd")

# [V6] vlim calculado de los datos reales (no fijo en ±15)
topo_data = np.concatenate([
    evoked_rest.data * 1e6,
    evoked_mi.data * 1e6
], axis=1)
vlim_abs = float(np.percentile(np.abs(topo_data), 98))
vlim_abs = max(vlim_abs, 2.0)
print(f"   vlim topomaps (98th pct): ±{vlim_abs:.1f} µV")

fig_topo, axes_topo = plt.subplots(
    2, len(topo_times), figsize=(18, 10), constrained_layout=True
)

# Fila 0: Rest
evoked_rest.plot_topomap(
    times=topo_times, axes=axes_topo[0, :],
    average=0.2, cmap="RdBu_r", vlim=(-20, 20),
    show=False, colorbar=False
)
# Fila 1: MI
evoked_mi.plot_topomap(
    times=topo_times, axes=axes_topo[1, :],
    average=0.2, cmap="RdBu_r", vlim=(-10, 10),
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
cbar.set_label(f"Amplitude (µV)  [vlim ±{vlim_abs:.1f}]", fontsize=11)
plt.suptitle(
    f"CNV Topographic Maps — {subject} | {session}\n",
    #f"Fila inferior = diferencia MI − Rest",
    fontsize=13, fontweight="bold"
)
#plt.show()


# ============================================================
# 11. RESUMEN FINAL
# ============================================================
X = epochs.get_data(picks="csd")
y = epochs.events[:, -1]

print("\n" + "="*60)
print("🚀  RESUMEN FINAL")
print("="*60)
print(f"   Feature matrix X : {X.shape}  (epochs × canales × muestras)")
print(f"   Labels vector  y : {y.shape}")
print(f"   Clases           : Rest={np.sum(y==event_dict['Rest (100)'])}"
      f"  |  MI={np.sum(y==event_dict['MI (200)'])}")
print(f"   EMG onset        : {avg_emg_onset:.3f} s ± {std_emg_onset:.3f} s"
      f"  ({n_detected} trials)")
print(f"   Fs               : {config.FS} Hz")
print(f"   Epoch window     : {epochs.tmin:.1f} → {epochs.tmax:.1f} s")
print(f"   Rejection        : {n_dropped}/{n_total} ({100*n_dropped/n_total:.1f}%)")
print(f"   Pipeline         : avg-ref → notch60 → BP(0.1-3Hz) → CSD")
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
    epochs      : mne.Epochs (post-CSD, con baseline corregida)
    event_dict  : dict con 'Rest (100)' y 'MI (200)'
    config.FS   : frecuencia de muestreo

Implementa:
    MODELO 1 — Estático:    54 features (9 puntos × 6 canales), 1 predicción
    MODELO 2 — Acumulativo: 9 clasificadores independientes, 1 por instante
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# CONFIGURACIÓN DE FEATURES
# ============================================================
PICKS_CNV    = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1']
T_START      = -2.5    # s — inicio de la ventana de features
T_END        =  0.0    # s — fin (instante de activación del guante)
N_TIMEPOINTS =  9      # puntos equidistantes en [T_START, T_END]
 
# Puntos temporales: −2.0, −1.75, −1.5, ..., 0.0 s
T_POINTS = np.linspace(T_START, T_END, N_TIMEPOINTS)
 
N_CHANNELS   = len(PICKS_CNV)
N_FEATURES   = N_TIMEPOINTS * N_CHANNELS   # 9 × 6 = 54
 
 
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
    X : np.ndarray (n_trials, n_features)  — en escala nativa CSD (µV/m²)
    y : np.ndarray (n_trials,)
    """
    times = epochs_obj.times
    pts   = t_points[:step] if step is not None else t_points
 
    # Índices del array de tiempo más cercanos a cada t_point
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
 
    ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
    ch_idx   = [ch_names.index(ch) for ch in picks if ch in ch_names]
 
    data = epochs_obj.get_data(picks="csd")   # (n_trials, n_ch, n_times)
 
    # Para cada trial: [ch1_t1, ch1_t2, ..., ch1_tk, ch2_t1, ..., ch6_tk]
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
        # K vecinos más cercanos — k=7 es un buen punto de partida
        # con ~80 trials (sqrt(84) ≈ 9, usamos impar cercano)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=7)),
        ])
    elif name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
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
 
X_full, y = extract_features(epochs, PICKS_CNV, T_POINTS)
n_rest = np.sum(y == event_dict["Rest (100)"])
n_mi   = np.sum(y == event_dict["MI (200)"])
 
print(f"   Canales        : {PICKS_CNV}")
print(f"   Puntos temp.   : {np.round(T_POINTS, 3)} s")
print(f"   Features total : {X_full.shape[1]}  ({N_CHANNELS} ch × {N_TIMEPOINTS} pts)")
print(f"   Trials         : Rest={n_rest}  |  MI={n_mi}")
print(f"   Shape X        : {X_full.shape}")
 
 
# ============================================================
# MODELO 1 — ESTÁTICO (54 features, una predicción por trial)
# ============================================================
print("\n" + "="*65)
print("📊  MODELO 1 — ESTÁTICO (54 features)")
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
# MODELO 2 — ACUMULATIVO (9 clasificadores independientes)
# ============================================================
print("\n" + "="*65)
print("⏱️   MODELO 2 — ACUMULATIVO (9 clasificadores × 3 modelos)")
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
ax_auc_bar.set_title("Modelo 1 — Estático\nAUC (54 features)", fontweight="bold")
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
ax_auc_seq.set_title("Modelo 2 — Acumulativo\nAUC vs instante temporal", fontweight="bold")
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
ax_acc_bar.set_title("Modelo 1 — Estático\nAccuracy (54 features)", fontweight="bold")
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
ax_acc_seq.set_title("Modelo 2 — Acumulativo\nAccuracy vs instante temporal",
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
plt.show()
 
 
# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*65)
print("🚀  RESUMEN DEL MODELO")
print("="*65)
 
print(f"\n   Modelo 1 — Estático (54 features):")
print(f"   {'Modelo':<6}  {'AUC':>8}  {'±std':>6}  {'Acc%':>7}  {'±std':>6}")
print("   " + "-"*42)
for clf_name in CLASSIFIERS:
    r = results_static[clf_name]
    print(f"   {clf_name:<6}  {r['auc_mean']:>8.3f}  {r['auc_std']:>6.3f}  "
          f"{r['acc_mean']:>6.1f}%  {r['acc_std']:>6.1f}%")
 
print(f"\n   Modelo 2 — Acumulativo (t=−2.0 s → t=0.0 s):")
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