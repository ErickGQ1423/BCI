"""
================================================================================
CNV BCI PIPELINE — Adaptado de Racz et al. 2023
================================================================================
Referencia: Racz et al., "Riemannian geometry-based detection of slow cortical
potentials during movement preparation", IEEE NER 2023.

DIFERENCIAS vs pipeline CSD anterior:
  [R1] Sin CSD Laplaciano — solo avg-ref (el paper no usa CSD)
  [R2] Filtro 0.1–2.0 Hz Butterworth 2° orden zero-phase (igual al paper)
  [R3] 9 canales frontocentrales: FC5, FC1, C3, Cz, CP5, CP1 + Fz, FCz, C1
       (el paper usa F1,Fz,F2,FC1,FCz,FC2,C1,Cz,C2 — adaptado a tu montaje)
  [R4] Rechazo por amplitud absoluta > 100 µV (el paper usa ±100 µV)
  [R5] Escala correcta en visualizaciones (µV sin multiplicar post avg-ref)
  [R6] Template matching para clasificador Riemanniano (pyriemann)
  [R7] Ventana epoch CNV: ventana prep→go completa (2 s en tu paradigma)

FIXES de escala vs versión anterior:
  [S1] get_mean_sem NO multiplica por 1e6 post-CSD (ya en µV/m²)
  [S2] topo_data NO multiplica por 1e6
  [S3] vlim_abs calculado en escala nativa
  [S4] Amplitudes en consola con decimales suficientes
================================================================================
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

try:
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from pyriemann.utils.mean import mean_covariance
    PYRIEMANN_OK = True
    print("✅  pyriemann disponible — clasificador Riemanniano activo")
except ImportError:
    PYRIEMANN_OK = False
    print("⚠️   pyriemann no instalado — solo clasificadores sklearn")
    print("     Instalar con: pip install pyriemann")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# 1. IDENTIDAD Y CONFIGURACIÓN
# ============================================================
subject = "CNV_PILOT_SUBJ_011"
session  = "S001_OFF"

#subject = "S26CLASS_SUBJ_008"
#session  = "S001OFFLINE_FES"

CHANNELS_TO_DROP        = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
CHANNELS_TO_INTERPOLATE = []

# [R3] 9 canales — los 6 originales + Fz, FCz, C1 si existen
PICKS_CNV_CORE  = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1']
PICKS_CNV_EXTRA = ['Fz', 'FCz', 'C1']   # se agregan si están disponibles
PICKS_CNV       = PICKS_CNV_CORE   # se actualiza después de cargar

CNV_WINDOW = (-2.0, 0.0)   # s — ventana de análisis

# [R4] Rechazo por amplitud absoluta (en µV, señal en avg-ref)
REJECT_UV  = 150.0   # µV — igual al paper

# Filtro — [R2]
BP_LOW  = 0.1
BP_HIGH = 1.0

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

print(f"\n📂  {subject} | {session}")
print(f"    Pipeline Racz 2023: avg-ref → BP({BP_LOW}–{BP_HIGH} Hz, zero-phase) → MDM Riemanniano")
mne.set_log_level("WARNING")


# ============================================================
# 2. CARGA Y PREPROCESAMIENTO POR ARCHIVO
# ============================================================
raw_list = []

for xdf_file in xdf_files:
    print(f"   └─ {os.path.basename(xdf_file)}")
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
    eeg_data_subset = eeg_data[valid_idx, :] / 1e6   # µV → V

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
        raw_tmp.info["bads"] = [ch for ch in CHANNELS_TO_INTERPOLATE
                                if ch in raw_tmp.ch_names]
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
print(f"✅  Raw: {raw.n_times / raw.info['sfreq']:.1f} s")


# ============================================================
# 3. DETECCIÓN DE EVENTOS
# ============================================================
events, event_id_map = mne.events_from_annotations(raw, verbose=False)
event_dict = {
    "Rest (100)": event_id_map["100"],
    "MI (200)":   event_id_map["200"],
}
mi_id   = event_id_map["200"]
rest_id = event_id_map["100"]
print(f"📌  Rest: {np.sum(events[:,2]==rest_id)}  |  MI: {np.sum(events[:,2]==mi_id)}")


# ============================================================
# 4. DETECCIÓN DE ONSET EMG (AUX7)
# ============================================================
avg_emg_onset = 0.5
std_emg_onset = 0.1
n_detected    = 0
all_onsets    = []

if "AUX7" in raw.ch_names:
    print("\n💪  Calculando onset EMG ...")
    raw_emg = raw.copy().pick(["emg"])
    raw_emg.filter(l_freq=20.0, h_freq=200.0, picks="all",
                   method="iir", phase="forward", verbose=False)
    raw_emg.notch_filter(freqs=[60.0], picks="all", verbose=False)

    raw_emg_filt  = raw_emg.copy()
    raw_env       = raw_emg.copy()
    raw_env._data = np.abs(raw_emg.get_data())
    raw_env.filter(l_freq=None, h_freq=10.0, picks="all",
                   method="iir", phase="forward", verbose=False)

    epochs_emg = mne.Epochs(raw_env, events, event_id={"MI": mi_id},
                            tmin=-2.0, tmax=5.0, baseline=None,
                            preload=True, verbose=False)
    epochs_emg_filt = mne.Epochs(raw_emg_filt, events, event_id={"MI": mi_id},
                                 tmin=-2.0, tmax=5.0, baseline=None,
                                 preload=True, verbose=False)

    emg_times     = epochs_emg.times
    emg_data      = epochs_emg.get_data()[:, 0, :] * 1e6
    emg_filt_data = epochs_emg_filt.get_data()[:, 0, :] * 1e6

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

    print(f"⏱️   EMG: {avg_emg_onset:.3f} s ± {std_emg_onset:.3f} s "
          f"({n_detected}/{len(emg_data)} trials)")

    # Plot EMG
    fig_emg = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig_emg, hspace=0.45, wspace=0.35)
    ax_bf = fig_emg.add_subplot(gs[0, :])
    ax_bf.plot(emg_times, emg_filt_data.T, color="gray", alpha=0.2, linewidth=0.5)
    ax_bf.plot(emg_times, np.mean(np.abs(emg_filt_data), axis=0),
               color="crimson", linewidth=2.2, label="Average |EMG|")
    ax_bf.axvline(0, color="black", ls="--", linewidth=1.5, label="Trigger")
    if all_onsets:
        ax_bf.axvline(avg_emg_onset, color="tab:green", linewidth=2.0,
                      label=f"Onset µ={avg_emg_onset:.3f} s")
        ax_bf.axvspan(avg_emg_onset-std_emg_onset, avg_emg_onset+std_emg_onset,
                      color="tab:green", alpha=0.15)
    ax_bf.set(xlabel="Time (s)", ylabel="Amplitude (µV)",
              title="All EMG trials (10–250 Hz)")
    ax_bf.legend(fontsize=9); ax_bf.grid(True, ls=":", alpha=0.4)

    ax_r = fig_emg.add_subplot(gs[1, 0])
    if all_onsets:
        for i, t in enumerate(all_onsets):
            ax_r.plot([t, t], [i-.4, i+.4], color="tab:green",
                      linewidth=1.5, solid_capstyle="round")
        ax_r.axvline(avg_emg_onset, color="crimson", ls="--", linewidth=1.5)
        ax_r.set(xlabel="Onset latency (s)", ylabel="Trial #",
                 title="EMG Latency Raster")
        ax_r.grid(True, ls=":", alpha=0.4)

    ax_h = fig_emg.add_subplot(gs[1, 1])
    if all_onsets:
        ax_h.hist(all_onsets, bins=12, color="tab:green", edgecolor="white", alpha=0.8)
        ax_h.axvline(avg_emg_onset, color="crimson", ls="--", linewidth=1.5)
        ax_h.set(xlabel="Onset latency (s)", ylabel="Frequency",
                 title="EMG Latency Distribution")
        ax_h.grid(True, ls=":", alpha=0.4)

    fig_emg.suptitle(f"EMG analysis — {subject} | {session}",
                     fontsize=13, fontweight="bold")
    plt.show()
else:
    print("⚠️   AUX7 no encontrado")


# ============================================================
# 5. PREPROCESAMIENTO EEG — RACZ 2023
# ============================================================
print("\n🎛️   Preprocesando EEG (Racz 2023) ...")

# [R1] Re-referencia a promedio — SIN CSD posterior 
# Basicamente esto es CAR
raw.set_eeg_reference("average", projection=False, verbose=False)
print("   ✓ avg-ref aplicada")

# Notch 60 Hz
raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
print("   ✓ Notch 60 Hz")

# [R2] Butterworth 2° orden zero-phase 0.1–2 Hz
# El paper usa zero-phase offline — correcto para análisis offline
raw.filter(
    l_freq=BP_LOW, h_freq=BP_HIGH,
    method="iir", iir_params=dict(order=2, ftype="butter"),
    phase="forward",
    picks="eeg", verbose=False
)
print(f"   ✓ Butterworth 2° orden zero-phase {BP_LOW}–{BP_HIGH} Hz")
print("   ✓ SIN CSD — señal en µV (avg-ref)")

# Actualizar PICKS_CNV con canales extra si existen
PICKS_CNV = PICKS_CNV_CORE.copy()
for ch in PICKS_CNV_EXTRA:
    if ch in raw.ch_names:
        PICKS_CNV.append(ch)
print(f"   ✓ Canales CNV: {PICKS_CNV} ({len(PICKS_CNV)} total)")


# ============================================================
# 6. EPOCHING — RACZ 2023 con rechazo localizado en PICKS_CNV
# ============================================================
# El paper aplica rechazo ±100 µV a todos los canales, pero eso
# rechaza todo cuando canales temporales (F7, F8) tienen artefactos
# musculares — que son irrelevantes para la CNV.
# Solución: epochar sin rechazo, luego aplicarlo manualmente
# solo sobre PICKS_CNV (igual que en el pipeline CSD).

epochs_all = mne.Epochs(
    raw, events,
    event_id = event_dict,
    tmin     = -3.0,
    tmax     =  5.0,
    baseline = (-2.5, -2.0),
    reject   = None,
    flat     = None,
    preload  = True,
    detrend  = None,
    verbose  = False,
)

# Filtrar PICKS_CNV a los que realmente existen
ch_names_eeg = epochs_all.copy().pick_types(eeg=True).ch_names
PICKS_CNV    = [ch for ch in PICKS_CNV if ch in ch_names_eeg]
print(f"   Canales CNV: {PICKS_CNV}")

# Peak-to-peak en µV solo sobre PICKS_CNV
pick_idx = [ch_names_eeg.index(ch) for ch in PICKS_CNV]
data_cnv = epochs_all.get_data()[:, pick_idx, :] * 1e6   # V → µV
pp       = data_cnv.max(axis=2) - data_cnv.min(axis=2)   # (n_epochs, n_ch)

reject_mask = pp.max(axis=1) > REJECT_UV
flat_mask   = pp.max(axis=1) < 1.0   # < 1 µV = señal plana
drop_mask   = reject_mask | flat_mask

# Diagnóstico
pp_max = pp.max(axis=1)
print(f"\n📊  Peak-to-peak en PICKS_CNV (µV):")
for p in [50, 75, 90, 95]:
    print(f"   {p:>3}th pct: {np.percentile(pp_max, p):.1f} µV")
print(f"   Máximo  : {pp_max.max():.1f} µV")
print(f"   Threshold: ±{REJECT_UV:.0f} µV")

# Drop log por canal
drop_counts = {}
for ch, ci in zip(PICKS_CNV, pick_idx):
    pp_ch = (data_cnv[:, PICKS_CNV.index(ch), :].max(axis=1) -
             data_cnv[:, PICKS_CNV.index(ch), :].min(axis=1))
    drop_counts[ch] = int(np.sum(pp_ch > REJECT_UV))

print("\n📋  Rechazados por canal (solo PICKS_CNV):")
max_bad = max(drop_counts.values()) if drop_counts else 1
for ch, n in drop_counts.items():
    bar = "█" * n + "░" * (max_bad - n)
    print(f"   {ch:>5} : {bar} {n}")

# Aplicar drop
drop_indices = np.where(drop_mask)[0].tolist()
epochs = epochs_all.copy()
epochs.drop(drop_indices, reason="MANUAL_REJECT")

n_rest    = len(epochs["Rest (100)"])
n_mi      = len(epochs["MI (200)"])
n_dropped = len(drop_indices)
n_total   = n_dropped + n_rest + n_mi

print(f"\n🛡️   Epochs — Rest: {n_rest}  |  MI: {n_mi}  |  Rechazados: {n_dropped}/{n_total} "
      f"({100*n_dropped/n_total:.1f}%)")

if n_rest == 0 or n_mi == 0:
    raise RuntimeError(
        f"❌  Todos rechazados. Sube REJECT_UV (actual: {REJECT_UV} µV)\n"
        f"   Máximo observado: {pp_max.max():.1f} µV"
    )


# ============================================================
# 7. ANÁLISIS CUANTITATIVO CNV
# ============================================================
print("\n📐  Análisis cuantitativo CNV (µV, escala avg-ref) ...")

t_mask = (epochs.times >= CNV_WINDOW[0]) & (epochs.times <= CNV_WINDOW[1])

print(f"\n   Ventana: {CNV_WINDOW} s")
print(f"   {'Canal':<6} {'Rest µV':>9} {'MI µV':>9} {'Δ µV':>9} {'p':>9} {'sig':>5}")
print("   " + "-"*47)

cnv_stats = {}
for ch in PICKS_CNV:
    if ch not in ch_names_eeg:
        continue
    idx = ch_names_eeg.index(ch)

    # [S1] Escala correcta: datos en V → µV multiplicando por 1e6
    # Aquí SÍ multiplicamos porque la señal está en V (avg-ref, sin CSD)
    d_rest = epochs["Rest (100)"].get_data()[:, idx, :] * 1e6
    d_mi   = epochs["MI (200)"].get_data()[:, idx, :] * 1e6

    amp_rest = d_rest[:, t_mask].mean(axis=1)
    amp_mi   = d_mi[:, t_mask].mean(axis=1)

    mu_r, mu_m   = amp_rest.mean(), amp_mi.mean()
    delta        = mu_m - mu_r
    _, p_val     = stats.ttest_ind(amp_rest, amp_mi, equal_var=False)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

    cnv_stats[ch] = dict(mu_rest=mu_r, mu_mi=mu_m, delta=delta, p=p_val, sig=sig)
    print(f"   {ch:<6} {mu_r:>9.3f} {mu_m:>9.3f} {delta:>9.3f} {p_val:>9.4f} {sig:>5}")


# ============================================================
# 8. VISUALIZACIÓN ERP
# ============================================================
print("\n🖥️   Generando plots ERP ...")

times = epochs.times

def get_mean_sem_uv(epochs_obj, condition, picks):
    """Media ± SEM en µV — datos en V multiplicados por 1e6."""
    ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
    ch_idx   = [ch_names.index(ch) for ch in picks if ch in ch_names]
    data = epochs_obj[condition].get_data()[:, ch_idx, :] * 1e6  # V → µV
    return np.mean(data, axis=0), np.std(data, axis=0) / np.sqrt(data.shape[0])

m_100, s_100 = get_mean_sem_uv(epochs, "Rest (100)", PICKS_CNV)
m_200, s_200 = get_mean_sem_uv(epochs, "MI (200)",   PICKS_CNV)

# ylim dinámico en µV reales
pick_idx_plot = [ch_names_eeg.index(ch) for ch in PICKS_CNV if ch in ch_names_eeg]
all_sig  = np.concatenate([m_100, m_200], axis=1)
ymax     = max(np.ceil(np.percentile(np.abs(all_sig), 99) * 1.3), 5.0)

channel_grid_base = [["FC5", "FC1"], ["C3", "Cz"], ["CP5", "CP1"]]
channel_grid = [row for row in channel_grid_base
                if any(ch in PICKS_CNV for ch in row)]

fig, axes = plt.subplots(len(channel_grid), 2, figsize=(14, 4*len(channel_grid)),
                         sharex=True, sharey=True)
if len(channel_grid) == 1:
    axes = axes[np.newaxis, :]

for row, ch_pair in enumerate(channel_grid):
    for col, ch in enumerate(ch_pair):
        ax = axes[row, col]
        if ch in PICKS_CNV:
            idx = PICKS_CNV.index(ch)
            ax.plot(times, m_100[idx], color="#2166ac",
                    label="Rest (100)", linewidth=2.0)
            ax.fill_between(times, m_100[idx]-s_100[idx],
                            m_100[idx]+s_100[idx], color="#2166ac", alpha=0.15)
            ax.plot(times, m_200[idx], color="#d6604d",
                    label="MI (200)", linewidth=2.5)
            ax.fill_between(times, m_200[idx]-s_200[idx],
                            m_200[idx]+s_200[idx], color="#d6604d", alpha=0.20)

            # Diferencia MI − Rest
            # diff     = m_200[idx] - m_100[idx]
            # sem_diff = np.sqrt(s_100[idx]**2 + s_200[idx]**2)
            # ax.plot(times, diff, color="#4d9221", lw=1.5,
            #         ls="--", alpha=0.85, label="MI − Rest")
            # ax.fill_between(times, diff-1.96*sem_diff, diff+1.96*sem_diff,
            #                 color="#4d9221", alpha=0.10)

            # EMG y ventana CNV
            ax.axvspan(avg_emg_onset-std_emg_onset, avg_emg_onset+std_emg_onset,
                       color="limegreen", alpha=0.15, label="EMG window")
            ax.axvline(avg_emg_onset, color="darkgreen", lw=2.0)
            ax.axvspan(CNV_WINDOW[0], CNV_WINDOW[1], color="gold", alpha=0.10, zorder=0)

            if ch in cnv_stats:
                s   = cnv_stats[ch]
                txt = (f"Rest: {s['mu_rest']:.3f} µV\n"
                       f"MI:   {s['mu_mi']:.3f} µV\n"
                       f"Δ: {s['delta']:.3f}  {s['sig']}")
                ax.text(0.02, 0.97, txt, transform=ax.transAxes,
                        fontsize=7.5, va="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.80, lw=0))

        ax.axvline(0,    color="black", ls="--", lw=1.5, label="Onset (0 s)")
        ax.axvline(-2.0, color="black", ls=":",  lw=1.2, label="Prep (−2 s)")
        ax.set_title(f"Ch: {ch}", fontweight="bold")
        ax.set_ylim(-ymax, ymax)
        ax.grid(True, ls=":", alpha=0.4)
        if col == 0: ax.set_ylabel("Amplitud (µV)")
        if row == len(channel_grid)-1: ax.set_xlabel("Time (s)")

handles, labels_leg = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="upper right",
           bbox_to_anchor=(0.99, 0.97), fontsize=9)
plt.suptitle(
    f"CNV — avg-ref + Butterworth {BP_LOW}–{BP_HIGH} Hz \n"
    f"{subject} | {session} | n_rest={n_rest}, n_mi={n_mi} "
    f"| ylim=±{ymax:.1f} µV",
    fontsize=12, fontweight="bold"
)
plt.subplots_adjust(left=0.09, right=0.95, top=0.87, bottom=0.08,
                    hspace=0.38, wspace=0.15)
plt.show()


# ============================================================
# 9. TOPOGRAPHIC MAPS — escala correcta
# ============================================================
print("\n🗺️   Generando topomaps ...")

topo_times  = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
evoked_rest = epochs["Rest (100)"].average(picks="eeg")
evoked_mi   = epochs["MI (200)"].average(picks="eeg")
#evoked_diff = evoked_mi.copy()
#evoked_diff.data -= evoked_rest.data

# Calcular vlim en µV directamente desde los datos convertidos
# Los datos de evoked están en V → multiplicar por 1e6 para µV
topo_data_uv = np.concatenate([
    evoked_rest.data * 1e6,
    evoked_mi.data   * 1e6
], axis=1)
vlim_uv = float(np.percentile(np.abs(topo_data_uv), 98))
vlim_uv = max(vlim_uv, 1.0)   # mínimo ±1 µV para que haya gradiente
print(f"   vlim (98th pct): ±{vlim_uv:.2f} µV")

fig_topo, axes_topo = plt.subplots(
    2, len(topo_times), figsize=(18, 11), constrained_layout=True
)

for evoked, ax_row, row_label in [
    (evoked_rest, axes_topo[0, :], "REST (100)"),
    (evoked_mi,   axes_topo[1, :], "MI  (200)"),
    #(evoked_diff, axes_topo[2, :], "MI − REST"),
]:
    # MNE plot_topomap espera vlim en las mismas unidades que scalings
    # Con scalings=dict(eeg=1e6) los datos se muestran en µV
    evoked.plot_topomap(
        times=topo_times, axes=ax_row,
        average=0.2, cmap="RdBu_r",
        vlim=(-vlim_uv, vlim_uv),
        scalings=dict(eeg=1e6),   # V → µV en el plot
        show=False, colorbar=False
    )
    ax_row[0].set_ylabel(row_label, fontsize=12, fontweight="bold")

im   = axes_topo[1, -1].images[0]
cbar = fig_topo.colorbar(im, ax=axes_topo.ravel().tolist(),
                         shrink=0.45, orientation="vertical", pad=0.02)
cbar.set_label(f"Amplitud (µV)  [vlim ±{vlim_uv:.2f} µV]", fontsize=11)
plt.suptitle(
    f"CNV Topomaps (Racz 2023) — {subject} | {session}",
    fontsize=13, fontweight="bold"
)
plt.show()


# ============================================================
# 10. CLASIFICACIÓN — RACZ 2023 + SKLEARN COMPARACIÓN
# ============================================================
print("\n🧠  Clasificación ...")

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
    MODELO 1 — Estático:    66 features (11 puntos × 6 canales), 1 predicción
    MODELO 2 — Acumulativo: 11 clasificadores independientes, 1 por instante

Opción B (Millan): ventana extendida −2.5 → 0.0 s con 11 puntos equidistantes
    T_POINTS = [−2.5, −2.25, −2.0, −1.75, −1.5, −1.25, −1.0, −0.75, −0.5, −0.25, 0.0]
    Los primeros 9 puntos cubren −2.5→−0.5 s (información anticipatoria extra)
    Los últimos 9 puntos cubren −2.0→0.0 s (ventana original)
    El overlap garantiza que no se pierde ninguna información previa
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
# 10. CLASIFICACIÓN — TODOS LOS MODELOS + MDM RIEMANNIANO
# ============================================================
print("\n🧠  Clasificación ...")

# Configuración — Opción B (Millan): ventana extendida −2.5 → 0.0 s
T_START      = -2.5
T_END        =  0.0
N_TIMEPOINTS =  11    # paso de 0.25 s
T_POINTS     = np.linspace(T_START, T_END, N_TIMEPOINTS)
N_CH         = len(PICKS_CNV)
N_FEATURES   = N_TIMEPOINTS * N_CH
# T_POINTS = [−2.5, −2.25, −2.0, −1.75, −1.5, −1.25, −1.0, −0.75, −0.5, −0.25, 0.0]


# ── Funciones de extracción ──────────────────────────────────

def extract_features(epochs_obj, picks, t_points, step=None):
    """
    Amplitud en µV en los puntos temporales indicados.
    Funciona con canales EEG (avg-ref) y CSD automáticamente.
    """
    times = epochs_obj.times
    pts   = t_points[:step] if step is not None else t_points
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0        # CSD ya en escala nativa
    except ValueError:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6        # V → µV
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def extract_raw_data(epochs_obj, picks, tmin, tmax):
    """
    Datos crudos en µV para template matching del MDM.
    Retorna (n_trials, n_ch, n_t) en µV.
    """
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


# ── Funciones Riemannianas ───────────────────────────────────

# Regularización Tikhonov — añade epsilon×I a la covarianza
# Garantiza positive definiteness cuando n_t < n_ch (pasos iniciales)
# Valor típico: 1e-6 a 1e-4. Más grande = más regularización = más sesgo.
COV_REG = 1e-4


def compute_cov_trace_norm(data_3d):
    """
    Covarianza trace-normalizada con regularización Tikhonov.
    Ecuación 1 de Racz 2023 + regularización para positive definiteness.
    data_3d : (n_trials, n_ch, n_t)
    """
    n, n_ch, n_t = data_3d.shape
    covs = np.zeros((n, n_ch, n_ch))
    for i in range(n):
        X  = data_3d[i].T          # (n_t, n_ch)
        C  = X.T @ X
        tr = np.trace(C)
        C  = C / tr if tr > 0 else C
        # Regularización: C + ε×I — garantiza positive definiteness
        C += COV_REG * np.eye(n_ch)
        covs[i] = C
    return covs


def build_template_covs(data_3d, template):
    """
    Concatena template a cada trial en eje de canales → covarianza extendida.
    Implementa template matching de Racz 2023.
    data_3d  : (n_trials, n_ch, n_t)
    template : (n_ch, n_t) — promedio de trials de entrenamiento
    retorna  : (n_trials, 2*n_ch, 2*n_ch)
    """
    tmpl_rep = np.tile(template[np.newaxis], (data_3d.shape[0], 1, 1))
    extended = np.concatenate([data_3d, tmpl_rep], axis=1)
    return compute_cov_trace_norm(extended)


# ── Función de validación cruzada unificada ──────────────────

def cross_val_metrics(clf_obj, X, y, n_splits=None,
                      is_riemann=False, raw_data=None):
    """
    Stratified K-Fold CV — AUC y Accuracy.
    Para MDM: is_riemann=True y raw_data=(n_trials, n_ch, n_t).
    """
    classes, counts = np.unique(y, return_counts=True)
    min_class = counts.min()
    k = min(10, min_class) if n_splits is None else min(n_splits, min_class)
    if k < 2:
        return np.array([0.5]), np.array([50.0]), 0

    cv   = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    aucs, accs = [], []

    for tr_idx, te_idx in cv.split(X, y):
        y_tr, y_te = y[tr_idx], y[te_idx]

        if is_riemann and PYRIEMANN_OK and raw_data is not None:
            template = raw_data[tr_idx].mean(axis=0)
            covs_tr  = build_template_covs(raw_data[tr_idx], template)
            covs_te  = build_template_covs(raw_data[te_idx], template)
            clf_obj.fit(covs_tr, y_tr)
            y_pred = clf_obj.predict(covs_te)
            accs.append(accuracy_score(y_te, y_pred) * 100.0)
            try:
                dists  = clf_obj.transform(covs_te)
                scores = -dists[:, 1] if dists.shape[1] > 1 else -dists[:, 0]
                aucs.append(roc_auc_score(y_te, scores))
            except Exception:
                aucs.append(0.5)
        else:
            clf_obj.fit(X[tr_idx], y[tr_idx])
            proba = clf_obj.predict_proba(X[te_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(y_te, proba))
            except ValueError:
                aucs.append(0.5)
            y_pred = clf_obj.predict(X[te_idx])
            accs.append(accuracy_score(y_te, y_pred) * 100.0)

    return np.array(aucs), np.array(accs), k


# ── Función para construir clasificadores sklearn ────────────

def make_clf(name):
    if name == "LDA":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis())])
    elif name == "LDA_shrink":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis(
                             solver="lsqr", shrinkage="auto"))])
    elif name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(
                             SVC(kernel="linear", C=1.0,
                                 probability=False, random_state=42),
                             cv=3, method="sigmoid"))])
    elif name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(
                             C=1.0, max_iter=1000, random_state=42))])
    elif name == "RF":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestClassifier(
                             n_estimators=100, max_depth=4,
                             min_samples_leaf=3, random_state=42))])
    elif name == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("clf", DecisionTreeClassifier(
                             max_depth=4, min_samples_leaf=5,
                             random_state=42))])
    elif name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))])
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(
                             hidden_layer_sizes=(64, 32), activation="relu",
                             max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.15))])
    else:
        raise ValueError(f"Desconocido: {name}")


# ── Extracción de datos ──────────────────────────────────────

X_full, y = extract_features(epochs, PICKS_CNV, T_POINTS)
n_rest_m  = np.sum(y == event_dict["Rest (100)"])
n_mi_m    = np.sum(y == event_dict["MI (200)"])

print(f"\n   T_POINTS : {np.round(T_POINTS, 2)} s")
print(f"   Features : {N_FEATURES} ({N_CH} ch × {N_TIMEPOINTS} pts) en µV")
print(f"   Trials   : Rest={n_rest_m} | MI={n_mi_m}")

# Datos crudos para MDM — ventana completa
raw_data_full, _ = extract_raw_data(epochs, PICKS_CNV, T_START, T_END)
print(f"   MDM data : {raw_data_full.shape}  (trials × ch × t)")

# Lista de clasificadores
SKL_CLFS = ["LDA", "LDA_shrink", "SVM", "LR", "RF", "DT", "KNN", "MLP"]
ALL_CLFS = SKL_CLFS + (["MDM_Riemann"] if PYRIEMANN_OK else [])

# Colores
COLORS = {
    "LDA":        "#2166ac",
    "LDA_shrink": "#7F77DD",
    "SVM":        "#d6604d",
    "LR":         "#f4a582",
    "RF":         "#1a9641",
    "DT":         "#a6d96a",
    "KNN":        "#969696",
    "MLP":        "#4d9221",
    "MDM_Riemann":"#333333",
}


# ============================================================
# MODELO 1 — ESTÁTICO
# ============================================================
print(f"\n{'='*68}")
print(f"📊  MODELO 1 — ESTÁTICO ({N_FEATURES} features = {N_TIMEPOINTS} pts × {N_CH} ch)")
print(f"{'='*68}")
print(f"   {'Modelo':<14} {'AUC':>7} {'±std':>6} {'Acc%':>7} {'±std':>6}  Folds")
print("   " + "-"*55)

results_static = {}
for clf_name in ALL_CLFS:
    is_r = (clf_name == "MDM_Riemann")
    if is_r:
        clf_obj = MDM(metric="riemann")
    else:
        clf_obj = make_clf(clf_name)

    aucs, accs, k = cross_val_metrics(
        clf_obj, X_full, y,
        is_riemann=is_r,
        raw_data=raw_data_full if is_r else None
    )
    results_static[clf_name] = dict(
        auc_mean=aucs.mean(), auc_std=aucs.std(),
        acc_mean=accs.mean(), acc_std=accs.std(), k=k
    )
    auc_str = f"{aucs.mean():>7.3f}" if not (is_r and aucs.mean() == 0.5) else "    N/A"
    print(f"   {clf_name:<14} {auc_str}  {aucs.std():>5.3f}  "
          f"{accs.mean():>6.1f}%  {accs.std():>5.1f}%  ({k}-fold)")

best = max(results_static, key=lambda c: results_static[c]["auc_mean"])
print(f"\n   Mejor: {best}  "
      f"AUC={results_static[best]['auc_mean']:.3f}  "
      f"Acc={results_static[best]['acc_mean']:.1f}%")


# ============================================================
# MODELO 2 — ACUMULATIVO (incluye MDM paso a paso)
# ============================================================
print(f"\n{'='*68}")
print(f"⏱️   MODELO 2 — ACUMULATIVO ({N_TIMEPOINTS} pasos × {len(ALL_CLFS)} modelos)")
print(f"{'='*68}")
print(f"   Paso 1: t={T_POINTS[0]:.2f} s ({N_CH} feat)  →  "
      f"Paso {N_TIMEPOINTS}: t={T_POINTS[-1]:.2f} s ({N_FEATURES} feat)")
print()

# Header
hdr = f"   {'t':>7} {'feat':>5}  "
hdr += "  ".join(f"{c[:10]:<12}" for c in ALL_CLFS)
print(hdr)
print(f"   {'':>7} {'':>5}  " +
      "  ".join(f"{'AUC / Acc%':<12}" for _ in ALL_CLFS))
print("   " + "-" * (7 + 5 + 14 * len(ALL_CLFS)))

results_seq = {c: [] for c in ALL_CLFS}

for step in range(1, N_TIMEPOINTS + 1):
    t_cur  = T_POINTS[step - 1]
    n_feat = step * N_CH
    X_step, _ = extract_features(epochs, PICKS_CNV, T_POINTS, step=step)

    # Para MDM: datos crudos de la ventana hasta t_cur
    raw_step, _ = extract_raw_data(
        epochs, PICKS_CNV, T_POINTS[0], t_cur
    ) if PYRIEMANN_OK else (None, None)

    row = f"   {t_cur:>7.3f} {n_feat:>5}  "
    for clf_name in ALL_CLFS:
        is_r = (clf_name == "MDM_Riemann")
        if is_r:
            clf_obj = MDM(metric="riemann")
        else:
            clf_obj = make_clf(clf_name)

        aucs, accs, k = cross_val_metrics(
            clf_obj, X_step, y,
            is_riemann=is_r,
            raw_data=raw_step if is_r else None
        )
        results_seq[clf_name].append(dict(
            t=t_cur, n_feat=n_feat,
            auc_mean=aucs.mean(), auc_std=aucs.std(),
            acc_mean=accs.mean(), acc_std=accs.std(), k=k
        ))
        auc_s = f"{aucs.mean():.3f}" if not (is_r and aucs.mean()==0.5) else " N/A"
        row += f"  {auc_s}/{accs.mean():5.1f}%  "
    print(row)


# ============================================================
# VISUALIZACIÓN — 2×2: AUC y Accuracy, estático y acumulativo
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
skl_plot  = [c for c in ALL_CLFS if c != "MDM_Riemann"]

# AUC barras (modelo 1)
ax = axes[0, 0]
means = [results_static[c]["auc_mean"] for c in skl_plot]
stds  = [results_static[c]["auc_std"]  for c in skl_plot]
bars  = ax.bar(skl_plot, means, yerr=stds,
               color=[COLORS[c] for c in skl_plot],
               edgecolor="white", linewidth=0.8,
               error_kw=dict(elinewidth=1.5, capsize=4))
if PYRIEMANN_OK:
    ax.axhline(results_static["MDM_Riemann"]["auc_mean"],
               color=COLORS["MDM_Riemann"], ls="-.", lw=2.0,
               label=f"MDM: {results_static['MDM_Riemann']['auc_mean']:.3f}")
ax.axhline(0.5, color="red",    ls="--", lw=1.2, label="Chance (0.5)")
ax.axhline(0.7, color="gray",   ls=":",  lw=1.0, label="Target (0.7)")
ax.axhline(0.74, color="purple", ls=":", lw=1.0, label="Racz 2023 (0.74)")
ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUC")
ax.set_title(f"Model 1 — AUC ({N_FEATURES} features)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.4, axis="y")
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

# Accuracy barras (modelo 1)
ax = axes[1, 0]
means_a = [results_static[c]["acc_mean"] for c in skl_plot]
stds_a  = [results_static[c]["acc_std"]  for c in skl_plot]
bars2   = ax.bar(skl_plot, means_a, yerr=stds_a,
                 color=[COLORS[c] for c in skl_plot],
                 edgecolor="white", linewidth=0.8,
                 error_kw=dict(elinewidth=1.5, capsize=4))
if PYRIEMANN_OK:
    ax.axhline(results_static["MDM_Riemann"]["acc_mean"],
               color=COLORS["MDM_Riemann"], ls="-.", lw=2.0,
               label=f"MDM: {results_static['MDM_Riemann']['acc_mean']:.1f}%")
ax.axhline(50.0, color="red",    ls="--", lw=1.2, label="Chance (50%)")
ax.axhline(70.0, color="gray",   ls=":",  lw=1.0, label="Target (70%)")
ax.axhline(74.01,color="purple", ls=":",  lw=1.0, label="Racz 2023 (74%)")
ax.set_ylim(30, 100); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Model 1 — Accuracy ({N_FEATURES} features)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.4, axis="y")
for bar, v in zip(bars2, means_a):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

# AUC curva acumulativa
ax = axes[0, 1]
for c in ALL_CLFS:
    t_v = [r["t"]        for r in results_seq[c]]
    a_v = [r["auc_mean"] for r in results_seq[c]]
    s_v = [r["auc_std"]  for r in results_seq[c]]
    ls  = "-." if c == "MDM_Riemann" else "-"
    ax.plot(t_v, a_v, "o"+ls, color=COLORS[c], lw=2.0, ms=5, label=c)
    ax.fill_between(t_v,
                    np.array(a_v) - np.array(s_v),
                    np.array(a_v) + np.array(s_v),
                    color=COLORS[c], alpha=0.08)
ax.axhline(0.5,  color="red",    ls="--", lw=1.2, label="Chance")
ax.axhline(0.7,  color="gray",   ls=":",  lw=1.0, label="Target (0.7)")
ax.axhline(0.74, color="purple", ls=":",  lw=0.8, label="Racz 2023")
ax.axvline(0.0,  color="black",  ls="--", lw=1.5, label="GO (0 s)")
ax.set_xlim(T_START - 0.1, T_END + 0.1)
ax.set_ylim(0.3, 1.0)
ax.set_xlabel("Available time (s)"); ax.set_ylabel("AUC")
ax.set_title("Model 2 — Cumulative AUC", fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.grid(True, ls=":", alpha=0.4)
ax.invert_xaxis()

# Accuracy curva acumulativa
ax = axes[1, 1]
for c in ALL_CLFS:
    t_v = [r["t"]        for r in results_seq[c]]
    a_v = [r["acc_mean"] for r in results_seq[c]]
    s_v = [r["acc_std"]  for r in results_seq[c]]
    ls  = "-." if c == "MDM_Riemann" else "-"
    ax.plot(t_v, a_v, "o"+ls, color=COLORS[c], lw=2.0, ms=5, label=c)
    ax.fill_between(t_v,
                    np.array(a_v) - np.array(s_v),
                    np.array(a_v) + np.array(s_v),
                    color=COLORS[c], alpha=0.08)
ax.axhline(50.0,  color="red",    ls="--", lw=1.2, label="Chance (50%)")
ax.axhline(70.0,  color="gray",   ls=":",  lw=1.0, label="Target (70%)")
ax.axhline(74.01, color="purple", ls=":",  lw=0.8, label="Racz 2023")
ax.axvline(0.0,   color="black",  ls="--", lw=1.5, label="GO (0 s)")
ax.set_xlim(T_START - 0.1, T_END + 0.1)
ax.set_ylim(30, 100)
ax.set_xlabel("Available time (s)"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Model 2 — Cumulative Accuracy", fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.grid(True, ls=":", alpha=0.4)
ax.invert_xaxis()

plt.suptitle(
    f"CNV BCI — {subject} | {session}\n"
    f"avg-ref + Butterworth {BP_LOW}–{BP_HIGH} Hz | "
    f"{N_CH} channels | {N_FEATURES} features | {n_rest_m+n_mi_m} trials",    
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.show()


# ============================================================
# RESUMEN FINAL
# ============================================================
print(f"\n{'='*68}")
print("🚀  RESUMEN FINAL")
print(f"{'='*68}")
print(f"   Sujeto/Sesión : {subject} | {session}")
print(f"   Pipeline      : avg-ref → notch60 → Butterworth({BP_LOW}–{BP_HIGH}Hz, zero-phase)")
print(f"   Canales       : {PICKS_CNV}")
print(f"   T_POINTS      : {np.round(T_POINTS, 2)} s")
print(f"   Features      : {N_FEATURES} ({N_CH} ch × {N_TIMEPOINTS} pts)")
print(f"   Trials        : Rest={n_rest_m} | MI={n_mi_m}")
print(f"   EMG onset     : {avg_emg_onset:.3f} s ± {std_emg_onset:.3f} s")
print(f"   Rechazo       : {n_dropped}/{n_total} ({100*n_dropped/n_total:.1f}%)")

print(f"\n   Modelo 1 — Estático:")
print(f"   {'Modelo':<14} {'AUC':>7} {'±std':>6} {'Acc%':>7} {'±std':>6}")
print("   " + "-"*50)
for c in ALL_CLFS:
    r = results_static[c]
    auc_s = f"{r['auc_mean']:>7.3f}" if not (c == "MDM_Riemann" and r['auc_mean'] == 0.5) else "    N/A"
    flag = " ← mejor AUC" if c == max(results_static, key=lambda x: results_static[x]["auc_mean"]) else ""
    print(f"   {c:<14} {auc_s}  {r['auc_std']:>5.3f}  "
          f"{r['acc_mean']:>6.1f}%  {r['acc_std']:>5.1f}%{flag}")

print(f"\n   Modelo 2 — Acumulativo: mejor AUC por modelo")
print(f"   {'Modelo':<14} {'t inicio':>9} {'AUC ini':>8} "
      f"{'t mejor':>8} {'AUC mejor':>10} {'AUC fin':>8}")
print("   " + "-"*62)
for c in ALL_CLFS:
    first = results_seq[c][0]
    last  = results_seq[c][-1]
    best  = max(results_seq[c], key=lambda r: r["auc_mean"])
    print(f"   {c:<14} {first['t']:>9.2f} {first['auc_mean']:>8.3f} "
          f"{best['t']:>8.2f} {best['auc_mean']:>10.3f} {last['auc_mean']:>8.3f}")

print(f"\n   Anticipación detectable (AUC > 0.65 / Acc > 60%):")
print(f"   {'Modelo':<14} {'AUC desde':>12} {'Acc desde':>12}")
print("   " + "-"*42)
for c in ALL_CLFS:
    auc_t = next((r['t'] for r in results_seq[c] if r["auc_mean"] > 0.65), None)
    acc_t = next((r['t'] for r in results_seq[c] if r["acc_mean"] > 60.0), None)
    print(f"   {c:<14} "
          f"{'t='+str(round(auc_t,2))+' s':>12}  "
          f"{'t='+str(round(acc_t,2))+' s':>12}"
          if (auc_t and acc_t) else
          f"   {c:<14} "
          f"{'t='+str(round(auc_t,2))+' s' if auc_t else 'nunca':>12}  "
          f"{'t='+str(round(acc_t,2))+' s' if acc_t else 'nunca':>12}")

print(f"\n   Referencia Racz 2023: Acc CNV = 74.01% (12 sujetos, 120 trials)")
print("="*68)

# ============================================================
# 11. MODELO 3 — VENTANAS DESLIZANTES
# ============================================================
# Ventana fija de 2 s deslizada en pasos de 0.05 s
# desde [−2.5, −0.5] hasta [−2.0, 0.0] → 11 ventanas
# Cada ventana: 9 puntos equidistantes × 7 canales = 63 features
#
# Diferencia vs Modelo 2 (acumulativo):
#   Modelo 2 — la ventana CRECE: empieza en un punto y se extiende hasta 0
#   Modelo 3 — la ventana SE DESPLAZA: tamaño fijo, se mueve en el tiempo
#
# Utilidad clínica: permite identificar en qué MOMENTO de la preparación
# motora la señal CNV es más discriminante, independientemente del tamaño
# de la ventana de análisis.
 
print(f"\n{'='*68}")
print("🪟  MODELO 3 — VENTANAS DESLIZANTES")
print(f"{'='*68}")
 
WIN_SIZE  = 2.0    # s — tamaño fijo de la ventana
WIN_STEP  = 0.05   # s — paso de desplazamiento
WIN_PTS   = 9      # puntos equidistantes dentro de cada ventana
WIN_FEATS = WIN_PTS * N_CH   # 9 × 7 = 63 features por ventana
 
# Generar todas las ventanas: [start, start + WIN_SIZE]
# La última ventana termina en T_END = 0.0
win_starts = np.arange(T_START, T_END - WIN_SIZE + 1e-9, WIN_STEP)
win_ends   = win_starts + WIN_SIZE
WINDOWS    = list(zip(np.round(win_starts, 4), np.round(win_ends, 4)))
 
print(f"   Ventana     : {WIN_SIZE} s  |  Paso: {WIN_STEP} s")
print(f"   N° ventanas : {len(WINDOWS)}")
print(f"   Features/ventana: {WIN_FEATS} ({WIN_PTS} pts × {N_CH} ch)")
print(f"   Primera: [{WINDOWS[0][0]:.2f}, {WINDOWS[0][1]:.2f}] s")
print(f"   Última:  [{WINDOWS[-1][0]:.2f}, {WINDOWS[-1][1]:.2f}] s")
print()
 
 
def extract_features_window(epochs_obj, picks, t_start, t_end, n_pts):
    """
    Extrae amplitud en n_pts equidistantes dentro de [t_start, t_end].
    Funciona con canales EEG (avg-ref) y CSD automáticamente.
    """
    times = epochs_obj.times
    pts   = np.linspace(t_start, t_end, n_pts)
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0
    except ValueError:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y
 
 
def extract_raw_data_window(epochs_obj, picks, t_start, t_end):
    """
    Datos crudos en µV para MDM en una ventana [t_start, t_end].
    """
    times  = epochs_obj.times
    t_mask = (times >= t_start) & (times <= t_end)
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
 
 
# ── Correr todas las ventanas ────────────────────────────────
 
# Header tabla
print(f"   {'Centro':>8}  {'Ventana':<18}  " +
      "  ".join(f"{c[:10]:<13}" for c in ALL_CLFS))
print(f"   {'':>8}  {'':18}  " +
      "  ".join(f"{'AUC / Acc%':<13}" for _ in ALL_CLFS))
print("   " + "-" * (8 + 18 + 15 * len(ALL_CLFS)))
 
results_sliding = {c: [] for c in ALL_CLFS}
 
for (w_start, w_end) in WINDOWS:
    w_center = round((w_start + w_end) / 2, 3)
    X_win, _ = extract_features_window(
        epochs, PICKS_CNV, w_start, w_end, WIN_PTS
    )
    raw_win, _ = extract_raw_data_window(
        epochs, PICKS_CNV, w_start, w_end
    ) if PYRIEMANN_OK else (None, None)
 
    row = f"   {w_center:>8.3f}  [{w_start:.2f}, {w_end:.2f}]  "
    for clf_name in ALL_CLFS:
        is_r = (clf_name == "MDM_Riemann")
        clf_obj = MDM(metric="riemann") if is_r else make_clf(clf_name)
 
        aucs, accs, k = cross_val_metrics(
            clf_obj, X_win, y,
            is_riemann=is_r,
            raw_data=raw_win if is_r else None
        )
        results_sliding[clf_name].append(dict(
            t_start=w_start, t_end=w_end, t_center=w_center,
            auc_mean=aucs.mean(), auc_std=aucs.std(),
            acc_mean=accs.mean(), acc_std=accs.std(), k=k
        ))
        auc_s = f"{aucs.mean():.3f}" if not (is_r and aucs.mean() == 0.5) else " N/A"
        row += f"  {auc_s}/{accs.mean():5.1f}%  "
    print(row)
 
 
# ── Visualization ────────────────────────────────────────────
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
# AUC by window position
ax = axes[0]
for c in ALL_CLFS:
    t_v = [r["t_center"]   for r in results_sliding[c]]
    a_v = [r["auc_mean"]   for r in results_sliding[c]]
    s_v = [r["auc_std"]    for r in results_sliding[c]]
    ls  = "-." if c == "MDM_Riemann" else "-"
    ax.plot(t_v, a_v, "o"+ls, color=COLORS[c], lw=2.0, ms=4, label=c)
    ax.fill_between(t_v,
                    np.array(a_v) - np.array(s_v),
                    np.array(a_v) + np.array(s_v),
                    color=COLORS[c], alpha=0.08)
ax.axhline(0.5,  color="red",    ls="--", lw=1.2, label="Chance (0.5)")
ax.axhline(0.7,  color="gray",   ls=":",  lw=1.0, label="Target (0.7)")
ax.axhline(0.74, color="purple", ls=":",  lw=0.8, label="Racz 2023")
ax.set_xlabel("Window center (s)")
ax.set_ylabel("AUC")
ax.set_title(f"Model 3 — AUC\nWindow={WIN_SIZE}s, step={WIN_STEP}s",
             fontweight="bold")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, ls=":", alpha=0.4)
 
# Accuracy by window position
ax = axes[1]
for c in ALL_CLFS:
    t_v = [r["t_center"]   for r in results_sliding[c]]
    a_v = [r["acc_mean"]   for r in results_sliding[c]]
    s_v = [r["acc_std"]    for r in results_sliding[c]]
    ls  = "-." if c == "MDM_Riemann" else "-"
    ax.plot(t_v, a_v, "o"+ls, color=COLORS[c], lw=2.0, ms=4, label=c)
    ax.fill_between(t_v,
                    np.array(a_v) - np.array(s_v),
                    np.array(a_v) + np.array(s_v),
                    color=COLORS[c], alpha=0.08)
ax.axhline(50.0,  color="red",    ls="--", lw=1.2, label="Chance (50%)")
ax.axhline(70.0,  color="gray",   ls=":",  lw=1.0, label="Target (70%)")
ax.axhline(74.01, color="purple", ls=":",  lw=0.8, label="Racz 2023")
ax.set_xlabel("Window center (s)")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Model 3 — Accuracy\nWindow={WIN_SIZE}s, step={WIN_STEP}s",
             fontweight="bold")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, ls=":", alpha=0.4)
 
plt.suptitle(
    f"CNV BCI — {subject} | {session}\n"
    f"Model 3: sliding windows  |  {N_CH} channels  |  "
    f"{WIN_FEATS} features/window  |  {n_rest_m+n_mi_m} trials",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.show()
 
 
# ── Resumen comparativo 3 modelos ────────────────────────────
 
print(f"\n{'='*68}")
print("📊  COMPARATIVO FINAL — MODELOS 1, 2 y 3")
print(f"{'='*68}")
 
print(f"\n   {'Modelo clf':<14}  {'M1 AUC':>8}  {'M2 mejor':>10}  "
      f"{'M2 en t':>8}  {'M3 mejor':>10}  {'M3 en t':>8}")
print("   " + "-"*65)
 
for c in ALL_CLFS:
    # Modelo 1 — estático
    m1_auc = results_static[c]["auc_mean"]
 
    # Modelo 2 — mejor punto acumulativo
    m2_best = max(results_seq[c], key=lambda r: r["auc_mean"])
 
    # Modelo 3 — mejor ventana deslizante
    m3_best = max(results_sliding[c], key=lambda r: r["auc_mean"])
 
    print(f"   {c:<14}  {m1_auc:>8.3f}  {m2_best['auc_mean']:>10.3f}  "
          f"{m2_best['t']:>8.2f}s  {m3_best['auc_mean']:>10.3f}  "
          f"[{m3_best['t_start']:.2f},{m3_best['t_end']:.2f}]s")
 
# Ganador por modelo
print(f"\n   Mejor clasificador por enfoque:")
for label, results_dict in [
    ("Modelo 1 (estático)",     results_static),
    ("Modelo 2 (acumulativo)",  {c: max(results_seq[c],
                                        key=lambda r: r["auc_mean"])
                                 for c in ALL_CLFS}),
    ("Modelo 3 (deslizante)",   {c: max(results_sliding[c],
                                        key=lambda r: r["auc_mean"])
                                 for c in ALL_CLFS}),
]:
    if label == "Modelo 1 (estático)":
        best_c = max(results_dict, key=lambda c: results_dict[c]["auc_mean"])
        best_v = results_dict[best_c]["auc_mean"]
    else:
        best_c = max(results_dict, key=lambda c: results_dict[c]["auc_mean"])
        best_v = results_dict[best_c]["auc_mean"]
    print(f"   {label:<26} → {best_c:<14} AUC={best_v:.3f}")
 
print(f"\n   Referencia Racz 2023: Acc CNV = 74.01% (12 sujetos, 120 trials)")
print("="*68)

# ============================================================
# 12. FEATURE SELECTION — REDUCCIÓN DE CANALES
# ============================================================
# Dos métodos para identificar los canales más informativos:
#   A) ANOVA F-score  — univariado, rápido, sin modelo
#   B) RFE con LDA_shrink — elimina canales iterativamente
#
# Después de seleccionar, se corren los 3 modelos con los
# canales reducidos y se compara vs baseline (7 canales).
 
print(f"\n{'='*68}")
print("🔍  SECCIÓN 12 — FEATURE SELECTION: REDUCCIÓN DE CANALES")
print(f"{'='*68}")
 
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
 
# ── Preparar datos: media por canal en ventana CNV ───────────
# Para channel selection usamos la amplitud media de cada canal
# en la ventana CNV completa [-2.5, 0] — un valor por canal por trial
# Esto da una matriz (n_trials, n_ch) limpia para rankear canales
 
t_mask_fs = (epochs.times >= T_START) & (epochs.times <= T_END)
try:
    ch_names_fs = epochs.copy().pick_types(csd=True).ch_names
    data_fs     = epochs.get_data(picks="csd")
    scale_fs    = 1.0
except ValueError:
    ch_names_fs = epochs.copy().pick_types(eeg=True).ch_names
    data_fs     = epochs.get_data()
    scale_fs    = 1e6
 
ch_idx_fs = [ch_names_fs.index(ch) for ch in PICKS_CNV if ch in ch_names_fs]
# X_ch: (n_trials, n_ch) — amplitud media por canal en ventana completa
X_ch = np.array([
    data_fs[:, ci, :][:, t_mask_fs].mean(axis=1) * scale_fs
    for ci in ch_idx_fs
]).T
 
print(f"\n   Canales base    : {PICKS_CNV}")
print(f"   Shape X_ch      : {X_ch.shape}  (trials × canales)")
print(f"   Labels          : Rest={np.sum(y==event_dict['Rest (100)'])} | "
      f"MI={np.sum(y==event_dict['MI (200)'])}")
 
 
# ── Método A: ANOVA F-score por canal ───────────────────────
print(f"\n{'─'*50}")
print("   Método A — ANOVA F-score por canal")
print(f"{'─'*50}")
 
f_scores, p_values = f_classif(X_ch, y)
 
print(f"\n   {'Canal':<8} {'F-score':>10} {'p-valor':>10} {'sig':>5}  {'Rank':>5}")
print("   " + "-"*42)
 
# Ordenar por F-score descendente
rank_anova = np.argsort(f_scores)[::-1]
anova_results = {}
for rank, ci in enumerate(rank_anova):
    ch   = PICKS_CNV[ci]
    f    = f_scores[ci]
    p    = p_values[ci]
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    anova_results[ch] = dict(f=f, p=p, sig=sig, rank=rank+1)
    bar = "█" * int(f / max(f_scores) * 20)
    print(f"   {ch:<8} {f:>10.3f} {p:>10.4f} {sig:>5}  #{rank+1:>2}  {bar}")
 
# Canales top por ANOVA
top_anova = [PICKS_CNV[i] for i in rank_anova]
print(f"\n   Ranking ANOVA: {top_anova}")
 
 
# ── Método B: RFE con LDA_shrink ────────────────────────────
print(f"\n{'─'*50}")
print("   Método B — RFE con LDA_shrink")
print(f"{'─'*50}")
 
# RFE necesita un estimador base — usamos LDA_shrink sin scaler
# porque RFE maneja el proceso de eliminación iterativa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_rfe
 
lda_rfe   = LDA_rfe(solver="lsqr", shrinkage="auto")
rfe       = RFE(estimator=lda_rfe, n_features_to_select=1, step=1)
X_scaled  = StandardScaler().fit_transform(X_ch)
rfe.fit(X_scaled, y)
 
rfe_ranking = rfe.ranking_   # 1 = mejor, N = peor
top_rfe = [PICKS_CNV[i] for i in np.argsort(rfe_ranking)]
 
print(f"\n   {'Canal':<8} {'RFE rank':>10}  {'Posición':>8}")
print("   " + "-"*30)
for ch, rk in sorted(zip(PICKS_CNV, rfe_ranking), key=lambda x: x[1]):
    bar = "█" * (len(PICKS_CNV) - rk + 1)
    print(f"   {ch:<8} {rk:>10}  #{rk:>2}  {bar}")
 
print(f"\n   Ranking RFE   : {top_rfe}")
 
 
# ── Comparación ANOVA vs RFE ─────────────────────────────────
print(f"\n{'─'*50}")
print("   Comparación ANOVA vs RFE")
print(f"{'─'*50}")
print(f"\n   {'Canal':<8} {'ANOVA rank':>11} {'RFE rank':>10}  Coincidencia")
print("   " + "-"*42)
 
for ch in PICKS_CNV:
    ra = anova_results[ch]["rank"]
    rr = int(rfe_ranking[PICKS_CNV.index(ch)])
    match = "✓ coinciden" if abs(ra - rr) <= 1 else ""
    print(f"   {ch:<8} #{ra:>2} (ANOVA)   #{rr:>2} (RFE)  {match}")
 
# Canales que aparecen en top-N de ambos métodos
for N in [2, 3, 4]:
    top_a = set(top_anova[:N])
    top_r = set(top_rfe[:N])
    inter = top_a & top_r
    print(f"\n   Top-{N} coincidentes: {sorted(inter)} "
          f"({len(inter)}/{N} canales)")
 
 
# ── Selección final de canales ───────────────────────────────
# Usar la intersección de top-3 de ambos métodos como conjunto reducido
# Si hay menos de 2 en común, usar top-3 de ANOVA (más estable)
N_SELECT   = 3
top_a3     = set(top_anova[:N_SELECT])
top_r3     = set(top_rfe[:N_SELECT])
selected   = sorted(top_a3 & top_r3)
 
if len(selected) < 2:
    selected = top_anova[:N_SELECT]
    print(f"\n   ⚠️  Poca coincidencia — usando top-{N_SELECT} ANOVA")
else:
    print(f"\n   ✓  Canales seleccionados (intersección top-{N_SELECT}): {selected}")
 
print(f"   Reducción: {len(PICKS_CNV)} → {len(selected)} canales "
      f"({len(selected)/len(PICKS_CNV)*100:.0f}% del original)")
print(f"   Features en Modelo 1: {N_TIMEPOINTS * len(selected)} "
      f"({N_TIMEPOINTS} pts × {len(selected)} ch)")
 
 
# ── Correr los 3 modelos con canales reducidos ───────────────
 
print(f"\n{'='*68}")
print(f"📊  MODELOS CON CANALES REDUCIDOS: {selected}")
print(f"{'='*68}")
 
PICKS_REDUCED = selected
N_CH_RED      = len(PICKS_REDUCED)
N_FEAT_RED    = N_TIMEPOINTS * N_CH_RED
 
# ── Modelo 1 reducido ────────────────────────────────────────
print(f"\n   Modelo 1 — Estático ({N_FEAT_RED} features = {N_TIMEPOINTS} pts × {N_CH_RED} ch)")
print(f"   {'Modelo':<14} {'AUC':>7} {'±std':>6} {'Acc%':>7} {'±std':>6}  vs baseline")
print("   " + "-"*62)
 
X_red, _ = extract_features(epochs, PICKS_REDUCED, T_POINTS)
raw_red_full, _ = extract_raw_data(epochs, PICKS_REDUCED, T_START, T_END)
 
results_static_red = {}
for clf_name in ALL_CLFS:
    is_r    = (clf_name == "MDM_Riemann")
    clf_obj = MDM(metric="riemann") if is_r else make_clf(clf_name)
    aucs, accs, k = cross_val_metrics(
        clf_obj, X_red, y,
        is_riemann=is_r,
        raw_data=raw_red_full if is_r else None
    )
    results_static_red[clf_name] = dict(
        auc_mean=aucs.mean(), auc_std=aucs.std(),
        acc_mean=accs.mean(), acc_std=accs.std()
    )
    baseline_auc = results_static[clf_name]["auc_mean"]
    delta        = aucs.mean() - baseline_auc
    delta_str    = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    auc_str      = f"{aucs.mean():>7.3f}" if not (is_r and aucs.mean()==0.5) else "    N/A"
    print(f"   {clf_name:<14} {auc_str}  {aucs.std():>5.3f}  "
          f"{accs.mean():>6.1f}%  {accs.std():>5.1f}%  ({delta_str})")
 
best_red = max(results_static_red, key=lambda c: results_static_red[c]["auc_mean"])
print(f"\n   Mejor reducido: {best_red}  "
      f"AUC={results_static_red[best_red]['auc_mean']:.3f}  "
      f"vs baseline={results_static[best_red]['auc_mean']:.3f}")
 
 
# ── Modelo 2 reducido (acumulativo) ─────────────────────────
print(f"\n   Modelo 2 — Acumulativo (canales reducidos)")
print(f"   {'t':>7} {'feat':>5}  " +
      "  ".join(f"{c[:10]:<12}" for c in ALL_CLFS))
print(f"   {'':>7} {'':>5}  " +
      "  ".join(f"{'AUC / Acc%':<12}" for _ in ALL_CLFS))
print("   " + "-" * (7 + 5 + 14 * len(ALL_CLFS)))
 
results_seq_red = {c: [] for c in ALL_CLFS}
 
for step in range(1, N_TIMEPOINTS + 1):
    t_cur    = T_POINTS[step - 1]
    n_feat   = step * N_CH_RED
    X_s, _   = extract_features(epochs, PICKS_REDUCED, T_POINTS, step=step)
    raw_s, _ = extract_raw_data(
        epochs, PICKS_REDUCED, T_POINTS[0], t_cur
    ) if PYRIEMANN_OK else (None, None)
 
    row = f"   {t_cur:>7.3f} {n_feat:>5}  "
    for clf_name in ALL_CLFS:
        is_r    = (clf_name == "MDM_Riemann")
        clf_obj = MDM(metric="riemann") if is_r else make_clf(clf_name)
        aucs, accs, k = cross_val_metrics(
            clf_obj, X_s, y,
            is_riemann=is_r,
            raw_data=raw_s if is_r else None
        )
        results_seq_red[clf_name].append(dict(
            t=t_cur, n_feat=n_feat,
            auc_mean=aucs.mean(), auc_std=aucs.std(),
            acc_mean=accs.mean(), acc_std=accs.std()
        ))
        auc_s = f"{aucs.mean():.3f}" if not (is_r and aucs.mean()==0.5) else " N/A"
        row += f"  {auc_s}/{accs.mean():5.1f}%  "
    print(row)
 
 
# ── Modelo 3 reducido (ventanas deslizantes) ─────────────────
print(f"\n   Modelo 3 — Ventanas deslizantes (canales reducidos)")
print(f"   {'Centro':>8}  {'Ventana':<18}  " +
      "  ".join(f"{c[:10]:<12}" for c in ALL_CLFS))
print(f"   {'':>8}  {'':18}  " +
      "  ".join(f"{'AUC / Acc%':<12}" for _ in ALL_CLFS))
print("   " + "-" * (8 + 18 + 14 * len(ALL_CLFS)))
 
results_sliding_red = {c: [] for c in ALL_CLFS}
 
for (w_start, w_end) in WINDOWS:
    w_center  = round((w_start + w_end) / 2, 3)
    X_w, _    = extract_features_window(
        epochs, PICKS_REDUCED, w_start, w_end, WIN_PTS)
    raw_w, _  = extract_raw_data_window(
        epochs, PICKS_REDUCED, w_start, w_end
    ) if PYRIEMANN_OK else (None, None)
 
    row = f"   {w_center:>8.3f}  [{w_start:.2f}, {w_end:.2f}]  "
    for clf_name in ALL_CLFS:
        is_r    = (clf_name == "MDM_Riemann")
        clf_obj = MDM(metric="riemann") if is_r else make_clf(clf_name)
        aucs, accs, k = cross_val_metrics(
            clf_obj, X_w, y,
            is_riemann=is_r,
            raw_data=raw_w if is_r else None
        )
        results_sliding_red[clf_name].append(dict(
            t_start=w_start, t_end=w_end, t_center=w_center,
            auc_mean=aucs.mean(), auc_std=aucs.std(),
            acc_mean=accs.mean(), acc_std=accs.std()
        ))
        auc_s = f"{aucs.mean():.3f}" if not (is_r and aucs.mean()==0.5) else " N/A"
        row += f"  {auc_s}/{accs.mean():5.1f}%  "
    print(row)
 
 
# ── Resumen comparativo baseline vs reducido ─────────────────
print(f"\n{'='*68}")
print(f"📊  COMPARATIVO: 7 canales vs {len(selected)} canales reducidos")
print(f"{'='*68}")
print(f"\n   Canales reducidos: {selected}")
print(f"\n   {'Modelo':<14}  {'M1 base':>8} {'M1 red':>8} {'Δ':>6}  "
      f"{'M2 base':>8} {'M2 red':>8} {'Δ':>6}  "
      f"{'M3 base':>8} {'M3 red':>8} {'Δ':>6}")
print("   " + "-"*80)
 
for c in ALL_CLFS:
    m1b = results_static[c]["auc_mean"]
    m1r = results_static_red[c]["auc_mean"]
    m2b = max(results_seq[c],         key=lambda r: r["auc_mean"])["auc_mean"]
    m2r = max(results_seq_red[c],     key=lambda r: r["auc_mean"])["auc_mean"]
    m3b = max(results_sliding[c],     key=lambda r: r["auc_mean"])["auc_mean"]
    m3r = max(results_sliding_red[c], key=lambda r: r["auc_mean"])["auc_mean"]
    d1  = m1r - m1b
    d2  = m2r - m2b
    d3  = m3r - m3b
    def fmt(v): return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"
    print(f"   {c:<14}  {m1b:>8.3f} {m1r:>8.3f} {fmt(d1):>6}  "
          f"{m2b:>8.3f} {m2r:>8.3f} {fmt(d2):>6}  "
          f"{m3b:>8.3f} {m3r:>8.3f} {fmt(d3):>6}")
 
# Ganador general
print(f"\n   Mejor absoluto con canales reducidos:")
for label, res in [("M1", results_static_red),
                   ("M2", {c: max(results_seq_red[c],
                                  key=lambda r: r["auc_mean"])
                            for c in ALL_CLFS}),
                   ("M3", {c: max(results_sliding_red[c],
                                  key=lambda r: r["auc_mean"])
                            for c in ALL_CLFS})]:
    if label == "M1":
        bc = max(res, key=lambda c: res[c]["auc_mean"])
        bv = res[bc]["auc_mean"]
    else:
        bc = max(res, key=lambda c: res[c]["auc_mean"])
        bv = res[bc]["auc_mean"]
    base_v = (max(results_static[bc]["auc_mean"],
                  max(r["auc_mean"] for r in results_seq[bc]),
                  max(r["auc_mean"] for r in results_sliding[bc])))
    print(f"   {label}: {bc:<14} AUC={bv:.3f}  "
          f"(baseline={base_v:.3f}, Δ={bv-base_v:+.3f})")
 
print("="*68)
 
# ============================================================
# 13. GUARDADO AUTOMÁTICO DE RESULTADOS
# ============================================================
import json
import datetime
 
SAVE_DIR = "/home/lab-admin/Documents/CNVStudy/logs"
os.makedirs(SAVE_DIR, exist_ok=True)
 
timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_name  = f"{subject}_{session}_{timestamp}"
path_json  = os.path.join(SAVE_DIR, base_name + ".json")
path_txt   = os.path.join(SAVE_DIR, base_name + ".txt")
 
# ── Construir diccionario de resultados ──────────────────────
results_dict = {
    "subject"      : subject,
    "session"      : session,
    "timestamp"    : timestamp,
    "pipeline"     : f"avg-ref → notch60 → Butterworth({BP_LOW}–{BP_HIGH}Hz, forward)",
    "channels"     : PICKS_CNV,
    "n_rest"       : int(n_rest_m),
    "n_mi"         : int(n_mi_m),
    "n_rejected"   : int(n_dropped),
    "pct_rejected" : round(100 * n_dropped / n_total, 1),
    "emg_onset_s"  : round(avg_emg_onset, 3),
    "emg_onset_std": round(std_emg_onset, 3),
 
    # Análisis CNV (sección 7)
    "cnv_stats": {
        ch: {
            "mu_rest" : round(v["mu_rest"], 3),
            "mu_mi"   : round(v["mu_mi"],   3),
            "delta"   : round(v["delta"],   3),
            "p"       : round(v["p"],       4),
            "sig"     : v["sig"]
        } for ch, v in cnv_stats.items()
    },
 
    # Modelo 1 — estático
    "model1_static": {
        c: {
            "auc_mean": round(v["auc_mean"], 3),
            "auc_std" : round(v["auc_std"],  3),
            "acc_mean": round(v["acc_mean"], 1),
            "acc_std" : round(v["acc_std"],  1),
        } for c, v in results_static.items()
    },
    "model1_best": {
        "clf": max(results_static, key=lambda c: results_static[c]["auc_mean"]),
        "auc": round(results_static[
            max(results_static, key=lambda c: results_static[c]["auc_mean"])
        ]["auc_mean"], 3)
    },
 
    # Modelo 2 — acumulativo
    "model2_best_per_clf": {
        c: {
            "auc"  : round(max(results_seq[c], key=lambda r: r["auc_mean"])["auc_mean"], 3),
            "t"    : round(max(results_seq[c], key=lambda r: r["auc_mean"])["t"], 2),
            "acc"  : round(max(results_seq[c], key=lambda r: r["auc_mean"])["acc_mean"], 1),
        } for c in ALL_CLFS
    },
    "model2_best": {
        "clf": max(ALL_CLFS, key=lambda c: max(
            r["auc_mean"] for r in results_seq[c])),
        "auc": round(max(
            max(r["auc_mean"] for r in results_seq[c]) for c in ALL_CLFS), 3),
        "t"  : round(max(
            (max(results_seq[c], key=lambda r: r["auc_mean"])
             for c in ALL_CLFS),
            key=lambda r: r["auc_mean"])["t"], 2)
    },
    "model2_anticipation": {
        c: {
            "auc_from": round(next(
                (r["t"] for r in results_seq[c] if r["auc_mean"] > 0.65),
                float("nan")), 2),
            "acc_from": round(next(
                (r["t"] for r in results_seq[c] if r["acc_mean"] > 60.0),
                float("nan")), 2),
        } for c in ALL_CLFS
    },
 
    # Modelo 3 — ventanas deslizantes
    "model3_best_per_clf": {
        c: {
            "auc"    : round(max(results_sliding[c], key=lambda r: r["auc_mean"])["auc_mean"], 3),
            "window" : [
                round(max(results_sliding[c], key=lambda r: r["auc_mean"])["t_start"], 2),
                round(max(results_sliding[c], key=lambda r: r["auc_mean"])["t_end"],   2),
            ],
            "acc"    : round(max(results_sliding[c], key=lambda r: r["auc_mean"])["acc_mean"], 1),
        } for c in ALL_CLFS
    },
    "model3_best": {
        "clf": max(ALL_CLFS, key=lambda c: max(
            r["auc_mean"] for r in results_sliding[c])),
        "auc": round(max(
            max(r["auc_mean"] for r in results_sliding[c]) for c in ALL_CLFS), 3),
    },
 
    # Feature selection (sección 12)
    "feature_selection": {
        "anova_ranking" : top_anova,
        "rfe_ranking"   : top_rfe,
        "selected_channels": PICKS_REDUCED,
 
        # Tablas completas con canales reducidos
        "model1_reduced": {
            c: {
                "auc_mean": round(v["auc_mean"], 3),
                "auc_std" : round(v["auc_std"],  3),
                "acc_mean": round(v["acc_mean"], 1),
                "acc_std" : round(v["acc_std"],  1),
                "delta_vs_baseline": round(
                    v["auc_mean"] - results_static[c]["auc_mean"], 3)
            } for c, v in results_static_red.items()
        },
        "model2_reduced": {
            c: {
                "auc": round(max(results_seq_red[c],
                                 key=lambda r: r["auc_mean"])["auc_mean"], 3),
                "t"  : round(max(results_seq_red[c],
                                 key=lambda r: r["auc_mean"])["t"], 2),
                "acc": round(max(results_seq_red[c],
                                 key=lambda r: r["auc_mean"])["acc_mean"], 1),
                "delta_vs_baseline": round(
                    max(results_seq_red[c], key=lambda r: r["auc_mean"])["auc_mean"] -
                    max(results_seq[c],     key=lambda r: r["auc_mean"])["auc_mean"], 3)
            } for c in ALL_CLFS
        },
        "model3_reduced": {
            c: {
                "auc"   : round(max(results_sliding_red[c],
                                    key=lambda r: r["auc_mean"])["auc_mean"], 3),
                "window": [
                    round(max(results_sliding_red[c],
                              key=lambda r: r["auc_mean"])["t_start"], 2),
                    round(max(results_sliding_red[c],
                              key=lambda r: r["auc_mean"])["t_end"], 2),
                ],
                "acc"   : round(max(results_sliding_red[c],
                                    key=lambda r: r["auc_mean"])["acc_mean"], 1),
                "delta_vs_baseline": round(
                    max(results_sliding_red[c], key=lambda r: r["auc_mean"])["auc_mean"] -
                    max(results_sliding[c],     key=lambda r: r["auc_mean"])["auc_mean"], 3)
            } for c in ALL_CLFS
        },
 
        "model1_reduced_best": {
            "clf": max(results_static_red,
                       key=lambda c: results_static_red[c]["auc_mean"]),
            "auc": round(results_static_red[
                max(results_static_red,
                    key=lambda c: results_static_red[c]["auc_mean"])
            ]["auc_mean"], 3)
        },
        "model2_reduced_best": {
            "clf": max(ALL_CLFS, key=lambda c: max(
                r["auc_mean"] for r in results_seq_red[c])),
            "auc": round(max(
                max(r["auc_mean"] for r in results_seq_red[c])
                for c in ALL_CLFS), 3)
        },
        "model3_reduced_best": {
            "clf": max(ALL_CLFS, key=lambda c: max(
                r["auc_mean"] for r in results_sliding_red[c])),
            "auc": round(max(
                max(r["auc_mean"] for r in results_sliding_red[c])
                for c in ALL_CLFS), 3)
        },
    }
}
 
# ── Guardar JSON — reemplazar NaN por null (JSON válido) ─────
def clean_nan(obj):
    """Convierte NaN e inf a None para JSON válido."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj
 
with open(path_json, "w") as f:
    json.dump(clean_nan(results_dict), f, indent=2)
print(f"\n💾  JSON guardado: {path_json}")
 
# ── Guardar TXT ──────────────────────────────────────────────
with open(path_txt, "w") as f:
    f.write(f"CNV BCI — RESULTADOS\n")
    f.write(f"{'='*68}\n")
    f.write(f"Sujeto/Sesión : {subject} | {session}\n")
    f.write(f"Timestamp     : {timestamp}\n")
    f.write(f"Pipeline      : avg-ref → notch60 → "
            f"Butterworth({BP_LOW}–{BP_HIGH}Hz, forward)\n")
    f.write(f"Canales       : {PICKS_CNV}\n")
    f.write(f"Trials        : Rest={n_rest_m} | MI={n_mi_m} "
            f"| Rechazados={n_dropped} ({results_dict['pct_rejected']}%)\n")
    f.write(f"EMG onset     : {avg_emg_onset:.3f} s "
            f"± {std_emg_onset:.3f} s\n")
 
    f.write(f"\n{'─'*50}\n")
    f.write("CNV Stats (ventana [-2.0, 0.0] s)\n")
    f.write(f"{'─'*50}\n")
    f.write(f"{'Canal':<8} {'Rest µV':>9} {'MI µV':>9} "
            f"{'Δ µV':>9} {'p':>9} {'sig':>5}\n")
    for ch, v in cnv_stats.items():
        f.write(f"{ch:<8} {v['mu_rest']:>9.3f} {v['mu_mi']:>9.3f} "
                f"{v['delta']:>9.3f} {v['p']:>9.4f} {v['sig']:>5}\n")
 
    f.write(f"\n{'─'*50}\n")
    f.write("Modelo 1 — Estático (7 canales, 77 features)\n")
    f.write(f"{'─'*50}\n")
    f.write(f"{'Modelo':<14} {'AUC':>7} {'±std':>6} {'Acc%':>7} {'±std':>6}\n")
    for c, v in results_static.items():
        flag = " ← mejor" if c == results_dict["model1_best"]["clf"] else ""
        f.write(f"{c:<14} {v['auc_mean']:>7.3f}  {v['auc_std']:>5.3f}  "
                f"{v['acc_mean']:>6.1f}%  {v['acc_std']:>5.1f}%{flag}\n")
 
    f.write(f"\n{'─'*50}\n")
    f.write("Modelo 2 — Acumulativo: mejor AUC por modelo\n")
    f.write(f"{'─'*50}\n")
    f.write(f"{'Modelo':<14} {'t mejor':>8} {'AUC mejor':>10} "
            f"{'AUC>0.65 desde':>16}\n")
    for c in ALL_CLFS:
        b   = results_dict["model2_best_per_clf"][c]
        ant = results_dict["model2_anticipation"][c]
        auc_t = f"t={ant['auc_from']:.2f}s" if not \
            (isinstance(ant["auc_from"], float) and
             ant["auc_from"] != ant["auc_from"]) else "nunca"
        f.write(f"{c:<14} {b['t']:>8.2f}s {b['auc']:>10.3f} "
                f"{auc_t:>16}\n")
 
    f.write(f"\n{'─'*50}\n")
    f.write("Modelo 3 — Ventanas deslizantes: mejor AUC por modelo\n")
    f.write(f"{'─'*50}\n")
    f.write(f"{'Modelo':<14} {'Ventana':>18} {'AUC mejor':>10}\n")
    for c in ALL_CLFS:
        b = results_dict["model3_best_per_clf"][c]
        f.write(f"{c:<14} [{b['window'][0]:.2f},{b['window'][1]:.2f}]s "
                f"{b['auc']:>10.3f}\n")
 
    f.write(f"\n{'─'*50}\n")
    f.write(f"Feature Selection — canales reducidos: {PICKS_REDUCED}\n")
    f.write(f"{'─'*50}\n")
    fs = results_dict["feature_selection"]
    f.write(f"ANOVA ranking : {fs['anova_ranking']}\n")
    f.write(f"RFE ranking   : {fs['rfe_ranking']}\n")
    f.write(f"Seleccionados : {fs['selected_channels']}\n")
 
    f.write(f"\nModelo 1 reducido ({N_FEAT_RED} features = "
            f"{N_TIMEPOINTS} pts × {N_CH_RED} ch):\n")
    f.write(f"{'Modelo':<14} {'AUC':>7} {'±std':>6} {'Acc%':>7} {'±std':>6}  vs baseline\n")
    f.write("-"*58 + "\n")
    for c in ALL_CLFS:
        r   = results_static_red[c]
        b   = results_static[c]["auc_mean"]
        d   = r["auc_mean"] - b
        ds  = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
        flag = " ← mejor" if c == fs["model1_reduced_best"]["clf"] else ""
        f.write(f"{c:<14} {r['auc_mean']:>7.3f}  {r['auc_std']:>5.3f}  "
                f"{r['acc_mean']:>6.1f}%  {r['acc_std']:>5.1f}%  "
                f"({ds}){flag}\n")
 
    f.write(f"\nModelo 2 reducido — mejor AUC por modelo:\n")
    f.write(f"{'Modelo':<14} {'t mejor':>8} {'AUC mejor':>10} {'Acc%':>7}  vs baseline\n")
    f.write("-"*50 + "\n")
    for c in ALL_CLFS:
        br  = max(results_seq_red[c], key=lambda r: r["auc_mean"])
        bb  = max(results_seq[c],     key=lambda r: r["auc_mean"])["auc_mean"]
        d   = br["auc_mean"] - bb
        ds  = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
        flag = " ← mejor" if c == fs["model2_reduced_best"]["clf"] else ""
        f.write(f"{c:<14} {br['t']:>8.2f}s {br['auc_mean']:>10.3f} "
                f"{br['acc_mean']:>6.1f}%  ({ds}){flag}\n")
 
    f.write(f"\nModelo 3 reducido — mejor ventana por modelo:\n")
    f.write(f"{'Modelo':<14} {'Ventana':>18} {'AUC mejor':>10} {'Acc%':>7}  vs baseline\n")
    f.write("-"*58 + "\n")
    for c in ALL_CLFS:
        br  = max(results_sliding_red[c], key=lambda r: r["auc_mean"])
        bb  = max(results_sliding[c],     key=lambda r: r["auc_mean"])["auc_mean"]
        d   = br["auc_mean"] - bb
        ds  = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
        flag = " ← mejor" if c == fs["model3_reduced_best"]["clf"] else ""
        f.write(f"{c:<14} [{br['t_start']:.2f},{br['t_end']:.2f}]s "
                f"{br['auc_mean']:>10.3f} {br['acc_mean']:>6.1f}%  "
                f"({ds}){flag}\n")
 
    f.write(f"\nModelo 2 reducido — tabla completa (AUC / Acc% por paso):\n")
    f.write(f"{'t':>7} {'feat':>5}  " +
            "  ".join(f"{c:<14}" for c in ALL_CLFS) + "\n")
    f.write("-"*(7 + 5 + 16 * len(ALL_CLFS)) + "\n")
    for i, step_r in enumerate(results_seq_red[ALL_CLFS[0]]):
        row = f"{step_r['t']:>7.3f} {step_r['n_feat']:>5}  "
        for c in ALL_CLFS:
            r = results_seq_red[c][i]
            auc_s = f"{r['auc_mean']:.3f}" if not (
                c == "MDM_Riemann" and r["auc_mean"] == 0.5) else " N/A"
            row += f"  {auc_s}/{r['acc_mean']:5.1f}%  "
        f.write(row + "\n")
 
    f.write(f"\nModelo 3 reducido — tabla completa (AUC / Acc% por ventana):\n")
    f.write(f"{'Centro':>8}  {'Ventana':<18}  " +
            "  ".join(f"{c:<14}" for c in ALL_CLFS) + "\n")
    f.write("-"*(8 + 18 + 16 * len(ALL_CLFS)) + "\n")
    for i, win_r in enumerate(results_sliding_red[ALL_CLFS[0]]):
        row = (f"{win_r['t_center']:>8.3f}  "
               f"[{win_r['t_start']:.2f},{win_r['t_end']:.2f}]  ")
        for c in ALL_CLFS:
            r = results_sliding_red[c][i]
            auc_s = f"{r['auc_mean']:.3f}" if not (
                c == "MDM_Riemann" and r["auc_mean"] == 0.5) else " N/A"
            row += f"  {auc_s}/{r['acc_mean']:5.1f}%  "
        f.write(row + "\n")
 
    f.write(f"\n{'='*68}\n")
    f.write(f"Referencia Racz 2023: Acc CNV = 74.01% "
            f"(12 sujetos, 120 trials)\n")
    f.write(f"{'='*68}\n")
 
print(f"📄  TXT guardado : {path_txt}")
print(f"\n✅  Archivos guardados en: {SAVE_DIR}")
