"""
================================================================================
CNV VISUAL DIAGNOSTIC — ERP + TOPOPLOTS ONLY
================================================================================

Versión ligera de Generate_Decoder_MotorCap_TimePoints.py.

Objetivo:
  - Generar las mismas figuras diagnósticas de ERP y topoplots.
  - Evitar entrenamiento/evaluación de modelos para revisar señales rápido.

Importante:
  - La señal de rechazo se calcula igual que el pipeline principal:
        CAR -> notch 60 Hz -> 0.1-1.0 Hz -> epochs -> rechazo en PICKS_CNV
  - La señal visual puede elegirse independientemente:
        CAR, NO_CAR, CAR_LAPLACIAN
  - Esto NO guarda modelos y NO modifica ningún archivo de datos.

Uso rápido:
  python Generate_CNV_ERP_Topoplots_Only.py

Uso con argumentos:
  python Generate_CNV_ERP_Topoplots_Only.py --subject CNV_PILOT_SUBJ_020 --session S001_OFFLINE_FES --ref CAR_LAPLACIAN
================================================================================
"""

import argparse
import os

import bci_runtime_env  # noqa: F401  # prepara paths/runtime del proyecto
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf


# ============================================================
# 1. CONFIGURACIÓN RÁPIDA
# ============================================================
subject = "CNV_PILOT_SUBJ_022"
session = "S001_OFFLINE"
base_dir = getattr(config, "DATA_DIR", "/home/lab-admin/Documents/CNVStudy")

CHANNELS_TO_DROP = ["M1", "M2", "T7", "T8", "Fp1", "Fpz", "Fp2"]
CHANNELS_TO_INTERPOLATE = []

PICKS_CNV = [
    "FC3", "FC1", "FCz",
    "C3",  "C1",  "Cz",
    "CP3", "CP1", "CPz",
]

CNV_WINDOW = (-2.0, 0.0)
ERP_YLIM = (-10.0, 10.0)
CSD_ERP_YLIM = None  # None = escala automática robusta para Laplacian/CSD

# Opciones: "CAR", "NO_CAR", "CAR_LAPLACIAN", "NO_CAR_LAPLACIAN"
VISUAL_EEG_REFERENCE_MODE = "CAR"

# Escala visual de topoplots.
# Si queda todo blanco, baja TOPO_VLIM_PERCENTILE o fija TOPO_VLIM_ABS.
# Ejemplos:
#   TOPO_VLIM_ABS = 5.0     # rango fijo ±5
#   TOPO_VLIM_ABS = None    # rango automático robusto
TOPO_VLIM_ABS = None
TOPO_VLIM_PERCENTILE = 90.0
TOPO_VLIM_MIN = 0.5

EEG_L_FREQ = 0.1
EEG_H_FREQ = 2.0

EMG_L_FREQ = 20.0
EMG_H_FREQ = 200.0
EMG_BASELINE_WINDOW = (-3.0, -2.1)
EMG_BASELINE_RMS_MAX_UV = 150.0

REJECT_THRESHOLD = dict(eeg=100e-6)
FLAT_THRESHOLD = dict(eeg=0.1e-6)

RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz",
    "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS = [100, 200]
INTERTRIAL_MARKERS = [600, 620]
INTERTRIAL_PLOT_WINDOW = (-2.5, 0.5)
LEGACY_PRETRIAL_DURATION = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera ERP y topoplots CNV sin entrenar modelos."
    )
    parser.add_argument("--subject", default=subject)
    parser.add_argument("--session", default=session)
    parser.add_argument("--base-dir", default=base_dir)
    parser.add_argument(
        "--ref",
        default=VISUAL_EEG_REFERENCE_MODE,
        choices=["CAR", "NO_CAR", "CAR_LAPLACIAN", "NO_CAR_LAPLACIAN"],
        help="Referencia visual para ERP/topoplots. No cambia la lógica de rechazo.",
    )
    return parser.parse_args()


def load_raw_session(args):
    xdf_dir = os.path.join(
        args.base_dir,
        f"sub-{args.subject}",
        f"ses-{args.session}",
        "eeg",
    )
    if not os.path.isdir(xdf_dir):
        raise FileNotFoundError(f"XDF directory does not exist: {xdf_dir}")

    xdf_files = sorted(
        os.path.join(xdf_dir, f)
        for f in os.listdir(xdf_dir)
        if f.endswith(".xdf")
    )
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {xdf_dir}")

    print(
        f"📂  Processing {len(xdf_files)} XDF file(s) — "
        f"subject: {args.subject} | session: {args.session}"
    )

    raw_list = []
    event_run_labels = []

    for run_idx, xdf_file in enumerate(xdf_files, start=1):
        print(f"   └─ Loading: {os.path.basename(xdf_file)}")
        eeg_s, marker_s = load_xdf(xdf_file)

        eeg_data = np.array(eeg_s["time_series"]).T
        eeg_timestamps = np.array(eeg_s["time_stamps"])
        channel_names = get_channel_names_from_xdf(eeg_s)

        marker_data_all = np.array([
            int(round(float(np.ravel(v)[0])))
            for v in marker_s["time_series"]
        ])
        marker_timestamps_all = np.array(marker_s["time_stamps"])

        keep = np.isin(marker_data_all, TARGET_MARKERS + INTERTRIAL_MARKERS)
        marker_data = marker_data_all[keep]
        marker_timestamps = marker_timestamps_all[keep]
        event_run_labels.extend([run_idx] * len(marker_data))

        valid_ch = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
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
            raw_tmp.info["bads"] = [
                ch for ch in CHANNELS_TO_INTERPOLATE if ch in raw_tmp.ch_names
            ]
            raw_tmp.interpolate_bads(reset_bads=True, verbose=False)

        t0 = eeg_timestamps[0]
        annot = mne.Annotations(
            onset=marker_timestamps - t0,
            duration=np.zeros(len(marker_data)),
            description=[str(m) for m in marker_data],
            orig_time=None,
        )
        raw_tmp.set_annotations(annot)
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    print(f"✅  Raw concatenado — {raw.n_times / raw.info['sfreq']:.1f} s totales")
    return raw, np.asarray(event_run_labels, dtype=int)


def get_events(raw, event_run_labels):
    events_all, event_id_map = mne.events_from_annotations(raw, verbose=False)
    if len(event_run_labels) != len(events_all):
        raise RuntimeError(
            "No se pudo alinear cada evento con su XDF de origen: "
            f"{len(event_run_labels)} etiquetas para {len(events_all)} eventos."
        )

    missing = [str(m) for m in TARGET_MARKERS if str(m) not in event_id_map]
    if missing:
        raise RuntimeError(f"Faltan triggers objetivo en el XDF: {missing}")

    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)": event_id_map["200"],
    }
    target_event_mask = np.isin(events_all[:, 2], list(event_dict.values()))
    events = events_all[target_event_mask]
    event_run_labels = event_run_labels[target_event_mask]
    mi_id = event_id_map["200"]

    print(
        f"📌  Eventos — Rest: {np.sum(events[:, 2] == event_dict['Rest (100)'])}  |"
        f"  MI: {np.sum(events[:, 2] == mi_id)}"
    )
    return events_all, event_id_map, events, event_dict, event_run_labels, mi_id


def estimate_emg_onset(raw, events, mi_id):
    print("\n💪  Calculando latencia de onset EMG desde AUX7 ...")

    avg_emg_onset = np.nan
    std_emg_onset = np.nan
    all_onsets = []

    if "AUX7" not in raw.ch_names:
        print("⚠️   AUX7 no encontrado — no se mostrará latencia EMG")
        return avg_emg_onset, std_emg_onset

    raw_emg = raw.copy().pick(["emg"])
    raw_emg.filter(
        l_freq=EMG_L_FREQ,
        h_freq=EMG_H_FREQ,
        picks="all",
        method="iir",
        phase="zero",
        verbose=False,
    )
    raw_emg.notch_filter(freqs=[60.0], picks="all", verbose=False)

    raw_emg_filt = raw_emg.copy()
    raw_env = raw_emg.copy()
    raw_env._data = np.abs(raw_emg.get_data())
    raw_env.filter(
        l_freq=None,
        h_freq=10.0,
        picks="all",
        method="iir",
        phase="forward",
        verbose=False,
    )

    epochs_emg = mne.Epochs(
        raw_env,
        events,
        event_id={"MI": mi_id},
        tmin=-2.0,
        tmax=5.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epochs_emg_filt = mne.Epochs(
        raw_emg_filt,
        events,
        event_id={"MI": mi_id},
        tmin=-2.0,
        tmax=5.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    emg_times = epochs_emg.times
    emg_data = epochs_emg.get_data()[:, 0, :] * 1e6
    emg_filt_data = epochs_emg_filt.get_data()[:, 0, :] * 1e6

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
            "⚠️   EMG no interpretable: actividad compatible con FES "
            f"(RMS pre-trigger mediano={emg_baseline_rms_uv:.1f} µV)."
        )
        return avg_emg_onset, std_emg_onset

    for trial in emg_data:
        idx_zero = np.argmin(np.abs(emg_times))
        threshold = np.mean(trial[:idx_zero]) + 5 * np.std(trial[:idx_zero])
        post = trial[idx_zero:]
        if np.any(post > threshold):
            all_onsets.append(emg_times[idx_zero + np.argmax(post > threshold)])

    if all_onsets:
        avg_emg_onset = float(np.mean(all_onsets))
        std_emg_onset = float(np.std(all_onsets))
        print(
            f"⏱️   EMG Onset: {avg_emg_onset:.3f} s ± "
            f"{std_emg_onset:.3f} s ({len(all_onsets)}/{len(emg_data)} trials)"
        )
    else:
        print("⚠️   No se detectó un onset EMG confiable.")

    return avg_emg_onset, std_emg_onset


def preprocess_for_rejection(raw):
    raw_car = raw.copy()
    raw_car.set_eeg_reference("average", projection=False, verbose=False)
    raw_car.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw_car.filter(
        l_freq=EEG_L_FREQ,
        h_freq=EEG_H_FREQ,
        method="iir",
        phase="zero",
        picks="eeg",
        verbose=False,
    )
    print("🎛️   Rechazo preparado con CAR → notch60 → 0.1–2 Hz")
    return raw_car


def preprocess_for_visual(raw, visual_mode):
    raw_visual = raw.copy()

    if visual_mode == "CAR":
        raw_visual.set_eeg_reference("average", projection=False, verbose=False)
        raw_visual.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
        raw_visual.filter(
            l_freq=EEG_L_FREQ,
            h_freq=EEG_H_FREQ,
            method="iir",
            phase="forward",
            picks="eeg",
            verbose=False,
        )
        label = "CAR"

    elif visual_mode == "NO_CAR":
        raw_visual.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
        raw_visual.filter(
            l_freq=EEG_L_FREQ,
            h_freq=EEG_H_FREQ,
            method="iir",
            phase="forward",
            picks="eeg",
            verbose=False,
        )
        label = "sin CAR"

    elif visual_mode == "CAR_LAPLACIAN":
        raw_visual.set_eeg_reference("average", projection=False, verbose=False)
        raw_visual = mne.preprocessing.compute_current_source_density(raw_visual)
        # Después de CSD/Laplacian los canales pasan de tipo "eeg" a "csd".
        raw_visual.notch_filter(freqs=[60.0], picks="data", method="iir", verbose=False)
        raw_visual.filter(
            l_freq=EEG_L_FREQ,
            h_freq=EEG_H_FREQ,
            method="iir",
            phase="forward",
            picks="data",
            verbose=False,
        )
        label = "CAR+Laplacian"

    elif visual_mode == "NO_CAR_LAPLACIAN":
        raw_visual = mne.preprocessing.compute_current_source_density(raw_visual)
        # Después de CSD/Laplacian los canales pasan de tipo "eeg" a "csd".
        raw_visual.notch_filter(freqs=[60.0], picks="data", method="iir", verbose=False)
        raw_visual.filter(
            l_freq=EEG_L_FREQ,
            h_freq=EEG_H_FREQ,
            method="iir",
            phase="forward",
            picks="data",
            verbose=False,
        )
        label = "Laplacian sin CAR explícito"

    else:
        raise ValueError(f"Modo visual desconocido: {visual_mode}")

    print(f"👁️   Visual preparado con ref={label}")
    return raw_visual, label


def make_epochs(raw_car, raw_visual, events, event_dict):
    epochs_all = mne.Epochs(
        raw_car,
        events,
        event_id=event_dict,
        tmin=-5.0,
        tmax=6.0,
        baseline=(-5.0, -3.0),
        reject=None,
        flat=None,
        preload=True,
        detrend=None,
        verbose=False,
    )

    pick_idx = [epochs_all.ch_names.index(ch) for ch in PICKS_CNV if ch in epochs_all.ch_names]
    if not pick_idx:
        raise RuntimeError(f"Ningún canal PICKS_CNV encontrado: {PICKS_CNV}")

    data_cnv = epochs_all.get_data()[:, pick_idx, :]
    pp = data_cnv.max(axis=2) - data_cnv.min(axis=2)

    reject_val = REJECT_THRESHOLD["eeg"]
    flat_val = FLAT_THRESHOLD["eeg"]
    reject_mask = pp.max(axis=1) > reject_val
    flat_mask = pp.max(axis=1) < flat_val
    drop_mask = reject_mask | flat_mask
    drop_indices = np.where(drop_mask)[0].tolist()

    epochs = epochs_all.copy()
    epochs.drop(drop_indices, reason="MANUAL_REJECT")

    epochs_visual_all = mne.Epochs(
        raw_visual,
        events,
        event_id=event_dict,
        tmin=-5.0,
        tmax=6.0,
        baseline=(-5.0, -3.0),
        reject=None,
        flat=None,
        preload=True,
        detrend=None,
        verbose=False,
    )
    epochs_visual = epochs_visual_all.copy()
    epochs_visual.drop(drop_indices, reason="MATCH_CAR_REJECT")

    n_rest = len(epochs["Rest (100)"])
    n_mi = len(epochs["MI (200)"])
    n_dropped = len(drop_indices)
    n_total = n_dropped + n_rest + n_mi

    print("\n🛡️   Rechazo localizado en canales CNV:")
    print(f"   Rechazados  : {n_dropped} / {n_total} ({100*n_dropped/n_total:.1f}%)")
    print(f"   Rest trials : {n_rest}")
    print(f"   MI trials   : {n_mi}")

    if n_rest == 0 or n_mi == 0:
        raise RuntimeError("Todos los epochs de una clase fueron rechazados.")

    return epochs, epochs_visual, drop_indices, n_rest, n_mi


def make_intertrial_epochs(raw_car, raw_visual, events_all, event_id_map, events, event_run_labels):
    sfreq = raw_car.info["sfreq"]
    neutral_event_code = 999

    if "600" in event_id_map:
        intertrial_begin_events = events_all[events_all[:, 2] == event_id_map["600"]]
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
        (neutral_event_samples + int(round(INTERTRIAL_PLOT_WINDOW[0] * sfreq)) >= raw_car.first_samp)
        & (
            neutral_event_samples + int(round(INTERTRIAL_PLOT_WINDOW[1] * sfreq))
            < raw_car.first_samp + raw_car.n_times
        )
    ]

    neutral_events = np.column_stack([
        neutral_event_samples,
        np.zeros(len(neutral_event_samples), dtype=int),
        np.full(len(neutral_event_samples), neutral_event_code, dtype=int),
    ])

    epochs_intertrial = mne.Epochs(
        raw_car,
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
        (neutral_pp.max(axis=1) > REJECT_THRESHOLD["eeg"])
        | (neutral_pp.max(axis=1) < FLAT_THRESHOLD["eeg"])
    )
    neutral_drop_indices = np.flatnonzero(neutral_drop_mask).tolist()
    epochs_intertrial.drop(neutral_drop_indices, reason="INTERTRIAL_REJECT")

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
        neutral_drop_indices,
        reason="MATCH_CAR_INTERTRIAL_REJECT",
    )

    print("\n⚪  Referencia intertrial para visualización:")
    print(f"   Fuente      : {intertrial_source}")
    print(
        f"   Ventana     : {INTERTRIAL_PLOT_WINDOW[0]:.1f} a "
        f"{INTERTRIAL_PLOT_WINDOW[1]:.1f} s"
    )
    print(
        f"   Conservados : {len(epochs_intertrial)} / "
        f"{len(neutral_events)} intervalos"
    )

    return epochs_intertrial_visual, len(epochs_intertrial)


def plot_erp(
    args,
    erp_epochs,
    erp_intertrial_epochs,
    n_rest,
    n_mi,
    n_intertrial,
    erp_reference_label,
    visual_mode,
    avg_emg_onset,
    std_emg_onset,
):
    print("\n🖥️   Generando ERP...")

    times = erp_epochs.times
    is_laplacian = "LAPLACIAN" in visual_mode
    visual_data_picks = "data" if is_laplacian else "eeg"
    visual_y_label = "CSD (scaled)" if is_laplacian else "Amplitude (µV)"

    def get_mean_sem(epochs_obj, condition):
        data = epochs_obj[condition].get_data(picks=visual_data_picks)
        mean = np.mean(data, axis=0) * 1e6
        sem = np.std(data, axis=0) / np.sqrt(data.shape[0]) * 1e6
        #std = np.std(data, axis=0) * 1e6
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
    erp_ch_names = erp_epochs.copy().pick(visual_data_picks).ch_names

    channel_grid = [
        PICKS_CNV[0:3],
        PICKS_CNV[3:6],
        PICKS_CNV[6:9],
    ]
    if is_laplacian:
        if CSD_ERP_YLIM is None:
            plot_indices = [erp_ch_names.index(ch) for ch in PICKS_CNV if ch in erp_ch_names]
            neutral_indices = [
                intertrial_ch_names.index(ch)
                for ch in PICKS_CNV
                if ch in intertrial_ch_names
            ]
            y_values = [
                m_100[plot_indices],
                m_200[plot_indices],
            ]
            if neutral_indices:
                y_values.append(intertrial_mean[neutral_indices])
            ylim_abs = float(np.percentile(np.abs(np.concatenate([v.ravel() for v in y_values])), 98))
            ylim_abs = max(ylim_abs, 1.0)
            erp_ylim = (-ylim_abs, ylim_abs)
        else:
            erp_ylim = CSD_ERP_YLIM
    else:
        erp_ylim = ERP_YLIM

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)

    for row in range(3):
        for col in range(3):
            ch = channel_grid[row][col]
            ax = axes[row, col]

            if ch in erp_ch_names:
                idx = erp_ch_names.index(ch)

                ax.plot(times, m_100[idx], color="#2166ac", label="Rest (100)", linewidth=2.0)
                ax.fill_between(
                    times,
                    m_100[idx] - s_100[idx],
                    m_100[idx] + s_100[idx],
                    color="#2166ac",
                    alpha=0.15,
                )
                ax.plot(times, m_200[idx], color="#d6604d", label="MI (200)", linewidth=2.5)
                ax.fill_between(
                    times,
                    m_200[idx] - s_200[idx],
                    m_200[idx] + s_200[idx],
                    color="#d6604d",
                    alpha=0.20,
                )

                if ch in intertrial_ch_names:
                    neutral_idx = intertrial_ch_names.index(ch)
                    ax.plot(
                        intertrial_times,
                        intertrial_mean[neutral_idx],
                        color="#4d4d4d",
                        label=f"Intertrial (n={n_intertrial})",
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

                if np.isfinite(avg_emg_onset):
                    ax.axvspan(
                        avg_emg_onset - std_emg_onset,
                        avg_emg_onset + std_emg_onset,
                        color="limegreen",
                        alpha=0.18,
                        label="EMG window",
                    )
                    ax.axvline(
                        avg_emg_onset,
                        color="darkgreen",
                        linestyle="-",
                        linewidth=2.0,
                        label=f"EMG µ = {avg_emg_onset:.2f} s",
                    )

                ax.axvspan(CNV_WINDOW[0], CNV_WINDOW[1], color="gold", alpha=0.10, zorder=0)

            ax.axvline(0, color="black", ls="--", linewidth=1.5, label="Onset (0 s)")
            ax.axvline(-2.0, color="black", ls=":", linewidth=1.2, label="Prep (−2 s)")
            ax.set_title(f"Ch: {ch}", fontweight="bold")
            ax.set_xlim(-3, 2)
            ax.set_ylim(*erp_ylim)
            ax.grid(True, ls=":", alpha=0.4)
            if col == 0:
                ax.set_ylabel(visual_y_label)
            if row == 2:
                ax.set_xlabel("Time (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.97), fontsize=9)
    plt.suptitle(
        f"CNV Validation — EEG + Muscle Latency + Intertrial Reference\n"
        f"{args.subject}  |  {args.session}  |  n_rest={n_rest}, n_mi={n_mi}  "
        f"|  ERP ref = {erp_reference_label}  "
        f"|  yellow area = CNV window {CNV_WINDOW}\n"
        "Gray intertrial: 3 s shifted onto the visual axis "
        f"{INTERTRIAL_PLOT_WINDOW}; does not share the MI/Rest onset",
        fontsize=13,
        fontweight="bold",
    )
    plt.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.84,
        bottom=0.08,
        hspace=0.38,
        wspace=0.15,
    )


def plot_topoplots(args, erp_epochs, erp_reference_label, visual_mode):
    print("\n🗺️   Generando topoplots...")

    is_laplacian = "LAPLACIAN" in visual_mode
    visual_data_picks = "data" if is_laplacian else "eeg"
    visual_y_label = "CSD (scaled)" if is_laplacian else "Amplitude (µV)"
    topo_times = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

    evoked_rest = erp_epochs["Rest (100)"].average(picks=visual_data_picks)
    evoked_mi = erp_epochs["MI (200)"].average(picks=visual_data_picks)

    # MNE dibuja topomaps usando las unidades internas del Evoked. Para que
    # el mapa y la barra de color usen exactamente la misma escala, creamos
    # copias escaladas manualmente y desactivamos el escalado automático.
    evoked_rest_plot = evoked_rest.copy()
    evoked_mi_plot = evoked_mi.copy()
    evoked_rest_plot.data *= 1e6
    evoked_mi_plot.data *= 1e6

    topo_data = np.concatenate([evoked_rest_plot.data, evoked_mi_plot.data], axis=1)
    if TOPO_VLIM_ABS is None:
        vlim_abs = float(np.percentile(np.abs(topo_data), TOPO_VLIM_PERCENTILE))
        vlim_source = f"{TOPO_VLIM_PERCENTILE:.0f}th pct"
    else:
        vlim_abs = float(TOPO_VLIM_ABS)
        vlim_source = "manual"
    vlim_abs = max(vlim_abs, TOPO_VLIM_MIN)
    print(f"   vlim topomaps ({vlim_source}): ±{vlim_abs:.1f}")

    fig_topo, axes_topo = plt.subplots(
        2,
        len(topo_times),
        figsize=(18, 10),
        constrained_layout=True,
    )

    evoked_rest_plot.plot_topomap(
        times=topo_times,
        axes=axes_topo[0, :],
        average=0.2,
        cmap="RdBu_r",
        vlim=(-vlim_abs, vlim_abs),
        scalings=dict(eeg=1.0, csd=1.0),
        show=False,
        colorbar=False,
    )
    evoked_mi_plot.plot_topomap(
        times=topo_times,
        axes=axes_topo[1, :],
        average=0.2,
        cmap="RdBu_r",
        vlim=(-vlim_abs, vlim_abs),
        scalings=dict(eeg=1.0, csd=1.0),
        show=False,
        colorbar=False,
    )

    axes_topo[0, 0].set_ylabel("REST (100)", fontsize=12, fontweight="bold")
    axes_topo[1, 0].set_ylabel("MI  (200)", fontsize=12, fontweight="bold")

    im = axes_topo[1, -1].images[0]
    cbar = fig_topo.colorbar(
        im,
        ax=axes_topo.ravel().tolist(),
        shrink=0.5,
        orientation="vertical",
        pad=0.02,
    )
    cbar.set_label(f"{visual_y_label}  [vlim ±{vlim_abs:.1f}]", fontsize=11)
    plt.suptitle(
        f"CNV Topographic Maps — {args.subject} | {args.session} | ref={erp_reference_label}\n",
        fontsize=13,
        fontweight="bold",
    )


def main():
    args = parse_args()
    mne.set_log_level("WARNING")

    visual_mode = str(args.ref).upper()
    raw, event_run_labels = load_raw_session(args)
    events_all, event_id_map, events, event_dict, event_run_labels, mi_id = get_events(
        raw,
        event_run_labels,
    )
    avg_emg_onset, std_emg_onset = estimate_emg_onset(raw, events, mi_id)

    raw_car = preprocess_for_rejection(raw)
    raw_visual, erp_reference_label = preprocess_for_visual(raw, visual_mode)

    _, epochs_visual, _, n_rest, n_mi = make_epochs(
        raw_car,
        raw_visual,
        events,
        event_dict,
    )
    epochs_intertrial_visual, n_intertrial = make_intertrial_epochs(
        raw_car,
        raw_visual,
        events_all,
        event_id_map,
        events,
        event_run_labels,
    )

    plot_erp(
        args,
        epochs_visual,
        epochs_intertrial_visual,
        n_rest,
        n_mi,
        n_intertrial,
        erp_reference_label,
        visual_mode,
        avg_emg_onset,
        std_emg_onset,
    )
    plot_topoplots(args, epochs_visual, erp_reference_label, visual_mode)

    print("\n🖼️  Figuras listas para visualización. No se calcularon modelos.")
    plt.show()


if __name__ == "__main__":
    main()
