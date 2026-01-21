#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mne
import pyxdf
import matplotlib.pyplot as plt
import config
from mne.preprocessing import compute_current_source_density

# ============================================================
# CONFIGURACIÓN DUAL
# ============================================================
XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf"

# "CSD" para la técnica del paper (Perrin et al.) o "MANUAL" para tu Laplaciano original
SPATIAL_FILTER_MODE = "MANUAL" 

EVENT_REST = int(config.TRIGGERS["REST_BEGIN"])
EVENT_MOV  = int(config.TRIGGERS["MI_BEGIN"])

TMIN, TMAX = -2.0, 4.0
BASELINE   = (-2.0, -1.0)
L_FREQ, H_FREQ = 0.1, 1.0
NOTCH = 60

PICKS = ["C3", "Cz"]
GRID_LAYOUT = [[None, "FC1", None], ["C3", None, "Cz"], [None, "CP1", None], ["P3", None, "Pz"]]

# Matriz para modo MANUAL
adjacency_matrix = {
    'FP1': ['FPZ', 'F3', 'FZ'], 'FPZ': ['FP1', 'FZ', 'FP2'], 'FP2': ['FPZ', 'FZ', 'F4'],
    'F7': ['FC5'], 'F3': ['FC5', 'FP1', 'FC1'], 'FZ': ['FP1', 'FPZ', 'FP2', 'FC1', 'FC2'],
    'F4': ['FP2', 'FC2', 'FC6'], 'F8': ['FC6'], 'FC5': ['F7', 'F3', 'T7', 'C3'],
    'FC1': ['F3', 'FZ', 'C3', 'CZ'], 'FC2': ['FZ', 'F4', 'CZ', 'C4'], 'FC6': ['F4', 'F8', 'C4', 'T8'],
    'T7': ['FC5', 'CP5'], 'C3': ['FC5', 'FC1', 'CP5', 'CP1'], 'CZ': ['FC1', 'FC2', 'CP1', 'CP2'],
    'C4': ['FC2', 'FC6', 'CP2', 'CP6'], 'T8': ['FC6', 'CP6'], 'CP5': ['T7', 'C3', 'P7', 'P3'],
    'CP1': ['C3', 'CZ', 'P3', 'PZ'], 'CP2': ['CZ', 'C4', 'PZ', 'P4'], 'CP6': ['C4', 'T8', 'P4', 'P8'],
    'P7': ['CP5'], 'P3': ['CP5', 'CP1'], 'PZ': ['CP1', 'CP2', 'POZ'], 'P4': ['CP2', 'CP6'],
    'P8': ['CP6'], 'POZ': ['PZ', 'O1', 'OZ', 'O2'], 'O1': ['POZ', 'OZ'], 'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['POZ', 'OZ']
}

# ============================================================
# PROCESAMIENTO
# ============================================================

def fix_channel_names(ch_names):
    mapping = {"FP1":"Fp1", "FPZ":"Fpz", "FP2":"Fp2", "FZ":"Fz", "CZ":"Cz", "PZ":"Pz", "OZ":"Oz", "POZ":"POz"}
    return [mapping.get(ch.upper(), ch) for ch in ch_names]

def apply_neighbor_operator(raw, adjacency):
    data = raw.get_data()
    new_data = data.copy()
    ch_map = {name.upper(): i for i, name in enumerate(raw.ch_names)}
    for ch, neighbors in adjacency.items():
        if ch.upper() not in ch_map: continue
        valid_nb = [ch_map[nb.upper()] for nb in neighbors if nb.upper() in ch_map]
        if valid_nb:
            new_data[ch_map[ch.upper()], :] -= np.mean(data[valid_nb, :], axis=0)
    raw._data[:] = new_data
    return raw

def make_evokeds(file_path):
    streams, _ = pyxdf.load_xdf(file_path)
    eeg = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
    markers = next(s for s in streams if s["info"]["type"][0].lower() in ["markers", "marker"])

    data = eeg["time_series"].T
    if np.max(np.abs(data)) > 1e3: data *= 1e-6
    ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    
    info = mne.create_info(ch_names=ch_names, sfreq=config.FS, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.drop_channels([ch for ch in raw.ch_names if any(x in ch.upper() for x in ["AUX", "TRIG", "M1", "M2"])])
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, fix_channel_names(raw.ch_names))})
    raw.set_montage("standard_1020")

    if SPATIAL_FILTER_MODE == "CSD":
        print("\n[INFO] Modo CSD: Aplicando Laplaciano Esférico (Perrin et al., 1987)...")
        raw = compute_current_source_density(raw)
    else:
        print("\n[INFO] Modo MANUAL: Aplicando CAR + Operador de Vecinos (Hjorth, 1975)...")
        raw.set_eeg_reference("average")
        apply_neighbor_operator(raw, adjacency_matrix)

    raw.notch_filter(NOTCH, verbose=False).filter(L_FREQ, H_FREQ, verbose=False)

    samples = np.round((markers["time_stamps"] - eeg["time_stamps"][0]) * config.FS).astype(int)
    m_ids = np.array([int(float(v[0])) for v in markers["time_series"]])
    events = np.c_[samples, np.zeros_like(samples), m_ids]
    events = events[np.isin(events[:, 2], [EVENT_REST, EVENT_MOV])]

    epochs = mne.Epochs(raw, events, {"REST": EVENT_REST, "MOV": EVENT_MOV}, TMIN, TMAX, baseline=BASELINE, preload=True)
    return epochs, epochs["REST"].average(), epochs["MOV"].average()

# ============================================================
# VISUALIZACIÓN REINTEGRADA (MEDIA ± STD)
# ============================================================

def plot_mean_std_comparison(epochs, picks):
    """Muestra media y desviación estándar para canales clave."""
    scale = 1e6
    for ch in picks:
        if ch not in epochs.ch_names: continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for cond, color in [("REST", "blue"), ("MOV", "red")]:
            data = epochs[cond].get_data(picks=ch)[:, 0, :] * scale
            ax.plot(epochs.times, np.mean(data, axis=0), label=cond, color=color)
            ax.fill_between(epochs.times, np.mean(data, axis=0) - np.std(data, axis=0), 
                            np.mean(data, axis=0) + np.std(data, axis=0), color=color, alpha=0.15)
        ax.axvline(0, color='k', ls='--')
        ax.set_title(f"MRCP ({SPATIAL_FILTER_MODE}) Mean ± STD @ {ch}")
        ax.legend()
        plt.show()

def plot_grid_mean_std(epochs, layout):
    """Grid de canales con media +/- std."""
    rows, cols = len(layout), len(layout[0])
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharex=True, sharey=True)
    for r in range(rows):
        for c in range(cols):
            ch = layout[r][c]
            ax = axes[r, c] if rows > 1 else axes[c]
            if ch and ch in epochs.ch_names:
                for cond, col in [("REST", "blue"), ("MOV", "red")]:
                    d = epochs[cond].get_data(picks=ch)[:, 0, :] * 1e6
                    ax.plot(epochs.times, np.mean(d, axis=0), color=col)
                ax.set_title(ch)
                ax.axvline(0, color='k', ls='--', lw=1)
            else:
                ax.axis("off")
    plt.tight_layout()
    plt.show()

def inspect_evoked_with_topomaps(ev_rest, ev_mov, times, v_min_max=None):
    """Topomaps duales (CSD/Manual) con escala ajustable."""
    ch_type = 'csd' if SPATIAL_FILTER_MODE == "CSD" else 'eeg'
    scale_val = 1e6
    vmax = float(v_min_max) if v_min_max else max(np.max(np.abs(ev_rest.data)), np.max(np.abs(ev_mov.data))) * scale_val * 0.7

    for label, ev in [('REST', ev_rest), ('MOV', ev_mov)]:
        fig = ev.plot_topomap(times=times, ch_type=ch_type, scalings={ch_type: scale_val},
                             vlim=(-vmax, vmax), cmap='RdBu_r', colorbar=True, show=False)
        fig.suptitle(f"Topografía MRCP ({SPATIAL_FILTER_MODE}): {label}")
        plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    epochs, ev_rest, ev_mov = make_evokeds(XDF_FILE)

    # 1. Media ± STD (Canales individuales)
    plot_mean_std_comparison(epochs, PICKS)

    # 2. Grid de canales
    plot_grid_mean_std(epochs, GRID_LAYOUT)

    # 3. Topomaps (Escala recomendada: ~30000 para CSD, ~40 para MANUAL)
    inspect_evoked_with_topomaps(ev_rest, ev_mov, times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], v_min_max=5)