#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mne
import pyxdf
import matplotlib.pyplot as plt
import config

from matplotlib.colors import Normalize
from mne.viz import plot_topomap

# ============================================================
# CONFIG
# ============================================================
XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf"

# Force trigger ids to int
EVENT_REST = int(config.TRIGGERS["REST_BEGIN"])  # e.g., 100
EVENT_MOV  = int(config.TRIGGERS["MI_BEGIN"])    # e.g., 200

# MRCP epoch window
TMIN, TMAX = -2.0, 4.0
BASELINE   = (-2.0, -1.0)

# MRCP band (slow potentials)
L_FREQ, H_FREQ = 0.1, 1.0
NOTCH = 60

# Channels of interest (edit if your montage labels differ)
PICKS = ["C3", "CZ"]

# ------------------------------------------------------------
# NEW: Spatial processing config (CAR + neighbor operator)
# ------------------------------------------------------------
USE_SPATIAL_OPERATOR = True
SPATIAL_MODE = "subtract"     # "subtract" replicates your old code: ch -= mean(neighbors)
EXCLUDE_FROM_SPATIAL = {"TRIGGER", "M1", "M2"}  # exclude AUX* automatically too

# ------------------------------------------------------------
# NEW: Plot config (Mean ± STD & Grid)
# ------------------------------------------------------------
PLOT_MEAN_STD = True   # show mean ± std bands on single-channel plots
PLOT_GRID = True       # show subplot grid like your image

GRID_LAYOUT = [
    #perdo["F3",  None, "FZ"],
    [None, "FC1", None],
    ["C3", None, "CZ"],
    [None, "CP1", None],
    ["P3", None, "PZ"],
]

# Your adjacency matrix (must match raw channel names)
adjacency_matrix = {
    'FP1': ['FPZ', 'F3', 'FZ'],
    'FPZ': ['FP1', 'FZ', 'FP2'],
    'FP2': ['FPZ', 'FZ', 'F4'],
    'F7': ['FC5'],
    'F3': ['FC5', 'FP1', 'FC1'],
    'FZ': ['FP1', 'FPZ', 'FP2', 'FC1', 'FC2'],
    'F4': ['FP2', 'FC2', 'FC6'],
    'F8': ['FC6'],
    'FC5': ['F7', 'F3', 'T7', 'C3'],
    'FC1': ['F3', 'FZ', 'C3', 'CZ'],
    'FC2': ['FZ', 'F4', 'CZ', 'C4'],
    'FC6': ['F4', 'F8', 'C4', 'T8'],
    'T7': ['FC5', 'CP5'],
    'C3': ['FC5', 'FC1', 'CP5', 'CP1'],
    'CZ': ['FC1', 'FC2', 'CP1', 'CP2'],
    'C4': ['FC2', 'FC6', 'CP2', 'CP6'],
    'T8': ['FC6', 'CP6'],
    'CP5': ['T7', 'C3', 'P7', 'P3'],
    'CP1': ['C3', 'CZ', 'P3', 'PZ'],
    'CP2': ['CZ', 'C4', 'PZ', 'P4'],
    'CP6': ['C4', 'T8', 'P4', 'P8'],
    'P7': ['CP5'],
    'P3': ['CP5', 'CP1'],
    'PZ': ['CP1', 'CP2', 'POZ'],
    'P4': ['CP2', 'CP6'],
    'P8': ['CP6'],
    'POZ': ['PZ', 'O1', 'OZ', 'O2'],
    'O1': ['POZ', 'OZ'],
    'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['POZ', 'OZ']
}

# Times (in seconds) where we want to plot MRCP topomaps
TOPOMAP_TIMES = [0.0]   # <--- Modifica aquí

# ============================================================
# XDF loading (pyxdf)
# ============================================================
def load_xdf_pyxdf(file_path):
    streams, header = pyxdf.load_xdf(file_path)

    print(f"Found {len(streams)} streams in the file.\n")
    for s in streams:
        name = s["info"]["name"][0]
        typ  = s["info"]["type"][0]
        rate = s["info"].get("nominal_srate", ["?"])[0]
        print(f"- {name} ({typ}), nominal rate: {rate} Hz")

    eeg = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")

    markers = None
    for s in streams:
        name = s["info"]["name"][0]
        typ  = s["info"]["type"][0]
        if typ.lower() in ("markers", "marker"):
            print("Marker stream encontrado:", name)
            if "eegosports" not in name.lower():
                markers = s
                break

    if markers is None:
        raise RuntimeError("⚠️ No se encontró ningún stream de markers en este archivo.")
    else:
        print(f"\n✅ Usando stream de markers: {markers['info']['name'][0]}")

    return eeg, markers


def _get_channel_names(eeg_stream, n_ch):
    """Try to read channel labels from XDF metadata; fallback to Ch0.."""
    try:
        ch_info = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ch_names = [c["label"][0] for c in ch_info]

        print("\n[DEBUG] Channel labels found in XDF metadata:")
        for i, name in enumerate(ch_names):
            print(f"  {i}: {name}")

        if len(ch_names) == n_ch:
            print(f"[DEBUG] Number of channel labels matches n_ch = {n_ch}")
            return ch_names
        else:
            print(
                f"[WARN] Channel label count ({len(ch_names)}) "
                f"does NOT match n_ch ({n_ch}). Falling back."
            )

    except Exception as e:
        print("\n[WARN] Could not read channel labels from XDF metadata.")
        print("Reason:", repr(e))

    fallback = [f"Ch{i}" for i in range(n_ch)]
    print("\n[DEBUG] Using fallback channel names:")
    print(fallback[:min(10, n_ch)], "..." if n_ch > 10 else "")
    return fallback


def xdf_to_raw_and_events(eeg, markers, fs):
    data = np.asarray(eeg["time_series"])
    ts   = np.asarray(eeg["time_stamps"])

    if data.ndim != 2:
        raise RuntimeError(f"EEG time_series unexpected shape: {data.shape}")

    # Asegurar (n_channels, n_samples)
    if data.shape[0] == len(ts):
        data = data.T

    # ---- SCALE FIX ----
    max_abs = np.max(np.abs(data))
    if max_abs > 1e3:
        print(f"[DEBUG] Data looks like uV (max={max_abs:.2f}). Converting uV -> V.")
        data = data * 1e-6

    n_ch, n_samp = data.shape
    ch_names = _get_channel_names(eeg, n_ch)

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    x = raw.get_data()
    print("\n[DEBUG] raw amplitude | max abs (V):", float(np.max(np.abs(x))))
    print("[DEBUG] raw amplitude | max abs (uV):", float(np.max(np.abs(x)) * 1e6))

    # ---- Markers -> events ----
    m_ts = np.asarray(markers["time_stamps"])
    m_2d = np.asarray(markers["time_series"])

    # You validated: ID is in col 0
    m_ids_raw = m_2d[:, 0] if (m_2d.ndim == 2) else m_2d

    # Force ids to int robustly (handles strings like "100", "100.0")
    m_ids = []
    for v in m_ids_raw:
        try:
            m_ids.append(int(float(v)))
        except Exception:
            m_ids.append(-999999)
    m_ids = np.asarray(m_ids, dtype=int)

    # Debug: show unique ids present
    uniq_ids, cnts = np.unique(m_ids[m_ids != -999999], return_counts=True)
    print("\n[DEBUG] Unique marker IDs (first 20):")
    for u, c in list(zip(uniq_ids, cnts))[:20]:
        print(f"  {u}: {c}")

    keep = np.isin(m_ids, [EVENT_REST, EVENT_MOV])
    m_ts_keep  = m_ts[keep]
    m_ids_keep = m_ids[keep]

    # timestamps -> samples (relative to EEG ts[0])
    samples = np.round((m_ts_keep - ts[0]) * fs).astype(int)
    valid = (samples >= 0) & (samples < raw.n_times)

    samples    = samples[valid]
    m_ids_keep = m_ids_keep[valid]

    events = np.c_[samples, np.zeros(len(samples), dtype=int), m_ids_keep].astype(int)

    # Print summary
    if len(events) == 0:
        print("\nEventos encontrados: {}  (VACÍO)")
    else:
        u, c = np.unique(events[:, 2], return_counts=True)
        print("\nEventos encontrados:", dict(zip(u, c)))

    # Extra debug
    print("[DEBUG] raw.n_times:", raw.n_times)
    if len(events):
        print("[DEBUG] events sample min/max:", int(events[:, 0].min()), int(events[:, 0].max()))

    return raw, events


# ============================================================
# Spatial operator (neighbor-based) replicating your old code
# ============================================================
def apply_neighbor_operator(raw, adjacency, picks=None, mode="subtract"):
    """
    Apply a neighbor-based spatial operator to MNE Raw data.

    mode="subtract": x_ch <- x_ch - mean(x_neighbors)   (replicates your old code)
    mode="add":      x_ch <- x_ch + mean(x_neighbors)   (optional smoothing)
    """
    if picks is None:
        picks = raw.ch_names

    picks_set = set(picks)
    ch_to_idx = {ch: i for i, ch in enumerate(raw.ch_names)}

    X = raw.get_data()          # (n_channels, n_times)
    X_new = X.copy()            # avoid order effects

    applied = 0
    skipped_missing = 0
    skipped_no_nb = 0

    for ch, neighbors in adjacency.items():
        if ch not in picks_set or ch not in ch_to_idx:
            skipped_missing += 1
            continue

        nb = [n for n in neighbors if (n in picks_set and n in ch_to_idx)]
        if len(nb) == 0:
            skipped_no_nb += 1
            continue

        ch_idx = ch_to_idx[ch]
        nb_idx = [ch_to_idx[n] for n in nb]
        nb_mean = X[nb_idx, :].mean(axis=0)

        if mode == "subtract":
            X_new[ch_idx, :] = X[ch_idx, :] - nb_mean
        elif mode == "add":
            X_new[ch_idx, :] = X[ch_idx, :] + nb_mean
        else:
            raise ValueError("mode must be 'subtract' or 'add'")

        applied += 1

    raw._data[:] = X_new

    print("\n[DEBUG] Spatial operator applied.")
    print(f"  mode: {mode}")
    print(f"  channels processed: {applied}")
    print(f"  skipped (missing in picks/raw): {skipped_missing}")
    print(f"  skipped (no valid neighbors): {skipped_no_nb}")

    return raw


def _rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x**2)))


# ============================================================
# MRCP epochs + evoked
# ============================================================
def make_evokeds(file_path):
    eeg, markers = load_xdf_pyxdf(file_path)
    raw, events = xdf_to_raw_and_events(eeg, markers, fs=config.FS)

    # Drop non-EEG auxiliary channels before montage
    drop_ch = [ch for ch in raw.ch_names if ch.upper().startswith("AUX") or ch.upper() in ["TRIGGER", "M1", "M2"]]
    if drop_ch:
        print(f"[INFO] Dropping non-EEG channels before montage: {drop_ch}")
        raw.drop_channels(drop_ch)

    # 0) Apply standard 10-20 montage (case-insensitive)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")

    if len(events) == 0:
        raise RuntimeError(
            "No se construyeron events para REST/MOV.\n"
            "Revisa que EVENT_REST/EVENT_MOV coincidan con los IDs reales que se imprimieron en [DEBUG]."
        )


    # -------------------------
    # Preprocessing
    # -------------------------
    # 1) CAR (Common Average Reference)
    raw.set_eeg_reference("average", verbose=False)

    # 2) Spatial operator (your neighbor subtraction)
    if USE_SPATIAL_OPERATOR:
        picks_spatial = [
            ch for ch in raw.ch_names
            if (ch not in EXCLUDE_FROM_SPATIAL and not ch.startswith("AUX"))
        ]

        # Optional debug: CZ RMS before/after spatial
        if "CZ" in raw.ch_names:
            cz_before = raw.get_data(picks=["CZ"])[0].copy()
            rms_before = _rms(cz_before)
        else:
            rms_before = None

        apply_neighbor_operator(raw, adjacency_matrix, picks=picks_spatial, mode=SPATIAL_MODE)

        if "CZ" in raw.ch_names and rms_before is not None:
            cz_after = raw.get_data(picks=["CZ"])[0]
            rms_after = _rms(cz_after)
            print(f"[DEBUG] CZ RMS before spatial: {rms_before:.6e} V | after: {rms_after:.6e} V")

    # 3) Notch
    raw.notch_filter([NOTCH], verbose=False)

    # 4) MRCP bandpass (slow potentials)
    raw.filter(L_FREQ, H_FREQ, verbose=False)

    event_id = {"REST": int(EVENT_REST), "MOV": int(EVENT_MOV)}

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        reject=None,
        reject_by_annotation=False,
        verbose=False
    )

    print("\n[DEBUG] Epochs created.")
    print("  total epochs:", len(epochs))
    print("  REST:", len(epochs["REST"]))
    print("  MOV :", len(epochs["MOV"]))
    print("  drop_log example (first 5):", epochs.drop_log[:5])

    ev_rest = epochs["REST"].average()
    ev_mov  = epochs["MOV"].average()
    return epochs, ev_rest, ev_mov


# ============================================================
# NEW: Mean ± STD plotting from Epochs
# ============================================================
def _plot_mean_std_from_epochs(ax, epochs_cond, ch, label, to_uV=True, alpha_fill=0.18):
    """
    Plot mean ± std across trials for a single channel for one condition on a given axis.
    epochs_cond: epochs["REST"] or epochs["MOV"]
    """
    if ch not in epochs_cond.ch_names:
        ax.set_title(f"{ch} (missing)")
        ax.axis("off")
        return

    X = epochs_cond.copy().pick([ch]).get_data()  # (n_epochs, 1, n_times)
    X = X[:, 0, :]                                # (n_epochs, n_times)
    tt = epochs_cond.times

    mean = X.mean(axis=0)
    std  = X.std(axis=0, ddof=0)

    if to_uV:
        mean = mean * 1e6
        std  = std * 1e6

    ax.plot(tt, mean, label=label, linewidth=2)
    ax.fill_between(tt, mean - std, mean + std, alpha=alpha_fill)


def plot_picks_mean_std(epochs, picks, title_prefix="MRCP (Mean ± STD)"):
    """
    One figure per channel in picks: REST and MOV mean ± std.
    """
    for ch in picks:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_mean_std_from_epochs(ax, epochs["REST"], ch, "REST")
        _plot_mean_std_from_epochs(ax, epochs["MOV"],  ch, "MOV")
        ax.axvline(0, linestyle="--", alpha=0.6)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{title_prefix} @ {ch}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_grid_mean_std(epochs, grid_layout, title="Left lateral + midline (Mean ± STD)"):
    """
    Grid layout like your screenshot. Each cell has REST/MOV mean ± std.
    grid_layout: list of rows, each row list of channel names or None.
    """
    n_rows = len(grid_layout)
    n_cols = len(grid_layout[0]) if n_rows else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c] if (n_rows > 1 and n_cols > 1) else axes[max(r, c)]
            ch = grid_layout[r][c]

            if ch is None:
                ax.axis("off")
                continue

            _plot_mean_std_from_epochs(ax, epochs["REST"], ch, "REST")
            _plot_mean_std_from_epochs(ax, epochs["MOV"],  ch, "MOV")

            ax.axvline(0, linestyle="--", alpha=0.6)
            ax.axhline(0, linewidth=1)
            ax.set_title(ch)
            ax.grid(True, linestyle=":", alpha=0.35)

            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_ylabel("Amplitude (µV)")

    # single global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


# ============================================================
# Existing plot (evoked) + optional call to mean±std
# ============================================================
def plot_grand_average(ev_rest, ev_mov, picks):
    picks_exist = [p for p in picks if p in ev_mov.ch_names]
    if not picks_exist:
        print("\n⚠️ Ningún pick coincide con tus nombres de canales.")
        print("Primeros 30 canales disponibles:")
        print(ev_mov.ch_names[:30])
        return

    # Compare evokeds (mean across picks)
    # mne.viz.plot_compare_evokeds(
    #    {"REST": ev_rest, "MOV": ev_mov},
    #    picks=picks_exist,
    #    combine=None,
    #    show=True
    #)

    # Channel-by-channel overlays in µV (mean only)
    for ch in picks_exist:
        plt.figure(figsize=(7, 4))
        y_mov  = ev_mov.copy().pick([ch]).data[0]  * 1e6
        y_rest = ev_rest.copy().pick([ch]).data[0] * 1e6

        plt.plot(ev_rest.times, y_rest, label="REST", linewidth=2)
        plt.plot(ev_mov.times,  y_mov,  label="MOV", linewidth=2)
        plt.axvline(0, linestyle="--", alpha=0.6)
        plt.title(f"Grand-average MRCP @ {ch}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ============================================================
# NEW: REST vs MOV topomap panel (2×N)
# ============================================================
# Plots a 2-row topomap grid at user-defined times:
#   Row 0 → REST evoked
#   Row 1 → MOV  evoked
#
# The `times` list defines seconds relative to movement onset (e.g. [-1, 0, 1, 2]).
# Requires a valid EEG montage and evoked objects (ev_rest, ev_mov).

def simple_topomaps(ev_rest, ev_mov, times):
    times = list(times)  # ensure list format

    print("\n✔ Generating REST topomaps...")
    fig1 = ev_rest.plot_topomap(
        times=times,
        ch_type='eeg',
        scalings=dict(eeg=1e6),  # Convert V → µV
        show=False
    )
    fig1.suptitle("REST Topomaps")

    print("\n✔ Generating MOV topomaps...")
    fig2 = ev_mov.plot_topomap(
        times=times,
        ch_type='eeg',
        scalings=dict(eeg=1e6),
        show=False
    )
    fig2.suptitle("MOV Topomaps")

    plt.show()

def topomaps_limited_scale(ev_rest, ev_mov, times, vmin=-5.0, vmax=5.0):
    """
    Panel 2xN de topoplots con escala fija en µV.
    Compatible con MNE 1.10.x (sin vmin/vmax en plot_topomap).
    """
    times = list(times)
    n_times = len(times)

    # Normalización de colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, n_times, figsize=(3*n_times, 6))
    fig.suptitle("MRCP REST vs MOV (escala fija)", fontsize=14)

    if n_times == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    last_im = None

    # --- Fila 0: REST ---
    for col, t in enumerate(times):
        ax = axes[0, col]

        idx = np.argmin(np.abs(ev_rest.times - t))
        data = ev_rest.data[:, idx] * 1e6  # V -> µV

        im = plot_topomap(
            data,
            ev_rest.info,
            axes=ax,
            cmap="RdBu_r",
            norm=norm,
            show=False
        )
        last_im = im
        ax.set_title(f"REST {t:.3f} s")

    # --- Fila 1: MOV ---
    for col, t in enumerate(times):
        ax = axes[1, col]

        idx = np.argmin(np.abs(ev_mov.times - t))
        data = ev_mov.data[:, idx] * 1e6  # V -> µV

        im = plot_topomap(
            data,
            ev_mov.info,
            axes=ax,
            cmap="RdBu_r",
            norm=norm,
            show=False
        )
        last_im = im
        ax.set_title(f"MOV {t:.3f} s")

    # Barra de color común
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # posición manual
        cbar = plt.colorbar(last_im[0], cax=cax)
        cbar.set_label("Amplitude (µV)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def print_topomap_vectors(evoked, times):
    """
    Imprime vectores (n_channels,) para cada tiempo solicitado.
    
    - evoked: objeto MNE Evoked (ej. ev_rest o ev_mov)
    - times: lista de tiempos en segundos
    """
    for t in times:
        # encontrar índice más cercano al tiempo t
        idx = (np.abs(evoked.times - t)).argmin()

        # extraer vector (n_channels,)
        vec = evoked.data[:, idx] * 1e6  # convertir a µV (opcional)

        print("\n-------------------------------")
        print(f"Time {t:.3f} s → index {idx}")
        print(f"Vector shape: {vec.shape}")

        # imprimir canal -> valor
        print("Canal → Valor (µV):")
        for ch, val in zip(evoked.ch_names, vec):
            print(f"{ch:>6}: {val: .3f}")
    print("-------------------------------")

def inspect_evoked_with_topomaps(ev_rest, ev_mov, times, title_prefix="MRCP",vmin=None, vmax=None, show_vectors=True):
    """
    Unifica:
      ✔ extracción de vectores en µV por canal
      ✔ impresión para REST y MOV
      ✔ topoplots por tiempo y condición
    """

    import numpy as np
    import mne
    import matplotlib.pyplot as plt

    # --- FIX: función interna para renombrar a 10-20 ---
    def fix_channel_names_for_1020(ch_names):
        rename_map = {
            "FP1":"Fp1", "FPZ":"Fpz", "FP2":"Fp2",
            "FZ":"Fz", "CZ":"Cz", "PZ":"Pz", "OZ":"Oz",
            "POZ":"POz"
        }
        return [rename_map.get(ch, ch) for ch in ch_names]

    print("\n===== INSPECCIÓN DE MRCP =====")

    for label, ev in [('REST', ev_rest), ('MOV', ev_mov)]:
        print(f"\n===== CONDICIÓN: {label} =====")

        for t in times:
            idx = (np.abs(ev.times - t)).argmin()
            vec = ev.data[:, idx] * 1e6  # convertir a µV

            if show_vectors:
                print("\n-------------------------------")
                print(f"Time {t:.3f} s → index {idx}")
                print(f"Vector shape: {vec.shape}")
                print("Canal → Valor (µV):")
                for ch, val in zip(ev.ch_names, vec):
                    print(f"{ch:>6}: {val: .3f}")
                print("-------------------------------")

            # ============================
            # Crear Evoked artificial para topoplot
            # ============================

            # 1) Arreglamos nombres para que el montage los reconozca
            ch_fixed = fix_channel_names_for_1020(ev.ch_names)

            # 2) Crear info con nombres corregidos
            info = mne.create_info(ch_names=ch_fixed, sfreq=1000, ch_types="eeg")

            # 3) Aplicar montaje 10-20
            mont = mne.channels.make_standard_montage("standard_1020")
            info.set_montage(mont)

            # 4) Convertir vector → Evoked artificial
            ev_temp = mne.EvokedArray(vec[:, np.newaxis], info, tmin=0.0)

            # 5) Render topoplot
            fig = mne.viz.plot_evoked_topomap(
                ev_temp,
                times=[0.0],
                scalings=dict(eeg=1),
                cmap="RdBu_r",
                colorbar=True,
                units=dict(eeg="µV"),
                vmin=vmin,
                vmax=vmax,
                show=False
            )
            fig.set_size_inches(6, 6)
            fig.suptitle(f"{title_prefix} | {label} | t = {t:.3f}s")
            plt.show(block=True)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    epochs, ev_rest, ev_mov = make_evokeds(XDF_FILE)

    #print("\nEpoch counts:")
    #print("  REST:", len(epochs["REST"]))
    #print("  MOV :", len(epochs["MOV"]))

    # 1) Existing evoked plots (mean only)
    #plot_grand_average(ev_rest, ev_mov, PICKS)

    # 2) NEW: Mean ± STD for PICKS (uses epochs)
    #if PLOT_MEAN_STD:
    #    plot_picks_mean_std(epochs, PICKS)

    # 3) NEW: Grid subplot like your image (uses epochs)
    # if PLOT_GRID:
    #     plot_grid_mean_std(epochs, GRID_LAYOUT)

    # 4) Topomap panel REST vs MOV (2×N)
    #plot_rest_mov_topomaps(ev_rest, ev_mov, TOPOMAP_TIMES)

    #simple_topomaps(ev_rest, ev_mov, TOPOMAP_TIMES)

    # En vez de simple_topomaps(...)
    #topomaps_limited_scale(ev_rest, ev_mov, times=[-1.0, -0.5, 0.0, 0.5, 1.0], vmin=-5.0, vmax=5.0)

    # print("\n===== VECTORES REST =====")
    # print_topomap_vectors(ev_rest, TOPOMAP_TIMES)

    # print("\n===== VECTORES MOV =====")
    # print_topomap_vectors(ev_mov, TOPOMAP_TIMES)

    inspect_evoked_with_topomaps(
    ev_rest,
    ev_mov,
    times=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.5],
    title_prefix="BCI MRCP",
    vmin=-7.0, 
    vmax=7.0,
    show_vectors=True
)
