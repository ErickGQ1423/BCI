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
#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf"

#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P009/ses-S001/eeg/sub-P009_ses-S001_task-Default_run-001_eeg.xdf"
XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S010/eeg/sub-P001_ses-S010_task-Default_run-001_eeg.xdf"

# "CSD" para la técnica del paper (Perrin et al.) o "MANUAL" para tu Laplaciano original
SPATIAL_FILTER_MODE = "CAR" 

EVENT_REST = int(config.TRIGGERS["REST_BEGIN"])
EVENT_MOV  = int(config.TRIGGERS["MI_BEGIN"])

TMIN, TMAX = -2.0, 4.0
BASELINE   = (-2.0, -1.5)
L_FREQ, H_FREQ = 0.1, 5.0
NOTCH = 60

PICKS = ["C3", "Cz", "FC1"]
GRID_LAYOUT = [[None, "FC1", None], 
               ["C3", None, "Cz"], 
               [None, "CP1", None], 
               ["P3", None, "Pz"]]

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
    print(f"\n[INFO] Cargando archivo: {file_path}")
    streams, header = pyxdf.load_xdf(file_path)
    
    # 1. Encontrar EEG y Marcadores (Igual que antes...)
    eeg = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
    print(f"[INFO] EEG Stream encontrado: {eeg['info']['name'][0]} ({len(eeg['time_series'])} muestras)")

    marker_stream = None
    for s in streams:
        if s["info"]["type"][0].lower() in ["markers", "marker", "events"]:
            try:
                unique_vals = np.unique([int(float(v[0])) for v in s["time_series"]])
                if EVENT_MOV in unique_vals or EVENT_REST in unique_vals:
                    marker_stream = s
                    break
            except: continue
    
    if marker_stream is None: raise ValueError(f"FATAL: No se encontraron triggers {EVENT_REST}/{EVENT_MOV}")

    # 2. Preparar Raw (Igual que antes...)
    data = eeg["time_series"].T
    if np.max(np.abs(data)) > 1e3: 
        print("   [AUTO] Detectado uV, convirtiendo a Volts...")
        data *= 1e-6
    
    try:
        ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    except:
        ch_names = [f"EEG_{i:02d}" for i in range(data.shape[0])]
    
    info = mne.create_info(ch_names=ch_names, sfreq=config.FS, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    
    # =========================================================================
    # CORRECCIÓN AQUÍ: ELIMINAR CANALES "BASURA" ANTES DEL MONTAJE
    # =========================================================================
    
    # 1. Normalizar nombres
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, fix_channel_names(raw.ch_names))})
    
    # 2. Identificar y borrar canales que NO son EEG (AUX, TRIGGER, etc.)
    # Esto evita el error "DigMontage is only a subset of info"
    #drops = [ch for ch in raw.ch_names if any(x in ch.upper() for x in [
    #    "AUX", "TRIG", "ACC", 
    #    "FP1", "FP2", "FPZ", "F8", # Los que ya tenías
    #    "FC1", "F4", "M2", "P8",   # <--- LOS NUEVOS CULPABLES (>200 uV)
    #    "F3", "F7"                 # <--- Opcional: También están altos (>170 uV)
    #])]
    
    # drops = [ch for ch in raw.ch_names if any(x in ch.upper() for x in [
    #     "AUX", "TRIG", "ACC", 
    #         "FP1", "FPZ", "FP2",      # Los que ya tenías
    #     "F7", "F3", "FZ", "F4", "F8",   # <--- LOS NUEVOS CULPABLES (>200 uV)
    #        "FC5","FC1","FC2","FC6",
    #     "T7", "C3", "CZ", "C4", "T8",
    #        "CP5","CP1","CP2","CP6",
    #     "P7", "P3", "PZ", "P4", "P8",
    #                 "POZ",
    #           "O1", "OZ", "O2"                  # <--- Opcional: También están altos (>170 uV)
    # ])]

    # drops = [ch for ch in raw.ch_names if any(x in ch.upper() for x in [
    #         "AUX", "TRIG", "ACC", 
    #         "FP1", "FPZ", "FP2",      # Los que ya tenías
    #     "F7", "F3", "FZ", "F4", "F8",   # <--- LOS NUEVOS CULPABLES (>200 uV)
           
    #     "P7", "P3", "PZ", "P4", "P8",
    #                 "POZ",
    #           "O1", "OZ", "O2"                  # <--- Opcional: También están altos (>170 uV)
    # ])]

    drops = [ch for ch in raw.ch_names if any(x in ch.upper() for x in [
             "AUX", "TRIG", "ACC"])]

    if drops: 
        print(f"   [FIX] Eliminando canales basura: {drops}")
        raw.drop_channels(drops)

    # 3. AHORA SÍ: Cargar el montaje (ya solo quedan canales de cerebro)
    try:
        raw.set_montage("standard_1020")
    except ValueError as e:
        print(f"   ⚠️ Advertencia de montaje: {e}")
        # Si falla por M1/M2, usamos on_missing='warn' para que no rompa el programa
        raw.set_montage("standard_1020", on_missing='warn')

    # =========================================================================

    # 5. FILTRADO ESPACIAL Y FRECUENCIA (Igual que antes...)
    if SPATIAL_FILTER_MODE == "CSD":
        print("\n[INFO] Modo CSD: Aplicando Laplaciano Esférico...")
        raw = compute_current_source_density(raw)
    elif SPATIAL_FILTER_MODE == "MANUAL":
        print("\n[INFO] Modo MANUAL: Aplicando Laplaciano Local...")
        raw.set_eeg_reference("average", projection=False)
        apply_neighbor_operator(raw, adjacency_matrix)
    else: 
        print("\n[INFO] Modo CAR: Referencia Promedio Común...")
        raw.set_eeg_reference("average", projection=False)

    print(f"[INFO] Filtrando: Notch={NOTCH}Hz | Bandpass={L_FREQ}-{H_FREQ}Hz")
    raw.notch_filter(NOTCH, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)

    # 6. Sincronización y Epochs (Igual que antes...)
    t_start = eeg["time_stamps"][0]
    samples = np.round((marker_stream["time_stamps"] - t_start) * config.FS).astype(int)
    m_ids = np.array([int(float(v[0])) for v in marker_stream["time_series"]])
    events_all = np.c_[samples, np.zeros_like(samples), m_ids]
    events = events_all[(events_all[:, 0] >= 0) & (np.isin(events_all[:, 2], [EVENT_REST, EVENT_MOV]))]

    reject_criteria = dict(eeg=250e-6) 
    
    print(f"[INFO] Cortando épocas...")
    epochs = mne.Epochs(raw, events, 
                        event_id={"REST": EVENT_REST, "MOV": EVENT_MOV}, 
                        tmin=TMIN, tmax=TMAX, 
                        baseline=BASELINE, 
                        reject=reject_criteria, # <--- Reactivamos el filtro
                        preload=True, 
                        verbose=False)
    
    print(f"   Epocas limpias: REST={len(epochs['REST'])} | MOV={len(epochs['MOV'])}")
    
    # [DIAGNÓSTICO] Ver la amplitud máxima
    data_max = np.max(np.abs(epochs.get_data(copy=False)))
    print(f"\n[DEBUG] Amplitud Máxima Peak-to-Peak en las épocas: {data_max:.2e} Volts")
    print(f"        (El límite anterior era 1.50e-04 Volts)")
    
    if data_max > 1.0:
        print("🔴 ALERTA: ¡Tus datos parecen estar en VOLTIOS o CONTEOS ENTEROS, no en microvolts!")
        print("   Revisa la conversión de unidades al principio del script.")
    elif data_max > 500e-6:
        print("🟠 ALERTA: Tienes artefactos muy fuertes (>500 uV). Probablemente un canal suelto.")

    # ==========================================
    # DETECTOR DE CANALES RUIDOSOS
    # ==========================================
    print("\n🔍 BUSCANDO EL CANAL CULPABLE...")
    
    # Obtener datos: (épocas, canales, tiempos)
    data = epochs.get_data(copy=True)
    
    # Calcular la amplitud pico a pico máxima para cada canal
    # (Maximo absoluto a través de todas las épocas y tiempos)
    max_per_channel = np.max(np.abs(data), axis=(0, 2))
    
    # Ordenar de peor a mejor
    bad_indices = np.argsort(max_per_channel)[::-1]
    
    print(f"{'CANAL':<10} | {'AMPLITUD MAX (uV)':<20} | {'ESTADO'}")
    print("-" * 45)
    
    for idx in bad_indices:
        val_uv = max_per_channel[idx] * 1e6  # Convertir a microvolts para leer mejor
        ch_name = epochs.ch_names[idx]
        
        status = "✅ OK"
        if val_uv > 150: status = "⚠️ Ruidoso (Ojos?)"
        if val_uv > 500: status = "❌ MUY MALO (Descartar)"
        
        # Solo imprimimos los peores 10 para no llenar la pantalla
        if val_uv > 100: 
            print(f"{ch_name:<10} | {val_uv:7.1f} uV           | {status}")
            
    print("-" * 45)
    # ==========================================

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
    """
    Muestra Topomaps de REST y MOV en la misma figura.
    CORRECCIÓN: Agrega una columna extra para la Colorbar de MNE.
    """
    scale_val = 1e6 # Convertir a uV
    
    # Calcular escala común
    if v_min_max:
        vmax = float(v_min_max)
    else:
        max_rest = np.max(np.abs(ev_rest.data)) * scale_val
        max_mov = np.max(np.abs(ev_mov.data)) * scale_val
        vmax = max(max_rest, max_mov) * 0.7

    print(f"[VIS] Generando topomaps con escala ±{vmax:.1f} uV")

    n_times = len(times)
    
    # --- CORRECCIÓN AQUÍ: ncols = n_times + 1 ---
    # MNE necesita n_times ejes para las cabezas + 1 eje para la barra de colores
    fig, axes = plt.subplots(nrows=2, ncols=n_times + 1, figsize=((n_times+1) * 2, 5), constrained_layout=True)
    
    # Fila 1: REST
    # Pasamos toda la fila de ejes (MNE usará el último para la colorbar)
    ev_rest.plot_topomap(times=times, ch_type='eeg', scalings={'eeg': scale_val},
                         vlim=(-vmax, vmax), cmap='RdBu_r', colorbar=True,
                         axes=axes[0], show=False)
    
    # Fila 2: MOV
    ev_mov.plot_topomap(times=times, ch_type='eeg', scalings={'eeg': scale_val},
                         vlim=(-vmax, vmax), cmap='RdBu_r', colorbar=True,
                         axes=axes[1], show=False)

    # Etiquetas de filas
    axes[0, 0].set_ylabel("REST", size='large', weight='bold', labelpad=15)
    axes[1, 0].set_ylabel("MOV", size='large', weight='bold', labelpad=15)
    
    fig.suptitle(f"Comparación Topográfica MRCP ({SPATIAL_FILTER_MODE})", fontsize=16)
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
    inspect_evoked_with_topomaps(ev_rest, ev_mov, times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], v_min_max=25)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import numpy as np
# import mne
# import pyxdf
# import matplotlib.pyplot as plt
# import config

# # ============================================================
# # CONFIGURACIÓN (SOLO FRECUENCIAS)
# # ============================================================

# #XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf"
# #XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P009/ses-S001/eeg/sub-P009_ses-S001_task-Default_run-001_eeg.xdf"

# XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S010/eeg/sub-P001_ses-S010_task-Default_run-001_eeg.xdf"



# # Triggers Originales
# EVENT_REST = int(config.TRIGGERS["REST_BEGIN"]) # 100
# EVENT_MOV  = int(config.TRIGGERS["MI_BEGIN"])   # 200

# # Filtros (Alineados con Phang et al., 2024)
# L_FREQ, H_FREQ = 0.1, 4.0 
# NOTCH = 60
# TMIN, TMAX = -2.0, 4.0
# BASELINE   = (-2.0, -1.5)

# PICKS = ["C3", "Cz", "FC1"]

# # ============================================================
# # PROCESAMIENTO
# # ============================================================

# def fix_channel_names(ch_names):
#     mapping = {"FP1":"Fp1", "FPZ":"Fpz", "FP2":"Fp2", "FZ":"Fz", "CZ":"Cz", "PZ":"Pz", "OZ":"Oz", "POZ":"POz"}
#     return [mapping.get(ch.upper(), ch) for ch in ch_names]

# def make_evokeds(file_path):
#     print(f"\n[INFO] Cargando archivo: {file_path}")
#     streams, header = pyxdf.load_xdf(file_path)
    
#     # 1. Encontrar EEG
#     eeg = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
    
#     # 2. Encontrar Marcadores
#     marker_stream = None
#     for s in streams:
#         if s["info"]["type"][0].lower() in ["markers", "marker", "events"]:
#             try:
#                 unique_vals = np.unique([int(float(v[0])) for v in s["time_series"]])
#                 if EVENT_MOV in unique_vals or EVENT_REST in unique_vals:
#                     marker_stream = s
#                     break
#             except: continue
    
#     if marker_stream is None: raise ValueError(f"FATAL: No se encontraron triggers.")

#     # 3. Preparar Raw
#     data = eeg["time_series"].T
#     if np.max(np.abs(data)) > 1e3: 
#         print("   [AUTO] Detectado uV, convirtiendo a Volts...")
#         data *= 1e-6
    
#     try:
#         ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
#     except:
#         ch_names = [f"EEG_{i:02d}" for i in range(data.shape[0])]
    
#     info = mne.create_info(ch_names=ch_names, sfreq=config.FS, ch_types="eeg")
#     raw = mne.io.RawArray(data, info)
    
#     # 4. Limpieza Básica
#     raw.rename_channels({old: new for old, new in zip(raw.ch_names, fix_channel_names(raw.ch_names))})
    
#     # Eliminamos canales que saturan la visualización (Opcional, pero recomendado para ver bien los topomaps)
#     # Como NO vamos a promediar referencias, borrar Fp1 no afecta a C3, así que es seguro.
#     bad_channels = [
#         "AUX", "TRIG", "ACC"
#     ]
#     drops = [ch for ch in raw.ch_names if any(x in ch.upper() for x in bad_channels)]
#     if drops: 
#         print(f"   [FIX] Eliminando canales visualmente ruidosos: {drops}")
#         raw.drop_channels(drops)

#     # 5. Montaje
#     try:
#         raw.set_montage("standard_1020")
#     except ValueError:
#         raw.set_montage("standard_1020", on_missing='warn')

#     # --- AQUÍ ESTÁ LA CLAVE: NO APLICAMOS NINGUNA REFERENCIA ---
#     print("\n[INFO] SPATIAL FILTER: NONE (Using Hardware Reference)")
#     # MNE usará la referencia original (A1/A2 o la que tuviera el gorro)
    
#     # 6. Filtros de Frecuencia (Canal por Canal)
#     print(f"[INFO] Filtrando: Notch={NOTCH}Hz | Bandpass={L_FREQ}-{H_FREQ}Hz")
#     raw.notch_filter(NOTCH, verbose=False)
#     raw.filter(L_FREQ, H_FREQ, verbose=False)

#     # 7. Epoching
#     t_start = eeg["time_stamps"][0]
#     samples = np.round((marker_stream["time_stamps"] - t_start) * config.FS).astype(int)
#     m_ids = np.array([int(float(v[0])) for v in marker_stream["time_series"]])
#     events_all = np.c_[samples, np.zeros_like(samples), m_ids]
    
#     events = events_all[(events_all[:, 0] >= 0) & (np.isin(events_all[:, 2], [EVENT_REST, EVENT_MOV]))]

#     reject_criteria = dict(eeg=1000-6) 
    
#     print(f"[INFO] Cortando épocas...")
#     epochs = mne.Epochs(raw, events, 
#                         event_id={"REST": EVENT_REST, "MOV": EVENT_MOV}, 
#                         tmin=TMIN, tmax=TMAX, 
#                         baseline=BASELINE, 
#                         reject=reject_criteria,
#                         preload=True, 
#                         verbose=False)
    
#     print(f"   Epocas limpias: REST={len(epochs['REST'])} | MOV={len(epochs['MOV'])}")
    
#     return epochs, epochs["REST"].average(), epochs["MOV"].average()

# # ============================================================
# # VISUALIZACIÓN
# # ============================================================

# def plot_mean_std_comparison(epochs, picks):
#     scale = 1e6
#     for ch in picks:
#         if ch not in epochs.ch_names: continue
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         for cond, color in [("REST", "blue"), ("MOV", "red")]:
#             data = epochs[cond].get_data(picks=ch)[:, 0, :] * scale
#             mean_signal = np.mean(data, axis=0)
#             std_signal = np.std(data, axis=0)
            
#             ax.plot(epochs.times, mean_signal, label=cond, color=color, linewidth=2)
#             ax.fill_between(epochs.times, mean_signal - std_signal, 
#                             mean_signal + std_signal, color=color, alpha=0.1)
                            
#         ax.axvline(0, color='k', ls='--', label="Cue Onset")
#         ax.set_title(f"MRCP en {ch} (Hardware Ref | 0.1-5Hz)")
#         ax.set_xlabel("Tiempo (s)")
#         ax.set_ylabel("Amplitud (uV)")
#         ax.legend()
#         plt.show()

# def inspect_evoked_with_topomaps(ev_rest, ev_mov, times, v_min_max=None):
#     scale_val = 1e6
    
#     if v_min_max:
#         vmax = float(v_min_max)
#     else:
#         if ev_rest.data.size == 0 or ev_mov.data.size == 0:
#              print("[WARN] Datos insuficientes para escala automática. Usando 20uV.")
#              vmax = 20.0
#         else:
#             max_rest = np.max(np.abs(ev_rest.data)) * scale_val
#             max_mov = np.max(np.abs(ev_mov.data)) * scale_val
#             vmax = max(max_rest, max_mov) * 0.7

#     print(f"[VIS] Generando topomaps (Solo Cz) con escala ±{vmax:.1f} uV")

#     n_times = len(times)
#     fig, axes = plt.subplots(nrows=2, ncols=n_times + 1, figsize=((n_times+1) * 2.5, 6), constrained_layout=True)
    
#     # --- TRUCO PYTHON: Función Lambda para filtrar nombres ---
#     # x es el nombre del canal. Si es 'Cz' lo deja pasar, si no devuelve None (oculto).
#     # Si quisieras ver 'Cz' y 'C3', pondrías: lambda x: x if x in ['Cz', 'C3'] else None
#     name_filter = lambda x: x if x == 'Cz' else None
    
#     # Fila 1: REST
#     ev_rest.plot_topomap(times=times, ch_type='eeg', scalings={'eeg': scale_val},
#                          vlim=(-vmax, vmax), cmap='RdBu_r', colorbar=True,
#                          show_names=name_filter,  # <--- APLICAMOS EL FILTRO
#                          axes=axes[0], show=False)
    
#     # Fila 2: MOV
#     ev_mov.plot_topomap(times=times, ch_type='eeg', scalings={'eeg': scale_val},
#                          vlim=(-vmax, vmax), cmap='RdBu_r', colorbar=True,
#                          show_names=name_filter,  # <--- APLICAMOS EL FILTRO
#                          axes=axes[1], show=False)

#     axes[0, 0].set_ylabel("REST", size='large', weight='bold', labelpad=15)
#     axes[1, 0].set_ylabel("MOV", size='large', weight='bold', labelpad=15)
    
#     fig.suptitle("Comparación Topográfica (Solo Cz)", fontsize=16)
#     plt.show()

# if __name__ == "__main__":
#     epochs, ev_rest, ev_mov = make_evokeds(XDF_FILE)
#     plot_mean_std_comparison(epochs, PICKS)
#     inspect_evoked_with_topomaps(ev_rest, ev_mov, times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], v_min_max=40)