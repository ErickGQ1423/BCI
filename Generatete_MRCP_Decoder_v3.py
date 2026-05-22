# import numpy as np
# import mne
# import pyxdf
# import matplotlib.pyplot as plt
# import config # Asegúrate de que este archivo esté en tu carpeta
# from scipy import signal

# # --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
# #XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S010/eeg/sub-P001_ses-S010_task-Default_run-001_eeg.xdf"
# #XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S001/eeg/sub-P_Claudia_ses-S001_task-Default_run-001_eeg_old1.xdf"
# XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S002/eeg/sub-P_Claudia_ses-S002_task-Default_run-001_eeg.xdf"

# L_FREQ, H_FREQ = 0.1, 4.0
# TMIN, TMAX = -2.0, 4.0
# BASELINE = (-2.0, -1.5)
# CANALES_MALOS = ['AF3', 'AF4', 'F3', 'P3'] # Detectados previamente

# # Matriz de Adyacencia para Filtro Laplaciano
# adjacency_matrix = {
#     'FC3': ['F3', 'C3', 'FC1'],
#     'FC1': ['FC3', 'F1', 'C1', 'FCz'],
#     'FCz': ['FC1', 'Fz', 'Cz', 'FC2'],
#     'FC2': ['FCz', 'F2', 'C2', 'FC4'],
#     'Cz':  ['C1', 'FCz', 'CPz', 'C2'],
#     'C3':  ['FC3', 'CP3', 'C1'],
#     'C1':  ['C3', 'FC1', 'CP1', 'Cz'],
#     'CPz': ['CP1', 'Cz', 'Pz', 'CP2'],
#     # ... puedes completar según tu lista anterior
# }

# # ============================================================
# # FUNCIONES DE FILTRADO (INTEGRACIÓN)
# # ============================================================

# def aplicar_filtros_interactivos(raw):
#     print("\n" + "="*40)
#     print("CONFIGURACIÓN DE FILTROS")
#     print("="*40)
#     do_car = input("¿Aplicar CAR (Common Average Reference)? (s/n): ").lower() == 's'
#     do_lap = input("¿Aplicar Laplaciano de Hjorth? (s/n): ").lower() == 's'
#     do_bp  = input(f"¿Aplicar Pasabanda ({L_FREQ}-{H_FREQ} Hz)? (s/n): ").lower() == 's'
    
#     # 1. CAR
#     if do_car:
#         print("[PROC] Aplicando CAR...")
#         raw.set_eeg_reference(ref_channels='average')

#     # 2. Laplaciano (Usando los datos de los vecinos)
#     if do_lap:
#         print("[PROC] Aplicando Filtro Laplaciano...")
#         data = raw.get_data()
#         ch_names = raw.ch_names
#         new_data = data.copy()
        
#         for i, ch in enumerate(ch_names):
#             if ch in adjacency_matrix:
#                 vecinos = [ch_names.index(v) for v in adjacency_matrix[ch] if v in ch_names]
#                 if vecinos:
#                     new_data[i] -= np.mean(data[vecinos, :], axis=0)
        
#         raw = mne.io.RawArray(new_data, raw.info)

#     # 3. Pasabanda
#     if do_bp:
#         print(f"[PROC] Aplicando Pasabanda {L_FREQ}-{H_FREQ} Hz...")
#         raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)
#         raw.notch_filter(60, verbose=False)

#     return raw

# # ============================================================
# # PROCESAMIENTO PRINCIPAL
# # ============================================================

# def procesar_bci_completo(file_path):
#     # 1. Carga de XDF
#     streams, _ = pyxdf.load_xdf(file_path)
#     eeg_s = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
#     m_s = next(s for s in streams if s["info"]["type"][0].lower() in ["markers", "events"])
    
#     # 2. Creación de Raw
#     data = eeg_s["time_series"].T
#     if np.max(np.abs(data)) > 1.0: data *= 1e-6
#     ch_names = [c["label"][0] for c in eeg_s["info"]["desc"][0]["channels"][0]["channel"]]
    
#     info = mne.create_info(ch_names=ch_names, sfreq=float(eeg_s["info"]["nominal_srate"][0]), ch_types="eeg")
#     raw = mne.io.RawArray(data, info)
    
#     # 3. Preparación de Canales
#     raw.drop_channels([ch for ch in raw.ch_names if any(x in ch.upper() for x in ["AUX", "TRIGGER"])])
#     raw.rename_channels(lambda x: x[:-1] + 'z' if x.endswith('Z') and len(x)<=3 else x)
#     raw.set_montage("standard_1020")
#     raw.info['bads'] = CANALES_MALOS # Marcamos los ruidosos

#     # 4. APLICACIÓN DE FILTROS INTERACTIVOS
#     raw = aplicar_filtros_interactivos(raw)

#     # ############################################################
#     # 5. Creación de Épocas (Trigger 100 y 200) - CORREGIDO
#     # ############################################################
    
#     # Obtenemos el tiempo inicial del stream EEG
#     t_start = eeg_s["time_stamps"][0]
    
#     # Calculamos el tiempo relativo de los marcadores respecto al inicio del EEG
#     # Usamos float64 para la resta y luego convertimos a int64 para evitar el overflow
#     marker_relative_times = m_s["time_stamps"] - t_start
    
#     # Convertimos los tiempos en segundos a índices de muestras (samples)
#     onset = np.round(marker_relative_times * raw.info['sfreq']).astype(np.int64)
    
#     # Extraemos los IDs de los triggers (100, 200)
#     ids = [int(float(v[0])) for v in m_s["time_series"]]
    
#     # Creamos la matriz de eventos (n_events, 3) forzando int64
#     # MNE convertirá esto internamente, pero enviarlo ya como int64 previene el error previo
#     events = np.c_[onset, np.zeros_like(onset, dtype=np.int64), ids].astype(np.int64)
    
#     # FILTRO DE SEGURIDAD: Eliminamos eventos que caigan fuera del rango del Raw
#     # (Por ejemplo, si un marcador se envió justo al final de la sesión)
#     valid_events_mask = (events[:, 0] >= 0) & (events[:, 0] < raw.n_times)
#     events = events[valid_events_mask]
    
#     print(f"[INFO] Eventos detectados tras corrección de overflow: {len(events)}")
    
#     event_dict = {
#         "REST": int(config.TRIGGERS["REST_BEGIN"]), 
#         "MI": int(config.TRIGGERS["MI_BEGIN"])
#     }
    
#     # Creación de épocas
#     epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=TMIN, tmax=TMAX, 
#                         baseline=BASELINE, preload=True, 
#                         reject=dict(eeg=250e-6), verbose=True)

#     return epochs

# # ============================================================
# # VISUALIZACIÓN 3x3 Y TOPOMAPS
# # ============================================================

# def plot_bci_dashboard(epochs, times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
#     # Topomaps Comparativos
#     evoked_mi = epochs["MI"].average()
#     evoked_rest = epochs["REST"].average()
    
#     fig_topo, axes_topo = plt.subplots(2, len(times) + 1, figsize=(18, 6), 
#                                        gridspec_kw={'width_ratios': [1]*len(times) + [0.1]})
    
#     v_lim = (-40, 40)
#     evoked_rest.plot_topomap(times=times, axes=axes_topo[0, :-1], colorbar=False, show=False, vlim=v_lim, cmap='RdBu_r')
#     evoked_mi.plot_topomap(times=times, axes=axes_topo[1, :-1], colorbar=False, show=False, vlim=v_lim, cmap='RdBu_r')

#     # Colorbar
#     norm = plt.Normalize(vmin=v_lim[0], vmax=v_lim[1])
#     sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
#     plt.colorbar(sm, cax=axes_topo[1, -1], label='µV')
#     axes_topo[0, -1].axis('off')
    
#     # Grid 3x3
#     channels_3x3 = [['FC3', 'FCz', 'FC1'], ['C3', 'Cz', 'C1'], ['CP3', 'CPz', 'CP1']]
#     fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
#     for r in range(3):
#         for c in range(3):
#             ch, ax = channels_3x3[r][c], axes[r, c]
#             for cond, col in zip(["REST", "MI"], ["#1f77b4", "#d62728"]):
#                 d = epochs[cond].get_data(picks=ch)[:, 0, :] * 1e6
#                 ax.plot(epochs.times, np.mean(d, axis=0), color=col, label=cond if (r==0 and c==0) else "")
#                 ax.fill_between(epochs.times, np.mean(d, axis=0)-np.std(d, axis=0), 
#                                 np.mean(d, axis=0)+np.std(d, axis=0), color=col, alpha=0.1)
#             ax.set_title(f"Canal: {ch}")
#             # FORZAR EJE Y (Ajusta los valores según necesites)
#             ax.set_ylim(-20, 20)
#             ax.axvline(0, color='k', ls='--')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     epochs = procesar_bci_completo(XDF_FILE)
#     if len(epochs) > 0:
#         plot_bci_dashboard(epochs)


import numpy as np
import mne
import pyxdf
import matplotlib.pyplot as plt
import config
from scipy import signal

# --- CONFIGURACIÓN ---
XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.xdf"
#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S001/eeg/sub-P_Claudia_ses-S001_task-Default_run-001_eeg_old1.xdf"
#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S002/eeg/sub-P_Claudia_ses-S002_task-Default_run-001_eeg.xdf"


ELECTRODOS_A_VER = ['FC3', 'FCz', 'O1', 
                    'C3', 'Cz', 'AF1', 
                    'CP3', 'CPz', 'AF2'] 
L_FREQ, H_FREQ = 0.1, 4.0
TMIN, TMAX = -2.0, 4.0
BASELINE = (-2.0, -1.5)
CANALES_MALOS = ['AF3', 'AF4', 'F3', 'P3'] 
REJECT_THRESHOLD = dict(eeg=500e-6)


adjacency_matrix = {
    'AF3': ['F3'],'AF4': ['F4'], 'F3': ['AF3', 'FC3', 'F1'],
    'F1': ['F3', 'FC1', 'FZ'], 'FZ': ['F1', 'FCZ', 'F2'],
    'F2': ['FZ', 'FC2', 'F4'], 'F4': ['F2', 'AF4', 'FC4'],
    'FC3': ['F3', 'C3', 'FC1'], 'FC1': ['FC3', 'F1', 'C1', 'FCZ'],
    'FCZ': ['FC1', 'FZ', 'CZ', 'FC2'], 'FC2': ['FCZ', 'F2', 'C2', 'FC4'],
    'FC4': ['FC2', 'F4', 'C4'], 'C3': ['FC3', 'CP3', 'C1'],
    'C1': ['C3', 'FC1', 'CP1', 'CZ'], 'CZ': ['C1', 'FCZ', 'CPZ', 'C2'], 'C2': ['CZ', 'FC2', 'CP2', 'C4'],
    'C4': ['C2', 'FC4', 'CP4'], 'CP3': ['C3', 'P3', 'CP1'],
    'CP1': ['CP3', 'C1', 'P1', 'CPZ'], 'CPZ': ['CP1', 'CZ', 'POZ', 'P2'],
    'CP2': ['CPZ', 'C2', 'P2', 'CP4'], 'CP4': ['CP2', 'C4', 'P4'],
    'P3': ['CP3', 'PO3', 'P1'], 'P1': ['P3', 'CP1', 'PZ'], 'PZ': ['P1', 'CPZ', 'POZ', 'P2'],
    'P2': ['PZ', 'CP2', 'P4'], 'P4': ['P2', 'CP4', 'PO4'], 'PO3': ['P3'],
    'POZ': ['PZ'], 'PO4': ['P4'], 'O1': [], 'O2': []
}

def aplicar_filtros_interactivos(raw):
    print("\n" + "="*40 + "\nCONFIGURACIÓN DE FILTROS\n" + "="*40)
    do_car = input("¿Aplicar CAR? (s/n): ").lower() == 's'
    do_lap = input("¿Aplicar Laplaciano? (s/n): ").lower() == 's'
    do_bp  = input(f"¿Aplicar Pasabanda ({L_FREQ}-{H_FREQ} Hz)? (s/n): ").lower() == 's'
    
    if do_car:
        raw.set_eeg_reference(ref_channels='average')
    if do_lap:
        data = raw.get_data()
        ch_names = raw.ch_names
        new_data = data.copy()
        for i, ch in enumerate(ch_names):
            if ch in adjacency_matrix:
                vecinos = [ch_names.index(v) for v in adjacency_matrix[ch] if v in ch_names]
                if vecinos: new_data[i] -= np.mean(data[vecinos, :], axis=0)
        raw = mne.io.RawArray(new_data, raw.info)
    if do_bp:
        raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)
        raw.notch_filter(60, verbose=False)
    return raw

def procesar_bci_completo(file_path):
    print(f"\n[INFO] Analizando archivo: {file_path}")
    streams, _ = pyxdf.load_xdf(file_path)
    
    # 1. Identificar Stream de EEG
    try:
        eeg_s = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
    except StopIteration:
        raise ValueError("FATAL: No se encontró ningún stream de tipo 'EEG'.")

    # 2. Identificar el Stream de Marcadores CORRECTO
    # Filtramos todos los streams que dicen ser de marcadores o eventos
    marker_streams = [s for s in streams if s["info"]["type"][0].lower() in ["markers", "events"]]
    
    if not marker_streams:
        raise ValueError("FATAL: No se encontró ningún stream de marcadores.")
    
    # Lógica de selección: El stream de triggers reales (100, 200) suele tener 
    # pocas muestras (decenas o cientos), no miles como el stream de sincronización.
    # Elegimos el que tenga menos muestras totales.
    m_s = min(marker_streams, key=lambda s: len(s["time_series"]))
    
    print(f"[OK] Stream EEG detectado: {eeg_s['info']['name'][0]}")
    print(f"[OK] Stream Marcadores elegido: '{m_s['info']['name'][0]}' con {len(m_s['time_series'])} eventos.")

    # 3. Preparar objeto Raw
    data = eeg_s["time_series"].T
    if np.max(np.abs(data)) > 1.0: data *= 1e-6
    ch_names = [c["label"][0] for c in eeg_s["info"]["desc"][0]["channels"][0]["channel"]]
    
    info = mne.create_info(ch_names=ch_names, sfreq=float(eeg_s["info"]["nominal_srate"][0]), ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    
    # Limpieza y Montaje
    raw.drop_channels([ch for ch in raw.ch_names if any(x in ch.upper() for x in ["AUX", "TRIGGER"])])
    raw.rename_channels(lambda x: x[:-1] + 'z' if x.endswith('Z') and len(x)<=3 else x)
    raw.set_montage("standard_1020")
    raw.info['bads'] = CANALES_MALOS
    
    # Aplicar Filtros (CAR, Laplaciano, BP)
    raw = aplicar_filtros_interactivos(raw)

    # 4. Sincronización de Eventos (Corrección de Overflow)
    t_start_eeg = eeg_s["time_stamps"][0]
    # Calculamos tiempos relativos usando float64 para evitar errores de precisión
    marker_relative_times = m_s["time_stamps"] - t_start_eeg
    onset = np.round(marker_relative_times * raw.info['sfreq']).astype(np.int64)
    
    # Extraer y Limpiar IDs
    raw_ids = [v[0] for v in m_s["time_series"]]
    ids = []
    for v in raw_ids:
        try:
            # Convertimos a float primero por si vienen como '100.0' y luego a int
            ids.append(int(float(v)))
        except (ValueError, TypeError):
            ids.append(-1)
            
    events = np.c_[onset, np.zeros_like(onset, dtype=np.int64), ids].astype(np.int64)
    
    # 5. Filtrar por Triggers de interés (100, 200)
    target_ids = [int(config.TRIGGERS["REST_BEGIN"]), int(config.TRIGGERS["MI_BEGIN"])]
    mask = np.isin(events[:, 2], target_ids)
    events = events[mask]
    
    print(f"[INFO] IDs únicos en este stream: {np.unique(ids)}")
    print(f"[INFO] Eventos válidos (100/200) encontrados: {len(events)}")
    
    if len(events) == 0:
        # Si esto falla, imprimimos todos los streams para que el usuario vea qué pasó
        print("\n--- STREAMS DISPONIBLES EN EL ARCHIVO ---")
        for i, s in enumerate(streams):
            print(f"Stream {i}: {s['info']['name'][0]} | Tipo: {s['info']['type'][0]} | Muestras: {len(s['time_series'])}")
        raise ValueError(f"No se encontraron triggers {target_ids}. Revisa los nombres de los streams arriba.")

    # 6. Crear Épocas
    event_dict = {"REST": target_ids[0], "MI": target_ids[1]}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=TMIN, tmax=TMAX, 
                        baseline=BASELINE, preload=True, reject=REJECT_THRESHOLD, verbose=True)
    
    return epochs

def plot_bci_dashboard(epochs, times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
    evoked_mi, evoked_rest = epochs["MI"].average(), epochs["REST"].average()
    
    # Topomaps
    fig_topo, axes_topo = plt.subplots(2, len(times) + 1, figsize=(20, 6), 
                                       gridspec_kw={'width_ratios': [1]*len(times) + [0.1]})
    v_lim = (-20, 20)
    evoked_rest.plot_topomap(times=times, axes=axes_topo[0, :-1], colorbar=False, show=False, vlim=v_lim, cmap='RdBu_r')
    evoked_mi.plot_topomap(times=times, axes=axes_topo[1, :-1], colorbar=False, show=False, vlim=v_lim, cmap='RdBu_r')
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=v_lim[0], vmax=v_lim[1]))
    plt.colorbar(sm, cax=axes_topo[1, -1], label='µV')
    axes_topo[0, -1].axis('off')
    
    # Grid 3x3
    channels_3x3 = [['FC3', 'FCz', 'FC1'], ['C3', 'Cz', 'C1'], ['CP3', 'CPz', 'CP1']]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    for r in range(3):
        for c in range(3):
            ch, ax = channels_3x3[r][c], axes[r, c]
            for cond, col in zip(["REST", "MI"], ["#1f77b4", "#d62728"]):
                d = epochs[cond].get_data(picks=ch)[:, 0, :] * 1e6
                ax.plot(epochs.times, np.mean(d, axis=0), color=col, lw=2, label=cond if (r==0 and c==0) else "")
                ax.fill_between(epochs.times, np.mean(d, axis=0)-np.std(d, axis=0), 
                                np.mean(d, axis=0)+np.std(d, axis=0), color=col, alpha=0.15)
            ax.axvline(0, color='k', ls='--')
            ax.set_ylim(-20, 20)
            ax.set_title(f"Canal: {ch}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        epochs = procesar_bci_completo(XDF_FILE)
        if len(epochs) > 0: plot_bci_dashboard(epochs)
    except Exception as e:
        print(f"[ERROR] {e}")