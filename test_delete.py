# import numpy as np
# import mne
# import pyxdf
# import matplotlib.pyplot as plt
# import os
# import glob

# # ==========================================
# # CONFIGURACIÓN DE PROCESAMIENTO
# # ==========================================
# FOLDER_PATH = "/home/lab-admin/Documents/CurrentStudy/sub-P002/ses-S001/eeg/"
# EVENTO_MI = '100.0' 
# CHANNELS_TO_PLOT = ["C3", "FC1", "Cz", "CP1"]
# T_MIN, T_MAX = -2.0, 4.0
# BASELINE = (-1.0, 0)
# REJECT_THRESHOLD = 50e-6 

# # --- VARIABLES DE CONTROL (Interruptores) ---
# # Si es None, no se aplica el filtro respectivo.
# FILTER_RANGE = [0.1, 3.0]      # Filtro Butterworth Pasabanda
# APPLY_CAR = True               # Common Average Reference
# APPLY_LAPLACIAN = None         # Filtro Laplaciano (CSD)

# # ==========================================
# # FUNCIÓN DE CARGA Y FILTRADO MULTI-ARCHIVO
# # ==========================================
# def process_and_combine_epochs(folder_path, picks, filter_freqs=None, car=None, laplacian=None):
#     search_path = os.path.join(folder_path, "*.xdf")
#     files = glob.glob(search_path)
#     all_epochs_list = []

#     for f_path in sorted(files):
#         try:
#             streams, _ = pyxdf.load_xdf(f_path, synchronize_clocks=False)
#             eeg_s = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
#             marker_s = next(s for s in streams if s["info"]["type"][0].lower() == "markers")
            
#             sfreq = float(eeg_s["info"]["nominal_srate"][0])
#             ch_names = [c["label"][0] for c in eeg_s["info"]["desc"][0]["channels"][0]["channel"]]
#             data = eeg_s["time_series"].T
#             if np.nanmax(np.abs(data)) > 1.0: data *= 1e-6
            
#             info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
#             raw = mne.io.RawArray(data, info, verbose=False)
#             raw.rename_channels(lambda x: x[:-1] + "z" if x.endswith("Z") and len(x) <= 3 else x)

#             # 1. Filtro Butterworth Opcional
#             if filter_freqs is not None:
#                 raw.filter(l_freq=filter_freqs[0], h_freq=filter_freqs[1], 
#                            method='iir', iir_params=dict(order=4, ftype='butter'), 
#                            phase='forward', verbose=False)

#             # 2. Filtro CAR Opcional
#             if car:
#                 raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

#             # 3. Filtro Laplaciano Opcional (Requiere montaje)
#             if laplacian:
#                 raw.set_montage('standard_1020')
#                 raw = mne.preprocessing.compute_current_source_density(raw)

#             # --- Procesamiento de Marcadores ---
#             t_start = eeg_s["time_stamps"][0]
#             labels = [str(l[0]) for l in marker_s["time_series"]]
#             event_dict = {label: i + 1 for i, label in enumerate(sorted(list(set(labels))))}
#             if EVENTO_MI not in event_dict: continue

#             onsets = np.round((marker_s["time_stamps"] - t_start) * sfreq).astype(np.int64)
#             ids = [event_dict[l] for l in labels]
#             events = np.c_[onsets, np.zeros_like(onsets), ids]

#             reject = {'eeg': REJECT_THRESHOLD}
#             ep = mne.Epochs(raw, events, event_id={EVENTO_MI: event_dict[EVENTO_MI]},
#                             tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, 
#                             picks=picks, preload=True, reject=reject, 
#                             event_repeated='drop', verbose=False)
#             all_epochs_list.append(ep)

#         except Exception as e:
#             print(f"Error en {f_path}: {e}")

#     return mne.concatenate_epochs(all_epochs_list) if all_epochs_list else None

# # ==========================================
# # EJECUCIÓN Y DASHBOARD
# # ==========================================
# if __name__ == "__main__":
#     epochs = process_and_combine_epochs(FOLDER_PATH, CHANNELS_TO_PLOT, 
#                                         filter_freqs=FILTER_RANGE, 
#                                         car=APPLY_CAR, 
#                                         laplacian=APPLY_LAPLACIAN)

#     if epochs:
#         times = epochs.times
#         n_total = len(epochs)
        
#         # Gráficas Individuales con Sombreado de Variabilidad
#         for ch_name in CHANNELS_TO_PLOT:
#             plt.figure(figsize=(10, 5))
#             data = epochs.get_data(picks=ch_name)[:, 0, :] * 1e6 
#             avg = np.mean(data, axis=0)
#             std = np.std(data, axis=0)
            
#             plt.fill_between(times, avg - std, avg + std, color='gray', alpha=0.2, label='±1 STD')
#             plt.plot(times, avg, color='black', lw=2, label='Promedio')
            
#             plt.axvline(0, color='red', linestyle='--')
#             plt.title(f"Detalle Sensor: {ch_name} (n={n_total})")
#             plt.ylabel("Amplitud (µV)")
#             plt.grid(True, alpha=0.2)
#             plt.legend()
#             plt.tight_layout()

#         # Gráfica Comparativa Superpuesta
#         plt.figure(figsize=(12, 7))
#         colors_resumen = plt.cm.Set1(np.linspace(0, 1, len(CHANNELS_TO_PLOT)))
        
#         for i, ch_name in enumerate(CHANNELS_TO_PLOT):
#             data_ch = epochs.get_data(picks=ch_name)[:, 0, :] * 1e6
#             avg_ch = np.mean(data_ch, axis=0)
#             plt.plot(times, avg_ch, color=colors_resumen[i], lw=2.5, label=f"Canal {ch_name}")

#         plt.axvline(0, color='black', linestyle='-', lw=1.5)
#         plt.axhline(0, color='black', linestyle=':', alpha=0.5)
#         plt.title(f"Comparativa: {', '.join(CHANNELS_TO_PLOT)} | Filtros Activos: " + 
#                   f"{'Butterworth ' if FILTER_RANGE else ''}{'CAR ' if APPLY_CAR else ''}" +
#                   f"{'Laplacian' if APPLY_LAPLACIAN else ''}", fontsize=12)
#         plt.xlabel("Tiempo (s)")
#         plt.ylabel("Amplitud (µV)")
#         plt.legend(loc='upper right', frameon=True, shadow=True)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         plt.show()

import numpy as np
import bci_runtime_env
import mne
import pyxdf
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
FOLDER_PATH = "/home/lab-admin/Documents/CurrentStudy/sub-P002/ses-S001/eeg/"

# Triggers de interés
EVENTO_MI = '200.0' 
EVENTO_REST = '100.0' # Ajustar si el trigger de descanso es diferente

CHANNELS_TO_PLOT = ["C3", "FC1", "Cz", "CP1"]

CHANNELS_TO_DROP = [
    'Fp1', 'Fp2', 'Fpz', 'M1', 'M2', 
    'AUX1', 'AUX2', 'AUX3', 'AUX7', 'AUX8', 'AUX9', 'TRIGGER'
]

T_MIN, T_MAX = -2.0, 4.0
BASELINE = (-1.0, 0)
REJECT_THRESHOLD = 50e-6 

FILTER_RANGE = [0.1, 3.0]
APPLY_CAR = True

# ==========================================
# 2. PROCESAMIENTO
# ==========================================
def process_bci_data(folder_path, picks, filter_freqs=None, car=None):
    search_path = os.path.join(folder_path, "*.xdf")
    files = glob.glob(search_path)
    mi_list, rest_list = [], []

    for f_path in sorted(files):
        try:
            print(f"\n--- Procesando: {os.path.basename(f_path)} ---")
            streams, _ = pyxdf.load_xdf(f_path, synchronize_clocks=False)
            eeg_s = next(s for s in streams if s["info"]["type"][0].lower() == "eeg")
            marker_s = next(s for s in streams if s["info"]["type"][0].lower() == "markers")
            
            sfreq = float(eeg_s["info"]["nominal_srate"][0])
            ch_names = [c["label"][0] for c in eeg_s["info"]["desc"][0]["channels"][0]["channel"]]
            data = eeg_s["time_series"].T
            if np.nanmax(np.abs(data)) > 1.0: data *= 1e-6
            
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info, verbose=False)

            # Limpieza y CPz
            raw.rename_channels(lambda x: x[:-1] + "z" if x.endswith("Z") and len(x) <= 3 else x)
            raw.rename_channels(lambda x: x.replace('FP', 'Fp'))
            if 'CPz' not in raw.ch_names:
                mne.add_reference_channels(raw, ref_channels=['CPz'])
            
            raw.drop_channels([ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names])

            if filter_freqs:
                raw.filter(l_freq=filter_freqs[0], h_freq=filter_freqs[1], method='iir', verbose=False)
            if car:
                raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

            # Marcadores
            t_start = eeg_s["time_stamps"][0]
            labels = [str(l[0]) for l in marker_s["time_series"]]
            event_dict = {label: i + 1 for i, label in enumerate(sorted(list(set(labels))))}
            
            onsets = np.round((marker_s["time_stamps"] - t_start) * sfreq).astype(np.int64)
            ids = [event_dict[label] for label in labels]
            events = np.c_[onsets, np.zeros_like(onsets), ids]

            # Épocas MI
            if EVENTO_MI in event_dict:
                ep_mi = mne.Epochs(raw, events, event_id={EVENTO_MI: event_dict[EVENTO_MI]},
                                   tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, picks=picks,
                                   preload=True, reject={'eeg': REJECT_THRESHOLD}, verbose=False)
                if len(ep_mi) > 0: mi_list.append(ep_mi)

            # Épocas Rest
            if EVENTO_REST in event_dict:
                ep_rest = mne.Epochs(raw, events, event_id={EVENTO_REST: event_dict[EVENTO_REST]},
                                     tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, picks=picks,
                                     preload=True, reject={'eeg': REJECT_THRESHOLD}, verbose=False)
                if len(ep_rest) > 0: rest_list.append(ep_rest)

        except Exception as e:
            print(f"❌ Error: {e}")

    return (mne.concatenate_epochs(mi_list) if mi_list else None, 
            mne.concatenate_epochs(rest_list) if rest_list else None)

# ==========================================
# 3. GENERACIÓN DE GRÁFICAS
# ==========================================
if __name__ == "__main__":
    epochs_mi, epochs_rest = process_bci_data(FOLDER_PATH, CHANNELS_TO_PLOT, FILTER_RANGE, APPLY_CAR)

    if epochs_mi and epochs_rest:
        times = epochs_mi.times
        
        # --- A. GRÁFICAS INDIVIDUALES (MI vs REST) ---
        for ch_name in CHANNELS_TO_PLOT:
            plt.figure(figsize=(10, 5))
            
            # Datos MI
            data_mi = epochs_mi.get_data(picks=ch_name)[:, 0, :] * 1e6
            avg_mi = np.mean(data_mi, axis=0)
            std_mi = np.std(data_mi, axis=0)
            
            # Datos Rest
            data_rest = epochs_rest.get_data(picks=ch_name)[:, 0, :] * 1e6
            avg_rest = np.mean(data_rest, axis=0)
            std_rest = np.std(data_rest, axis=0)
            
            plt.fill_between(times, avg_mi - std_mi, avg_mi + std_mi, color='blue', alpha=0.1)
            plt.plot(times, avg_mi, color='blue', lw=2, label=f'{ch_name} Imaginación Motora')
            
            plt.fill_between(times, avg_rest - std_rest, avg_rest + std_rest, color='green', alpha=0.1)
            plt.plot(times, avg_rest, color='green', lw=2, label=f'{ch_name} Reposo (Rest)')
            
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f"Comparativa: {ch_name} (MI vs Rest)")
            plt.ylabel("Amplitud (µV)")
            plt.legend()
            plt.grid(True, alpha=0.2)

        # --- B. DASHBOARD N-ELECTRODOS (POR SEPARADO) ---
        # Gráfica MI
        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(CHANNELS_TO_PLOT)))
        for i, ch_name in enumerate(CHANNELS_TO_PLOT):
            plt.plot(times, np.mean(epochs_mi.get_data(picks=ch_name) * 1e6, axis=(0, 1)), 
                     color=colors[i], lw=2, label=ch_name)
        plt.axvline(0, color='black')
        plt.title("Dashboard: Todos los canales en IMAGINACIÓN MOTORA")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfica Rest
        plt.figure(figsize=(12, 6))
        for i, ch_name in enumerate(CHANNELS_TO_PLOT):
            plt.plot(times, np.mean(epochs_rest.get_data(picks=ch_name) * 1e6, axis=(0, 1)), 
                     color=colors[i], lw=2, label=ch_name, linestyle='--')
        plt.axvline(0, color='black')
        plt.title("Dashboard: Todos los canales en REPOSO (REST)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()