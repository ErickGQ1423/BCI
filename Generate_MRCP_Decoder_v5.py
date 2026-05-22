import pyxdf
import numpy as np
import matplotlib.pyplot as plt

# Archivo

#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf" # Claudia en Diciembre

#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P009/ses-S001/eeg/sub-P009_ses-S001_task-Default_run-001_eeg.xdf" # Erick Enero

#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S010/eeg/sub-P001_ses-S010_task-Default_run-001_eeg.xdf" # Sharon Febrero
#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S001/eeg/sub-P_Claudia_ses-S001_task-Default_run-001_eeg_old1.xdf" # Claudia Febrero
#XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-P_Claudia/ses-S002/eeg/sub-P_Claudia_ses-S002_task-Default_run-001_eeg.xdf"     # Claudia Febrero

XDF_FILE = "/home/lab-admin/Documents/CurrentStudy/sub-Pxxx/ses-S005/eeg/sub-Pxxx_ses-S005_task-Default_run-001_eeg.xdf" #Registro muuuuy corto de prueba

# Cargar
streams, header = pyxdf.load_xdf(XDF_FILE)

# Identificar el stream EEG automáticamente para evitar el IndexError
eeg_stream = next(s for s in streams if s['info']['type'][0].upper() == 'EEG')
t = np.asarray(eeg_stream["time_stamps"])          # (N,)
X = np.asarray(eeg_stream["time_series"])          # (N, C)

# ==========================================================
# --- PARÁMETROS DE VENTANA (AJUSTA AQUÍ) ---
# ==========================================================
t_start_view = 40.0   # Segundos desde que inició el registro
t_end_view   = 80.0  # Segundos hasta donde quieres ver
n_ch_plot    = 39     # Cuántos canales quieres ver (ej. los primeros 10)
# ==========================================================

# Calcular el tiempo relativo (empezando en 0)
t_relativo = t - t[0]

# Crear la máscara para filtrar el tiempo
mask = (t_relativo >= t_start_view) & (t_relativo <= t_end_view)

t_win = t_relativo[mask]
X_win = X[mask, :n_ch_plot]

# Normalizar para que se vean bien apiladas
Xn = X_win - np.mean(X_win, axis=0, keepdims=True)
std = np.std(Xn, axis=0, keepdims=True)
std[std == 0] = 1
Xn = Xn / std

# Offset para apilar
offset = np.arange(n_ch_plot) * 5.0
Y = Xn + offset

# Graficar
plt.figure(figsize=(12, 8))
for ch in range(n_ch_plot):
    plt.plot(t_win, Y[:, ch], linewidth=0.8)

plt.yticks(offset, [f"ch{ch}" for ch in range(n_ch_plot)])
plt.xlabel("Tiempo desde el inicio (s)")
plt.ylabel("Canales")
plt.title(f"Vista EEG: Ventana {t_start_view}s - {t_end_view}s")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()