import pyxdf
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
file_path = "/home/lab-admin/Documents/CurrentStudy/sub-ErickGuzman/ses-S001/eeg/sub-ErickGuzman_ses-S001_task-Default_run-001_eeg.xdf" # <-- pon aquí tu archivo .xdf

# === CARGA DEL ARCHIVO ===
streams, header = pyxdf.load_xdf(file_path)
print(f"Found {len(streams)} streams in the file.\n")
for s in streams:
    name = s['info']['name'][0]
    typ  = s['info']['type'][0]
    rate = s['info'].get('nominal_srate', ['?'])[0]
    print(f"- {name} ({typ}), nominal rate: {rate} Hz")

# --- identificar EEG y marcadores ---
eeg = next(s for s in streams if s['info']['type'][0].lower() == 'eeg')
markers = next((s for s in streams if s['info']['type'][0].lower() in ('markers','marker')), None)

data = np.asarray(eeg['time_series'])
ts   = np.asarray(eeg['time_stamps'])

# --- nombres de canales ---
try:
    ch_info = eeg['info']['desc'][0]['channels'][0]['channel']
    ch_names = [c['label'][0] for c in ch_info]
    n_channels_meta = len(ch_names)
except Exception:
    ch_names = None
    n_channels_meta = None

print("\nShapes antes:", data.shape, ts.shape, "n_channels_meta:", n_channels_meta)

# --- asegurar forma (n_channels, n_samples) ---
if data.ndim == 2:
    if n_channels_meta is not None:
        if data.shape[0] != n_channels_meta and data.shape[1] == n_channels_meta:
            data = data.T
    else:
        if data.shape[0] == len(ts) and data.shape[1] != len(ts):
            data = data.T

print("Shapes después:", data.shape, ts.shape)

# === VISUALIZACIÓN: STACK DE CANALES ===
N = 39           # cuántos canales mostrar
FIRST = 0       # canal inicial
SCALE = 1.0     # ajusta si ves números enormes (0.1 o 0.01)
DECIM = max(1, len(ts)//30000)

sel = slice(FIRST, FIRST + N)
t = (ts - ts[0])[::DECIM]
Y = (data[sel, ::DECIM] * SCALE)

# normaliza cada canal
Y = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-12)

offset = 5.0
plt.figure(figsize=(14, 6))
for i in range(N):
    lbl = ch_names[FIRST + i] if ch_names else f"Ch{FIRST+i}"
    plt.plot(t, Y[i] + i*offset, linewidth=0.8, label=lbl)

# marcadores
if markers is not None:
    m_ts = np.asarray(markers['time_stamps'])
    for mt in m_ts:
        plt.axvline(mt - ts[0], color='k', alpha=0.08)

plt.yticks([i*offset for i in range(N)],
           [ch_names[FIRST+i] if ch_names else f"Ch{FIRST+i}" for i in range(N)])
plt.xlabel("Tiempo (s)")
plt.title(f"EEG – canales {FIRST}..{FIRST+N-1}")
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
plt.show()