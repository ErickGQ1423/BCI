import pyxdf
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
file_path = "/home/lab-admin/Documents/CurrentStudy/sub-P001/ses-S002/eeg/sub-P001_ses-S002_task-Default_run-001_eeg.xdf"

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

markers = None
for s in streams:
    name = s['info']['name'][0]
    typ  = s['info']['type'][0]
    if typ.lower() in ('markers', 'marker'):
        print("Marker stream encontrado:", name)
        # Prefiere el que NO tenga 'eegosports' en el nombre
        if 'eegosports' not in name.lower():
            markers = s
            break

if markers is None:
    print("\n⚠️ No se encontró ningún stream de markers en este archivo.")
else:
    print(f"\nUsando stream de markers: {markers['info']['name'][0]}")

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

# --- asegurar forma (n_channels, n_samples) ---
if data.ndim == 2:
    if n_channels_meta is not None:
        if data.shape[0] != n_channels_meta and data.shape[1] == n_channels_meta:
            data = data.T
    else:
        if data.shape[0] == len(ts) and data.shape[1] != len(ts):
            data = data.T

print("Shapes finales:", data.shape, ts.shape)

# =============================================================
# === RECORTAR SOLO UN FRAGMENTO DE TIEMPO =====================
# =============================================================
# t0, t1 = 20, 100  # segundos a visualizar
# mask = (ts >= ts[0] + t0) & (ts <= ts[0] + t1)

# --- definir ventana automáticamente alrededor de markers ---
if markers is not None:
    m_ts = np.asarray(markers['time_stamps'])
    if len(m_ts) > 0:
        t0 = max(0, (m_ts.min() - ts[0]) - 2.0)   # 2 s antes
        t1 = (m_ts.max() - ts[0]) + 2.0           # 2 s después
    else:
        t0, t1 = 0, 40
else:
    t0, t1 = 0, 40

mask = (ts >= ts[0] + t0) & (ts <= ts[0] + t1)

ts_seg = ts[mask]
data_seg = data[:, mask]
t_rel = ts_seg - ts_seg[0]   # tiempo relativo dentro del segmento

# =============================================================
# === IMPRIMIR Y RESUMIR EVENTOS DESDE MARKERS ================
# =============================================================
if markers is not None:
    m_ts = np.asarray(markers['time_stamps'])
    m_2d = np.asarray(markers['time_series'])  # shape (n_events, 4)
    m_lab = m_2d[:, 0]  # SOLO el ID del marker (100/200/etc)

    print("\n[DEBUG] MarkerStream vs EEG timestamps:")
    print("  n_markers =", len(m_ts))
    if len(m_ts) > 0:
        print("  Marker min ts =", m_ts.min())
        print("  Marker max ts =", m_ts.max())
    print("  EEG ts[0] =", ts[0])
    print("  EEG ts[-1] =", ts[-1])
    print("  Segmento EEG ts_seg[0], ts_seg[-1] =", ts_seg[0], ts_seg[-1])

    if len(m_ts) > 0:
        print("  Primeros 5 markers (ts, id):")
        for i in range(min(5, len(m_ts))):
            print("   ", i, "ts=", m_ts[i], "id=", int(m_lab[i]))


    # markers dentro del segmento t0–t1
    mask_m = (m_ts >= ts_seg[0]) & (m_ts <= ts_seg[-1])

    m_ts_seg  = m_ts[mask_m]
    m_lab_seg = m_lab[mask_m]

    print("\nResumen de markers en el segmento:")
    if len(m_ts_seg) == 0:
        print("  (No hay markers dentro de este intervalo de tiempo)")
    else:
        # 1) ver algunos labels crudos
        print("  Primeros 10 labels crudos:")
        for t_ev, lab in list(zip(m_ts_seg, m_lab_seg))[:10]:
            print(f"    t = {t_ev - ts_seg[0]:.3f} s  ->  {lab}")

        # 2) ver valores únicos
        uniq, counts = np.unique(m_lab_seg, return_counts=True)
        print("\n  Valores únicos y sus cuentas en el segmento:")
        for u, c in zip(uniq, counts):
            print(f"    {u} : {c} veces")

        # 3) quedarnos solo con los cambios de estado (cuando cambia el label)
        if len(m_lab_seg) > 1:
            change_idx = np.where(
                np.r_[True, m_lab_seg[1:] != m_lab_seg[:-1]]
            )[0]
        else:
            change_idx = np.array([0], dtype=int)

        t_events_rel = m_ts_seg[change_idx] - ts_seg[0]
        lab_events   = m_lab_seg[change_idx]

        print("\n  Eventos (solo cambios de estado):")
        for t_ev, lab in zip(t_events_rel, lab_events):
            print(f"    t = {t_ev:.3f} s  ->  {lab}")

        # 4) Opcional: graficar cambios de estado con códigos pequeños
        #    (asignamos 1,2,3,... a cada label distinto)
        label_to_code = {lab: i + 1 for i, lab in enumerate(uniq)}
        codes_plot = np.array([label_to_code[lab] for lab in lab_events])

        plt.figure(figsize=(8, 3))
        plt.stem(t_events_rel, codes_plot)
        plt.xlabel("Tiempo relativo en el segmento (s)")
        plt.ylabel("Código")
        plt.title("Cambios de estado del experimento (markers)")
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.show()
else:
    print("\nNo se imprimen eventos porque no hay stream de markers.")


# =============================================================
# === VISUALIZACIÓN: STACK DE CANALES EEG =====================
# =============================================================
TOTAL = data.shape[0]

N = 39           # cuántos canales quieres ver
FIRST = TOTAL - N
if FIRST < 0:
    FIRST = 0

SCALE = 1.0
DECIM = max(1, len(ts_seg) // 5000)   # decimado más agresivo

sel = slice(FIRST, min(FIRST + N, TOTAL))

t_plot = t_rel[::DECIM]
Y = (data_seg[sel, ::DECIM] * SCALE)

# asegurar que Y sea 2D
if Y.ndim == 1:
    Y = Y[np.newaxis, :]

# normaliza cada canal
Y = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-12)

n_sel = Y.shape[0]
offset = 5.0

plt.figure(figsize=(14, 6))
for i in range(n_sel):
    idx_ch = FIRST + i
    lbl = ch_names[idx_ch] if ch_names and idx_ch < len(ch_names) else f"Ch{idx_ch}"
    plt.plot(t_plot, Y[i] + i * offset, linewidth=0.8, label=lbl, rasterized=True)

# --- dibujar markers como líneas verticales en la gráfica de EEG ---
if markers is not None:
    m_ts = np.asarray(markers['time_stamps'])
    m_seg = m_ts[(m_ts >= ts[0] + t0) & (m_ts <= ts[0] + t1)]
    for mt in m_seg:
        plt.axvline(mt - ts_seg[0], color='k', alpha=0.08)

plt.yticks(
    [i * offset for i in range(n_sel)],
    [ch_names[FIRST + i] if ch_names and (FIRST + i) < len(ch_names) else f"Ch{FIRST + i}"
     for i in range(n_sel)]
)
plt.xlabel("Tiempo (s) relativo al inicio del segmento")
plt.title(f"EEG – canales {FIRST}..{FIRST + n_sel - 1} (segmento {t0}-{t1}s)")
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
plt.show()
