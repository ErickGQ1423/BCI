import numpy as np
import matplotlib.pyplot as plt
import sys
import os

try:
    import config
    from Utils.stream_utils import load_xdf
    FS = config.FS
except ImportError:
    print("❌ Error: revisa el entorno del proyecto")
    sys.exit(1)


def extract_triggers(xdf_path, keep_ids):
    eeg_stream, marker_stream = load_xdf(xdf_path)

    ts = np.asarray(marker_stream["time_stamps"])
    labels = np.asarray(marker_stream["time_series"])

    events = []

    for col in [0, 1]:
        if labels.ndim <= col:
            continue

        for i in range(len(ts)):
            try:
                trig_id = int(float(labels[i, col]))
                if trig_id in keep_ids:
                    events.append((ts[i], trig_id))
            except:
                continue

        if events:
            break

    if not events:
        return None

    return np.array(events, dtype=[("ts", float), ("id", int)])


def build_series_aligned(events, t0, FS, N):
    """Construye un vector discreto alineado a un t0 común y longitud N."""
    t_rel = events["ts"] - t0
    idx = np.round(t_rel * FS).astype(int)

    series = np.zeros(N, dtype=int)

    # Por seguridad: filtrar índices fuera de rango (por redondeo o eventos raros)
    valid = (idx >= 0) & (idx < N)
    series[idx[valid]] = events["id"][valid]

    return series


def plot_dual_trigger_series_aligned(xdf_path):
    print(f"📂 Analizando: {os.path.basename(xdf_path)}")

    main_ids = {100, 110, 120}
    aux_ids  = {200, 210, 220}

    main_events = extract_triggers(xdf_path, main_ids)
    aux_events  = extract_triggers(xdf_path, aux_ids)

    if main_events is None and aux_events is None:
        print("❌ No se encontraron triggers en ninguno de los dos grupos.")
        return

    # ✅ t0 común: el primer evento global (del que ocurra antes)
    t0_candidates = []
    if main_events is not None: t0_candidates.append(main_events["ts"][0])
    if aux_events  is not None: t0_candidates.append(aux_events["ts"][0])
    t0 = min(t0_candidates)

    # ✅ Longitud común N: hasta el último evento global (del que ocurra más tarde)
    t_last_candidates = []
    if main_events is not None: t_last_candidates.append(main_events["ts"][-1])
    if aux_events  is not None: t_last_candidates.append(aux_events["ts"][-1])
    t_last = max(t_last_candidates)

    N = int(np.round((t_last - t0) * FS)) + 1
    t = np.arange(N) / FS

    # Construir ambas series alineadas
    if main_events is not None:
        main_series = build_series_aligned(main_events, t0, FS, N)
    else:
        main_series = np.zeros(N, dtype=int)

    if aux_events is not None:
        aux_series = build_series_aligned(aux_events, t0, FS, N)
    else:
        aux_series = np.zeros(N, dtype=int)

    # === GRÁFICA ===
    plt.figure(figsize=(15, 4))
    plt.step(t, main_series, where="post", label="Triggers 100/120 (Rest)", linewidth=1.8)
    plt.step(t, aux_series,  where="post", label="Triggers 200/220 (Movement)", linewidth=1.5, alpha=0.85)

    plt.xlabel("Tiempo desde el inicio (t0 global) (s)")
    plt.ylabel("Trigger ID (0 = sin evento)")
    plt.title("Triggers superpuestos (alineados correctamente en el tiempo)")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Debug rápido: mostrar diferencias entre eventos consecutivos si existen ===
    if main_events is not None and aux_events is not None:
        # ejemplo: primer par cercano de 100->120
        print("✅ Alineación lista. Si quieres, ahora sí podemos medir offsets 100→120 y 200→220 automáticamente.")


if __name__ == "__main__":
    # XDF_FILE = (
    #    "/home/lab-admin/BCI_project/CurrentStudy/sub-Prueba/training_data/"
    #    "sub-CLASS_SUBJ_1032_ses-S001OFFLINE_FES_task-Default_run-001_eeg.xdf"
    #)
    XDF_FILE = ("/home/lab-admin/Documents/CurrentStudy/" \
    "sub-S26CLASS_SUBJ_008/ses-borrar/eeg/" \
    "sub-S26CLASS_SUBJ_008_ses-borrar_task-Default_run-001_eeg.xdf")
    plot_dual_trigger_series_aligned(XDF_FILE)
