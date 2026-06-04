"""
================================================================================
Generate_Decoder_TransferLearning.py
================================================================================
Evalúa transfer learning entre sujetos para BCI CNV.

Entrenamiento : SUBJ_001–006  (CNVStudy, sesión S001OFFLINE_GLOVE)
Prueba        : SUBJ_010, SUBJ_011  (nunca vistos por el modelo base)

Enfoque A — M2 + SVM/LR + reentrenamiento adaptativo cada N trials
Enfoque B — MDM Riemanniano + recentrado geodésico trial a trial (Racz 2023)

Para cada sujeto de prueba se mide:
  (1) AUC inicial sin adaptación
  (2) Curva AUC vs número de trials usados para adaptar

Preprocesamiento : avg-ref → notch 60 Hz → BP 0.1–3 Hz → CSD
Features A       : amplitud CSD en T_POINTS (ventana [-1.5, 0.0] s)
Features B       : matrices de covarianza CSD (ventana [-1.5, 0.0] s)
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import bci_runtime_env
import mne
from mne.preprocessing import compute_current_source_density

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

# ============================================================
# CONFIGURACIÓN
# ============================================================

XDF_BASE = "/home/lab-admin/Documents/CNVStudy"

TRAIN_SUBJECTS = [
    ("CNV_PILOT_SUBJ_001", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_002", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_003", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_004", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_005", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_006", "S001OFFLINE_GLOVE"),
]

TEST_SUBJECTS = [
    ("CNV_PILOT_SUBJ_010", "S001"),
    ("CNV_PILOT_SUBJ_011", "S001_OFF"),
]

PICKS_4CH    = ['FC5', 'FC1', 'Cz', 'CP1']
T_START_FEAT = -1.5
T_END_FEAT   =  0.0
N_TIMEPOINTS =  7
T_POINTS     = np.linspace(T_START_FEAT, T_END_FEAT, N_TIMEPOINTS)

BP_LOW    = 0.1
BP_HIGH   = 3.0
REJECT_CSD_MAX = 0.20   # V/m²  — threshold for CSD peak-to-peak artifact rejection
REJECT_CSD_MIN = 1e-7   # V/m²  — flat-line sanity check

# Pasos de adaptación simulados
ADAPT_STEPS     = [0, 10, 20, 30, 40, 50]
MIN_EVAL_TRIALS = 15   # mínimo de trials para calcular AUC válido

CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
    "FCZ": "FCz", "CPZ": "CPz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]


# ============================================================
# CARGA Y PREPROCESAMIENTO
# ============================================================

def load_subject(subject, session):
    """
    Carga XDF y aplica: avg-ref → notch 60 Hz → BP 0.1–3 Hz → CSD → epochs.
    Retorna: epochs (MNE), event_dict, n_rest, n_mi
    """
    xdf_dir = os.path.join(XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/")
    xdf_files = sorted([os.path.join(xdf_dir, f)
                        for f in os.listdir(xdf_dir)
                        if f.endswith(".xdf") and "_old" not in f])
    if not xdf_files:
        raise FileNotFoundError(f"No XDF en: {xdf_dir}")

    raw_list = []
    for xdf_file in xdf_files:
        eeg_s, marker_s = load_xdf(xdf_file)
        eeg_data        = np.array(eeg_s["time_series"]).T
        eeg_timestamps  = np.array(eeg_s["time_stamps"])
        channel_names   = get_channel_names_from_xdf(eeg_s)

        marker_data       = np.array([int(v[0]) for v in marker_s["time_series"]])
        marker_timestamps = np.array(marker_s["time_stamps"])
        keep              = np.isin(marker_data, TARGET_MARKERS)
        marker_data       = marker_data[keep]
        marker_timestamps = marker_timestamps[keep]

        valid_ch  = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data  = eeg_data[valid_idx, :] / 1e6

        info    = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data, info, verbose=False)

        renames = {k: v for k, v in RENAME_DICT.items() if k in raw_tmp.ch_names}
        if renames:
            raw_tmp.rename_channels(renames)

        raw_tmp.set_montage(mne.channels.make_standard_montage("standard_1020"),
                            on_missing="warn")
        for ch in CHANNELS_TO_DROP:
            if ch in raw_tmp.ch_names:
                raw_tmp.drop_channels([ch])

        t0    = eeg_timestamps[0]
        annot = mne.Annotations(
            onset       = marker_timestamps - t0,
            duration    = np.zeros(len(marker_data)),
            description = [str(m) for m in marker_data],
            orig_time   = None,
        )
        raw_tmp.set_annotations(annot)
        raw_list.append(raw_tmp)

    raw = mne.concatenate_raws(raw_list)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
    raw.filter(
        l_freq=BP_LOW, h_freq=BP_HIGH,
        method="iir", iir_params=dict(order=2, ftype="butter"),
        phase="forward", picks="eeg", verbose=False,
    )

    # CSD — excluir canales sin posición
    eeg_chs = [ch for ch in raw.ch_names
               if raw.get_channel_types(picks=ch)[0] == "eeg"]
    no_pos  = [ch for ch in eeg_chs
               if np.allclose(raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3], 0)]
    if no_pos:
        raw.info["bads"] += [ch for ch in no_pos if ch not in raw.info["bads"]]
    raw = compute_current_source_density(raw)

    # Eventos y epochs
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)":   event_id_map["200"],
    }
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=1.0,
        baseline=(-3.0, -2.0),
        reject=None, flat=None,
        preload=True, detrend=None, verbose=False,
    )

    # Rechazo por peak-to-peak en PICKS_4CH
    all_ch      = epochs.ch_names
    picks_avail = [ch for ch in PICKS_4CH if ch in all_ch]
    if picks_avail:
        pidx    = [all_ch.index(ch) for ch in picks_avail]
        data_pp = epochs.get_data()[:, pidx, :]          # V/m² — CSD units
        pp      = data_pp.max(axis=2) - data_pp.min(axis=2)
        drop_mask = (pp.max(axis=1) > REJECT_CSD_MAX) | (pp.max(axis=1) < REJECT_CSD_MIN)
        epochs.drop(np.where(drop_mask)[0].tolist(), reason="MANUAL_REJECT")

    n_rest = len(epochs["Rest (100)"])
    n_mi   = len(epochs["MI (200)"])
    return epochs, event_dict, n_rest, n_mi


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def extract_features_A(epochs):
    """Amplitud CSD en T_POINTS → (X: n×28, y: n)"""
    times       = epochs.times
    t_idx       = [np.argmin(np.abs(times - t)) for t in T_POINTS]
    all_ch      = epochs.ch_names
    picks_avail = [ch for ch in PICKS_4CH if ch in all_ch]
    ch_idx      = [all_ch.index(ch) for ch in picks_avail]
    data        = epochs.get_data()[:, ch_idx, :]
    X = np.hstack([data[:, i, :][:, t_idx] for i in range(data.shape[1])])
    y = epochs.events[:, -1]
    return X, y


def extract_features_B(epochs):
    """
    Matrices de covarianza OAS sobre ventana [T_START_FEAT, T_END_FEAT].
    Retorna (C: n×4×4, y: n)
    """
    times       = epochs.times
    t_mask      = (times >= T_START_FEAT) & (times <= T_END_FEAT)
    all_ch      = epochs.ch_names
    picks_avail = [ch for ch in PICKS_4CH if ch in all_ch]
    ch_idx      = [all_ch.index(ch) for ch in picks_avail]
    data        = epochs.get_data()[:, ch_idx, :][:, :, t_mask]
    C = Covariances(estimator="oas").fit_transform(data)
    y = epochs.events[:, -1]
    return C, y


def recenter_riemann(C_eval, C_ref):
    """
    Recentrado geodésico: blanquea C_eval respecto a la media Riemanniana de C_ref.
    C_ref  : (n_ref, 4, 4) — covarianzas del sujeto de prueba acumuladas hasta ahora
    C_eval : (n_eval, 4, 4) — covarianzas a clasificar
    """
    M         = mean_covariance(C_ref, metric="riemann")
    M_invsqrt = invsqrtm(M)
    return np.array([M_invsqrt @ c @ M_invsqrt for c in C_eval])


# ============================================================
# CARGA DE DATOS
# ============================================================

print(f"\n{'='*65}")
print("🔬  TRANSFER LEARNING — CNV BCI")
print(f"{'='*65}")
mne.set_log_level("WARNING")

print(f"\n{'─'*40}")
print("📂  Cargando sujetos de entrenamiento (SUBJ_001–006) ...")

X_A_list, C_B_list, y_list = [], [], []

for subj, sess in TRAIN_SUBJECTS:
    epochs, event_dict, n_r, n_m = load_subject(subj, sess)
    Xa, ya = extract_features_A(epochs)
    Cb, yb = extract_features_B(epochs)
    X_A_list.append(Xa)
    C_B_list.append(Cb)
    y_list.append(ya)
    print(f"   {subj.split('_')[-1]}: REST={n_r}  MI={n_m}")

X_train_A = np.vstack(X_A_list)
C_train_B = np.vstack(C_B_list)
y_train   = np.concatenate(y_list)

MI_ID   = event_dict["MI (200)"]
REST_ID = event_dict["Rest (100)"]

n_rest_tr = np.sum(y_train == REST_ID)
n_mi_tr   = np.sum(y_train == MI_ID)
print(f"\n   Total entrenamiento: {len(y_train)} trials  (REST={n_rest_tr}, MI={n_mi_tr})")
print(f"   Features A : {X_train_A.shape[1]}  ({len(PICKS_4CH)} ch × {N_TIMEPOINTS} pts)")
print(f"   Features B : covarianzas {C_train_B.shape[1]}×{C_train_B.shape[2]}")

print(f"\n{'─'*40}")
print("📂  Cargando sujetos de prueba (SUBJ_010, 011) ...")

test_data = {}
for subj, sess in TEST_SUBJECTS:
    epochs, event_dict, n_r, n_m = load_subject(subj, sess)
    Xa, ya = extract_features_A(epochs)
    Cb, yb = extract_features_B(epochs)
    test_data[subj] = {"X_A": Xa, "y_A": ya, "C_B": Cb, "y_B": yb}
    print(f"   {subj.split('_')[-1]}: REST={n_r}  MI={n_m}  total={n_r+n_m}")


# ============================================================
# ENFOQUE A — SVM / LR + REENTRENAMIENTO ADAPTATIVO
# ============================================================

print(f"\n{'='*65}")
print("🧠  ENFOQUE A — SVM/LR + reentrenamiento adaptativo")
print(f"{'─'*65}")
print(f"   Base: {len(y_train)} trials de SUBJ_001-006")
print(f"   Adaptación: combinar base + primeros N trials del sujeto de prueba")
print(f"   Evaluación: trials restantes (N+1 en adelante)")

def make_svm():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=42))])

def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])

results_A = {}

for subj, _ in TEST_SUBJECTS:
    sname  = subj.split("_")[-1]
    X_test = test_data[subj]["X_A"]
    y_test = test_data[subj]["y_A"]
    n_test = len(y_test)

    print(f"\n   ── SUBJ_{sname}  ({n_test} trials) ──")
    print(f"   {'n_adapt':>8}  {'SVM AUC':>9}  {'LR AUC':>9}  {'n_eval':>7}")

    auc_svm, auc_lr, steps_used = [], [], []

    for n_adapt in ADAPT_STEPS:
        n_eval = n_test - n_adapt
        if n_eval < MIN_EVAL_TRIALS:
            break

        X_eval = X_test[n_adapt:]
        y_eval = y_test[n_adapt:]

        if n_adapt == 0:
            X_fit, y_fit = X_train_A, y_train
        else:
            X_fit = np.vstack([X_train_A, X_test[:n_adapt]])
            y_fit = np.concatenate([y_train, y_test[:n_adapt]])

        svm = make_svm(); svm.fit(X_fit, y_fit)
        lr  = make_lr();  lr.fit(X_fit, y_fit)

        a_svm = roc_auc_score(y_eval, svm.predict_proba(X_eval)[:, 1])
        a_lr  = roc_auc_score(y_eval, lr.predict_proba(X_eval)[:, 1])

        auc_svm.append(a_svm)
        auc_lr.append(a_lr)
        steps_used.append(n_adapt)
        print(f"   {n_adapt:>8d}  {a_svm:>9.3f}  {a_lr:>9.3f}  {n_eval:>7d}")

    results_A[subj] = {"steps": steps_used, "auc_svm": auc_svm, "auc_lr": auc_lr}


# ============================================================
# ENFOQUE B — MDM RIEMANNIANO + RECENTRADO GEODÉSICO
# ============================================================

print(f"\n{'='*65}")
print("🧠  ENFOQUE B — MDM Riemanniano + recentrado geodésico")
print(f"{'─'*65}")
print(f"   Modelo base: MDM ajustado en {len(y_train)} trials de SUBJ_001-006")
print(f"   Adaptación : primeros N trials del sujeto de prueba → estimación M_ref")
print(f"   Recentrado : C_eval ← M_ref^(-1/2) @ C_eval @ M_ref^(-1/2)")

mdm = MDM(metric="riemann")
mdm.fit(C_train_B, y_train)
print(f"   MDM entrenado. Clases: {mdm.classes_}")

results_B = {}

for subj, _ in TEST_SUBJECTS:
    sname  = subj.split("_")[-1]
    C_test = test_data[subj]["C_B"]
    y_test = test_data[subj]["y_B"]
    n_test = len(y_test)

    print(f"\n   ── SUBJ_{sname}  ({n_test} trials) ──")
    print(f"   {'n_adapt':>8}  {'MDM AUC':>9}  {'n_eval':>7}")

    auc_mdm, steps_used = [], []

    for n_adapt in ADAPT_STEPS:
        n_eval = n_test - n_adapt
        if n_eval < MIN_EVAL_TRIALS:
            break

        C_eval = C_test[n_adapt:]
        y_eval = y_test[n_adapt:]

        if n_adapt == 0:
            C_classify = C_eval
        else:
            C_ref      = C_test[:n_adapt]
            C_classify = recenter_riemann(C_eval, C_ref)

        mi_col = list(mdm.classes_).index(MI_ID)
        proba  = mdm.predict_proba(C_classify)
        a_mdm  = roc_auc_score(y_eval, proba[:, mi_col])

        auc_mdm.append(a_mdm)
        steps_used.append(n_adapt)
        print(f"   {n_adapt:>8d}  {a_mdm:>9.3f}  {n_eval:>7d}")

    results_B[subj] = {"steps": steps_used, "auc_mdm": auc_mdm}


# ============================================================
# RESUMEN
# ============================================================

print(f"\n{'='*65}")
print("🏆  RESUMEN — AUC inicial (sin adaptación)")
print(f"{'─'*65}")
print(f"   {'Sujeto':<12}  {'SVM':>8}  {'LR':>8}  {'MDM':>8}")
print(f"   {'─'*45}")
for subj, _ in TEST_SUBJECTS:
    sname = subj.split("_")[-1]
    s = results_A[subj]["auc_svm"][0] if results_A[subj]["auc_svm"] else float("nan")
    l = results_A[subj]["auc_lr"][0]  if results_A[subj]["auc_lr"]  else float("nan")
    m = results_B[subj]["auc_mdm"][0] if results_B[subj]["auc_mdm"] else float("nan")
    print(f"   SUBJ_{sname:<7}  {s:>8.3f}  {l:>8.3f}  {m:>8.3f}")
print(f"{'='*65}")


# ============================================================
# VISUALIZACIÓN
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

c_svm = "#d6604d"
c_lr  = "#f4a582"
c_mdm = "#1a9641"

for ax_idx, (subj, _) in enumerate(TEST_SUBJECTS):
    ax    = axes[ax_idx]
    sname = subj.split("_")[-1]

    steps_A = results_A[subj]["steps"]
    steps_B = results_B[subj]["steps"]

    if steps_A:
        ax.plot(steps_A, results_A[subj]["auc_svm"], "o-",
                color=c_svm, lw=2.5, ms=8, label="A: SVM")
        ax.plot(steps_A, results_A[subj]["auc_lr"],  "s--",
                color=c_lr,  lw=2.5, ms=8, label="A: LR")
    if steps_B:
        ax.plot(steps_B, results_B[subj]["auc_mdm"], "^-",
                color=c_mdm, lw=2.5, ms=8, label="B: MDM (recentering)")

    ax.axhline(0.5, color="red",  ls="--", lw=1.2, label="Azar (0.5)")
    ax.axhline(0.7, color="gray", ls=":",  lw=1.0, label="Objetivo (0.7)")

    ax.set_xlabel("Trials del sujeto usados para adaptación")
    if ax_idx == 0:
        ax.set_ylabel("AUC (ROC)")
    ax.set_title(f"SUBJ_{sname}", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    max_step = max((steps_A[-1] if steps_A else 0),
                   (steps_B[-1] if steps_B else 0))
    ax.set_xlim(-3, max_step + 5)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, ls=":", alpha=0.4)

    # Anotar AUC inicial
    if results_A[subj]["auc_svm"]:
        ax.annotate(f"AUC₀={results_A[subj]['auc_svm'][0]:.3f}",
                    xy=(0, results_A[subj]["auc_svm"][0]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=8, color=c_svm)
    if results_B[subj]["auc_mdm"]:
        ax.annotate(f"AUC₀={results_B[subj]['auc_mdm'][0]:.3f}",
                    xy=(0, results_B[subj]["auc_mdm"][0]),
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=c_mdm)

plt.suptitle(
    f"Transfer Learning CNV BCI\n"
    f"Modelo base: SUBJ_001–006 ({len(y_train)} trials) "
    f"→ Prueba: SUBJ_010, SUBJ_011 (sin overlap)\n"
    f"Preprocesamiento: avg-ref → notch → BP(0.1–3 Hz) → CSD  |  "
    f"Canales: {PICKS_4CH}  |  Ventana: [{T_START_FEAT}, {T_END_FEAT}] s",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figures_TransferLearning.png")
os.makedirs(os.path.dirname(fig_path) if os.path.dirname(fig_path) else ".", exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n📊  Figura guardada: {fig_path}")
plt.show()
