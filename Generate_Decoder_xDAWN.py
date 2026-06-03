"""
================================================================================
Generate_Decoder_xDAWN.py
================================================================================
Compara cuatro pipelines de filtrado espacial para clasificación CNV:
  A) Sin filtro     — canales PICKS_CNV, amplitudes en T_POINTS → LDA_shrink
  B) CSD            — Laplaciano de superficie sobre PICKS_CNV → LDA_shrink
  C) xDAWN-PICKS    — filtro xDAWN entrenado sobre PICKS_CNV → LDA_shrink
  D) xDAWN-ALL      — filtro xDAWN entrenado sobre todos los canales EEG

xDAWN se ajusta DENTRO de cada fold CV para evitar data leakage.

Referencia: Rivet et al. (2009) — xDAWN algorithm for evoked potentials
================================================================================
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUBJECT  = "CNV_PILOT_SUBJ_001"
SESSION  = "S001OFFLINE"
XDF_BASE = "/home/lab-admin/Documents/CurrentStudy"

PICKS_CNV = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1']

T_START            = -2.5
T_END              =  0.0
N_TIMEPOINTS       =  9
T_POINTS           = np.linspace(T_START, T_END, N_TIMEPOINTS)
N_XDAWN_COMPONENTS =  4
N_CV_FOLDS         =  5

BP_LOW    = 0.1
BP_HIGH   = 3.0
REJECT_UV = 150.0   # µV peak-to-peak

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

def load_and_preprocess(subject, session):
    """Carga XDF y aplica: ref. promedio, notch 60 Hz, filtro BP. Sin CSD."""
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

        valid_ch        = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx       = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info    = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        existing_renames = {k: v for k, v in RENAME_DICT.items()
                            if k in raw_tmp.ch_names}
        if existing_renames:
            raw_tmp.rename_channels(existing_renames)

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
    return raw


def make_epochs(raw):
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)":   event_id_map["200"],
    }
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=5.0,
        baseline=(-3.0, -2.0),
        reject=None, flat=None,
        preload=True, detrend=None, verbose=False,
    )

    # Rechazo por peak-to-peak en PICKS_CNV
    ch_names = epochs.copy().pick_types(eeg=True).ch_names
    p_avail  = [ch for ch in PICKS_CNV if ch in ch_names]
    p_idx    = [ch_names.index(ch) for ch in p_avail]
    data     = epochs.get_data()[:, p_idx, :] * 1e6
    pp       = data.max(axis=2) - data.min(axis=2)
    drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
    epochs.drop(np.where(drop_mask)[0].tolist(), reason="MANUAL_REJECT")

    n_rest = len(epochs["Rest (100)"])
    n_mi   = len(epochs["MI (200)"])
    n_drop = drop_mask.sum()
    print(f"   Rest={n_rest}  MI={n_mi}  rechazados={n_drop}")
    return epochs, event_dict


# ============================================================
# FUNCIONES xDAWN
# ============================================================

def fit_xdawn_filter(X_mi, X_all, n_components):
    """
    Calcula filtros xDAWN.
    X_mi  : (n_mi,  n_ch, n_t) — epochs MI para covarianza de señal
    X_all : (n_all, n_ch, n_t) — todos los epochs para covarianza de ruido
    Retorna W: (n_ch, n_components)
    """
    n_ch = X_mi.shape[1]

    # Covarianza de señal: respuesta evocada promedio
    evoked = X_mi.mean(axis=0)       # (n_ch, n_t)
    Cs = evoked @ evoked.T           # (n_ch, n_ch)

    # Covarianza de ruido: todos los epochs
    Cn = np.zeros((n_ch, n_ch))
    for ep in X_all:
        Cn += ep @ ep.T
    Cn /= (X_all.shape[0] * X_all.shape[2])
    Cn += 1e-6 * np.eye(n_ch)        # regularización mínima

    # Eigenvalores generalizados: Cs W = λ Cn W
    eigenvalues, eigenvectors = eigh(Cs, Cn)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]]   # (n_ch, n_components)


def apply_xdawn(X_3d, W):
    """X_3d: (n, n_ch, n_t) → (n, n_comp, n_t)"""
    return np.einsum('ect,cn->ent', X_3d, W)


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def extract_tp_features(data_3d, times, t_points):
    """
    data_3d: (n_epochs, n_signals, n_times)
    Retorna: (n_epochs, n_signals × len(t_points)) — amplitudes en T_POINTS
    """
    t_idx = [np.argmin(np.abs(times - t)) for t in t_points]
    return np.hstack([data_3d[:, i, :][:, t_idx]
                      for i in range(data_3d.shape[1])])


# ============================================================
# CROSS-VALIDATION
# ============================================================

def make_lda():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def cv_standard(X_2d, y, n_splits=N_CV_FOLDS):
    """CV estándar (sin xDAWN)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []
    for tr, te in cv.split(X_2d, y):
        clf = make_lda()
        clf.fit(X_2d[tr], y[tr])
        accs.append(np.mean(clf.predict(X_2d[te]) == y[te]) * 100)
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X_2d[te])[:, 1]))
        except Exception:
            aucs.append(0.5)
    return np.array(accs), np.array(aucs)


def cv_xdawn(X_3d, y, mi_id, n_components, times, t_points, n_splits=N_CV_FOLDS):
    """
    CV con xDAWN ajustado dentro de cada fold (sin data leakage).
    X_3d: (n_epochs, n_ch, n_times)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []
    for tr, te in cv.split(X_3d, y):
        # xDAWN ajustado solo sobre MI de entrenamiento
        W = fit_xdawn_filter(X_3d[tr][y[tr] == mi_id], X_3d[tr], n_components)

        X_tr_feat = extract_tp_features(apply_xdawn(X_3d[tr], W), times, t_points)
        X_te_feat = extract_tp_features(apply_xdawn(X_3d[te], W), times, t_points)

        clf = make_lda()
        clf.fit(X_tr_feat, y[tr])
        accs.append(np.mean(clf.predict(X_te_feat) == y[te]) * 100)
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X_te_feat)[:, 1]))
        except Exception:
            aucs.append(0.5)
    return np.array(accs), np.array(aucs)


def cv_xdawn_temporal(X_3d, y, mi_id, n_components, times, t_points_all,
                      n_splits=N_CV_FOLDS):
    """
    CV xDAWN acumulativo: evalúa a cada instante t_points_all[k]
    usando solo los features hasta ese punto.
    Retorna listas de acc y auc por instante.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Precompute xDAWN filters for each fold (fit once, reuse for all time steps)
    fold_data = []
    for tr, te in cv.split(X_3d, y):
        W = fit_xdawn_filter(X_3d[tr][y[tr] == mi_id], X_3d[tr], n_components)
        fold_data.append((tr, te, W))

    accs_t, aucs_t = [], []
    for step in range(1, len(t_points_all) + 1):
        t_sub = t_points_all[:step]
        accs_fold, aucs_fold = [], []
        for tr, te, W in fold_data:
            X_tr_feat = extract_tp_features(apply_xdawn(X_3d[tr], W), times, t_sub)
            X_te_feat = extract_tp_features(apply_xdawn(X_3d[te], W), times, t_sub)
            clf = make_lda()
            clf.fit(X_tr_feat, y[tr])
            accs_fold.append(np.mean(clf.predict(X_te_feat) == y[te]) * 100)
            try:
                aucs_fold.append(roc_auc_score(y[te], clf.predict_proba(X_te_feat)[:, 1]))
            except Exception:
                aucs_fold.append(0.5)
        accs_t.append(np.mean(accs_fold))
        aucs_t.append(np.mean(aucs_fold))
    return np.array(accs_t), np.array(aucs_t)


def cv_standard_temporal(X_2d_full, y, times, t_points_all, n_splits=N_CV_FOLDS):
    """CV estándar acumulativo temporal (para pipelines A y B)."""
    n_signals = X_2d_full.shape[1] // len(t_points_all)
    accs_t, aucs_t = [], []
    for step in range(1, len(t_points_all) + 1):
        # Reconstruir X con solo los primeros `step` timepoints
        # Los features están ordenados como [ch1_t1,..,ch1_tk, ch2_t1,..,ch2_tk, ...]
        # Necesitamos reindexar
        t_sub   = t_points_all[:step]
        t_idx   = [np.argmin(np.abs(times - t)) for t in t_sub]
        # Placeholder: se pasa X_3d de cada pipeline para consistencia
        accs_t.append(0)
        aucs_t.append(0)
    return np.array(accs_t), np.array(aucs_t)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

print(f"\n{'='*65}")
print("🔬  COMPARACIÓN PIPELINES ESPACIALES — CNV BCI")
print(f"{'='*65}")
print(f"   Sujeto     : {SUBJECT} | {SESSION}")
print(f"   Canales    : {PICKS_CNV}")
print(f"   xDAWN      : {N_XDAWN_COMPONENTS} componentes")
print(f"   CV         : {N_CV_FOLDS}-fold estratificado (xDAWN ajustado por fold)")
print(f"   Ventana    : [{T_START}, {T_END}]s  ({N_TIMEPOINTS} puntos)")

mne.set_log_level("WARNING")

# ── Cargar datos ────────────────────────────────────────────
print(f"\n{'─'*40}")
print("📂  Cargando y preprocesando ...")
raw = load_and_preprocess(SUBJECT, SESSION)
epochs, event_dict = make_epochs(raw)

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]
y       = epochs.events[:, -1]
times   = epochs.times

# Índices de canales EEG con posición (para CSD y xDAWN-ALL)
all_eeg_ch  = [ch for ch in epochs.ch_names
               if epochs.get_channel_types(picks=ch)[0] == "eeg"]
no_pos_ch   = [ch for ch in all_eeg_ch
               if np.allclose(epochs.info["chs"][epochs.ch_names.index(ch)]["loc"][:3], 0)]
valid_eeg_ch = [ch for ch in all_eeg_ch if ch not in no_pos_ch]
valid_eeg_idx = [epochs.ch_names.index(ch) for ch in valid_eeg_ch]

if no_pos_ch:
    print(f"   ⚠️  Canales sin posición 3D (excluidos de CSD/xDAWN-ALL): {no_pos_ch}")

picks_avail   = [ch for ch in PICKS_CNV if ch in valid_eeg_ch]
picks_idx     = [epochs.ch_names.index(ch) for ch in picks_avail]

# Arrays 3D base
data_picks = epochs.get_data()[:, picks_idx, :] * 1e6       # (n, picks, t)
data_all   = epochs.get_data()[:, valid_eeg_idx, :] * 1e6   # (n, all_eeg, t)

# ── Pipeline A: Sin filtro ──────────────────────────────────
print(f"\n{'─'*40}")
print("📊  A — Sin filtro espacial (baseline) ...")
X_a        = extract_tp_features(data_picks, times, T_POINTS)
acc_a, auc_a = cv_standard(X_a, y)
print(f"   Acc: {acc_a.mean():.1f}% ± {acc_a.std():.1f}%  |  AUC: {auc_a.mean():.3f} ± {auc_a.std():.3f}")
print(f"   Features: {X_a.shape[1]}  ({len(picks_avail)} ch × {N_TIMEPOINTS} pts)")

# ── Pipeline B: CSD ─────────────────────────────────────────
print(f"\n{'─'*40}")
print("📊  B — CSD (Laplaciano de superficie) ...")
from mne.preprocessing import compute_current_source_density
epochs_csd     = compute_current_source_density(epochs.copy())
ch_names_csd   = [ch for ch in epochs_csd.ch_names
                  if epochs_csd.get_channel_types(picks=ch)[0] == "csd"]
picks_csd_avail = [ch for ch in picks_avail if ch in ch_names_csd]
picks_csd_idx   = [ch_names_csd.index(ch) for ch in picks_csd_avail]
data_csd        = epochs_csd.get_data(picks="csd")[:, picks_csd_idx, :] * 1e6

X_b        = extract_tp_features(data_csd, times, T_POINTS)
acc_b, auc_b = cv_standard(X_b, y)
print(f"   Acc: {acc_b.mean():.1f}% ± {acc_b.std():.1f}%  |  AUC: {auc_b.mean():.3f} ± {auc_b.std():.3f}")
print(f"   Features: {X_b.shape[1]}  ({len(picks_csd_avail)} ch × {N_TIMEPOINTS} pts)")

# ── Pipeline C: xDAWN sobre PICKS_CNV ──────────────────────
print(f"\n{'─'*40}")
print("📊  C — xDAWN (sobre PICKS_CNV) ...")
acc_c, auc_c = cv_xdawn(data_picks, y, MI_ID, N_XDAWN_COMPONENTS, times, T_POINTS)
print(f"   Acc: {acc_c.mean():.1f}% ± {acc_c.std():.1f}%  |  AUC: {auc_c.mean():.3f} ± {auc_c.std():.3f}")
print(f"   Features: {N_XDAWN_COMPONENTS * N_TIMEPOINTS}  ({N_XDAWN_COMPONENTS} comp × {N_TIMEPOINTS} pts)")

# ── Pipeline D: xDAWN sobre todos los canales ──────────────
print(f"\n{'─'*40}")
print(f"📊  D — xDAWN (todos los {len(valid_eeg_ch)} canales EEG) ...")
acc_d, auc_d = cv_xdawn(data_all, y, MI_ID, N_XDAWN_COMPONENTS, times, T_POINTS)
print(f"   Acc: {acc_d.mean():.1f}% ± {acc_d.std():.1f}%  |  AUC: {auc_d.mean():.3f} ± {auc_d.std():.3f}")
print(f"   Features: {N_XDAWN_COMPONENTS * N_TIMEPOINTS}  ({N_XDAWN_COMPONENTS} comp × {N_TIMEPOINTS} pts)")

# ── Resumen ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("🏆  COMPARACIÓN FINAL")
print(f"{'─'*65}")
print(f"   {'Pipeline':<22}  {'Acc':>8}  {'±':>5}  {'AUC':>7}  {'±':>5}")
print(f"   {'─'*52}")
results = {
    "A: Sin filtro":     (acc_a, auc_a),
    "B: CSD":            (acc_b, auc_b),
    "C: xDAWN-PICKS":    (acc_c, auc_c),
    "D: xDAWN-ALL":      (acc_d, auc_d),
}
for name, (accs, aucs) in results.items():
    marker = " ◀" if aucs.mean() == max(r[1].mean() for r in results.values()) else ""
    print(f"   {name:<22}  {accs.mean():>7.1f}%  {accs.std():>5.1f}%  "
          f"{aucs.mean():>7.3f}  {aucs.std():>5.3f}{marker}")
print(f"{'='*65}")


# ============================================================
# ANÁLISIS TEMPORAL (cómo evoluciona la discriminabilidad)
# ============================================================
print(f"\n{'─'*40}")
print("⏱️   Evaluando evolución temporal (acumulativo) ...")

# Para A y B: recalcular con ventana temporal acumulativa
def cv_standard_3d(X_3d, y, times, t_points_all, n_splits=N_CV_FOLDS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs_t, aucs_t = [], []
    for step in range(1, len(t_points_all) + 1):
        X_step = extract_tp_features(X_3d, times, t_points_all[:step])
        fold_accs, fold_aucs = [], []
        for tr, te in cv.split(X_step, y):
            clf = make_lda()
            clf.fit(X_step[tr], y[tr])
            fold_accs.append(np.mean(clf.predict(X_step[te]) == y[te]) * 100)
            try:
                fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(X_step[te])[:, 1]))
            except Exception:
                fold_aucs.append(0.5)
        accs_t.append(np.mean(fold_accs))
        aucs_t.append(np.mean(fold_aucs))
    return np.array(accs_t), np.array(aucs_t)

acc_a_t, auc_a_t = cv_standard_3d(data_picks, y, times, T_POINTS)
acc_b_t, auc_b_t = cv_standard_3d(data_csd,   y, times, T_POINTS)
acc_c_t, auc_c_t = cv_xdawn_temporal(data_picks, y, MI_ID, N_XDAWN_COMPONENTS, times, T_POINTS)
acc_d_t, auc_d_t = cv_xdawn_temporal(data_all,   y, MI_ID, N_XDAWN_COMPONENTS, times, T_POINTS)

print("   ✅  Análisis temporal completo.")


# ============================================================
# VISUALIZACIÓN
# ============================================================
pipeline_names = ["A: Sin filtro", "B: CSD", "C: xDAWN-PICKS", "D: xDAWN-ALL"]
colors = ["#2166ac", "#d6604d", "#1a9641", "#fdae61"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ── Accuracy bar ─────────────────────────────────────────────
ax = axes[0, 0]
accs_m = [results[n][0].mean() for n in pipeline_names]
accs_s = [results[n][0].std()  for n in pipeline_names]
bars = ax.bar(range(4), accs_m, yerr=accs_s, color=colors, edgecolor="white",
              linewidth=0.8, error_kw=dict(elinewidth=1.5, capsize=5))
ax.axhline(50, color="red",  ls="--", lw=1.2, label="Azar (50%)")
ax.axhline(70, color="gray", ls=":",  lw=1.0, label="Objetivo (70%)")
ax.set_xticks(range(4)); ax.set_xticklabels(pipeline_names, rotation=12, ha="right")
ax.set_ylim(30, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy por pipeline", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.4, axis="y")
for bar, val in zip(bars, accs_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── AUC bar ──────────────────────────────────────────────────
ax = axes[0, 1]
aucs_m = [results[n][1].mean() for n in pipeline_names]
aucs_s = [results[n][1].std()  for n in pipeline_names]
bars2 = ax.bar(range(4), aucs_m, yerr=aucs_s, color=colors, edgecolor="white",
               linewidth=0.8, error_kw=dict(elinewidth=1.5, capsize=5))
ax.axhline(0.5, color="red",  ls="--", lw=1.2, label="Azar (0.5)")
ax.axhline(0.7, color="gray", ls=":",  lw=1.0, label="Objetivo (0.7)")
ax.set_xticks(range(4)); ax.set_xticklabels(pipeline_names, rotation=12, ha="right")
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUC (ROC)")
ax.set_title("AUC por pipeline", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.4, axis="y")
for bar, val in zip(bars2, aucs_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── Accuracy temporal ─────────────────────────────────────────
ax = axes[1, 0]
t_plot = T_POINTS
for accs_t, name, color in zip(
        [acc_a_t, acc_b_t, acc_c_t, acc_d_t], pipeline_names, colors):
    ax.plot(t_plot, accs_t, "o-", color=color, linewidth=2.0,
            markersize=6, label=name)
ax.axhline(50, color="red",  ls="--", lw=1.2, label="Azar")
ax.axhline(70, color="gray", ls=":",  lw=1.0, label="Objetivo")
ax.axvline(0,  color="black", ls="--", lw=1.5)
ax.set_xlim(T_START - 0.1, T_END + 0.1)
ax.set_ylim(30, 100)
ax.invert_xaxis()
ax.set_xlabel("Tiempo acumulado (s)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy acumulativa en el tiempo", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.4)

# ── AUC temporal ─────────────────────────────────────────────
ax = axes[1, 1]
for aucs_t, name, color in zip(
        [auc_a_t, auc_b_t, auc_c_t, auc_d_t], pipeline_names, colors):
    ax.plot(t_plot, aucs_t, "o-", color=color, linewidth=2.0,
            markersize=6, label=name)
ax.axhline(0.5, color="red",  ls="--", lw=1.2, label="Azar")
ax.axhline(0.7, color="gray", ls=":",  lw=1.0, label="Objetivo")
ax.axvline(0,   color="black", ls="--", lw=1.5)
ax.set_xlim(T_START - 0.1, T_END + 0.1)
ax.set_ylim(0.3, 1.0)
ax.invert_xaxis()
ax.set_xlabel("Tiempo acumulado (s)")
ax.set_ylabel("AUC (ROC)")
ax.set_title("AUC acumulativa en el tiempo", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.4)

plt.suptitle(
    f"Comparación Filtros Espaciales — {SUBJECT} | {SESSION}\n"
    f"LDA_shrink | {N_CV_FOLDS}-fold CV estratificado | "
    f"n_rest={len(epochs['Rest (100)'])}  n_mi={len(epochs['MI (200)'])}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.show()
