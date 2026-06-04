"""
================================================================================
Generate_Decoder_Validation.py
================================================================================
Valida el pipeline de Generate_Decoder_Total.py en sujetos nuevos no vistos.
Compara dos enfoques de adaptación:

  Enfoque A (Batch): reentrenamiento en lote
    Para cada N en [0,10,20,30,40,50]: entrena con base + primeros N trials
    del sujeto, evalúa AUC en los trials restantes.
    Responde: "¿qué AUC obtengo si tengo N trials de calibración?"

  Enfoque B (Rolling, online): reentrenamiento incremental trial a trial
    Procesa trials en orden cronológico. Predice con el modelo actual, luego
    reentrena con base + trials acumulados cada RETRAIN_EVERY trials.
    MDM: recentering geodésico (α=0.05) después de cada trial.
    Responde: "¿cómo evolucionaría el AUC durante una sesión real?"

Entrenamiento : SUBJ_001, 003, 004, 005, 006  (S001OFFLINE_GLOVE)
Prueba        : SUBJ_011 (S001_OFF), SUBJ_012 (S001OFFLINE)
Clasificadores: LDA, LDA_shrink, LR, SVM + MDM Riemanniano
================================================================================
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bci_runtime_env
import mne

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

try:
    from pyriemann.classification import MDM
    PYRIEMANN_OK = True
except ImportError:
    PYRIEMANN_OK = False
    print("⚠️   pyriemann no instalado — MDM desactivado")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# CONFIGURACIÓN
# ============================================================

XDF_BASE = "/home/lab-admin/Documents/CNVStudy"

TRAIN_SUBJECTS = [
    ("CNV_PILOT_SUBJ_001", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_003", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_004", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_005", "S001OFFLINE_GLOVE"),
    ("CNV_PILOT_SUBJ_006", "S001OFFLINE_GLOVE"),
]

TEST_SUBJECTS = [
    ("CNV_PILOT_SUBJ_011", "S001_OFF"),
    ("CNV_PILOT_SUBJ_012", "S001OFFLINE"),
]

PICKS_LOSO      = ['FC1', 'Cz']
T_START         = -2.5
T_END           =  0.0
N_TIMEPOINTS    =  11
T_POINTS        = np.linspace(T_START, T_END, N_TIMEPOINTS)

BP_LOW, BP_HIGH = 0.1, 1.0
REJECT_UV       = 150.0
COV_REG         = 1e-4
RETRAIN_EVERY   = 10
ALPHA_MDM       = 0.05
ADAPT_STEPS     = [0, 10, 20, 30, 40, 50]
MIN_EVAL        = 15

CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
    "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]

SKL_CLFS = ["LDA", "LDA_shrink", "LR", "SVM"]


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def load_and_preprocess(subject, session):
    xdf_dir   = os.path.join(XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/")
    xdf_files = sorted([os.path.join(xdf_dir, f)
                        for f in os.listdir(xdf_dir)
                        if f.endswith(".xdf") and "_old" not in f])
    if not xdf_files:
        raise FileNotFoundError(f"No XDF en: {xdf_dir}")

    raw_list = []
    for xdf_file in xdf_files:
        eeg_s, marker_s   = load_xdf(xdf_file)
        eeg_data          = np.array(eeg_s["time_series"]).T
        eeg_timestamps    = np.array(eeg_s["time_stamps"])
        channel_names     = get_channel_names_from_xdf(eeg_s)

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

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

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

    ch_names_eeg = epochs.copy().pick_types(eeg=True).ch_names
    picks_avail  = [ch for ch in PICKS_LOSO if ch in ch_names_eeg]
    drop_idx = []
    if picks_avail:
        pick_idx  = [ch_names_eeg.index(ch) for ch in picks_avail]
        data_cnv  = epochs.get_data()[:, pick_idx, :] * 1e6
        pp        = data_cnv.max(axis=2) - data_cnv.min(axis=2)
        drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
        drop_idx  = np.where(drop_mask)[0].tolist()
        epochs.drop(drop_idx, reason="MANUAL_REJECT")

    print(f"      {subject.split('_')[-1]}: "
          f"Rest={len(epochs['Rest (100)'])}  "
          f"MI={len(epochs['MI (200)'])}  "
          f"rechazados={len(drop_idx)}")
    return epochs, event_dict


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def get_eeg_data(epochs_obj, picks):
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0
    except Exception:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    return data, ch_idx, scale


def extract_features(epochs_obj, picks, t_points, step=None):
    times = epochs_obj.times
    pts   = t_points[:step] if step is not None else t_points
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
    data, ch_idx, scale = get_eeg_data(epochs_obj, picks)
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def extract_raw_data(epochs_obj, picks, tmin, tmax):
    times  = epochs_obj.times
    t_mask = (times >= tmin) & (times <= tmax)
    data, ch_idx, scale = get_eeg_data(epochs_obj, picks)
    y = epochs_obj.events[:, -1]
    return data[:, ch_idx, :][:, :, t_mask] * scale, y


# ============================================================
# COVARIANZAS RIEMANNIANAS
# ============================================================

def compute_cov_trace_norm(data_3d):
    n, n_ch, n_t = data_3d.shape
    covs = np.zeros((n, n_ch, n_ch))
    for i in range(n):
        X  = data_3d[i].T
        C  = X.T @ X
        tr = np.trace(C)
        C  = C / tr if tr > 0 else C
        C += COV_REG * np.eye(n_ch)
        covs[i] = C
    return covs


def build_template_covs(data_3d, template):
    tmpl_rep = np.tile(template[np.newaxis], (data_3d.shape[0], 1, 1))
    extended = np.concatenate([data_3d, tmpl_rep], axis=1)
    return compute_cov_trace_norm(extended)


def geodesic_riemann(A, B, alpha):
    try:
        A_sqrt  = np.linalg.cholesky(A)
        A_isqrt = np.linalg.inv(A_sqrt)
        M       = A_isqrt @ B @ A_isqrt.T
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)
        M_alpha = eigvecs @ np.diag(eigvals ** alpha) @ eigvecs.T
        return A_sqrt @ M_alpha @ A_sqrt.T
    except Exception:
        return (1 - alpha) * A + alpha * B


# ============================================================
# CLASIFICADORES SKLEARN
# ============================================================

def make_clf(name):
    if name == "LDA":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis())])
    elif name == "LDA_shrink":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis(
                             solver="lsqr", shrinkage="auto"))])
    elif name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(
                             C=1.0, max_iter=1000, random_state=42))])
    elif name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(
                             SVC(kernel="linear", C=1.0,
                                 probability=False, random_state=42),
                             cv=3, method="sigmoid"))])
    raise ValueError(f"Desconocido: {name}")


# ============================================================
# CARGA DE DATOS
# ============================================================

print(f"\n{'='*65}")
print("🔬  VALIDACIÓN CROSS-SUBJECT — CNV BCI")
print(f"{'='*65}")
mne.set_log_level("WARNING")

print(f"\n{'─'*45}")
print("📂  Cargando sujetos de entrenamiento ...")
print(f"{'─'*45}")

all_epochs_tr = {}
for subj, sess in TRAIN_SUBJECTS:
    epochs, event_dict = load_and_preprocess(subj, sess)
    all_epochs_tr[subj] = epochs

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]

# Features M1 y M2 de entrenamiento
X_tr_m1_list, y_tr_list, raw_tr_list = [], [], []
X_tr_m2_steps   = [[] for _ in range(N_TIMEPOINTS)]
raw_tr_m2_steps  = [[] for _ in range(N_TIMEPOINTS)]

for subj in all_epochs_tr:
    ep = all_epochs_tr[subj]
    X1, y1 = extract_features(ep, PICKS_LOSO, T_POINTS)
    X_tr_m1_list.append(X1); y_tr_list.append(y1)
    raw_ep, _ = extract_raw_data(ep, PICKS_LOSO, T_START, T_END)
    raw_tr_list.append(raw_ep)
    for step in range(N_TIMEPOINTS):
        Xs, _ = extract_features(ep, PICKS_LOSO, T_POINTS, step=step + 1)
        X_tr_m2_steps[step].append(Xs)
        rs, _ = extract_raw_data(ep, PICKS_LOSO, T_POINTS[0], T_POINTS[step])
        raw_tr_m2_steps[step].append(rs)

X_tr_m1  = np.vstack(X_tr_m1_list)
y_tr     = np.concatenate(y_tr_list)
raw_tr   = np.vstack(raw_tr_list)
X_tr_m2  = [np.vstack(X_tr_m2_steps[s])   for s in range(N_TIMEPOINTS)]
raw_tr_m2 = [np.vstack(raw_tr_m2_steps[s]) for s in range(N_TIMEPOINTS)]

n_rest_tr = np.sum(y_tr == REST_ID)
n_mi_tr   = np.sum(y_tr == MI_ID)
print(f"\n   Total entrenamiento: {len(y_tr)} trials  (REST={n_rest_tr}, MI={n_mi_tr})")

print(f"\n{'─'*45}")
print("📂  Cargando sujetos de prueba ...")
print(f"{'─'*45}")

all_epochs_te = {}
for subj, sess in TEST_SUBJECTS:
    epochs, _ = load_and_preprocess(subj, sess)
    all_epochs_te[subj] = epochs


# ============================================================
# ENTRENAMIENTO MODELOS BASE
# ============================================================

# Sklearn M1 y M2
skl_m1_base = {c: make_clf(c) for c in SKL_CLFS}
for c in SKL_CLFS:
    skl_m1_base[c].fit(X_tr_m1, y_tr)

skl_m2_base = [{c: make_clf(c) for c in SKL_CLFS} for _ in range(N_TIMEPOINTS)]
for step in range(N_TIMEPOINTS):
    for c in SKL_CLFS:
        skl_m2_base[step][c].fit(X_tr_m2[step], y_tr)

# MDM M1
template_base  = raw_tr.mean(axis=0)
covs_tr_m1     = build_template_covs(raw_tr, template_base)
if PYRIEMANN_OK:
    mdm_m1_base = MDM(metric="riemann")
    mdm_m1_base.fit(covs_tr_m1, y_tr)
    centers_m1_base = {
        label: mdm_m1_base.covmeans_[i].copy()
        for i, label in enumerate(mdm_m1_base.classes_)
    }

# MDM M2
if PYRIEMANN_OK:
    mdm_m2_base     = []
    centers_m2_base = []
    for step in range(N_TIMEPOINTS):
        tmpl = raw_tr_m2[step].mean(axis=0)
        covs = build_template_covs(raw_tr_m2[step], tmpl)
        m    = MDM(metric="riemann")
        m.fit(covs, y_tr)
        mdm_m2_base.append(m)
        centers_m2_base.append({
            label: m.covmeans_[i].copy()
            for i, label in enumerate(m.classes_)
        })

print(f"\n   Modelos base entrenados — sklearn {SKL_CLFS}"
      + (f" + MDM Riemanniano" if PYRIEMANN_OK else ""))


# ============================================================
# ENFOQUE A — BATCH SNAPSHOT (M2, mejor paso por clf)
# ============================================================

def eval_batch(subj):
    """Para cada n_adapt: entrenar con base + primeros n_adapt trials,
    evaluar AUC en los trials restantes. Usa M2 (mejor paso)."""
    ep          = all_epochs_te[subj]
    X_te_m1, y_te = extract_features(ep, PICKS_LOSO, T_POINTS)
    raw_te, _   = extract_raw_data(ep, PICKS_LOSO, T_START, T_END)
    n_trials    = len(y_te)

    X_te_m2   = [extract_features(ep, PICKS_LOSO, T_POINTS, step=s+1)[0]
                 for s in range(N_TIMEPOINTS)]
    raw_te_m2 = [extract_raw_data(ep, PICKS_LOSO, T_POINTS[0], T_POINTS[s])[0]
                 for s in range(N_TIMEPOINTS)]

    results = {c: {"fix": {}, "adp": {}} for c in SKL_CLFS}
    if PYRIEMANN_OK:
        results["MDM"] = {"fix": {}, "adp": {}}

    for n_adapt in ADAPT_STEPS:
        n_eval = n_trials - n_adapt
        if n_eval < MIN_EVAL:
            break
        y_eval = y_te[n_adapt:]
        key    = str(n_adapt)

        # ── Sklearn M2 ────────────────────────────────────────
        for c in SKL_CLFS:
            aucs_fix, aucs_adp = [], []
            for step in range(N_TIMEPOINTS):
                X_eval = X_te_m2[step][n_adapt:]

                p = skl_m2_base[step][c].predict_proba(X_eval)[:, 1]
                aucs_fix.append(roc_auc_score(y_eval, p))

                if n_adapt == 0:
                    clf_adp = skl_m2_base[step][c]
                else:
                    Xc = np.vstack([X_tr_m2[step], X_te_m2[step][:n_adapt]])
                    yc = np.concatenate([y_tr, y_te[:n_adapt]])
                    clf_adp = make_clf(c)
                    clf_adp.fit(Xc, yc)
                p = clf_adp.predict_proba(X_eval)[:, 1]
                aucs_adp.append(roc_auc_score(y_eval, p))

            results[c]["fix"][key] = round(max(aucs_fix), 3)
            results[c]["adp"][key] = round(max(aucs_adp), 3)

        if not PYRIEMANN_OK:
            continue

        # ── MDM M2 fijo ───────────────────────────────────────
        aucs_fix, aucs_adp = [], []
        for step in range(N_TIMEPOINTS):
            raw_eval = raw_te_m2[step][n_adapt:]
            tmpl_s   = raw_tr_m2[step].mean(axis=0)
            covs_s   = build_template_covs(raw_eval, tmpl_s)
            mi_col   = list(mdm_m2_base[step].classes_).index(MI_ID)
            p = mdm_m2_base[step].predict_proba(covs_s)[:, mi_col]
            aucs_fix.append(roc_auc_score(y_eval, p))

            # Adaptativo: recentering con n_adapt trials
            ctr = copy.deepcopy(centers_m2_base[step])
            for i in range(n_adapt):
                cov_i = build_template_covs(raw_te_m2[step][i:i+1], tmpl_s)[0]
                for lbl in ctr:
                    ctr[lbl] = geodesic_riemann(ctr[lbl], cov_i, ALPHA_MDM)
            scores_adp = []
            for i in range(len(raw_eval)):
                cov_i = build_template_covs(raw_eval[i:i+1], tmpl_s)[0]
                d_r = np.linalg.norm(cov_i - ctr[REST_ID], 'fro')
                d_m = np.linalg.norm(cov_i - ctr[MI_ID],   'fro')
                scores_adp.append(-(d_m / (d_r + d_m + 1e-10)))
            aucs_adp.append(roc_auc_score(y_eval, scores_adp))

        results["MDM"]["fix"][key] = round(max(aucs_fix), 3)
        results["MDM"]["adp"][key] = round(max(aucs_adp), 3)

    return results, y_te


# ============================================================
# ENFOQUE B — ROLLING ONLINE (M1, trial a trial)
# ============================================================

def eval_rolling(subj):
    """Procesa trials en orden cronológico.
    Sklearn: predice con modelo actual, reentrena cada RETRAIN_EVERY trials.
    MDM: recentering geodésico (α=ALPHA_MDM) después de cada trial.
    Devuelve AUC acumulativo por trial para cada clasificador."""
    ep           = all_epochs_te[subj]
    X_te, y_te   = extract_features(ep, PICKS_LOSO, T_POINTS)
    raw_te, _    = extract_raw_data(ep, PICKS_LOSO, T_START, T_END)
    n_trials     = len(y_te)

    # Modelos adaptativos — copia para no modificar los base
    skl_adp = {c: make_clf(c) for c in SKL_CLFS}
    for c in SKL_CLFS:
        skl_adp[c].fit(X_tr_m1, y_tr)

    centers_adp = copy.deepcopy(centers_m1_base) if PYRIEMANN_OK else None

    # Acumuladores
    raw_scores  = {c: [] for c in SKL_CLFS}
    mdm_scores  = []
    X_accum     = []
    y_accum     = []

    for i in range(n_trials):
        x_i   = X_te[i]
        raw_i = raw_te[i:i+1]

        # ── Sklearn: predice con modelo actual ────────────────
        for c in SKL_CLFS:
            try:
                p = skl_adp[c].predict_proba(x_i.reshape(1, -1))[0, 1]
            except Exception:
                p = 0.5
            raw_scores[c].append(p)

        # ── MDM: predice con centroides actuales ──────────────
        if PYRIEMANN_OK:
            cov_i = build_template_covs(raw_i, template_base)[0]
            d_r   = np.linalg.norm(cov_i - centers_adp[REST_ID], 'fro')
            d_m   = np.linalg.norm(cov_i - centers_adp[MI_ID],   'fro')
            mdm_scores.append(-(d_m / (d_r + d_m + 1e-10)))
            # Actualizar centroides DESPUÉS de predecir
            for lbl in centers_adp:
                centers_adp[lbl] = geodesic_riemann(
                    centers_adp[lbl], cov_i, ALPHA_MDM)

        # Acumular trial
        X_accum.append(x_i)
        y_accum.append(y_te[i])

        # ── Sklearn: reentrenar cada RETRAIN_EVERY trials ─────
        if len(X_accum) % RETRAIN_EVERY == 0:
            Xc = np.vstack([X_tr_m1, np.array(X_accum)])
            yc = np.concatenate([y_tr, np.array(y_accum)])
            for c in SKL_CLFS:
                try:
                    clf_new = make_clf(c)
                    clf_new.fit(Xc, yc)
                    skl_adp[c] = clf_new
                except Exception:
                    pass

    # AUC acumulativo por trial (None si < MIN_EVAL)
    rolling = {c: [] for c in SKL_CLFS}
    if PYRIEMANN_OK:
        rolling["MDM"] = []

    for i in range(n_trials):
        labels_so_far = y_te[:i + 1]
        for c in SKL_CLFS:
            if i + 1 < MIN_EVAL:
                rolling[c].append(None)
            else:
                try:
                    rolling[c].append(
                        round(roc_auc_score(labels_so_far, raw_scores[c]), 3))
                except Exception:
                    rolling[c].append(None)
        if PYRIEMANN_OK:
            if i + 1 < MIN_EVAL:
                rolling["MDM"].append(None)
            else:
                try:
                    rolling["MDM"].append(
                        round(roc_auc_score(labels_so_far, mdm_scores), 3))
                except Exception:
                    rolling["MDM"].append(None)

    return rolling, y_te


# ============================================================
# CORRER EVALUACIONES
# ============================================================

print(f"\n{'='*65}")
print("🧪  Evaluando sujetos de prueba ...")
print(f"{'='*65}")

batch_results   = {}
rolling_results = {}

for subj, _ in TEST_SUBJECTS:
    sname = subj.split("_")[-1]
    n_te  = len(all_epochs_te[subj].events)
    print(f"\n{'─'*45}")
    print(f"   SUBJ_{sname}  ({n_te} trials)")
    print(f"{'─'*45}")

    print("   → Enfoque A (batch) ...")
    b_res, y_te = eval_batch(subj)
    batch_results[subj] = (b_res, y_te)

    print("   → Enfoque B (rolling online) ...")
    r_res, _    = eval_rolling(subj)
    rolling_results[subj] = (r_res, y_te)


# ============================================================
# RESUMEN NUMÉRICO
# ============================================================

print(f"\n{'='*65}")
print("📊  RESUMEN COMPARATIVO")
print(f"{'='*65}")

for subj, _ in TEST_SUBJECTS:
    sname = subj.split("_")[-1]
    b_res, y_te = batch_results[subj]
    r_res, _    = rolling_results[subj]
    n_trials    = len(y_te)

    print(f"\n  SUBJ_{sname}  ({n_trials} trials)")
    print(f"  {'─'*60}")
    print(f"  {'Clf':<12}  "
          f"{'A-fix n=0':>9} {'A-fix best':>10} {'A-adp best':>10}  "
          f"{'B-final AUC':>12}")
    print(f"  {'─'*60}")

    clfs = SKL_CLFS + (["MDM"] if PYRIEMANN_OK else [])
    for c in clfs:
        if c not in b_res:
            continue
        steps = list(b_res[c]["fix"].keys())
        a_fix_0    = b_res[c]["fix"].get("0", float("nan"))
        a_fix_best = max(b_res[c]["fix"].values())
        a_adp_best = max(b_res[c]["adp"].values())

        # Rolling: último AUC válido
        r_vals = [v for v in r_res[c] if v is not None]
        b_final = r_vals[-1] if r_vals else float("nan")

        # Retraining markers for rolling
        retrain_marker = ""
        if len(r_vals) > 0:
            last_retrain = (len(r_vals) // RETRAIN_EVERY) * RETRAIN_EVERY
            retrain_marker = f"(último retrain: trial {last_retrain})"

        print(f"  {c:<12}  "
              f"{a_fix_0:>9.3f} {a_fix_best:>10.3f} {a_adp_best:>10.3f}  "
              f"{b_final:>12.3f}")

    print(f"\n  Referencia LOSO within-group: ~0.635–0.651 (adaptado)")


# ============================================================
# FIGURA — 2 filas × 2 columnas
# Fila 0: Enfoque A (batch snapshot)
# Fila 1: Enfoque B (rolling online)
# ============================================================

COLORS = {
    "LDA":        "#1f77b4",
    "LDA_shrink": "#2196F3",
    "LR":         "#ff7f0e",
    "SVM":        "#2ca02c",
    "MDM":        "#d62728",
}

n_subj  = len(TEST_SUBJECTS)
fig, axes = plt.subplots(2, n_subj, figsize=(7 * n_subj, 11))
if n_subj == 1:
    axes = axes.reshape(2, 1)

clfs_plot = SKL_CLFS + (["MDM"] if PYRIEMANN_OK else [])

for col, (subj, _) in enumerate(TEST_SUBJECTS):
    sname   = subj.split("_")[-1]
    b_res, y_te_b = batch_results[subj]
    r_res, y_te_r = rolling_results[subj]
    n_trials = len(y_te_b)

    # ── Fila 0: Enfoque A (Batch) ─────────────────────────────
    ax = axes[0, col]
    steps_str = list(next(iter(b_res.values()))["fix"].keys())
    steps_int = [int(s) for s in steps_str]

    for c in clfs_plot:
        if c not in b_res:
            continue
        clr = COLORS[c]
        vals_fix = [b_res[c]["fix"][s] for s in steps_str]
        vals_adp = [b_res[c]["adp"][s] for s in steps_str]
        ax.plot(steps_int, vals_fix, color=clr, ls="--",
                marker="o", ms=5, lw=1.6, alpha=0.7,
                label=f"{c} fijo")
        ax.plot(steps_int, vals_adp, color=clr, ls="-",
                marker="^", ms=6, lw=2.0,
                label=f"{c} adapt")

    ax.axhline(0.5,   color="gray",  ls=":", lw=1.2, label="Azar (0.5)")
    ax.axhline(0.635, color="black", ls=":", lw=1.0, label="Ref LOSO (0.635)")
    ax.set_xlabel("Trials de calibración usados (n_adapt)", fontsize=10)
    ax.set_ylabel("AUC (ROC) — evaluado en trials restantes", fontsize=9)
    ax.set_title(f"SUBJ_{sname} — Enfoque A: Batch snapshot (M2)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, ls=":", alpha=0.4)

    # ── Fila 1: Enfoque B (Rolling) ───────────────────────────
    ax = axes[1, col]
    trial_axis = list(range(1, n_trials + 1))

    for c in clfs_plot:
        if c not in r_res:
            continue
        clr  = COLORS[c]
        vals = r_res[c]
        # Graficar solo donde hay AUC válido (>= MIN_EVAL trials)
        xs = [t for t, v in zip(trial_axis, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color=clr, lw=2.0, label=c)

    # Marcar los puntos de reentrenamiento
    for rt in range(RETRAIN_EVERY, n_trials + 1, RETRAIN_EVERY):
        ax.axvline(rt, color="silver", ls=":", lw=0.9)

    ax.axhline(0.5,   color="gray",  ls=":", lw=1.2, label="Azar (0.5)")
    ax.axhline(0.635, color="black", ls=":", lw=1.0, label="Ref LOSO (0.635)")
    ax.set_xlabel(f"Trial (líneas grises = reentrenamiento cada {RETRAIN_EVERY})",
                  fontsize=10)
    ax.set_ylabel("AUC acumulativo", fontsize=9)
    ax.set_title(f"SUBJ_{sname} — Enfoque B: Rolling online (M1)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(0, n_trials + 2)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", alpha=0.4)

plt.suptitle(
    "Validación cross-subject — CNV BCI\n"
    "Modelo base: SUBJ_001/003/004/005/006  →  Prueba: SUBJ_011, SUBJ_012\n"
    "A: batch snapshot (M2, mejor paso) | B: rolling online (M1, retrain cada 10 trials)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figures_Validation.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n📊  Figura guardada: {fig_path}")
