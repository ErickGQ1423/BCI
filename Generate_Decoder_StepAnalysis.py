"""
AUC por paso temporal — LDA_shrink vs MDM
Carga el modelo base ya entrenado y evalúa cada uno de los 11 pasos
en SUBJ_011 (out-of-sample) y SUBJ_012 (in-sample, referencia).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = (
    "/home/lab-admin/Documents/CurrentStudy/"
    "sub-CNV_PILOT_SUBJ_012/models/sub-CNV_PILOT_SUBJ_012_model.pkl"
)

XDF_BASE = "/home/lab-admin/Documents/CNVStudy"

TEST_SUBJECTS = [
    ("CNV_PILOT_SUBJ_011", "S001_OFF",   "out-of-sample"),
    ("CNV_PILOT_SUBJ_012", "S001OFFLINE","in-sample (calibración)"),
]

CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
    "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]
COV_REG          = 1e-4


# ============================================================
# CARGAR MODELO
# ============================================================

with open(MODEL_PATH, "rb") as f:
    pkg = pickle.load(f)

PICKS        = pkg["picks"]
T_POINTS     = pkg["t_points"]
T_START      = pkg["t_start"]
N_TIMEPOINTS = pkg["n_timepoints"]
REST_ID      = pkg["REST_ID"]
MI_ID        = pkg["MI_ID"]
skl_models   = pkg["skl_models"]
mdm_models   = pkg["mdm_models"]
mdm_templates= pkg["mdm_templates"]
MDM_OK       = pkg["mdm_available"] and len(mdm_models) == N_TIMEPOINTS

print(f"\n{'='*60}")
print("📊  ANÁLISIS POR PASO — LDA_shrink vs MDM")
print(f"{'='*60}")
print(f"   Modelo  : {pkg['model_type']}")
print(f"   Canales : {PICKS}")
print(f"   Pasos   : {N_TIMEPOINTS}  ({T_START}→{T_POINTS[-1]}s, Δ=0.25s)")
print(f"   MDM     : {'disponible' if MDM_OK else 'no disponible'}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def load_subject(subject, session):
    xdf_dir   = os.path.join(XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/")
    xdf_files = sorted([os.path.join(xdf_dir, f)
                        for f in os.listdir(xdf_dir)
                        if f.endswith(".xdf") and "_old" not in f])
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

        valid_ch  = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
        valid_idx = [channel_names.index(ch) for ch in valid_ch]
        eeg_data  = eeg_data[valid_idx, :] / 1e6

        info    = mne.create_info(ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data, info, verbose=False)
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
    raw.filter(l_freq=0.1, h_freq=1.0, method="iir",
               iir_params=dict(order=2, ftype="butter"),
               phase="forward", picks="eeg", verbose=False)

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {"Rest (100)": event_id_map["100"], "MI (200)": event_id_map["200"]}
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=5.0, baseline=(-3.0, -2.0),
        reject=None, flat=None, preload=True, detrend=None, verbose=False,
    )

    ch_names_eeg = epochs.copy().pick_types(eeg=True).ch_names
    picks_avail  = [ch for ch in PICKS if ch in ch_names_eeg]
    if picks_avail:
        pidx = [ch_names_eeg.index(ch) for ch in picks_avail]
        pp   = (epochs.get_data()[:, pidx, :] * 1e6).max(axis=2) - \
               (epochs.get_data()[:, pidx, :] * 1e6).min(axis=2)
        drop = np.where((pp.max(axis=1) > 150) | (pp.max(axis=1) < 1))[0].tolist()
        epochs.drop(drop, reason="MANUAL_REJECT")

    return epochs


def extract_features(epochs, picks, t_points, step):
    times  = epochs.times
    t_idx  = [np.argmin(np.abs(times - t)) for t in t_points[:step]]
    ch     = epochs.copy().pick_types(eeg=True).ch_names
    ch_idx = [ch.index(p) for p in picks if p in ch]
    data   = epochs.get_data() * 1e6
    X = np.hstack([data[:, ci, :][:, t_idx] for ci in ch_idx])
    y = epochs.events[:, -1]
    return X, y


def extract_raw(epochs, picks, t_start, t_end):
    times  = epochs.times
    mask   = (times >= t_start) & (times <= t_end)
    ch     = epochs.copy().pick_types(eeg=True).ch_names
    ch_idx = [ch.index(p) for p in picks if p in ch]
    data   = epochs.get_data() * 1e6
    y      = epochs.events[:, -1]
    return data[:, ch_idx, :][:, :, mask], y


def build_cov(data_3d, template):
    n, n_ch, _ = data_3d.shape
    tmpl_rep   = np.tile(template[np.newaxis], (n, 1, 1))
    ext        = np.concatenate([data_3d, tmpl_rep], axis=1)
    covs       = np.zeros((n, ext.shape[1], ext.shape[1]))
    for i in range(n):
        X  = ext[i].T
        C  = X.T @ X
        tr = np.trace(C)
        C  = C / tr if tr > 0 else C
        C += COV_REG * np.eye(C.shape[0])
        covs[i] = C
    return covs


# ============================================================
# EVALUACIÓN POR PASO
# ============================================================

mne.set_log_level("WARNING")
results = {}

for subj, sess, label in TEST_SUBJECTS:
    sname = subj.split("_")[-1]
    print(f"\n{'─'*45}")
    print(f"   SUBJ_{sname}  [{label}]")
    print(f"{'─'*45}")

    epochs = load_subject(subj, sess)
    n      = len(epochs.events)
    y      = epochs.events[:, -1]
    print(f"   {n} trials  (REST={np.sum(y==REST_ID)}, MI={np.sum(y==MI_ID)})")

    auc_lda = []
    auc_mdm = []

    print(f"\n   {'Paso':>5} {'t(s)':>7}  {'AUC LDA':>9}  {'AUC MDM':>9}  {'Δ':>7}")
    print(f"   {'-'*45}")

    for step in range(1, N_TIMEPOINTS + 1):
        # LDA_shrink
        X, _ = extract_features(epochs, PICKS, T_POINTS, step)
        p_lda = skl_models[step-1].predict_proba(X)[:, 1]
        a_lda = round(roc_auc_score(y, p_lda), 3)
        auc_lda.append(a_lda)

        # MDM
        a_mdm = float("nan")
        if MDM_OK:
            raw, _ = extract_raw(epochs, PICKS, T_START, T_POINTS[step-1])
            tmpl   = mdm_templates[step-1]
            covs   = build_cov(raw, tmpl)
            mi_col = list(mdm_models[step-1].classes_).index(MI_ID)
            try:
                p_mdm = mdm_models[step-1].predict_proba(covs)[:, mi_col]
                a_mdm = round(roc_auc_score(y, p_mdm), 3)
            except Exception:
                a_mdm = float("nan")
        auc_mdm.append(a_mdm)

        delta = a_mdm - a_lda if not np.isnan(a_mdm) else float("nan")
        winner = "MDM↑" if delta > 0.01 else ("LDA↑" if delta < -0.01 else "≈")
        print(f"   {step:>5d} {T_POINTS[step-1]:>7.2f}  "
              f"{a_lda:>9.3f}  {a_mdm:>9.3f}  "
              f"{delta:>+6.3f}  {winner}")

    results[sname] = {"lda": auc_lda, "mdm": auc_mdm, "label": label}

    best_lda = max(auc_lda)
    best_mdm = max([v for v in auc_mdm if not np.isnan(v)]) if MDM_OK else float("nan")
    print(f"\n   Mejor LDA: {best_lda:.3f} (paso {auc_lda.index(best_lda)+1})")
    if MDM_OK:
        print(f"   Mejor MDM: {best_mdm:.3f} "
              f"(paso {[v for v in auc_mdm].index(best_mdm)+1})")


# ============================================================
# FIGURA
# ============================================================

steps  = list(range(1, N_TIMEPOINTS + 1))
t_axis = [T_POINTS[s-1] for s in steps]

fig, axes = plt.subplots(1, len(TEST_SUBJECTS), figsize=(7*len(TEST_SUBJECTS), 5),
                         sharey=True)
if len(TEST_SUBJECTS) == 1:
    axes = [axes]

for ax, (subj, _, label) in zip(axes, TEST_SUBJECTS):
    sname = subj.split("_")[-1]
    r = results[sname]

    ax.plot(t_axis, r["lda"], "o-", color="#1f77b4", lw=2.2, ms=7,
            label="LDA_shrink")
    if MDM_OK:
        mdm_vals = [v if not np.isnan(v) else None for v in r["mdm"]]
        ax.plot(t_axis, mdm_vals, "s-", color="#d62728", lw=2.2, ms=7,
                label="MDM Riemanniano")

    ax.axhline(0.5,   color="gray",  ls=":", lw=1.2, label="Azar (0.5)")
    ax.axhline(0.635, color="black", ls=":", lw=1.0, label="Ref LOSO (0.635)")

    # Shading — zona donde la CNV es más activa
    ax.axvspan(-1.0, 0.0, alpha=0.08, color="green", label="CNV activa")

    ax.set_xlabel("Tiempo del paso (s desde trigger)", fontsize=11)
    ax.set_ylabel("AUC (ROC)", fontsize=11)
    ax.set_title(f"SUBJ_{sname} — {label}", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xticks(t_axis)
    ax.set_xticklabels([f"{t:.2f}" for t in t_axis], rotation=45, fontsize=8)

plt.suptitle(
    "AUC por paso temporal — LDA_shrink vs MDM Riemanniano\n"
    f"Modelo base: SUBJ_001/003/004/005/006 + SUBJ_012  |  "
    f"Canales: {PICKS}  |  M2 acumulativo",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figures_StepAnalysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n📊  Figura guardada: {fig_path}")
print(f"{'='*60}")
