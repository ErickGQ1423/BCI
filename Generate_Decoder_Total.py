# """
# ================================================================================
# CNV BCI — LEAVE-ONE-SUBJECT-OUT (LOSO)
# ================================================================================
# Script independiente de transfer learning cross-subject.
# Evalúa qué tan bien generaliza el modelo a sujetos no vistos.

# Sujetos válidos: CNV_PILOT_SUBJ_001, 003, 004, 005
# Sesión         : S001OFFLINE_GLOVE
# Canales fijos  : FC5 + Cz + CP1  (más consistentes entre sujetos)

# Esquema:
#   Fold 1: entrenar en 003+004+005, evaluar en 001
#   Fold 2: entrenar en 001+004+005, evaluar en 003
#   Fold 3: entrenar en 001+003+005, evaluar en 004
#   Fold 4: entrenar en 001+003+004, evaluar en 005

# Modelos evaluados:
#   M1 — Estático     : 33 features (11 pts × 3 ch)
#   M2 — Acumulativo  : 11 pasos, features crecientes
#   M3 — Deslizante   : ventana 2s, paso 0.05s, 11 ventanas

# Clasificadores: LDA, LDA_shrink, SVM, LR, RF, DT, KNN, MLP, MDM_Riemann
# ================================================================================
# """

# import os
# import json
# import datetime
# import numpy as np
# import mne
# from scipy import stats

# import config
# from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

# try:
#     from pyriemann.classification import MDM
#     PYRIEMANN_OK = True
#     print("✅  pyriemann disponible")
# except ImportError:
#     PYRIEMANN_OK = False
#     print("⚠️   pyriemann no instalado — MDM desactivado")

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.calibration import CalibratedClassifierCV


# # ============================================================
# # CONFIGURACIÓN GLOBAL
# # ============================================================

# SUBJECTS = [
#     ("CNV_PILOT_SUBJ_001", "S001OFFLINE_GLOVE"),
#     ("CNV_PILOT_SUBJ_003", "S001OFFLINE_GLOVE"),
#     ("CNV_PILOT_SUBJ_004", "S001OFFLINE_GLOVE"),
#     ("CNV_PILOT_SUBJ_005", "S001OFFLINE_GLOVE"),
#     ("CNV_PILOT_SUBJ_006", "S001OFFLINE_GLOVE"),
# ]

# XDF_BASE     = "/home/lab-admin/Documents/CNVStudy"
# SAVE_DIR     = "/home/lab-admin/Documents/CNVStudy/logs_2"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # Canales fijos para transfer learning
# PICKS_LOSO   = ['FC5', 'FC1', 'Cz', 'CP1', 'Fz']

# # Preprocesamiento
# CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
# RENAME_DICT = {
#     "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
#     "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
#     "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
# }
# NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
# TARGET_MARKERS   = [100, 200]
# REJECT_UV        = 150.0
# BP_LOW, BP_HIGH  = 0.1, 1.0

# # T_POINTS — Opción B (Millan): −2.5 → 0.0 s, 11 puntos
# T_START      = -2.5
# T_END        =  0.0
# N_TIMEPOINTS =  11
# T_POINTS     = np.linspace(T_START, T_END, N_TIMEPOINTS)
# N_CH         = len(PICKS_LOSO)
# N_FEATURES   = N_TIMEPOINTS * N_CH   # 33

# # Ventanas deslizantes
# WIN_SIZE  = 2.0
# WIN_STEP  = 0.05
# WIN_PTS   = 9
# win_starts = np.arange(T_START, T_END - WIN_SIZE + 1e-9, WIN_STEP)
# WINDOWS    = list(zip(np.round(win_starts, 4),
#                       np.round(win_starts + WIN_SIZE, 4)))

# # Regularización MDM
# COV_REG = 1e-4

# print(f"\n{'='*68}")
# print("🔄  CNV BCI — LEAVE-ONE-SUBJECT-OUT")
# print(f"{'='*68}")
# print(f"   Sujetos  : {[s[0].split('_')[-1] for s in SUBJECTS]}")
# print(f"   Canales  : {PICKS_LOSO}")
# print(f"   Features : {N_FEATURES} ({N_CH} ch × {N_TIMEPOINTS} pts)")
# print(f"   Ventanas : {len(WINDOWS)} (tamaño={WIN_SIZE}s, paso={WIN_STEP}s)")


# # ============================================================
# # FUNCIONES DE PREPROCESAMIENTO
# # ============================================================

# def load_and_preprocess(subject, session):
#     """Carga, filtra y epocha datos de un sujeto."""
#     xdf_dir = os.path.join(
#         XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/"
#     )
#     xdf_files = sorted(
#         [os.path.join(xdf_dir, f)
#          for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
#     )
#     if not xdf_files:
#         raise FileNotFoundError(f"No XDF en: {xdf_dir}")

#     raw_list = []
#     for xdf_file in xdf_files:
#         eeg_s, marker_s = load_xdf(xdf_file)

#         eeg_data       = np.array(eeg_s["time_series"]).T
#         eeg_timestamps = np.array(eeg_s["time_stamps"])
#         channel_names  = get_channel_names_from_xdf(eeg_s)

#         marker_data       = np.array([int(v[0]) for v in marker_s["time_series"]])
#         marker_timestamps = np.array(marker_s["time_stamps"])
#         keep              = np.isin(marker_data, TARGET_MARKERS)
#         marker_data       = marker_data[keep]
#         marker_timestamps = marker_timestamps[keep]

#         valid_ch        = [ch for ch in channel_names
#                            if ch not in NON_EEG_CHANNELS]
#         valid_idx       = [channel_names.index(ch) for ch in valid_ch]
#         eeg_data_subset = eeg_data[valid_idx, :] / 1e6

#         info    = mne.create_info(
#             ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
#         raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

#         if "AUX7" in raw_tmp.ch_names:
#             raw_tmp.set_channel_types({"AUX7": "emg"})

#         existing_renames = {k: v for k, v in RENAME_DICT.items()
#                             if k in raw_tmp.ch_names}
#         if existing_renames:
#             raw_tmp.rename_channels(existing_renames)

#         raw_tmp.set_montage(
#             mne.channels.make_standard_montage("standard_1020"),
#             on_missing="warn")

#         drop_targets = [ch for ch in CHANNELS_TO_DROP
#                         if ch in raw_tmp.ch_names]
#         if drop_targets:
#             raw_tmp.drop_channels(drop_targets)

#         t0    = eeg_timestamps[0]
#         annot = mne.Annotations(
#             onset       = marker_timestamps - t0,
#             duration    = np.zeros(len(marker_data)),
#             description = [str(m) for m in marker_data],
#             orig_time   = None,
#         )
#         raw_tmp.set_annotations(annot)
#         raw_list.append(raw_tmp)

#     raw = mne.concatenate_raws(raw_list)

#     # Preprocesamiento
#     raw.set_eeg_reference("average", projection=False, verbose=False)
#     raw.notch_filter(freqs=[60.0], picks="eeg", method="iir", verbose=False)
#     raw.filter(
#         l_freq=BP_LOW, h_freq=BP_HIGH,
#         method="iir", iir_params=dict(order=2, ftype="butter"),
#         phase="forward", picks="eeg", verbose=False
#     )

#     # Eventos y epoching
#     events, event_id_map = mne.events_from_annotations(raw, verbose=False)
#     event_dict = {
#         "Rest (100)": event_id_map["100"],
#         "MI (200)":   event_id_map["200"],
#     }

#     epochs_all = mne.Epochs(
#         raw, events, event_id=event_dict,
#         tmin=-3.0, tmax=5.0,
#         baseline=(-3.0, -2.0),
#         reject=None, flat=None,
#         preload=True, detrend=None, verbose=False,
#     )

#     # Rechazo manual en canales LOSO
#     ch_names_eeg = epochs_all.copy().pick_types(eeg=True).ch_names
#     picks_available = [ch for ch in PICKS_LOSO if ch in ch_names_eeg]

#     if picks_available:
#         pick_idx = [ch_names_eeg.index(ch) for ch in picks_available]
#         data_cnv = epochs_all.get_data()[:, pick_idx, :] * 1e6
#         pp       = data_cnv.max(axis=2) - data_cnv.min(axis=2)
#         drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
#         drop_idx  = np.where(drop_mask)[0].tolist()
#         epochs_all.drop(drop_idx, reason="MANUAL_REJECT")

#     n_rest = len(epochs_all["Rest (100)"])
#     n_mi   = len(epochs_all["MI (200)"])
#     print(f"      {subject.split('_')[-1]}: Rest={n_rest} MI={n_mi} "
#           f"rechazados={len(drop_idx) if picks_available else 0}")

#     return epochs_all, event_dict


# # ============================================================
# # FUNCIONES DE EXTRACCIÓN DE FEATURES
# # ============================================================

# def extract_features(epochs_obj, picks, t_points, step=None):
#     """Amplitud en µV en puntos temporales discretos."""
#     times = epochs_obj.times
#     pts   = t_points[:step] if step is not None else t_points
#     t_idx = [np.argmin(np.abs(times - t)) for t in pts]
#     try:
#         ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
#         data     = epochs_obj.get_data(picks="csd")
#         scale    = 1.0
#     except ValueError:
#         ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
#         data     = epochs_obj.get_data()
#         scale    = 1e6
#     ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
#     X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
#     y = epochs_obj.events[:, -1]
#     return X, y


# def extract_raw_data(epochs_obj, picks, tmin, tmax):
#     """Datos crudos en µV para MDM template matching."""
#     times  = epochs_obj.times
#     t_mask = (times >= tmin) & (times <= tmax)
#     try:
#         ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
#         data     = epochs_obj.get_data(picks="csd")
#         scale    = 1.0
#     except ValueError:
#         ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
#         data     = epochs_obj.get_data()
#         scale    = 1e6
#     ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
#     y      = epochs_obj.events[:, -1]
#     return data[:, ch_idx, :][:, :, t_mask] * scale, y


# def extract_features_window(epochs_obj, picks, t_start, t_end, n_pts):
#     """Amplitud en n_pts equidistantes dentro de [t_start, t_end]."""
#     times = epochs_obj.times
#     pts   = np.linspace(t_start, t_end, n_pts)
#     t_idx = [np.argmin(np.abs(times - t)) for t in pts]
#     try:
#         ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
#         data     = epochs_obj.get_data(picks="csd")
#         scale    = 1.0
#     except ValueError:
#         ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
#         data     = epochs_obj.get_data()
#         scale    = 1e6
#     ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
#     X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
#     y = epochs_obj.events[:, -1]
#     return X, y


# def extract_raw_data_window(epochs_obj, picks, t_start, t_end):
#     """Datos crudos en µV para MDM en ventana deslizante."""
#     times  = epochs_obj.times
#     t_mask = (times >= t_start) & (times <= t_end)
#     try:
#         ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
#         data     = epochs_obj.get_data(picks="csd")
#         scale    = 1.0
#     except ValueError:
#         ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
#         data     = epochs_obj.get_data()
#         scale    = 1e6
#     ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
#     y      = epochs_obj.events[:, -1]
#     return data[:, ch_idx, :][:, :, t_mask] * scale, y


# # ============================================================
# # FUNCIONES RIEMANNIANAS
# # ============================================================

# def compute_cov_trace_norm(data_3d):
#     """Covarianza trace-normalizada + regularización Tikhonov."""
#     n, n_ch, n_t = data_3d.shape
#     covs = np.zeros((n, n_ch, n_ch))
#     for i in range(n):
#         X  = data_3d[i].T
#         C  = X.T @ X
#         tr = np.trace(C)
#         C  = C / tr if tr > 0 else C
#         C += COV_REG * np.eye(n_ch)
#         covs[i] = C
#     return covs


# def build_template_covs(data_3d, template):
#     """Template matching de Racz 2023 — covarianza extendida."""
#     tmpl_rep = np.tile(template[np.newaxis], (data_3d.shape[0], 1, 1))
#     extended = np.concatenate([data_3d, tmpl_rep], axis=1)
#     return compute_cov_trace_norm(extended)


# # ============================================================
# # FUNCIONES DE CLASIFICADORES
# # ============================================================

# def make_clf(name):
#     if name == "LDA":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", LinearDiscriminantAnalysis())])
#     elif name == "LDA_shrink":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", LinearDiscriminantAnalysis(
#                              solver="lsqr", shrinkage="auto"))])
#     elif name == "SVM":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", CalibratedClassifierCV(
#                              SVC(kernel="linear", C=1.0,
#                                  probability=False, random_state=42),
#                              cv=3, method="sigmoid"))])
#     elif name == "LR":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", LogisticRegression(
#                              C=1.0, max_iter=1000, random_state=42))])
#     elif name == "RF":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", RandomForestClassifier(
#                              n_estimators=100, max_depth=4,
#                              min_samples_leaf=3, random_state=42))])
#     elif name == "DT":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", DecisionTreeClassifier(
#                              max_depth=4, min_samples_leaf=5,
#                              random_state=42))])
#     elif name == "KNN":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", KNeighborsClassifier(n_neighbors=7))])
#     elif name == "MLP":
#         return Pipeline([("sc", StandardScaler()),
#                          ("clf", MLPClassifier(
#                              hidden_layer_sizes=(64, 32), activation="relu",
#                              max_iter=500, random_state=42,
#                              early_stopping=True, validation_fraction=0.15))])
#     else:
#         raise ValueError(f"Desconocido: {name}")


# SKL_CLFS = ["LDA", "LDA_shrink", "SVM", "LR", "RF", "DT", "KNN", "MLP"]
# ALL_CLFS = SKL_CLFS + (["MDM_Riemann"] if PYRIEMANN_OK else [])


# def evaluate_loso(X_tr, y_tr, X_te, y_te,
#                   raw_tr=None, raw_te=None):
#     """
#     Entrena en (X_tr, y_tr) y evalúa en (X_te, y_te).
#     Para MDM usa raw_tr y raw_te directamente.
#     Retorna dict {clf_name: {auc, acc}}.
#     """
#     results = {}
#     for clf_name in ALL_CLFS:
#         is_r = (clf_name == "MDM_Riemann")
#         try:
#             if is_r and PYRIEMANN_OK and raw_tr is not None:
#                 clf_obj  = MDM(metric="riemann")
#                 template = raw_tr.mean(axis=0)
#                 covs_tr  = build_template_covs(raw_tr, template)
#                 covs_te  = build_template_covs(raw_te, template)
#                 clf_obj.fit(covs_tr, y_tr)
#                 y_pred   = clf_obj.predict(covs_te)
#                 acc      = accuracy_score(y_te, y_pred) * 100.0
#                 try:
#                     dists  = clf_obj.transform(covs_te)
#                     scores = -dists[:, 1] if dists.shape[1] > 1 \
#                              else -dists[:, 0]
#                     auc    = roc_auc_score(y_te, scores)
#                 except Exception:
#                     auc = 0.5
#             else:
#                 clf_obj = make_clf(clf_name)
#                 clf_obj.fit(X_tr, y_tr)
#                 proba   = clf_obj.predict_proba(X_te)[:, 1]
#                 try:
#                     auc = roc_auc_score(y_te, proba)
#                 except ValueError:
#                     auc = 0.5
#                 y_pred = clf_obj.predict(X_te)
#                 acc    = accuracy_score(y_te, y_pred) * 100.0
#         except Exception as e:
#             auc, acc = 0.5, 50.0
#             print(f"      ⚠️  {clf_name}: {e}")

#         results[clf_name] = dict(auc=round(auc, 3), acc=round(acc, 1))
#     return results


# # ============================================================
# # CARGA DE DATOS — todos los sujetos
# # ============================================================

# print(f"\n{'─'*50}")
# print("📂  Cargando y preprocesando sujetos ...")
# print(f"{'─'*50}")

# mne.set_log_level("WARNING")
# all_epochs   = {}   # subject -> epochs
# all_labels   = {}   # subject -> y (event codes)

# for subj, sess in SUBJECTS:
#     print(f"   {subj.split('_')[-1]} | {sess}")
#     epochs, event_dict = load_and_preprocess(subj, sess)
#     all_epochs[subj]   = epochs
#     _, y = extract_features(epochs, PICKS_LOSO, T_POINTS)
#     all_labels[subj]   = y

# REST_ID = event_dict["Rest (100)"]
# MI_ID   = event_dict["MI (200)"]


# # ============================================================
# # LOSO — BUCLE PRINCIPAL
# # ============================================================

# print(f"\n{'='*68}")
# print("🔄  LEAVE-ONE-SUBJECT-OUT")
# print(f"{'='*68}")
# print(f"   Canales   : {PICKS_LOSO}")
# print(f"   Features  : {N_FEATURES} (M1) | acumulativo (M2) | "
#       f"{len(WINDOWS)} ventanas (M3)")

# results_loso = {}   # subject_eval -> {M1, M2, M3}

# for i, (subj_eval, sess_eval) in enumerate(SUBJECTS):
#     subj_train = [s for s, _ in SUBJECTS if s != subj_eval]
#     label_eval = subj_eval.split("_")[-1]

#     print(f"\n{'─'*50}")
#     print(f"   Fold {i+1}/4 — evaluar: {label_eval}")
#     print(f"   Entrenar: {[s.split('_')[-1] for s in subj_train]}")
#     print(f"{'─'*50}")

#     epochs_eval = all_epochs[subj_eval]
#     y_te        = all_labels[subj_eval]
#     n_rest_te   = int(np.sum(y_te == REST_ID))
#     n_mi_te     = int(np.sum(y_te == MI_ID))
#     print(f"   Eval  : Rest={n_rest_te} MI={n_mi_te} trials")

#     # Concatenar datos de entrenamiento
#     X_tr_list, y_tr_list = [], []
#     raw_tr_list           = []
#     for subj_t in subj_train:
#         X_t, y_t = extract_features(
#             all_epochs[subj_t], PICKS_LOSO, T_POINTS)
#         X_tr_list.append(X_t)
#         y_tr_list.append(y_t)
#         raw_t, _ = extract_raw_data(
#             all_epochs[subj_t], PICKS_LOSO, T_START, T_END)
#         raw_tr_list.append(raw_t)

#     X_tr    = np.vstack(X_tr_list)
#     y_tr    = np.concatenate(y_tr_list)
#     raw_tr  = np.vstack(raw_tr_list)
#     n_rest_tr = int(np.sum(y_tr == REST_ID))
#     n_mi_tr   = int(np.sum(y_tr == MI_ID))
#     print(f"   Train : Rest={n_rest_tr} MI={n_mi_tr} trials "
#           f"({len(subj_train)} sujetos)")

#     results_loso[subj_eval] = {"fold": i+1}

#     # ── Modelo 1 — Estático ──────────────────────────────────
#     print(f"\n   📊  M1 Estático ({N_FEATURES} features)")
#     X_te, _ = extract_features(epochs_eval, PICKS_LOSO, T_POINTS)
#     raw_te, _ = extract_raw_data(epochs_eval, PICKS_LOSO, T_START, T_END)

#     m1 = evaluate_loso(X_tr, y_tr, X_te, y_te,
#                         raw_tr=raw_tr, raw_te=raw_te)
#     results_loso[subj_eval]["M1"] = m1

#     print(f"   {'Modelo':<14} {'AUC':>7} {'Acc%':>7}")
#     print("   " + "-"*30)
#     for c in ALL_CLFS:
#         flag = " ←" if m1[c]["auc"] == max(
#             m1[cc]["auc"] for cc in ALL_CLFS) else ""
#         print(f"   {c:<14} {m1[c]['auc']:>7.3f} "
#               f"{m1[c]['acc']:>6.1f}%{flag}")

#     # ── Modelo 2 — Acumulativo ───────────────────────────────
#     print(f"\n   ⏱️   M2 Acumulativo")
#     print(f"   {'t':>7} {'feat':>5}  " +
#           "  ".join(f"{c[:8]:<10}" for c in ALL_CLFS))
#     print("   " + "-"*(7+5+12*len(ALL_CLFS)))

#     m2 = {c: [] for c in ALL_CLFS}
#     for step in range(1, N_TIMEPOINTS + 1):
#         t_cur   = T_POINTS[step - 1]
#         n_feat  = step * N_CH

#         # Entrenar
#         X_tr_s_list = []
#         raw_tr_s_list = []
#         for subj_t in subj_train:
#             Xs, _ = extract_features(
#                 all_epochs[subj_t], PICKS_LOSO, T_POINTS, step=step)
#             X_tr_s_list.append(Xs)
#             rs, _ = extract_raw_data(
#                 all_epochs[subj_t], PICKS_LOSO, T_POINTS[0], t_cur)
#             raw_tr_s_list.append(rs)
#         X_tr_s   = np.vstack(X_tr_s_list)
#         raw_tr_s = np.vstack(raw_tr_s_list)

#         # Evaluar
#         X_te_s, _ = extract_features(
#             epochs_eval, PICKS_LOSO, T_POINTS, step=step)
#         raw_te_s, _ = extract_raw_data(
#             epochs_eval, PICKS_LOSO, T_POINTS[0], t_cur)

#         step_res = evaluate_loso(
#             X_tr_s, y_tr, X_te_s, y_te,
#             raw_tr=raw_tr_s, raw_te=raw_te_s)

#         row = f"   {t_cur:>7.3f} {n_feat:>5}  "
#         for c in ALL_CLFS:
#             m2[c].append(dict(
#                 t=t_cur, n_feat=n_feat,
#                 auc=step_res[c]["auc"],
#                 acc=step_res[c]["acc"]
#             ))
#             row += f"  {step_res[c]['auc']:.3f}/{step_res[c]['acc']:5.1f}%"
#         print(row)

#     results_loso[subj_eval]["M2"] = m2

#     # ── Modelo 3 — Ventanas deslizantes ─────────────────────
#     print(f"\n   🪟  M3 Ventanas deslizantes")
#     print(f"   {'Centro':>8}  {'Ventana':<18}  " +
#           "  ".join(f"{c[:8]:<10}" for c in ALL_CLFS))
#     print("   " + "-"*(8+18+12*len(ALL_CLFS)))

#     m3 = {c: [] for c in ALL_CLFS}
#     for (w_start, w_end) in WINDOWS:
#         w_center = round((w_start + w_end) / 2, 3)

#         X_tr_w_list   = []
#         raw_tr_w_list = []
#         for subj_t in subj_train:
#             Xw, _ = extract_features_window(
#                 all_epochs[subj_t], PICKS_LOSO, w_start, w_end, WIN_PTS)
#             X_tr_w_list.append(Xw)
#             rw, _ = extract_raw_data_window(
#                 all_epochs[subj_t], PICKS_LOSO, w_start, w_end)
#             raw_tr_w_list.append(rw)
#         X_tr_w   = np.vstack(X_tr_w_list)
#         raw_tr_w = np.vstack(raw_tr_w_list)

#         X_te_w, _ = extract_features_window(
#             epochs_eval, PICKS_LOSO, w_start, w_end, WIN_PTS)
#         raw_te_w, _ = extract_raw_data_window(
#             epochs_eval, PICKS_LOSO, w_start, w_end)

#         win_res = evaluate_loso(
#             X_tr_w, y_tr, X_te_w, y_te,
#             raw_tr=raw_tr_w, raw_te=raw_te_w)

#         row = f"   {w_center:>8.3f}  [{w_start:.2f},{w_end:.2f}]  "
#         for c in ALL_CLFS:
#             m3[c].append(dict(
#                 t_start=w_start, t_end=w_end, t_center=w_center,
#                 auc=win_res[c]["auc"],
#                 acc=win_res[c]["acc"]
#             ))
#             row += f"  {win_res[c]['auc']:.3f}/{win_res[c]['acc']:5.1f}%"
#         print(row)

#     results_loso[subj_eval]["M3"] = m3


# # ============================================================
# # RESUMEN LOSO
# # ============================================================

# print(f"\n{'='*68}")
# print("🚀  RESUMEN LOSO — CROSS-SUBJECT TRANSFER")
# print(f"{'='*68}")
# print(f"   Canales fijos: {PICKS_LOSO}")

# print(f"\n   {'Sujeto eval':<16} {'M1 mejor':>10} {'M1 clf':>12} "
#       f"{'M2 mejor':>10} {'M2 clf':>12} {'M2 en t':>8} "
#       f"{'M3 mejor':>10} {'M3 clf':>12}")
# print("   " + "-"*95)

# auc_m1_all, auc_m2_all, auc_m3_all = [], [], []

# for subj_eval, _ in SUBJECTS:
#     label = subj_eval.split("_")[-1]
#     r     = results_loso[subj_eval]

#     # M1
#     best_m1_clf = max(r["M1"], key=lambda c: r["M1"][c]["auc"])
#     best_m1_auc = r["M1"][best_m1_clf]["auc"]
#     auc_m1_all.append(best_m1_auc)

#     # M2
#     best_m2_clf = max(r["M2"], key=lambda c:
#                       max(x["auc"] for x in r["M2"][c]))
#     best_m2_auc = max(x["auc"] for x in r["M2"][best_m2_clf])
#     best_m2_t   = max(r["M2"][best_m2_clf],
#                       key=lambda x: x["auc"])["t"]
#     auc_m2_all.append(best_m2_auc)

#     # M3
#     best_m3_clf = max(r["M3"], key=lambda c:
#                       max(x["auc"] for x in r["M3"][c]))
#     best_m3_auc = max(x["auc"] for x in r["M3"][best_m3_clf])
#     auc_m3_all.append(best_m3_auc)

#     print(f"   {label:<16} {best_m1_auc:>10.3f} {best_m1_clf:>12} "
#           f"{best_m2_auc:>10.3f} {best_m2_clf:>12} {best_m2_t:>8.2f}s "
#           f"{best_m3_auc:>10.3f} {best_m3_clf:>12}")

# print("   " + "-"*95)
# print(f"   {'PROMEDIO':<16} {np.mean(auc_m1_all):>10.3f} {'':>12} "
#       f"{np.mean(auc_m2_all):>10.3f} {'':>12} {'':>8} "
#       f"{np.mean(auc_m3_all):>10.3f}")

# # AUC por clasificador promediado entre folds
# print(f"\n   AUC promedio cross-subject por clasificador:")
# print(f"   {'Modelo':<14} {'M1 avg':>8} {'M2 avg':>8} {'M3 avg':>8}  "
#       f"{'Mejor modelo':>12}")
# print("   " + "-"*55)

# for c in ALL_CLFS:
#     m1_avg = np.mean([results_loso[s]["M1"][c]["auc"]
#                       for s, _ in SUBJECTS])
#     m2_avg = np.mean([max(x["auc"] for x in results_loso[s]["M2"][c])
#                       for s, _ in SUBJECTS])
#     m3_avg = np.mean([max(x["auc"] for x in results_loso[s]["M3"][c])
#                       for s, _ in SUBJECTS])
#     best_m = "M1" if m1_avg >= m2_avg and m1_avg >= m3_avg else \
#              "M2" if m2_avg >= m3_avg else "M3"
#     print(f"   {c:<14} {m1_avg:>8.3f} {m2_avg:>8.3f} {m3_avg:>8.3f}  "
#           f"{best_m:>12}")

# print(f"\n   Referencia within-subject (pipeline principal):")
# print(f"   SUBJ_001: M2=0.758 | SUBJ_003: M2=0.704 | "
#       f"SUBJ_004: M2=0.583 | SUBJ_005: M2=0.618")
# print(f"   Caída esperada cross-subject: 5–15 AUC points")
# print("="*68)


# # ============================================================
# # GUARDADO
# # ============================================================

# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# base_name = f"LOSO_4subj_{timestamp}"

# def clean_nan(obj):
#     if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
#         return None
#     elif isinstance(obj, dict):
#         return {k: clean_nan(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [clean_nan(v) for v in obj]
#     return obj

# # JSON
# path_json = os.path.join(SAVE_DIR, base_name + ".json")
# save_dict = {
#     "timestamp"    : timestamp,
#     "subjects"     : [s for s, _ in SUBJECTS],
#     "channels"     : PICKS_LOSO,
#     "n_features_m1": N_FEATURES,
#     "n_windows_m3" : len(WINDOWS),
#     "results"      : {}
# }
# for subj_eval, _ in SUBJECTS:
#     r     = results_loso[subj_eval]
#     label = subj_eval.split("_")[-1]
#     save_dict["results"][label] = {
#         "fold": r["fold"],
#         "M1"  : r["M1"],
#         "M2_best_per_clf": {
#             c: {
#                 "auc": max(x["auc"] for x in r["M2"][c]),
#                 "t"  : max(r["M2"][c], key=lambda x: x["auc"])["t"]
#             } for c in ALL_CLFS
#         },
#         "M3_best_per_clf": {
#             c: {
#                 "auc"   : max(x["auc"] for x in r["M3"][c]),
#                 "window": [
#                     max(r["M3"][c], key=lambda x: x["auc"])["t_start"],
#                     max(r["M3"][c], key=lambda x: x["auc"])["t_end"],
#                 ]
#             } for c in ALL_CLFS
#         },
#     }

# with open(path_json, "w") as f:
#     json.dump(clean_nan(save_dict), f, indent=2)
# print(f"\n💾  JSON guardado: {path_json}")

# # TXT
# path_txt = os.path.join(SAVE_DIR, base_name + ".txt")
# with open(path_txt, "w") as f:
#     f.write(f"CNV BCI — LOSO CROSS-SUBJECT TRANSFER\n")
#     f.write(f"{'='*68}\n")
#     f.write(f"Timestamp  : {timestamp}\n")
#     f.write(f"Sujetos    : {[s.split('_')[-1] for s,_ in SUBJECTS]}\n")
#     f.write(f"Canales    : {PICKS_LOSO}\n")
#     f.write(f"Features   : M1={N_FEATURES} | M2=acumulativo | "
#             f"M3={len(WINDOWS)} ventanas\n\n")

#     for subj_eval, _ in SUBJECTS:
#         label = subj_eval.split("_")[-1]
#         r     = results_loso[subj_eval]
#         f.write(f"{'─'*50}\n")
#         f.write(f"Fold {r['fold']} — evaluar: {label}\n")
#         f.write(f"{'─'*50}\n")

#         f.write(f"M1 — Estático ({N_FEATURES} features):\n")
#         f.write(f"{'Modelo':<14} {'AUC':>7} {'Acc%':>7}\n")
#         for c in ALL_CLFS:
#             flag = " ← mejor" if c == max(
#                 r["M1"], key=lambda cc: r["M1"][cc]["auc"]) else ""
#             f.write(f"{c:<14} {r['M1'][c]['auc']:>7.3f} "
#                     f"{r['M1'][c]['acc']:>6.1f}%{flag}\n")

#         f.write(f"\nM2 — Acumulativo (mejor por modelo):\n")
#         f.write(f"{'Modelo':<14} {'t mejor':>8} {'AUC':>8}\n")
#         for c in ALL_CLFS:
#             best = max(r["M2"][c], key=lambda x: x["auc"])
#             flag = " ← mejor" if best["auc"] == max(
#                 max(x["auc"] for x in r["M2"][cc])
#                 for cc in ALL_CLFS) else ""
#             f.write(f"{c:<14} {best['t']:>8.2f}s {best['auc']:>8.3f}{flag}\n")

#         f.write(f"\nM2 — tabla completa (AUC por paso):\n")
#         f.write(f"{'t':>7} {'feat':>5}  " +
#                 "  ".join(f"{c:<14}" for c in ALL_CLFS) + "\n")
#         for step_i, step_r in enumerate(r["M2"][ALL_CLFS[0]]):
#             row = f"{step_r['t']:>7.3f} {step_r['n_feat']:>5}  "
#             for c in ALL_CLFS:
#                 row += f"  {r['M2'][c][step_i]['auc']:.3f}/" \
#                        f"{r['M2'][c][step_i]['acc']:5.1f}%  "
#             f.write(row + "\n")

#         f.write(f"\nM3 — Ventanas deslizantes (mejor por modelo):\n")
#         f.write(f"{'Modelo':<14} {'Ventana':>18} {'AUC':>8}\n")
#         for c in ALL_CLFS:
#             best = max(r["M3"][c], key=lambda x: x["auc"])
#             f.write(f"{c:<14} [{best['t_start']:.2f},{best['t_end']:.2f}]s "
#                     f"{best['auc']:>8.3f}\n")

#         f.write(f"\nM3 — tabla completa (AUC por ventana):\n")
#         f.write(f"{'Centro':>8}  {'Ventana':<18}  " +
#                 "  ".join(f"{c:<14}" for c in ALL_CLFS) + "\n")
#         for win_i, win_r in enumerate(r["M3"][ALL_CLFS[0]]):
#             row = (f"{win_r['t_center']:>8.3f}  "
#                    f"[{win_r['t_start']:.2f},{win_r['t_end']:.2f}]  ")
#             for c in ALL_CLFS:
#                 row += f"  {r['M3'][c][win_i]['auc']:.3f}/" \
#                        f"{r['M3'][c][win_i]['acc']:5.1f}%  "
#             f.write(row + "\n")
#         f.write("\n")

#     f.write(f"{'='*68}\n")
#     f.write("RESUMEN CROSS-SUBJECT\n")
#     f.write(f"{'='*68}\n")
#     f.write(f"{'Modelo':<14} {'M1 avg':>8} {'M2 avg':>8} {'M3 avg':>8}\n")
#     f.write("-"*42 + "\n")
#     for c in ALL_CLFS:
#         m1a = np.mean([results_loso[s]["M1"][c]["auc"] for s,_ in SUBJECTS])
#         m2a = np.mean([max(x["auc"] for x in results_loso[s]["M2"][c])
#                        for s,_ in SUBJECTS])
#         m3a = np.mean([max(x["auc"] for x in results_loso[s]["M3"][c])
#                        for s,_ in SUBJECTS])
#         f.write(f"{c:<14} {m1a:>8.3f} {m2a:>8.3f} {m3a:>8.3f}\n")
#     f.write(f"\nReferencia within-subject promedio:\n")
#     f.write(f"  M1=0.638 | M2=0.666 | M3=0.678\n")
#     f.write(f"{'='*68}\n")

# print(f"📄  TXT guardado: {path_txt}")
# print(f"\n✅  LOSO completo — archivos en: {SAVE_DIR}")

# # ============================================================
# # SECCIÓN 2 — ADAPTIVE RECENTERING + REENTRENAMIENTO ACUMULATIVO
# # ============================================================
# # MDM Riemanniano: adaptive recentering trial a trial (Racz 2023)
# #   - α = 0.05
# #   - Centroide se actualiza ANTES de clasificar cada trial
# #   - Geodésica Riemanniana entre centroide actual y covarianza del trial
# #
# # Sklearn: reentrenamiento acumulativo cada RETRAIN_EVERY trials
# #   - Reentrenar con datos originales + trials nuevos acumulados
# #   - Comparar vs modelo fijo sin adaptación
 
# print(f"\n{'='*68}")
# print("🔄  SECCIÓN 2 — ADAPTIVE RECENTERING")
# print(f"{'='*68}")
 
# ALPHA        = 0.05   # factor de aprendizaje MDM (Racz 2023)
# RETRAIN_EVERY = 10    # reentrenar sklearn cada N trials nuevos
# MIN_WINDOW   = 20     # mínimo de trials para calcular AUC rolling
 
# from pyriemann.utils.mean import mean_covariance
 
 
# def geodesic_riemann(A, B, alpha):
#     """
#     Punto en la geodésica Riemanniana entre A y B con paso alpha.
#     Implementa: A^(1/2) (A^(-1/2) B A^(-1/2))^alpha A^(1/2)
#     """
#     try:
#         A_sqrt  = np.linalg.cholesky(A)
#         A_isqrt = np.linalg.inv(A_sqrt)
#         M       = A_isqrt @ B @ A_isqrt.T
#         eigvals, eigvecs = np.linalg.eigh(M)
#         eigvals = np.maximum(eigvals, 1e-10)
#         M_alpha = eigvecs @ np.diag(eigvals ** alpha) @ eigvecs.T
#         return A_sqrt @ M_alpha @ A_sqrt.T
#     except Exception:
#         return (1 - alpha) * A + alpha * B
 
 
# def adaptive_recentering_loso(subj_eval, subj_train_list,
#                                alpha=ALPHA, retrain_every=RETRAIN_EVERY):
#     """
#     Adaptive recentering LOSO para un fold.
#     Procesa trials del sujeto de evaluación en orden cronológico.
 
#     Retorna:
#         mdm_fixed   : AUC acumulativo sin recentering (referencia)
#         mdm_adapt   : AUC acumulativo con recentering
#         skl_fixed   : {clf: AUC acumulativo sin reentrenamiento}
#         skl_adapt   : {clf: AUC acumulativo con reentrenamiento}
#     """
#     epochs_eval = all_epochs[subj_eval]
#     y_te_full   = all_labels[subj_eval]
 
#     # Datos de entrenamiento concatenados
#     X_tr_list, y_tr_list, raw_tr_list = [], [], []
#     for subj_t in subj_train_list:
#         X_t, y_t = extract_features(all_epochs[subj_t], PICKS_LOSO, T_POINTS)
#         X_tr_list.append(X_t)
#         y_tr_list.append(y_t)
#         raw_t, _ = extract_raw_data(
#             all_epochs[subj_t], PICKS_LOSO, T_START, T_END)
#         raw_tr_list.append(raw_t)
 
#     X_tr    = np.vstack(X_tr_list)
#     y_tr    = np.concatenate(y_tr_list)
#     raw_tr  = np.vstack(raw_tr_list)
 
#     # Features del sujeto de evaluación
#     X_te_full, _ = extract_features(epochs_eval, PICKS_LOSO, T_POINTS)
#     raw_te_full, _ = extract_raw_data(
#         epochs_eval, PICKS_LOSO, T_START, T_END)
#     n_trials = len(y_te_full)
 
#     # ── MDM: entrenar modelo base ────────────────────────────
#     template_base = raw_tr.mean(axis=0)
#     covs_tr       = build_template_covs(raw_tr, template_base)
 
#     mdm_fixed_obj = MDM(metric="riemann")
#     mdm_fixed_obj.fit(covs_tr, y_tr)
 
#     # Copiar centroides para la versión adaptativa
#     centers_adapt = {
#         label: mdm_fixed_obj.covmeans_[i].copy()
#         for i, label in enumerate(mdm_fixed_obj.classes_)
#     }
 
#     # ── Sklearn: entrenar modelos base ───────────────────────
#     skl_clfs_fixed = {}
#     skl_clfs_adapt = {}
#     for clf_name in SKL_CLFS:
#         clf_f = make_clf(clf_name)
#         clf_f.fit(X_tr, y_tr)
#         skl_clfs_fixed[clf_name] = clf_f
 
#         clf_a = make_clf(clf_name)
#         clf_a.fit(X_tr, y_tr)
#         skl_clfs_adapt[clf_name] = clf_a
 
#     # ── Procesamiento trial a trial ──────────────────────────
#     mdm_scores_fixed = []
#     mdm_scores_adapt = []
#     mdm_labels       = []
 
#     skl_scores_fixed = {c: [] for c in SKL_CLFS}
#     skl_scores_adapt = {c: [] for c in SKL_CLFS}
#     skl_labels       = {c: [] for c in SKL_CLFS}
 
#     X_accum   = []   # trials nuevos acumulados para sklearn
#     y_accum   = []
 
#     for trial_i in range(n_trials):
#         x_trial   = X_te_full[trial_i]
#         raw_trial = raw_te_full[trial_i:trial_i+1]
#         y_trial   = y_te_full[trial_i]
 
#         # Covarianza del trial nuevo con template base
#         cov_trial = build_template_covs(raw_trial, template_base)[0]
 
#         # ── MDM fijo ────────────────────────────────────────
#         cov_trial_batch = cov_trial[np.newaxis]
#         try:
#             dists  = mdm_fixed_obj.transform(cov_trial_batch)
#             score  = -dists[0, 1] if dists.shape[1] > 1 else -dists[0, 0]
#         except Exception:
#             score = 0.0
#         mdm_scores_fixed.append(score)
#         mdm_labels.append(y_trial)
 
#         # ── MDM adaptativo: clasificar ANTES de actualizar ──
#         # Calcular distancias con centroides actuales
#         try:
#             dist_rest = np.linalg.norm(
#                 cov_trial - centers_adapt[REST_ID], 'fro')
#             dist_mi   = np.linalg.norm(
#                 cov_trial - centers_adapt[MI_ID],   'fro')
#             score_adapt = -(dist_mi / (dist_rest + dist_mi + 1e-10))
#         except Exception:
#             score_adapt = 0.0
#         mdm_scores_adapt.append(score_adapt)
 
#         # Actualizar centroides con geodésica Riemanniana
#         for label in centers_adapt:
#             centers_adapt[label] = geodesic_riemann(
#                 centers_adapt[label], cov_trial, alpha)
 
#         # ── Sklearn fijo ─────────────────────────────────────
#         for clf_name in SKL_CLFS:
#             try:
#                 proba = skl_clfs_fixed[clf_name].predict_proba(
#                     x_trial.reshape(1, -1))[0, 1]
#             except Exception:
#                 proba = 0.5
#             skl_scores_fixed[clf_name].append(proba)
#             skl_labels[clf_name].append(y_trial)
 
#         # ── Sklearn adaptativo: acumular y reentrenar ────────
#         X_accum.append(x_trial)
#         y_accum.append(y_trial)
 
#         for clf_name in SKL_CLFS:
#             try:
#                 proba = skl_clfs_adapt[clf_name].predict_proba(
#                     x_trial.reshape(1, -1))[0, 1]
#             except Exception:
#                 proba = 0.5
#             skl_scores_adapt[clf_name].append(proba)
 
#         # Reentrenar sklearn cada RETRAIN_EVERY trials
#         if len(X_accum) % retrain_every == 0 and len(X_accum) >= retrain_every:
#             X_combined = np.vstack([X_tr] + [np.array(X_accum)])
#             y_combined = np.concatenate([y_tr, np.array(y_accum)])
#             for clf_name in SKL_CLFS:
#                 try:
#                     clf_new = make_clf(clf_name)
#                     clf_new.fit(X_combined, y_combined)
#                     skl_clfs_adapt[clf_name] = clf_new
#                 except Exception:
#                     pass
 
#     # ── Calcular AUC rolling (ventana acumulativa) ───────────
#     def rolling_auc(scores, labels, min_w=MIN_WINDOW):
#         aucs = []
#         for i in range(len(scores)):
#             if i + 1 < min_w:
#                 aucs.append(None)
#                 continue
#             try:
#                 a = roc_auc_score(labels[:i+1], scores[:i+1])
#             except Exception:
#                 a = 0.5
#             aucs.append(round(a, 3))
#         return aucs
 
#     mdm_auc_fixed = rolling_auc(mdm_scores_fixed, mdm_labels)
#     mdm_auc_adapt = rolling_auc(mdm_scores_adapt, mdm_labels)
 
#     skl_auc_fixed = {}
#     skl_auc_adapt = {}
#     for clf_name in SKL_CLFS:
#         skl_auc_fixed[clf_name] = rolling_auc(
#             skl_scores_fixed[clf_name], skl_labels[clf_name])
#         skl_auc_adapt[clf_name] = rolling_auc(
#             skl_scores_adapt[clf_name], skl_labels[clf_name])
 
#     # AUC final (todos los trials)
#     def final_auc(scores, labels):
#         try:
#             return round(roc_auc_score(labels, scores), 3)
#         except Exception:
#             return 0.5
 
#     return dict(
#         n_trials      = n_trials,
#         mdm_fixed_auc = final_auc(mdm_scores_fixed, mdm_labels),
#         mdm_adapt_auc = final_auc(mdm_scores_adapt, mdm_labels),
#         mdm_auc_fixed_curve = mdm_auc_fixed,
#         mdm_auc_adapt_curve = mdm_auc_adapt,
#         skl_fixed_auc = {c: final_auc(skl_scores_fixed[c], skl_labels[c])
#                          for c in SKL_CLFS},
#         skl_adapt_auc = {c: final_auc(skl_scores_adapt[c], skl_labels[c])
#                          for c in SKL_CLFS},
#         skl_auc_fixed_curve = skl_auc_fixed,
#         skl_auc_adapt_curve = skl_auc_adapt,
#     )
 
 
# # ── Ejecutar adaptive recentering para todos los folds ───────
 
# print(f"\n   α MDM       : {ALPHA}")
# print(f"   Reentrenar  : cada {RETRAIN_EVERY} trials (sklearn)")
# print(f"   Ventana min : {MIN_WINDOW} trials para AUC rolling")
 
# results_adapt = {}
 
# for i, (subj_eval, _) in enumerate(SUBJECTS):
#     subj_train = [s for s, _ in SUBJECTS if s != subj_eval]
#     label_eval = subj_eval.split("_")[-1]
 
#     print(f"\n{'─'*50}")
#     print(f"   Fold {i+1}/{len(SUBJECTS)} — evaluar: {label_eval} "
#           f"| entrenar: {[s.split('_')[-1] for s in subj_train]}")
#     print(f"{'─'*50}")
 
#     r = adaptive_recentering_loso(subj_eval, subj_train)
#     results_adapt[subj_eval] = r
 
#     delta_mdm = r["mdm_adapt_auc"] - r["mdm_fixed_auc"]
#     print(f"\n   MDM fijo    : AUC={r['mdm_fixed_auc']:.3f}")
#     print(f"   MDM adapt   : AUC={r['mdm_adapt_auc']:.3f}  "
#           f"({'+'if delta_mdm>=0 else ''}{delta_mdm:.3f})")
 
#     print(f"\n   {'Modelo':<14} {'Fijo':>7} {'Adapt':>7} {'Δ':>7}")
#     print("   " + "-"*35)
#     for c in SKL_CLFS:
#         f_auc = r["skl_fixed_auc"][c]
#         a_auc = r["skl_adapt_auc"][c]
#         delta = a_auc - f_auc
#         flag  = " ←" if a_auc == max(
#             list(r["skl_adapt_auc"].values()) +
#             [r["mdm_adapt_auc"]]) else ""
#         print(f"   {c:<14} {f_auc:>7.3f} {a_auc:>7.3f} "
#               f"{'+'if delta>=0 else ''}{delta:.3f}{flag}")
 
 
# # ── Resumen cross-subject adaptive recentering ────────────────
 
# print(f"\n{'='*68}")
# print("🚀  RESUMEN ADAPTIVE RECENTERING — CROSS-SUBJECT")
# print(f"{'='*68}")
# print(f"   α={ALPHA} | reentrenar cada {RETRAIN_EVERY} trials")
 
# print(f"\n   {'Sujeto':<8} {'MDM fijo':>9} {'MDM adapt':>10} {'ΔMDM':>6}  "
#       f"{'Mejor fijo':>11} {'Mejor adapt':>12} {'ΔBest':>7}")
# print("   " + "-"*70)
 
# all_mdm_fixed, all_mdm_adapt = [], []
# all_best_fixed, all_best_adapt = [], []
 
# for subj_eval, _ in SUBJECTS:
#     label = subj_eval.split("_")[-1]
#     r     = results_adapt[subj_eval]
 
#     mdm_f = r["mdm_fixed_auc"]
#     mdm_a = r["mdm_adapt_auc"]
#     d_mdm = mdm_a - mdm_f
 
#     best_f = max(list(r["skl_fixed_auc"].values()) + [mdm_f])
#     best_a = max(list(r["skl_adapt_auc"].values()) + [mdm_a])
#     d_best = best_a - best_f
 
#     all_mdm_fixed.append(mdm_f)
#     all_mdm_adapt.append(mdm_a)
#     all_best_fixed.append(best_f)
#     all_best_adapt.append(best_a)
 
#     print(f"   {label:<8} {mdm_f:>9.3f} {mdm_a:>10.3f} "
#           f"{'+'if d_mdm>=0 else ''}{d_mdm:.3f}  "
#           f"{best_f:>11.3f} {best_a:>12.3f} "
#           f"{'+'if d_best>=0 else ''}{d_best:.3f}")
 
# print("   " + "-"*70)
# print(f"   {'PROMEDIO':<8} {np.mean(all_mdm_fixed):>9.3f} "
#       f"{np.mean(all_mdm_adapt):>10.3f} "
#       f"{'+'if np.mean(all_mdm_adapt)>=np.mean(all_mdm_fixed) else ''}"
#       f"{np.mean(all_mdm_adapt)-np.mean(all_mdm_fixed):.3f}  "
#       f"{np.mean(all_best_fixed):>11.3f} {np.mean(all_best_adapt):>12.3f} "
#       f"{'+'if np.mean(all_best_adapt)>=np.mean(all_best_fixed) else ''}"
#       f"{np.mean(all_best_adapt)-np.mean(all_best_fixed):.3f}")
 
# print(f"\n   AUC promedio por clasificador (fijo vs adaptativo):")
# print(f"   {'Modelo':<14} {'Fijo avg':>9} {'Adapt avg':>10} {'Δ avg':>7}")
# print("   " + "-"*44)
# for c in SKL_CLFS:
#     f_avg = np.mean([results_adapt[s]["skl_fixed_auc"][c]
#                      for s, _ in SUBJECTS])
#     a_avg = np.mean([results_adapt[s]["skl_adapt_auc"][c]
#                      for s, _ in SUBJECTS])
#     delta = a_avg - f_avg
#     print(f"   {c:<14} {f_avg:>9.3f} {a_avg:>10.3f} "
#           f"{'+'if delta>=0 else ''}{delta:.3f}")
 
# mdm_f_avg = np.mean(all_mdm_fixed)
# mdm_a_avg = np.mean(all_mdm_adapt)
# print(f"   {'MDM_Riemann':<14} {mdm_f_avg:>9.3f} {mdm_a_avg:>10.3f} "
#       f"{'+'if mdm_a_avg>=mdm_f_avg else ''}{mdm_a_avg-mdm_f_avg:.3f}")
 
# print(f"\n   Referencia LOSO sin adaptación (M1): "
#       f"AUC promedio = {np.mean([results_loso[s]['M1'][c]['auc'] for s,_ in SUBJECTS for c in ALL_CLFS if c=='MDM_Riemann']):.3f} (MDM)")
# print("="*68)
 
 
# # ── Guardar resultados adaptativos ────────────────────────────
 
# timestamp_a = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# base_name_a = f"LOSO_adaptive_{timestamp_a}"
 
# path_json_a = os.path.join(SAVE_DIR, base_name_a + ".json")
# path_txt_a  = os.path.join(SAVE_DIR, base_name_a + ".txt")
 
# save_adapt = {
#     "timestamp"     : timestamp_a,
#     "subjects"      : [s for s, _ in SUBJECTS],
#     "channels"      : PICKS_LOSO,
#     "alpha_mdm"     : ALPHA,
#     "retrain_every" : RETRAIN_EVERY,
#     "results"       : {}
# }
# for subj_eval, _ in SUBJECTS:
#     label = subj_eval.split("_")[-1]
#     r     = results_adapt[subj_eval]
#     save_adapt["results"][label] = {
#         "n_trials"     : r["n_trials"],
#         "mdm_fixed_auc": r["mdm_fixed_auc"],
#         "mdm_adapt_auc": r["mdm_adapt_auc"],
#         "skl_fixed_auc": r["skl_fixed_auc"],
#         "skl_adapt_auc": r["skl_adapt_auc"],
#         "mdm_auc_fixed_curve": r["mdm_auc_fixed_curve"],
#         "mdm_auc_adapt_curve": r["mdm_auc_adapt_curve"],
#     }
 
# with open(path_json_a, "w") as f:
#     json.dump(clean_nan(save_adapt), f, indent=2)
# print(f"\n💾  JSON adaptativo: {path_json_a}")
 
# with open(path_txt_a, "w") as f:
#     f.write(f"CNV BCI — ADAPTIVE RECENTERING LOSO\n")
#     f.write(f"{'='*68}\n")
#     f.write(f"Timestamp     : {timestamp_a}\n")
#     f.write(f"Sujetos       : {[s.split('_')[-1] for s,_ in SUBJECTS]}\n")
#     f.write(f"Canales       : {PICKS_LOSO}\n")
#     f.write(f"α MDM         : {ALPHA}\n")
#     f.write(f"Reentrenar    : cada {RETRAIN_EVERY} trials (sklearn)\n\n")
 
#     for subj_eval, _ in SUBJECTS:
#         label = subj_eval.split("_")[-1]
#         r     = results_adapt[subj_eval]
#         f.write(f"{'─'*50}\n")
#         f.write(f"Sujeto eval: {label} | {r['n_trials']} trials\n")
#         f.write(f"{'─'*50}\n")
#         f.write(f"MDM fijo    : {r['mdm_fixed_auc']:.3f}\n")
#         f.write(f"MDM adapt   : {r['mdm_adapt_auc']:.3f}  "
#                 f"(Δ={'+'if r['mdm_adapt_auc']>=r['mdm_fixed_auc'] else ''}"
#                 f"{r['mdm_adapt_auc']-r['mdm_fixed_auc']:.3f})\n\n")
#         f.write(f"{'Modelo':<14} {'Fijo':>7} {'Adapt':>7} {'Δ':>7}\n")
#         f.write("-"*35 + "\n")
#         for c in SKL_CLFS:
#             fa = r["skl_fixed_auc"][c]
#             aa = r["skl_adapt_auc"][c]
#             d  = aa - fa
#             f.write(f"{c:<14} {fa:>7.3f} {aa:>7.3f} "
#                     f"{'+'if d>=0 else ''}{d:.3f}\n")
#         f.write(f"\nCurva AUC acumulativa — MDM:\n")
#         f.write(f"{'Trial':>7} {'Fijo':>8} {'Adapt':>8}\n")
#         for ti, (af, aa) in enumerate(zip(
#                 r["mdm_auc_fixed_curve"],
#                 r["mdm_auc_adapt_curve"])):
#             if af is not None and aa is not None:
#                 f.write(f"{ti+1:>7} {af:>8.3f} {aa:>8.3f}\n")
#         f.write("\n")
 
#     f.write(f"{'='*68}\n")
#     f.write("RESUMEN PROMEDIO CROSS-SUBJECT\n")
#     f.write(f"{'='*68}\n")
#     f.write(f"{'Modelo':<14} {'Fijo avg':>9} {'Adapt avg':>10} {'Δ':>7}\n")
#     f.write("-"*44 + "\n")
#     for c in SKL_CLFS:
#         fa = np.mean([results_adapt[s]["skl_fixed_auc"][c]
#                       for s,_ in SUBJECTS])
#         aa = np.mean([results_adapt[s]["skl_adapt_auc"][c]
#                       for s,_ in SUBJECTS])
#         f.write(f"{c:<14} {fa:>9.3f} {aa:>10.3f} "
#                 f"{'+'if aa>=fa else ''}{aa-fa:.3f}\n")
#     mf = np.mean([results_adapt[s]["mdm_fixed_auc"] for s,_ in SUBJECTS])
#     ma = np.mean([results_adapt[s]["mdm_adapt_auc"] for s,_ in SUBJECTS])
#     f.write(f"{'MDM_Riemann':<14} {mf:>9.3f} {ma:>10.3f} "
#             f"{'+'if ma>=mf else ''}{ma-mf:.3f}\n")
#     f.write(f"{'='*68}\n")
 
# print(f"📄  TXT adaptativo: {path_txt_a}")
# print(f"\n✅  Adaptive recentering completo")
 

"""
================================================================================
CNV BCI — ADAPTIVE RECENTERING LOSO — M1, M2, M3
================================================================================
Script independiente de adaptive recentering cross-subject.
Evalúa mejora de adaptación trial a trial en los 3 modelos temporales.

Esquema LOSO: 5 sujetos, dejar uno fuera en cada iteración.
Canales fijos: FC5, FC1, Cz, CP1, Fz

MDM Riemanniano:
  - Recentering una vez por trial (al final del trial)
  - α = 0.05 (Racz 2023)
  - Geodésica Riemanniana: M_new = geodesic(M_old, C_trial, α)

Sklearn (LDA, LDA_shrink, SVM, LR, RF, DT, KNN, MLP):
  - Reentrenamiento acumulativo cada RETRAIN_EVERY trials
  - Datos originales + trials nuevos acumulados

Modelos evaluados:
  M1 — Estático     : 55 features (11 pts × 5 ch), 1 clasificación por trial
  M2 — Acumulativo  : 11 pasos, clasificación en cada paso temporal
  M3 — Deslizante   : 11 ventanas de 2s, clasificación en cada ventana
================================================================================
"""

import os
import json
import datetime
import numpy as np
import mne
from scipy import stats

import config
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

try:
    from pyriemann.classification import MDM
    from pyriemann.utils.mean import mean_covariance
    PYRIEMANN_OK = True
    print("✅  pyriemann disponible")
except ImportError:
    PYRIEMANN_OK = False
    print("⚠️   pyriemann no instalado — MDM desactivado")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# CONFIGURACIÓN
# ============================================================

SUBJECTS = [
    ("CNV_PILOT_SUBJ_001", "S001OFFLINE_GLOVE"), #Erick 
    ("CNV_PILOT_SUBJ_003", "S001OFFLINE_GLOVE"), #Claudia_1
    ("CNV_PILOT_SUBJ_004", "S001OFFLINE_GLOVE"), #Claudia_2
    ("CNV_PILOT_SUBJ_005", "S001OFFLINE_GLOVE"), #Alex
    ("CNV_PILOT_SUBJ_006", "S001OFFLINE_GLOVE"), #Sharon
]

XDF_BASE      = "/home/lab-admin/Documents/CNVStudy"
SAVE_DIR      = "/home/lab-admin/Documents/CNVStudy/logs_2"
os.makedirs(SAVE_DIR, exist_ok=True)

PICKS_LOSO    = ['FC5', 'FC1', 'Cz', 'CP1', 'Fz']

# Preprocesamiento
CHANNELS_TO_DROP = ['M1', 'M2', 'T7', 'T8', 'Fp1', 'Fpz', 'Fp2']
RENAME_DICT = {
    "FP1": "Fp1", "FPz": "Fpz", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ":  "Fz",  "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz",
    "OZ":  "Oz",  "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX8", "AUX9", "TRIGGER"}
TARGET_MARKERS   = [100, 200]
REJECT_UV        = 150.0
BP_LOW, BP_HIGH  = 0.1, 1.0

# T_POINTS
T_START      = -2.5
T_END        =  0.0
N_TIMEPOINTS =  11
T_POINTS     = np.linspace(T_START, T_END, N_TIMEPOINTS)
N_CH         = len(PICKS_LOSO)
N_FEATURES   = N_TIMEPOINTS * N_CH   # 55

# Ventanas deslizantes M3
WIN_SIZE  = 2.0
WIN_STEP  = 0.05
WIN_PTS   = 9
win_starts = np.arange(T_START, T_END - WIN_SIZE + 1e-9, WIN_STEP)
WINDOWS    = list(zip(np.round(win_starts, 4),
                      np.round(win_starts + WIN_SIZE, 4)))
N_WINDOWS  = len(WINDOWS)

# Adaptive recentering
ALPHA         = 0.05   # factor de aprendizaje MDM (Racz 2023)
RETRAIN_EVERY = 10     # reentrenar sklearn cada N trials
MIN_WINDOW    = 20     # mínimo trials para AUC rolling
COV_REG       = 1e-4

print(f"\n{'='*68}")
print("🔄  CNV BCI — ADAPTIVE RECENTERING (M1, M2, M3)")
print(f"{'='*68}")
print(f"   Sujetos  : {[s[0].split('_')[-1] for s in SUBJECTS]}")
print(f"   Canales  : {PICKS_LOSO}")
print(f"   α MDM    : {ALPHA} | Reentrenar sklearn: cada {RETRAIN_EVERY} trials")
print(f"   M1: {N_FEATURES} features | M2: {N_TIMEPOINTS} pasos | M3: {N_WINDOWS} ventanas")


# ============================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================

def load_and_preprocess(subject, session):
    xdf_dir = os.path.join(
        XDF_BASE, f"sub-{subject}", f"ses-{session}", "eeg/"
    )
    xdf_files = sorted(
        [os.path.join(xdf_dir, f)
         for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
    )
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

        valid_ch        = [ch for ch in channel_names
                           if ch not in NON_EEG_CHANNELS]
        valid_idx       = [channel_names.index(ch) for ch in valid_ch]
        eeg_data_subset = eeg_data[valid_idx, :] / 1e6

        info    = mne.create_info(
            ch_names=valid_ch, sfreq=config.FS, ch_types="eeg")
        raw_tmp = mne.io.RawArray(eeg_data_subset, info, verbose=False)

        if "AUX7" in raw_tmp.ch_names:
            raw_tmp.set_channel_types({"AUX7": "emg"})

        existing_renames = {k: v for k, v in RENAME_DICT.items()
                            if k in raw_tmp.ch_names}
        if existing_renames:
            raw_tmp.rename_channels(existing_renames)

        raw_tmp.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            on_missing="warn")

        drop_targets = [ch for ch in CHANNELS_TO_DROP
                        if ch in raw_tmp.ch_names]
        if drop_targets:
            raw_tmp.drop_channels(drop_targets)

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
        phase="forward", picks="eeg", verbose=False
    )

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    event_dict = {
        "Rest (100)": event_id_map["100"],
        "MI (200)":   event_id_map["200"],
    }

    epochs_all = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-3.0, tmax=5.0,
        baseline=(-3.0, -2.0),
        reject=None, flat=None,
        preload=True, detrend=None, verbose=False,
    )

    ch_names_eeg = epochs_all.copy().pick_types(eeg=True).ch_names
    picks_avail  = [ch for ch in PICKS_LOSO if ch in ch_names_eeg]

    drop_idx = []
    if picks_avail:
        pick_idx = [ch_names_eeg.index(ch) for ch in picks_avail]
        data_cnv = epochs_all.get_data()[:, pick_idx, :] * 1e6
        pp       = data_cnv.max(axis=2) - data_cnv.min(axis=2)
        drop_mask = (pp.max(axis=1) > REJECT_UV) | (pp.max(axis=1) < 1.0)
        drop_idx  = np.where(drop_mask)[0].tolist()
        epochs_all.drop(drop_idx, reason="MANUAL_REJECT")

    print(f"      {subject.split('_')[-1]}: "
          f"Rest={len(epochs_all['Rest (100)'])} "
          f"MI={len(epochs_all['MI (200)'])} "
          f"rechazados={len(drop_idx)}")

    return epochs_all, event_dict


# ============================================================
# FUNCIONES DE EXTRACCIÓN
# ============================================================

def get_eeg_data(epochs_obj, picks):
    try:
        ch_names = epochs_obj.copy().pick_types(csd=True).ch_names
        data     = epochs_obj.get_data(picks="csd")
        scale    = 1.0
    except ValueError:
        ch_names = epochs_obj.copy().pick_types(eeg=True).ch_names
        data     = epochs_obj.get_data()
        scale    = 1e6
    ch_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    return data, ch_idx, scale, ch_names


def extract_features(epochs_obj, picks, t_points, step=None):
    times  = epochs_obj.times
    pts    = t_points[:step] if step is not None else t_points
    t_idx  = [np.argmin(np.abs(times - t)) for t in pts]
    data, ch_idx, scale, _ = get_eeg_data(epochs_obj, picks)
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def extract_raw_data(epochs_obj, picks, tmin, tmax):
    times  = epochs_obj.times
    t_mask = (times >= tmin) & (times <= tmax)
    data, ch_idx, scale, _ = get_eeg_data(epochs_obj, picks)
    y = epochs_obj.events[:, -1]
    return data[:, ch_idx, :][:, :, t_mask] * scale, y


def extract_features_window(epochs_obj, picks, t_start, t_end, n_pts):
    times = epochs_obj.times
    pts   = np.linspace(t_start, t_end, n_pts)
    t_idx = [np.argmin(np.abs(times - t)) for t in pts]
    data, ch_idx, scale, _ = get_eeg_data(epochs_obj, picks)
    X = np.hstack([data[:, ci, :][:, t_idx] * scale for ci in ch_idx])
    y = epochs_obj.events[:, -1]
    return X, y


def extract_raw_data_window(epochs_obj, picks, t_start, t_end):
    times  = epochs_obj.times
    t_mask = (times >= t_start) & (times <= t_end)
    data, ch_idx, scale, _ = get_eeg_data(epochs_obj, picks)
    y = epochs_obj.events[:, -1]
    return data[:, ch_idx, :][:, :, t_mask] * scale, y


# ============================================================
# FUNCIONES RIEMANNIANAS
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
    elif name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(
                             SVC(kernel="linear", C=1.0,
                                 probability=False, random_state=42),
                             cv=3, method="sigmoid"))])
    elif name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(
                             C=1.0, max_iter=1000, random_state=42))])
    elif name == "RF":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestClassifier(
                             n_estimators=100, max_depth=4,
                             min_samples_leaf=3, random_state=42))])
    elif name == "DT":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", DecisionTreeClassifier(
                             max_depth=4, min_samples_leaf=5,
                             random_state=42))])
    elif name == "KNN":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))])
    elif name == "MLP":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(
                             hidden_layer_sizes=(64, 32), activation="relu",
                             max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.15))])
    else:
        raise ValueError(f"Desconocido: {name}")


SKL_CLFS = ["LDA", "LDA_shrink", "SVM", "LR", "RF", "DT", "KNN", "MLP"]
ALL_CLFS = SKL_CLFS + (["MDM_Riemann"] if PYRIEMANN_OK else [])


# ============================================================
# FUNCIÓN AUXILIAR — AUC ROLLING ACUMULATIVO
# ============================================================

def rolling_auc(scores, labels, min_w=MIN_WINDOW):
    aucs = []
    for i in range(len(scores)):
        if i + 1 < min_w:
            aucs.append(None)
            continue
        try:
            a = roc_auc_score(labels[:i+1], scores[:i+1])
        except Exception:
            a = 0.5
        aucs.append(round(a, 3))
    return aucs


def final_auc(scores, labels):
    try:
        return round(roc_auc_score(labels, scores), 3)
    except Exception:
        return 0.5


# ============================================================
# CARGA DE DATOS
# ============================================================

print(f"\n{'─'*50}")
print("📂  Cargando sujetos ...")
print(f"{'─'*50}")

mne.set_log_level("WARNING")
all_epochs = {}
all_labels = {}

for subj, sess in SUBJECTS:
    print(f"   {subj.split('_')[-1]} | {sess}")
    epochs, event_dict = load_and_preprocess(subj, sess)
    all_epochs[subj]   = epochs
    _, y = extract_features(epochs, PICKS_LOSO, T_POINTS)
    all_labels[subj]   = y

REST_ID = event_dict["Rest (100)"]
MI_ID   = event_dict["MI (200)"]


# ============================================================
# ADAPTIVE RECENTERING LOSO — M1, M2, M3
# ============================================================

def run_adaptive_fold(subj_eval, subj_train_list):
    """
    Corre adaptive recentering para un fold en M1, M2 y M3.
    Procesa trials en orden cronológico.
    MDM: recentering al final de cada trial (Opción A).
    Sklearn: reentrenamiento cada RETRAIN_EVERY trials.

    Retorna dict con resultados de M1, M2, M3 para fijo y adaptativo.
    """
    epochs_eval  = all_epochs[subj_eval]
    y_te_full    = all_labels[subj_eval]
    n_trials     = len(y_te_full)

    # ── Datos de entrenamiento ───────────────────────────────
    X_tr_list, y_tr_list, raw_tr_list = [], [], []
    for subj_t in subj_train_list:
        Xt, yt = extract_features(all_epochs[subj_t], PICKS_LOSO, T_POINTS)
        X_tr_list.append(Xt)
        y_tr_list.append(yt)
        rt, _  = extract_raw_data(
            all_epochs[subj_t], PICKS_LOSO, T_START, T_END)
        raw_tr_list.append(rt)

    X_tr   = np.vstack(X_tr_list)
    y_tr   = np.concatenate(y_tr_list)
    raw_tr = np.vstack(raw_tr_list)

    # ── Features del sujeto de evaluación ───────────────────
    X_te_full, _ = extract_features(epochs_eval, PICKS_LOSO, T_POINTS)
    raw_te_full, _ = extract_raw_data(
        epochs_eval, PICKS_LOSO, T_START, T_END)

    # M2 — features por paso
    X_te_steps = []
    raw_te_steps = []
    for step in range(1, N_TIMEPOINTS + 1):
        Xs, _ = extract_features(
            epochs_eval, PICKS_LOSO, T_POINTS, step=step)
        X_te_steps.append(Xs)
        rs, _ = extract_raw_data(
            epochs_eval, PICKS_LOSO, T_POINTS[0], T_POINTS[step-1])
        raw_te_steps.append(rs)

    # M2 training features por paso
    X_tr_steps = []
    raw_tr_steps = []
    for step in range(1, N_TIMEPOINTS + 1):
        Xs_list, rs_list = [], []
        for subj_t in subj_train_list:
            Xs, _ = extract_features(
                all_epochs[subj_t], PICKS_LOSO, T_POINTS, step=step)
            Xs_list.append(Xs)
            rs, _ = extract_raw_data(
                all_epochs[subj_t], PICKS_LOSO,
                T_POINTS[0], T_POINTS[step-1])
            rs_list.append(rs)
        X_tr_steps.append(np.vstack(Xs_list))
        raw_tr_steps.append(np.vstack(rs_list))

    # M3 — features por ventana
    X_te_wins = []
    raw_te_wins = []
    for (w_start, w_end) in WINDOWS:
        Xw, _ = extract_features_window(
            epochs_eval, PICKS_LOSO, w_start, w_end, WIN_PTS)
        X_te_wins.append(Xw)
        rw, _ = extract_raw_data_window(
            epochs_eval, PICKS_LOSO, w_start, w_end)
        raw_te_wins.append(rw)

    X_tr_wins = []
    raw_tr_wins = []
    for (w_start, w_end) in WINDOWS:
        Xw_list, rw_list = [], []
        for subj_t in subj_train_list:
            Xw, _ = extract_features_window(
                all_epochs[subj_t], PICKS_LOSO, w_start, w_end, WIN_PTS)
            Xw_list.append(Xw)
            rw, _ = extract_raw_data_window(
                all_epochs[subj_t], PICKS_LOSO, w_start, w_end)
            rw_list.append(rw)
        X_tr_wins.append(np.vstack(Xw_list))
        raw_tr_wins.append(np.vstack(rw_list))

    # ── Entrenar modelos base ────────────────────────────────
    template_base = raw_tr.mean(axis=0)
    covs_tr_full  = build_template_covs(raw_tr, template_base)

    # MDM M1
    mdm_m1_fixed = MDM(metric="riemann")
    mdm_m1_fixed.fit(covs_tr_full, y_tr)
    centers_m1 = {
        label: mdm_m1_fixed.covmeans_[i].copy()
        for i, label in enumerate(mdm_m1_fixed.classes_)
    }

    # MDM M2 — un modelo por paso
    mdm_m2_fixed = []
    centers_m2   = []
    for step in range(N_TIMEPOINTS):
        m = MDM(metric="riemann")
        covs_s = build_template_covs(
            raw_tr_steps[step],
            raw_tr_steps[step].mean(axis=0))
        m.fit(covs_s, y_tr)
        mdm_m2_fixed.append(m)
        centers_m2.append({
            label: m.covmeans_[i].copy()
            for i, label in enumerate(m.classes_)
        })

    # MDM M3 — un modelo por ventana
    mdm_m3_fixed = []
    centers_m3   = []
    for wi in range(N_WINDOWS):
        m = MDM(metric="riemann")
        covs_w = build_template_covs(
            raw_tr_wins[wi],
            raw_tr_wins[wi].mean(axis=0))
        m.fit(covs_w, y_tr)
        mdm_m3_fixed.append(m)
        centers_m3.append({
            label: m.covmeans_[i].copy()
            for i, label in enumerate(m.classes_)
        })

    # Sklearn — M1, M2 (por paso), M3 (por ventana)
    skl_m1_fixed = {c: make_clf(c) for c in SKL_CLFS}
    skl_m1_adapt = {c: make_clf(c) for c in SKL_CLFS}
    for c in SKL_CLFS:
        skl_m1_fixed[c].fit(X_tr, y_tr)
        skl_m1_adapt[c].fit(X_tr, y_tr)

    skl_m2_fixed = [{c: make_clf(c) for c in SKL_CLFS}
                    for _ in range(N_TIMEPOINTS)]
    skl_m2_adapt = [{c: make_clf(c) for c in SKL_CLFS}
                    for _ in range(N_TIMEPOINTS)]
    for step in range(N_TIMEPOINTS):
        for c in SKL_CLFS:
            skl_m2_fixed[step][c].fit(X_tr_steps[step], y_tr)
            skl_m2_adapt[step][c].fit(X_tr_steps[step], y_tr)

    skl_m3_fixed = [{c: make_clf(c) for c in SKL_CLFS}
                    for _ in range(N_WINDOWS)]
    skl_m3_adapt = [{c: make_clf(c) for c in SKL_CLFS}
                    for _ in range(N_WINDOWS)]
    for wi in range(N_WINDOWS):
        for c in SKL_CLFS:
            skl_m3_fixed[wi][c].fit(X_tr_wins[wi], y_tr)
            skl_m3_adapt[wi][c].fit(X_tr_wins[wi], y_tr)

    # ── Procesamiento trial a trial ──────────────────────────
    # Acumuladores de scores por modelo
    m1_mdm_fix_scores,  m1_mdm_adp_scores  = [], []
    m2_mdm_fix_scores  = [[] for _ in range(N_TIMEPOINTS)]
    m2_mdm_adp_scores  = [[] for _ in range(N_TIMEPOINTS)]
    m3_mdm_fix_scores  = [[] for _ in range(N_WINDOWS)]
    m3_mdm_adp_scores  = [[] for _ in range(N_WINDOWS)]

    m1_skl_fix = {c: [] for c in SKL_CLFS}
    m1_skl_adp = {c: [] for c in SKL_CLFS}
    m2_skl_fix = {c: [[] for _ in range(N_TIMEPOINTS)] for c in SKL_CLFS}
    m2_skl_adp = {c: [[] for _ in range(N_TIMEPOINTS)] for c in SKL_CLFS}
    m3_skl_fix = {c: [[] for _ in range(N_WINDOWS)]    for c in SKL_CLFS}
    m3_skl_adp = {c: [[] for _ in range(N_WINDOWS)]    for c in SKL_CLFS}

    labels_all = list(y_te_full)

    X_accum_m1 = []
    X_accum_m2 = [[] for _ in range(N_TIMEPOINTS)]
    X_accum_m3 = [[] for _ in range(N_WINDOWS)]
    y_accum    = []

    for trial_i in range(n_trials):
        y_trial   = y_te_full[trial_i]
        raw_trial = raw_te_full[trial_i:trial_i+1]

        # Covarianza del trial completo para M1 recentering
        cov_trial_full = build_template_covs(
            raw_trial, template_base)[0]

        # ── M1 MDM fijo ──────────────────────────────────────
        try:
            dists = mdm_m1_fixed.transform(cov_trial_full[np.newaxis])
            s = -dists[0, 1] if dists.shape[1] > 1 else -dists[0, 0]
        except Exception:
            s = 0.0
        m1_mdm_fix_scores.append(s)

        # ── M1 MDM adaptativo ────────────────────────────────
        try:
            d_rest = np.linalg.norm(
                cov_trial_full - centers_m1[REST_ID], 'fro')
            d_mi   = np.linalg.norm(
                cov_trial_full - centers_m1[MI_ID],   'fro')
            s_adp  = -(d_mi / (d_rest + d_mi + 1e-10))
        except Exception:
            s_adp = 0.0
        m1_mdm_adp_scores.append(s_adp)

        # Actualizar centroides M1 al final del trial
        for label in centers_m1:
            centers_m1[label] = geodesic_riemann(
                centers_m1[label], cov_trial_full, ALPHA)

        # ── M1 Sklearn ───────────────────────────────────────
        x_trial_m1 = X_te_full[trial_i]
        X_accum_m1.append(x_trial_m1)
        y_accum.append(y_trial)

        for c in SKL_CLFS:
            try:
                p_fix = skl_m1_fixed[c].predict_proba(
                    x_trial_m1.reshape(1,-1))[0, 1]
                p_adp = skl_m1_adapt[c].predict_proba(
                    x_trial_m1.reshape(1,-1))[0, 1]
            except Exception:
                p_fix = p_adp = 0.5
            m1_skl_fix[c].append(p_fix)
            m1_skl_adp[c].append(p_adp)

        # Reentrenar sklearn M1 cada RETRAIN_EVERY trials
        if len(X_accum_m1) % RETRAIN_EVERY == 0:
            X_comb = np.vstack([X_tr] + [np.array(X_accum_m1)])
            y_comb = np.concatenate([y_tr, np.array(y_accum)])
            for c in SKL_CLFS:
                try:
                    clf_new = make_clf(c)
                    clf_new.fit(X_comb, y_comb)
                    skl_m1_adapt[c] = clf_new
                except Exception:
                    pass

        # ── M2 — cada paso temporal ──────────────────────────
        for step in range(N_TIMEPOINTS):
            raw_step = raw_te_steps[step][trial_i:trial_i+1]
            tmpl_s   = raw_tr_steps[step].mean(axis=0)
            cov_s    = build_template_covs(raw_step, tmpl_s)[0]

            # MDM fijo
            try:
                dists = mdm_m2_fixed[step].transform(cov_s[np.newaxis])
                s = -dists[0, 1] if dists.shape[1] > 1 else -dists[0, 0]
            except Exception:
                s = 0.0
            m2_mdm_fix_scores[step].append(s)

            # MDM adaptativo
            try:
                d_rest = np.linalg.norm(
                    cov_s - centers_m2[step][REST_ID], 'fro')
                d_mi   = np.linalg.norm(
                    cov_s - centers_m2[step][MI_ID],   'fro')
                s_adp  = -(d_mi / (d_rest + d_mi + 1e-10))
            except Exception:
                s_adp = 0.0
            m2_mdm_adp_scores[step].append(s_adp)

            # Sklearn M2
            x_step = X_te_steps[step][trial_i]
            X_accum_m2[step].append(x_step)

            for c in SKL_CLFS:
                try:
                    p_fix = skl_m2_fixed[step][c].predict_proba(
                        x_step.reshape(1,-1))[0, 1]
                    p_adp = skl_m2_adapt[step][c].predict_proba(
                        x_step.reshape(1,-1))[0, 1]
                except Exception:
                    p_fix = p_adp = 0.5
                m2_skl_fix[c][step].append(p_fix)
                m2_skl_adp[c][step].append(p_adp)

            # Reentrenar sklearn M2 cada RETRAIN_EVERY trials
            if len(X_accum_m2[step]) % RETRAIN_EVERY == 0:
                X_comb = np.vstack(
                    [X_tr_steps[step]] + [np.array(X_accum_m2[step])])
                y_comb = np.concatenate([y_tr, np.array(y_accum)])
                for c in SKL_CLFS:
                    try:
                        clf_new = make_clf(c)
                        clf_new.fit(X_comb, y_comb)
                        skl_m2_adapt[step][c] = clf_new
                    except Exception:
                        pass

        # Actualizar centroides M2 al final del trial (ventana completa)
        for step in range(N_TIMEPOINTS):
            raw_step = raw_te_steps[step][trial_i:trial_i+1]
            tmpl_s   = raw_tr_steps[step].mean(axis=0)
            cov_s    = build_template_covs(raw_step, tmpl_s)[0]
            for label in centers_m2[step]:
                centers_m2[step][label] = geodesic_riemann(
                    centers_m2[step][label], cov_s, ALPHA)

        # ── M3 — cada ventana deslizante ─────────────────────
        for wi, (w_start, w_end) in enumerate(WINDOWS):
            raw_win = raw_te_wins[wi][trial_i:trial_i+1]
            tmpl_w  = raw_tr_wins[wi].mean(axis=0)
            cov_w   = build_template_covs(raw_win, tmpl_w)[0]

            # MDM fijo
            try:
                dists = mdm_m3_fixed[wi].transform(cov_w[np.newaxis])
                s = -dists[0, 1] if dists.shape[1] > 1 else -dists[0, 0]
            except Exception:
                s = 0.0
            m3_mdm_fix_scores[wi].append(s)

            # MDM adaptativo
            try:
                d_rest = np.linalg.norm(
                    cov_w - centers_m3[wi][REST_ID], 'fro')
                d_mi   = np.linalg.norm(
                    cov_w - centers_m3[wi][MI_ID],   'fro')
                s_adp  = -(d_mi / (d_rest + d_mi + 1e-10))
            except Exception:
                s_adp = 0.0
            m3_mdm_adp_scores[wi].append(s_adp)

            # Sklearn M3
            x_win = X_te_wins[wi][trial_i]
            X_accum_m3[wi].append(x_win)

            for c in SKL_CLFS:
                try:
                    p_fix = skl_m3_fixed[wi][c].predict_proba(
                        x_win.reshape(1,-1))[0, 1]
                    p_adp = skl_m3_adapt[wi][c].predict_proba(
                        x_win.reshape(1,-1))[0, 1]
                except Exception:
                    p_fix = p_adp = 0.5
                m3_skl_fix[c][wi].append(p_fix)
                m3_skl_adp[c][wi].append(p_adp)

            # Reentrenar sklearn M3 cada RETRAIN_EVERY trials
            if len(X_accum_m3[wi]) % RETRAIN_EVERY == 0:
                X_comb = np.vstack(
                    [X_tr_wins[wi]] + [np.array(X_accum_m3[wi])])
                y_comb = np.concatenate([y_tr, np.array(y_accum)])
                for c in SKL_CLFS:
                    try:
                        clf_new = make_clf(c)
                        clf_new.fit(X_comb, y_comb)
                        skl_m3_adapt[wi][c] = clf_new
                    except Exception:
                        pass

        # Actualizar centroides M3 al final del trial
        for wi, (w_start, w_end) in enumerate(WINDOWS):
            raw_win = raw_te_wins[wi][trial_i:trial_i+1]
            tmpl_w  = raw_tr_wins[wi].mean(axis=0)
            cov_w   = build_template_covs(raw_win, tmpl_w)[0]
            for label in centers_m3[wi]:
                centers_m3[wi][label] = geodesic_riemann(
                    centers_m3[wi][label], cov_w, ALPHA)

    # ── Calcular AUC final por modelo ────────────────────────
    def best_auc_across(scores_list, labels):
        """Mejor AUC entre todos los pasos/ventanas."""
        aucs = []
        for s in scores_list:
            aucs.append(final_auc(s, labels))
        return max(aucs), np.argmax(aucs)

    # M1
    m1_res = {
        "MDM_fix" : final_auc(m1_mdm_fix_scores, labels_all),
        "MDM_adp" : final_auc(m1_mdm_adp_scores, labels_all),
        "skl_fix" : {c: final_auc(m1_skl_fix[c], labels_all)
                     for c in SKL_CLFS},
        "skl_adp" : {c: final_auc(m1_skl_adp[c], labels_all)
                     for c in SKL_CLFS},
        "MDM_fix_curve": rolling_auc(m1_mdm_fix_scores, labels_all),
        "MDM_adp_curve": rolling_auc(m1_mdm_adp_scores, labels_all),
    }

    # M2 — mejor paso para cada clasificador
    m2_res = {"MDM_fix": {}, "MDM_adp": {}, "skl_fix": {}, "skl_adp": {}}
    for step in range(N_TIMEPOINTS):
        t = T_POINTS[step]
        auc_mf = final_auc(m2_mdm_fix_scores[step], labels_all)
        auc_ma = final_auc(m2_mdm_adp_scores[step], labels_all)
        m2_res["MDM_fix"][round(t,3)] = auc_mf
        m2_res["MDM_adp"][round(t,3)] = auc_ma
        for c in SKL_CLFS:
            if c not in m2_res["skl_fix"]:
                m2_res["skl_fix"][c] = {}
                m2_res["skl_adp"][c] = {}
            m2_res["skl_fix"][c][round(t,3)] = final_auc(
                m2_skl_fix[c][step], labels_all)
            m2_res["skl_adp"][c][round(t,3)] = final_auc(
                m2_skl_adp[c][step], labels_all)

    # M3 — mejor ventana para cada clasificador
    m3_res = {"MDM_fix": {}, "MDM_adp": {}, "skl_fix": {}, "skl_adp": {}}
    for wi, (w_start, w_end) in enumerate(WINDOWS):
        key = f"[{w_start:.2f},{w_end:.2f}]"
        m3_res["MDM_fix"][key] = final_auc(
            m3_mdm_fix_scores[wi], labels_all)
        m3_res["MDM_adp"][key] = final_auc(
            m3_mdm_adp_scores[wi], labels_all)
        for c in SKL_CLFS:
            if c not in m3_res["skl_fix"]:
                m3_res["skl_fix"][c] = {}
                m3_res["skl_adp"][c] = {}
            m3_res["skl_fix"][c][key] = final_auc(
                m3_skl_fix[c][wi], labels_all)
            m3_res["skl_adp"][c][key] = final_auc(
                m3_skl_adp[c][wi], labels_all)

    return dict(
        n_trials = n_trials,
        M1       = m1_res,
        M2       = m2_res,
        M3       = m3_res,
    )


# ============================================================
# EJECUTAR LOSO ADAPTIVE
# ============================================================

print(f"\n{'='*68}")
print("🔄  LEAVE-ONE-SUBJECT-OUT — ADAPTIVE RECENTERING")
print(f"{'='*68}")

results_all = {}

for i, (subj_eval, _) in enumerate(SUBJECTS):
    subj_train = [s for s, _ in SUBJECTS if s != subj_eval]
    label      = subj_eval.split("_")[-1]

    print(f"\n{'─'*50}")
    print(f"   Fold {i+1}/{len(SUBJECTS)} — evaluar: {label}")
    print(f"   Entrenar: {[s.split('_')[-1] for s in subj_train]}")
    print(f"{'─'*50}")

    r = run_adaptive_fold(subj_eval, subj_train)
    results_all[subj_eval] = r

    # Resumen del fold
    best_m1_fix = max(list(r["M1"]["skl_fix"].values()) + [r["M1"]["MDM_fix"]])
    best_m1_adp = max(list(r["M1"]["skl_adp"].values()) + [r["M1"]["MDM_adp"]])
    best_m2_fix = max(
        [max(r["M2"]["skl_fix"][c].values()) for c in SKL_CLFS] +
        [max(r["M2"]["MDM_fix"].values())])
    best_m2_adp = max(
        [max(r["M2"]["skl_adp"][c].values()) for c in SKL_CLFS] +
        [max(r["M2"]["MDM_adp"].values())])
    best_m3_fix = max(
        [max(r["M3"]["skl_fix"][c].values()) for c in SKL_CLFS] +
        [max(r["M3"]["MDM_fix"].values())])
    best_m3_adp = max(
        [max(r["M3"]["skl_adp"][c].values()) for c in SKL_CLFS] +
        [max(r["M3"]["MDM_adp"].values())])

    print(f"\n   {'Modelo':<6} {'M1 fijo':>8} {'M1 adp':>8} {'ΔM1':>6}  "
          f"{'M2 fijo':>8} {'M2 adp':>8} {'ΔM2':>6}  "
          f"{'M3 fijo':>8} {'M3 adp':>8} {'ΔM3':>6}")
    print("   " + "-"*75)

    # MDM
    mdm_m2_fix = max(r["M2"]["MDM_fix"].values())
    mdm_m2_adp = max(r["M2"]["MDM_adp"].values())
    mdm_m3_fix = max(r["M3"]["MDM_fix"].values())
    mdm_m3_adp = max(r["M3"]["MDM_adp"].values())
    print(f"   {'MDM':<6} "
          f"{r['M1']['MDM_fix']:>8.3f} {r['M1']['MDM_adp']:>8.3f} "
          f"{'+'if r['M1']['MDM_adp']>=r['M1']['MDM_fix'] else ''}"
          f"{r['M1']['MDM_adp']-r['M1']['MDM_fix']:>5.3f}  "
          f"{mdm_m2_fix:>8.3f} {mdm_m2_adp:>8.3f} "
          f"{'+'if mdm_m2_adp>=mdm_m2_fix else ''}"
          f"{mdm_m2_adp-mdm_m2_fix:>5.3f}  "
          f"{mdm_m3_fix:>8.3f} {mdm_m3_adp:>8.3f} "
          f"{'+'if mdm_m3_adp>=mdm_m3_fix else ''}"
          f"{mdm_m3_adp-mdm_m3_fix:>5.3f}")

    # Sklearn
    for c in SKL_CLFS:
        m1f = r["M1"]["skl_fix"][c]
        m1a = r["M1"]["skl_adp"][c]
        m2f = max(r["M2"]["skl_fix"][c].values())
        m2a = max(r["M2"]["skl_adp"][c].values())
        m3f = max(r["M3"]["skl_fix"][c].values())
        m3a = max(r["M3"]["skl_adp"][c].values())
        print(f"   {c[:6]:<6} "
              f"{m1f:>8.3f} {m1a:>8.3f} "
              f"{'+'if m1a>=m1f else ''}{m1a-m1f:>5.3f}  "
              f"{m2f:>8.3f} {m2a:>8.3f} "
              f"{'+'if m2a>=m2f else ''}{m2a-m2f:>5.3f}  "
              f"{m3f:>8.3f} {m3a:>8.3f} "
              f"{'+'if m3a>=m3f else ''}{m3a-m3f:>5.3f}")

    print(f"\n   Mejor fijo  : M1={best_m1_fix:.3f} M2={best_m2_fix:.3f} "
          f"M3={best_m3_fix:.3f}")
    print(f"   Mejor adapt : M1={best_m1_adp:.3f} M2={best_m2_adp:.3f} "
          f"M3={best_m3_adp:.3f}")


# ============================================================
# RESUMEN FINAL
# ============================================================

print(f"\n{'='*68}")
print("🚀  RESUMEN FINAL — ADAPTIVE RECENTERING M1, M2, M3")
print(f"{'='*68}")
print(f"   α={ALPHA} | reentrenar sklearn cada {RETRAIN_EVERY} trials\n")

# Tabla por sujeto
print(f"   {'Suj':<5} {'M1fix':>6} {'M1adp':>6} {'ΔM1':>5}  "
      f"{'M2fix':>6} {'M2adp':>6} {'ΔM2':>5}  "
      f"{'M3fix':>6} {'M3adp':>6} {'ΔM3':>5}")
print("   " + "-"*65)

m1_fix_all, m1_adp_all = [], []
m2_fix_all, m2_adp_all = [], []
m3_fix_all, m3_adp_all = [], []

for subj_eval, _ in SUBJECTS:
    label = subj_eval.split("_")[-1]
    r     = results_all[subj_eval]

    m1f = max(list(r["M1"]["skl_fix"].values()) + [r["M1"]["MDM_fix"]])
    m1a = max(list(r["M1"]["skl_adp"].values()) + [r["M1"]["MDM_adp"]])
    m2f = max([max(r["M2"]["skl_fix"][c].values()) for c in SKL_CLFS] +
              [max(r["M2"]["MDM_fix"].values())])
    m2a = max([max(r["M2"]["skl_adp"][c].values()) for c in SKL_CLFS] +
              [max(r["M2"]["MDM_adp"].values())])
    m3f = max([max(r["M3"]["skl_fix"][c].values()) for c in SKL_CLFS] +
              [max(r["M3"]["MDM_fix"].values())])
    m3a = max([max(r["M3"]["skl_adp"][c].values()) for c in SKL_CLFS] +
              [max(r["M3"]["MDM_adp"].values())])

    m1_fix_all.append(m1f); m1_adp_all.append(m1a)
    m2_fix_all.append(m2f); m2_adp_all.append(m2a)
    m3_fix_all.append(m3f); m3_adp_all.append(m3a)

    print(f"   {label:<5} {m1f:>6.3f} {m1a:>6.3f} "
          f"{'+'if m1a>=m1f else ''}{m1a-m1f:>4.3f}  "
          f"{m2f:>6.3f} {m2a:>6.3f} "
          f"{'+'if m2a>=m2f else ''}{m2a-m2f:>4.3f}  "
          f"{m3f:>6.3f} {m3a:>6.3f} "
          f"{'+'if m3a>=m3f else ''}{m3a-m3f:>4.3f}")

print("   " + "-"*65)
print(f"   {'AVG':<5} {np.mean(m1_fix_all):>6.3f} {np.mean(m1_adp_all):>6.3f} "
      f"{'+'if np.mean(m1_adp_all)>=np.mean(m1_fix_all) else ''}"
      f"{np.mean(m1_adp_all)-np.mean(m1_fix_all):>4.3f}  "
      f"{np.mean(m2_fix_all):>6.3f} {np.mean(m2_adp_all):>6.3f} "
      f"{'+'if np.mean(m2_adp_all)>=np.mean(m2_fix_all) else ''}"
      f"{np.mean(m2_adp_all)-np.mean(m2_fix_all):>4.3f}  "
      f"{np.mean(m3_fix_all):>6.3f} {np.mean(m3_adp_all):>6.3f} "
      f"{'+'if np.mean(m3_adp_all)>=np.mean(m3_fix_all) else ''}"
      f"{np.mean(m3_adp_all)-np.mean(m3_fix_all):>4.3f}")

# Tabla MDM específicamente
print(f"\n   MDM Riemanniano — promedio cross-subject:")
print(f"   {'Modelo':<6} {'M1fix':>6} {'M1adp':>6} {'ΔM1':>5}  "
      f"{'M2fix':>6} {'M2adp':>6} {'ΔM2':>5}  "
      f"{'M3fix':>6} {'M3adp':>6} {'ΔM3':>5}")
print("   " + "-"*65)
mdm_vals = {
    "M1fix": np.mean([results_all[s]["M1"]["MDM_fix"] for s,_ in SUBJECTS]),
    "M1adp": np.mean([results_all[s]["M1"]["MDM_adp"] for s,_ in SUBJECTS]),
    "M2fix": np.mean([max(results_all[s]["M2"]["MDM_fix"].values())
                      for s,_ in SUBJECTS]),
    "M2adp": np.mean([max(results_all[s]["M2"]["MDM_adp"].values())
                      for s,_ in SUBJECTS]),
    "M3fix": np.mean([max(results_all[s]["M3"]["MDM_fix"].values())
                      for s,_ in SUBJECTS]),
    "M3adp": np.mean([max(results_all[s]["M3"]["MDM_adp"].values())
                      for s,_ in SUBJECTS]),
}
print(f"   {'MDM':<6} "
      f"{mdm_vals['M1fix']:>6.3f} {mdm_vals['M1adp']:>6.3f} "
      f"{'+'if mdm_vals['M1adp']>=mdm_vals['M1fix'] else ''}"
      f"{mdm_vals['M1adp']-mdm_vals['M1fix']:>4.3f}  "
      f"{mdm_vals['M2fix']:>6.3f} {mdm_vals['M2adp']:>6.3f} "
      f"{'+'if mdm_vals['M2adp']>=mdm_vals['M2fix'] else ''}"
      f"{mdm_vals['M2adp']-mdm_vals['M2fix']:>4.3f}  "
      f"{mdm_vals['M3fix']:>6.3f} {mdm_vals['M3adp']:>6.3f} "
      f"{'+'if mdm_vals['M3adp']>=mdm_vals['M3fix'] else ''}"
      f"{mdm_vals['M3adp']-mdm_vals['M3fix']:>4.3f}")

print(f"\n   Referencia within-subject M2: 0.666 promedio")
print("="*68)


# ============================================================
# GUARDADO
# ============================================================

def clean_nan(obj):
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"adaptive_M1M2M3_{timestamp}"

path_json = os.path.join(SAVE_DIR, base_name + ".json")
path_txt  = os.path.join(SAVE_DIR, base_name + ".txt")

save_dict = {
    "timestamp"    : timestamp,
    "subjects"     : [s for s, _ in SUBJECTS],
    "channels"     : PICKS_LOSO,
    "alpha_mdm"    : ALPHA,
    "retrain_every": RETRAIN_EVERY,
    "results"      : {
        subj_eval.split("_")[-1]: results_all[subj_eval]
        for subj_eval, _ in SUBJECTS
    }
}

with open(path_json, "w") as f:
    json.dump(clean_nan(save_dict), f, indent=2)
print(f"\n💾  JSON: {path_json}")

with open(path_txt, "w") as f:
    f.write("CNV BCI — ADAPTIVE RECENTERING M1, M2, M3\n")
    f.write(f"{'='*68}\n")
    f.write(f"Timestamp  : {timestamp}\n")
    f.write(f"Sujetos    : {[s.split('_')[-1] for s,_ in SUBJECTS]}\n")
    f.write(f"Canales    : {PICKS_LOSO}\n")
    f.write(f"α MDM      : {ALPHA}\n")
    f.write(f"Reentrenar : cada {RETRAIN_EVERY} trials (sklearn)\n\n")

    for subj_eval, _ in SUBJECTS:
        label = subj_eval.split("_")[-1]
        r     = results_all[subj_eval]
        f.write(f"{'─'*50}\n")
        f.write(f"Sujeto: {label} | {r['n_trials']} trials\n")
        f.write(f"{'─'*50}\n")

        f.write(f"\nM1 — Estático:\n")
        f.write(f"{'Modelo':<14} {'Fijo':>7} {'Adapt':>7} {'Δ':>6}\n")
        f.write("-"*35 + "\n")
        f.write(f"{'MDM_Riemann':<14} {r['M1']['MDM_fix']:>7.3f} "
                f"{r['M1']['MDM_adp']:>7.3f} "
                f"{'+'if r['M1']['MDM_adp']>=r['M1']['MDM_fix'] else ''}"
                f"{r['M1']['MDM_adp']-r['M1']['MDM_fix']:>5.3f}\n")
        for c in SKL_CLFS:
            mf = r["M1"]["skl_fix"][c]
            ma = r["M1"]["skl_adp"][c]
            f.write(f"{c:<14} {mf:>7.3f} {ma:>7.3f} "
                    f"{'+'if ma>=mf else ''}{ma-mf:>5.3f}\n")

        f.write(f"\nM2 — Acumulativo (mejor paso):\n")
        f.write(f"{'Modelo':<14} {'Fijo':>7} {'t':>7} {'Adapt':>7} {'t':>7} {'Δ':>6}\n")
        f.write("-"*50 + "\n")
        mdm_m2f = max(r["M2"]["MDM_fix"].values())
        mdm_m2a = max(r["M2"]["MDM_adp"].values())
        t_m2f   = max(r["M2"]["MDM_fix"], key=r["M2"]["MDM_fix"].get)
        t_m2a   = max(r["M2"]["MDM_adp"], key=r["M2"]["MDM_adp"].get)
        f.write(f"{'MDM_Riemann':<14} {mdm_m2f:>7.3f} {t_m2f:>7} "
                f"{mdm_m2a:>7.3f} {t_m2a:>7} "
                f"{'+'if mdm_m2a>=mdm_m2f else ''}{mdm_m2a-mdm_m2f:>5.3f}\n")
        for c in SKL_CLFS:
            mf  = max(r["M2"]["skl_fix"][c].values())
            ma  = max(r["M2"]["skl_adp"][c].values())
            tf  = max(r["M2"]["skl_fix"][c], key=r["M2"]["skl_fix"][c].get)
            ta  = max(r["M2"]["skl_adp"][c], key=r["M2"]["skl_adp"][c].get)
            f.write(f"{c:<14} {mf:>7.3f} {tf:>7} {ma:>7.3f} {ta:>7} "
                    f"{'+'if ma>=mf else ''}{ma-mf:>5.3f}\n")

        f.write(f"\nM3 — Ventanas deslizantes (mejor ventana):\n")
        f.write(f"{'Modelo':<14} {'Fijo':>7} {'Adapt':>7} {'Δ':>6}\n")
        f.write("-"*35 + "\n")
        mdm_m3f = max(r["M3"]["MDM_fix"].values())
        mdm_m3a = max(r["M3"]["MDM_adp"].values())
        f.write(f"{'MDM_Riemann':<14} {mdm_m3f:>7.3f} {mdm_m3a:>7.3f} "
                f"{'+'if mdm_m3a>=mdm_m3f else ''}{mdm_m3a-mdm_m3f:>5.3f}\n")
        for c in SKL_CLFS:
            mf = max(r["M3"]["skl_fix"][c].values())
            ma = max(r["M3"]["skl_adp"][c].values())
            f.write(f"{c:<14} {mf:>7.3f} {ma:>7.3f} "
                    f"{'+'if ma>=mf else ''}{ma-mf:>5.3f}\n")
        f.write("\n")

    f.write(f"{'='*68}\n")
    f.write("RESUMEN PROMEDIO CROSS-SUBJECT\n")
    f.write(f"{'='*68}\n")
    f.write(f"{'Modelo':<14} {'M1fix':>7} {'M1adp':>7} {'ΔM1':>5}  "
            f"{'M2fix':>7} {'M2adp':>7} {'ΔM2':>5}  "
            f"{'M3fix':>7} {'M3adp':>7} {'ΔM3':>5}\n")
    f.write("-"*72 + "\n")
    for c in SKL_CLFS + ["MDM_Riemann"]:
        if c == "MDM_Riemann":
            m1f = np.mean([results_all[s]["M1"]["MDM_fix"] for s,_ in SUBJECTS])
            m1a = np.mean([results_all[s]["M1"]["MDM_adp"] for s,_ in SUBJECTS])
            m2f = np.mean([max(results_all[s]["M2"]["MDM_fix"].values())
                           for s,_ in SUBJECTS])
            m2a = np.mean([max(results_all[s]["M2"]["MDM_adp"].values())
                           for s,_ in SUBJECTS])
            m3f = np.mean([max(results_all[s]["M3"]["MDM_fix"].values())
                           for s,_ in SUBJECTS])
            m3a = np.mean([max(results_all[s]["M3"]["MDM_adp"].values())
                           for s,_ in SUBJECTS])
        else:
            m1f = np.mean([results_all[s]["M1"]["skl_fix"][c]
                           for s,_ in SUBJECTS])
            m1a = np.mean([results_all[s]["M1"]["skl_adp"][c]
                           for s,_ in SUBJECTS])
            m2f = np.mean([max(results_all[s]["M2"]["skl_fix"][c].values())
                           for s,_ in SUBJECTS])
            m2a = np.mean([max(results_all[s]["M2"]["skl_adp"][c].values())
                           for s,_ in SUBJECTS])
            m3f = np.mean([max(results_all[s]["M3"]["skl_fix"][c].values())
                           for s,_ in SUBJECTS])
            m3a = np.mean([max(results_all[s]["M3"]["skl_adp"][c].values())
                           for s,_ in SUBJECTS])
        f.write(f"{c:<14} {m1f:>7.3f} {m1a:>7.3f} "
                f"{'+'if m1a>=m1f else ''}{m1a-m1f:>4.3f}  "
                f"{m2f:>7.3f} {m2a:>7.3f} "
                f"{'+'if m2a>=m2f else ''}{m2a-m2f:>4.3f}  "
                f"{m3f:>7.3f} {m3a:>7.3f} "
                f"{'+'if m3a>=m3f else ''}{m3a-m3f:>4.3f}\n")
    f.write(f"\nReferencia within-subject: M2=0.666 promedio\n")
    f.write(f"{'='*68}\n")

print(f"📄  TXT: {path_txt}")
print(f"\n✅  Adaptive recentering M1+M2+M3 completo")