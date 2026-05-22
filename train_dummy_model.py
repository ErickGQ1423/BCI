import pickle
import numpy as np
from pyriemann.classification import MDM

# Simular matrices de covarianza de 5 canales
n_trials = 20
n_channels = 5  # FC5, FC1, Cz, CP1, Fz
rng = np.random.RandomState(42)

# Generar matrices SPD aleatorias
covs = []
for _ in range(n_trials):
    A = rng.randn(n_channels, n_channels)
    C = A @ A.T + np.eye(n_channels) * 0.1
    covs.append(C)
covs = np.array(covs)

# Labels alternados Rest/MI
labels = np.array([100, 200] * (n_trials // 2))

# Entrenar MDM
model = MDM(metric="riemann")
model.fit(covs, labels)

# Guardar
import os
os.makedirs("/home/lab-admin/Documents/CurrentStudy/sub-S26CLASS_SUBJ_008/models", exist_ok=True)
with open("/home/lab-admin/Documents/CurrentStudy/sub-S26CLASS_SUBJ_008/models/sub-S26CLASS_SUBJ_008_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo dummy guardado")