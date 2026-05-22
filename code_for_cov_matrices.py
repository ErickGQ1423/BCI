# Constants

N_EEG_CHANNELS = 32
N_SAMPLES_PER_WINDOW = 512

N_WINDOWS_PER_SUBJECT_1 = 200
N_WINDOWS_PER_SUBJECT_2 = 100

SHRINKAGE_VALUE = 0.02 # This is what Arman uses, depending of the data this might need to be adjusted

# Imports

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Shrinkage
from pyriemann.preprocessing import Whitening

# Create fake data (several subjects)
eeg_window_subject_1 = np.random.randn(N_WINDOWS_PER_SUBJECT_1, N_EEG_CHANNELS, N_SAMPLES_PER_WINDOW)
eeg_window_subject_2 = np.random.randn(N_WINDOWS_PER_SUBJECT_2, N_EEG_CHANNELS, N_SAMPLES_PER_WINDOW)

labels_subject_1 = np.random.randint(0, 1, N_WINDOWS_PER_SUBJECT)
labels_subject_2 = np.random.randint(0, 1, N_WINDOWS_PER_SUBJECT)

list_subjects_eeg = [eeg_window_subject_1, eeg_window_subject_2]
list_subjects_labels = [labels_subject_1, labels_subject_2]

# Create cov matrices (windows -> cov matrices -> norm cov matrices -> shrink norm cov matrices -> recentered shrink norm cov matrices)
# We are doing all operations (shrinkage, recentering) per subject
list_subject_features = []
list_subject_labels = []
for idx, subject in enumerate(list_subjects_eeg):
    this_labels = list_subjects_labels[idx]

    # windows -> cov matrices (N_WINDOWS_PER_SUBJECT, N_EEG_CHANNELS, N_SAMPLES_PER_WINDOW) -> (N_WINDOWS_PER_SUBJECT, N_EEG_CHANNELS, N_EEG_CHANNELS)
    this_subject_features = np.array([window @ window.T for window in subject])

    # cov matrices -> norm cov matrices (same input and output shape)
    this_subject_features = np.array([cov / np.trace(cov) for cov in this_subject_features])

    # norm cov matrices -> shrink norm cov matrices (same input and output shape)
    this_subject_shrinker = Shrinkage(shrinkage=SHRINKAGE_VALUE)
    this_subject_features = this_subject_shrinker.fit_transform(this_subject_features)

    # Note:
    #   - You can use other shrinkage methods that might work better or worse (e.g. Ledoit-Wolf, OAS)
    #     Depending on what you do this code might change a bit (e.g. Ledoit works with windows not cov matrices)

    # shrink norm cov matrices -> recentered shrink norm cov matrices (same input and output shape)
    this_subject_whitener = Whitening(metric='riemann')
    this_subject_features = this_subject_whitener.fit_transform(this_subject_features)

    # Save this subject
    list_subject_features.append(this_subject_features)
    list_subject_labels.append(this_labels)

# Create matrix of features (N_WINDOWS_PER_SUBJECT_1 + N_WINDOWS_PER_SUBJECT_2, N_EEG_CHANNELS, N_EEG_CHANNELS)
all_features = np.concatenate(list_subject_features, axis=0)

# Create matrix of labels (N_WINDOWS_PER_SUBJECT_1 + N_WINDOWS_PER_SUBJECT_2)
all_labels = np.concatenate(list_subject_labels, axis=0)

# Train MDM
model = MDM(metric='riemann')
model = model.fit(all_features, all_labels)

# Note:
#   - For testing offline you can do the same pipeline, for online you can't, as instead of recenter in group
#     you need to estimate that recentering operation (the convergence thing we talked about) (see Arman's code)