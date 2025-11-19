import os
import config

eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
print("Buscando en:", eeg_dir)

files = [f for f in os.listdir(eeg_dir) if f.endswith(".xdf")]
print("Archivos XDF encontrados:", files)