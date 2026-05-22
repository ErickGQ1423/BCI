"""
MockEEG_CNV.py
Simula stream EEG con canales CNV para probar la interfaz online sin headset.
"""
import time
import random
import numpy as np
from pylsl import StreamInfo, StreamOutlet

def main():
    # Canales que espera EEGStreamState para CNV
    ch_names = [
        'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
        'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
        'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'M1', 'M2'
    ]
    n_channels = len(ch_names)
    fs = 512

    # Crear stream con metadatos de canales
    info = StreamInfo('MockEEG', 'EEG', n_channels, fs, 'float32', 'mock_cnv_001')

    # Agregar nombres de canales a los metadatos
    chns = info.desc().append_child("channels")
    for ch in ch_names:
        chn = chns.append_child("channel")
        chn.append_child_value("label", ch)
        chn.append_child_value("type", "EEG")
        chn.append_child_value("unit", "microvolts")

    outlet = StreamOutlet(info)

    print(f"✅ MockEEG CNV activo — {n_channels} canales a {fs} Hz")
    print(f"   Canales: {ch_names}")
    print("   Puedes correr ExperimentDriver_Online.py ahora.")

    while True:
        # Ruido aleatorio simulando µV
        sample = [random.gauss(0, 10) for _ in range(n_channels)]
        outlet.push_sample(sample)
        time.sleep(1.0 / fs)

if __name__ == '__main__':
    main()