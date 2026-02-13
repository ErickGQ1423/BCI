"""
MockEEG.py
Simula un stream de EEG para probar el sistema sin el casco.
"""
import time
import random
from pylsl import StreamInfo, StreamOutlet

def main():
    # 1. Crear la información del stream (Debe coincidir con lo que busca tu driver)
    # Nombre: BioSemi, Tipo: EEG, Canales: 8, Hz: 250, Formato: float32
    info = StreamInfo('BioSemi', 'EEG', 8, 250, 'float32', 'myuid34234')

    # 2. Crear la "salida" (Outlet)
    outlet = StreamOutlet(info)

    print("✅ Simulador de EEG activo. Enviando datos falsos...")
    print("Ahora puedes correr tu ExperimentDriver.")

    # 3. Enviar ruido aleatorio infinitamente
    while True:
        # Generar 8 números aleatorios (simulando 8 canales)
        sample = [random.random() for _ in range(8)]
        
        # Enviar al sistema LSL
        outlet.push_sample(sample)
        
        # Esperar un poquito para simular los 250Hz (0.004s)
        time.sleep(0.004)

if __name__ == '__main__':
    main()