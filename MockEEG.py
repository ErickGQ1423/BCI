"""
MockEEG.py
Simulates an EEG stream to test the system without the headset.
"""
import time
import random
from pylsl import StreamInfo, StreamOutlet

def main():
    # 1. Create stream information (must match what your driver expects)
    # Name: BioSemi, Type: EEG, Channels: 8, Hz: 250, Format: float32
    info = StreamInfo('BioSemi', 'EEG', 8, 250, 'float32', 'myuid34234')

    # 2. Create the output outlet
    outlet = StreamOutlet(info)

    print("✅ EEG simulator active. Sending fake data...")
    print("You can now run your ExperimentDriver.")

    # 3. Send random noise indefinitely
    while True:
        # Generate 8 random numbers (simulating 8 channels)
        sample = [random.random() for _ in range(8)]
        
        # Send to the LSL system
        outlet.push_sample(sample)
        
        # Wait a bit to simulate 250 Hz (0.004 s)
        time.sleep(0.004)

if __name__ == '__main__':
    main()