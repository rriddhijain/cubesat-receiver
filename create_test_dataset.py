import numpy as np
import json
import os
from pathlib import Path

def create_test_dataset():
    """Create synthetic dataset for testing."""
    
    # Create directory structure
    base_path = Path("cubesat_dataset")
    
    phases = [
        ("phase1_timing", ["snr_10db", "snr_5db", "snr_0db"]),
        ("phase2_snr", ["snr_0db", "snr_10db"]),
        ("phase3_coding", ["reed_solomon", "convolutional"]),
        ("phase4_doppler", ["doppler_100hz", "doppler_500hz"])
    ]
    
    for phase_dir, sample_types in phases:
        phase_path = base_path / phase_dir
        
        for sample_type in sample_types:
            for sample_num in range(3):  # Create 3 samples each
                sample_dir = phase_path / f"{sample_type}" / f"sample_{sample_num:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate synthetic signal
                num_bits = 1000
                samples_per_symbol = 4
                snr_db = 10 if "10db" in sample_type else (5 if "5db" in sample_type else 0)
                
                # Create BPSK signal
                bits = np.random.randint(0, 2, num_bits)
                symbols = 2 * bits - 1  # Map to +1/-1
                upsampled = np.repeat(symbols, samples_per_symbol)
                
                # Add noise
                signal_power = np.mean(np.abs(upsampled)**2)
                noise_power = signal_power / (10**(snr_db/10))
                noise = np.sqrt(noise_power/2) * (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled)))
                
                rx_signal = upsampled + noise
                
                # Save rx.npy
                np.save(sample_dir / "rx.npy", rx_signal)
                
                # Save meta.json
                metadata = {
                    "snr": snr_db,
                    "num_bits": num_bits,
                    "samples_per_symbol": samples_per_symbol,
                    "ground_truth_bits": bits.tolist()
                }
                
                with open(sample_dir / "meta.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Created {sample_dir}")

if __name__ == "__main__":
    create_test_dataset()
    print("Test dataset created!")