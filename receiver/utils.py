import numpy as np
import json
from pathlib import Path

def load_dataset(sample_dir: Path):
    """
    Load dataset sample from a directory.
    Finds the first .npy file (ignoring 'decoded_bits.npy') and the first .json file.
    """
    
    data = {}
    print(f"DEBUG: --- Processing directory: {sample_dir.name} ---")
    
    # Find and load the first .npy file in the directory, IGNORING previous output
    try:
        print("DEBUG: Searching for .npy file...")
        # Find all .npy files that are NOT named 'decoded_bits.npy'
        valid_npy_files = [p for p in sample_dir.glob('*.npy') if p.name != 'decoded_bits.npy']
        
        if not valid_npy_files:
            raise StopIteration # No valid input files found

        rx_path = valid_npy_files[0] # Use the first valid file
        print(f"DEBUG: Found valid .npy file: {rx_path.name}")
        data['rx_samples'] = np.load(rx_path)
        print(f"DEBUG: Loaded 'rx_samples' with shape {data['rx_samples'].shape}")

    except StopIteration:
        print("DEBUG: No valid input .npy file found in this directory.")
        pass

    # Find and load the first .json file in the directory
    try:
        print("DEBUG: Searching for .json file...")
        meta_path = next(sample_dir.glob('*.json'))
        print(f"DEBUG: Found .json file: {meta_path.name}")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            data.update(metadata)
            print(f"DEBUG: Loaded metadata with keys: {list(metadata.keys())}")
    except StopIteration:
        print("DEBUG: No .json file found in this directory.")
        pass
    
    print(f"DEBUG: Final loaded data keys: {list(data.keys())}\n")
    return data

def generate_test_signal(num_bits=1000, samples_per_symbol=4, snr_db=10):
    """Generate test BPSK signal for development."""
    
    # Generate random bits
    bits = np.random.randint(0, 2, num_bits)
    
    # Map to BPSK symbols (+1, -1)
    symbols = 2 * bits - 1
    
    # Upsample
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    # Add noise
    signal_power = np.mean(np.abs(upsampled)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled)))
    
    noisy_signal = upsampled + noise
    
    return {
        'rx_samples': noisy_signal,
        'ground_truth_bits': bits,
        'snr': snr_db
    }

