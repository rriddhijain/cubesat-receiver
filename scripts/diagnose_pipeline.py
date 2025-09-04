import numpy as np
import matplotlib.pyplot as plt
from receiver.io import load_dataset
from receiver.synchronization import TimingRecovery
from receiver.synchronization import FrequencyRecovery

def bits_from_symbols(symbols):
    """
    Hard-decision slicer for BPSK.
    Maps real(symbol) > 0 → 1, else 0.
    """
    return (np.real(symbols) > 0).astype(int)

def compute_ber(bits, ref_bits):
    errors = np.sum(bits[:len(ref_bits)] != ref_bits)
    return errors / len(ref_bits)

def diagnose(folder, sps=2):
    # --- Load dataset ---
    rx, meta = load_dataset(folder)
    ref_bits = np.array(meta["bits"])  # ground truth provided in meta.json?

    print(f"[INFO] Loaded {len(rx)} samples, target SNR={meta.get('snr_db', 'unknown')} dB")

    # --- Stage 1: Matched Filter ---
    mf = np.ones(sps) / sps
    mf_out = np.convolve(rx, mf, mode="same")

    plt.figure()
    plt.scatter(np.real(mf_out[:2000]), np.imag(mf_out[:2000]), s=5)
    plt.title("Constellation after Matched Filter")
    plt.grid(True)

    # Test naive downsample BER
    symbols_ds = mf_out[::sps]
    bits_ds = bits_from_symbols(symbols_ds)
    ber_ds = compute_ber(bits_ds, ref_bits)
    print(f"BER after naive downsample (no loops): {ber_ds:.3e}")

    # --- Stage 2: Timing Recovery ---
    timing = TimingRecovery(samples_per_symbol=sps, loop_bandwidth=0.001)
    aligned = timing.recover(mf_out)

    plt.figure()
    plt.scatter(np.real(aligned[:2000]), np.imag(aligned[:2000]), s=5)
    plt.title("Constellation after Timing Recovery")
    plt.grid(True)

    bits_timing = bits_from_symbols(aligned)
    ber_timing = compute_ber(bits_timing, ref_bits)
    print(f"BER after timing recovery: {ber_timing:.3e}")

    # --- Stage 3: Frequency Recovery ---
    freqrec = FrequencyRecovery(loop_bandwidth=0.001)
    corrected = freqrec.recover(aligned)

    plt.figure()
    plt.scatter(np.real(corrected[:2000]), np.imag(corrected[:2000]), s=5)
    plt.title("Constellation after Frequency Recovery")
    plt.grid(True)

    bits_final = bits_from_symbols(corrected)
    ber_final = compute_ber(bits_final, ref_bits)
    print(f"BER after frequency recovery: {ber_final:.3e}")

    plt.show()

if __name__ == "__main__":
    # Example: change to one dataset sample path
    folder = "../cubesat_dataset/phase1_timing/sample1"
    diagnose(folder, sps=2)
