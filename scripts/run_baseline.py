import numpy as np
from receiver.io import load_dataset

def ber(a, b):
    n = min(len(a), len(b))
    return np.mean((a[:n] ^ b[:n]).astype(np.uint8))

def rectangular_mf(x, sps=2):
    h = np.ones(sps) / sps
    return np.convolve(x, h, mode="same")

def brute_force_timing(x, sps=2):
    # Try all possible sample phases [0..sps-1], pick the one with largest eye opening
    best_phase, best_score, best_syms = 0, -1, None
    for k in range(sps):
        syms = x[k::sps]
        # Score: absolute mean / std (SNR-ish). Use real part since BPSK is real after correction.
        score = np.abs(np.mean(np.real(syms))) / (np.std(np.real(syms)) + 1e-8)
        if score > best_score:
            best_score, best_phase, best_syms = score, k, syms
    return best_syms, best_phase, best_score

def coarse_cfo_squaring(x, sps=2):
    """
    BPSK: square the signal -> data is removed, remaining tone at 2*f_off.
    Estimate with 1-lag method; divide by 2. Returns corrected signal and f_hat (rad/sample).
    """
    y = x**2
    # Phase slope estimator (arg of autocorrelation at lag 1)
    phi = np.angle(np.sum(y[1:] * np.conj(y[:-1])))
    w2 = phi  # rad/sample for the squared signal
    w = w2 / 2.0
    n = np.arange(len(x), dtype=np.float64)
    x_corr = x * np.exp(-1j * w * n)
    return x_corr, w

def fine_phase_from_symbols(syms):
    # Phase = angle of mean symbol (robust if BER not terrible)
    m = np.mean(syms)
    return np.angle(m) if np.abs(m) > 1e-12 else 0.0

def decide_bits_bpsk(syms):
    return (np.real(syms) > 0).astype(np.uint8)

def run_baseline(folder, sps=2):
    rx, meta = load_dataset(folder)
    ref_bits = np.array(meta.get("bits", []), dtype=np.uint8)

    # 0) Coarse CFO first (important if Doppler is big)
    rx_cfo, w_hat = coarse_cfo_squaring(rx, sps=sps)
    # 1) Matched filter
    mf = rectangular_mf(rx_cfo, sps=sps)
    # 2) Brute-force timing
    syms, phase, score = brute_force_timing(mf, sps=sps)
    # 3) Fine phase (one-shot, not a loop)
    theta = fine_phase_from_symbols(syms)
    syms_rot = syms * np.exp(-1j * theta)
    # 4) Decide
    bits = decide_bits_bpsk(syms_rot)

    # Optional: auto-flip if inverted (BPSK 180° ambiguity)
    if ref_bits.size:
        e = ber(bits, ref_bits)
        if e > 0.45:
            bits = 1 - bits
            e = ber(bits, ref_bits)
        print(f"[INFO] CFO rad/samp ~ {w_hat:.4e}, best timing phase={phase}, score={score:.3f}, fine phase={theta:.3f} rad")
        print(f"[RESULT] BER baseline = {e:.3e}")
    else:
        print("[WARN] No reference bits in meta.json; cannot compute BER.")
    return bits

if __name__ == "__main__":
    folder = "../cubesat_dataset/phase1_timing/sample1"  # change to your path
    run_baseline(folder, sps=2)
