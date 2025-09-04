# test_demod_debug.py
import numpy as np
import matplotlib.pyplot as plt
from receiver.demodulator import BPSKDemodulator
from receiver.synchronization import TimingRecovery, FrequencyRecovery  # optional access

# RRC helper (same convention as demodulator)
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1 - beta + 4*beta/np.pi
        elif abs(abs(4*beta*ti) - 1.0) < 1e-6:
            term1 = (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
            term2 = (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            h[i] = (beta/np.sqrt(2))*(term1 + term2)
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    return h/np.sqrt(np.sum(h**2))

def simulate_and_test(
    sps=8, num_data=2000, preamble_len=400,
    beta=0.35, span=10, freq_offset=0.01, noise_std=0.08,
    timing_bw=None, phase_bw=None):

    # 1) TX: preamble + random payload
    preamble_bits = np.tile([1,0], preamble_len//2).astype(np.uint8)  # alternating known pattern
    data_bits = np.random.randint(0,2,num_data).astype(np.uint8)
    tx_bits = np.concatenate([preamble_bits, data_bits]).astype(np.uint8)

    # BPSK mapping: bit 0 -> +1, bit 1 -> -1  (match your demodulator decision mapping)
    tx_symbols = (1 - 2*tx_bits).astype(np.complex64)

    # Upsample
    tx_ups = np.zeros(len(tx_symbols)*sps, dtype=np.complex64)
    tx_ups[::sps] = tx_symbols

    # RRC transmit filter
    rrc = rrc_filter(beta=beta, sps=sps, span=span)
    from scipy.signal import lfilter
    tx_filtered = lfilter(rrc, 1, tx_ups)

    # Channel: freq offset + AWGN
    n = np.arange(len(tx_filtered))
    rx = tx_filtered * np.exp(1j*freq_offset*n)
    rx += noise_std*(np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))

    # 2) Run your demodulator but also extract intermediate corrected symbols
    demod = BPSKDemodulator(samples_per_symbol=sps)

    # Optionally override loop bandwidths if demod supports arguments or by replacing loops:
    if timing_bw is not None:
        demod.timing_recovery = TimingRecovery(sps, loop_bandwidth=timing_bw)
    if phase_bw is not None:
        demod.freq_recovery = FrequencyRecovery(loop_bandwidth=phase_bw)

    # Prepare sample_data
    sample_data = {"rx_samples": rx, "ground_truth_bits": tx_bits}

    # Run demodulator.process() to get bits (and if process() returns symbols, it'll be used)
    out = demod.process(sample_data)
    if isinstance(out, tuple) and len(out) == 2:
        bits, _ = out
    else:
        bits = out

    # Also compute corrected_symbols by running loops directly so we can inspect:
    demod.timing_recovery.reset()
    demod.freq_recovery.reset()
    mf = np.convolve(rx, demod.matched_filter, mode="same")
    timed = demod.timing_recovery.recover(mf)
    corrected_symbols = demod.freq_recovery.recover(timed)

    # Align true bits to decisions (skip preamble region)
    true_data = tx_bits[preamble_len : preamble_len + len(bits)]

    print("Lengths: decided_bits =", len(bits), "true_data =", len(true_data),
          "timed symbols =", len(timed), "corrected_symbols =", len(corrected_symbols))

    # BER and flipped BER
    if len(true_data) == 0:
        print("No overlapping bits - increase num_data or reduce preamble_len.")
        return None

    ber = np.mean(bits != true_data)
    ber_flipped = np.mean((1-bits) != true_data)
    print(f"BER: {ber:.4f}  BER_if_flipped: {ber_flipped:.4f}")

    # Quick diagnostics print
    cs_head = corrected_symbols[:60]
    print("first 20 corrected symbols (real):", np.round(np.real(cs_head[:20]), 4))
    print("first 20 corrected symbols (imag):", np.round(np.imag(cs_head[:20]), 4))
    print("mean abs corrected symbol:", np.mean(np.abs(corrected_symbols)))
    phases = np.angle(corrected_symbols + 1e-12)
    print("phase (min/mean/max):", np.min(phases), np.mean(phases), np.max(phases))

    # Constellation plot
    plt.figure(figsize=(4,4))
    plt.scatter(np.real(cs_head), np.imag(cs_head), s=12)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.title(f"Constellation (first {len(cs_head)} symbols) BER={ber:.4f}")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.grid(True)
    plt.show()

    return {
        "ber": ber, "ber_flipped": ber_flipped,
        "bits": bits, "true_data": true_data,
        "corrected_symbols": corrected_symbols
    }

if __name__ == "__main__":
    out = simulate_and_test()
