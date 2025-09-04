import numpy as np
import json
import os
from pathlib import Path

# Assuming these modules exist from your project structure
from config import Config
from receiver.demodulator import BPSKDemodulator
from evaluation.metrics import calculate_ber
from evaluation.plotting import plot_constellation
def best_align_ber(decoded_bits: np.ndarray,
                   gt_bits: np.ndarray,
                   max_shift: int = 16):
    """
    Find the best small alignment (±max_shift) and optional inversion that
    minimizes BER between decoded_bits and gt_bits.
    Returns: (best_ber, best_shift, inverted_flag)
    """
    best = (1.0, 0, False)  # (ber, shift, inverted)
    if decoded_bits.size == 0:
        return best

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            L = min(decoded_bits.size - shift, gt_bits.size)
            if L <= 0: 
                continue
            d = decoded_bits[shift:shift + L]
            g = gt_bits[:L]
        else:
            L = min(decoded_bits.size, gt_bits.size + shift)
            if L <= 0:
                continue
            d = decoded_bits[:L]
            g = gt_bits[-shift:-shift + L]

        ber = np.mean(d != g)
        ber_flip = np.mean((1 - d) != g)

        if ber < best[0]:
            best = (ber, shift, False)
        if ber_flip < best[0]:
            best = (ber_flip, shift, True)

    return best



def load_dataset(sample_dir):
    """Loads a single sample directory, explicitly finding rx.npy and a .json file."""
    sample_data = {}

    # --- FOOLPROOF FILE LOADING ---
    # Construct the explicit path to the 'rx.npy' file to avoid ambiguity
    # with other .npy files like 'decoded_bits.npy'.
    npy_path = sample_dir / 'rx.npy'

    # Check if the specific file exists and load it.
    if npy_path.exists():
        sample_data['rx_samples'] = np.load(npy_path)

    # Find and load the corresponding .json metadata file.
    json_file = next(sample_dir.glob('*.json'), None)
    if json_file:
        with open(json_file, 'r') as f:
            sample_data.update(json.load(f))

    return sample_data


def main():
    """Main processing pipeline for CubeSat receiver challenge."""
    config = Config()
    for phase in range(1, 5):
        print(f"\n=== Processing Phase {phase} ===")
        process_phase(phase, config)


def process_phase(phase_num, config):
    """Process a specific challenge phase with acquire→track Costas loop and aligned BER."""
    # ---- Tunables for this run (safe defaults for low SNR) ----
    PREAMBLE_LEN = 800          # longer warm-up at low SNR (skip these in BER)
    ACQUIRE_BW   = 0.06         # wide Costas BW to grab CFO on preamble
    TRACK_BW     = 0.003       # narrow Costas BW for data tracking (less jitter)
    VERBOSE_FIRST = True         # print demod diagnostics for the first sample only
    SHOW_CONSTELLATION_FOR_FIRST = True
    # -----------------------------------------------------------

    # Phase configuration
    phase_config = config.get_phase_config(phase_num)
    dataset_path = Path(phase_config['dataset_path'])
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # Loop bandwidth defaults (used to construct the demod; per-call we override for acquire→track)
    timing_loop_bw = phase_config.get('timing_bw', 0.002)
    phase_loop_bw  = phase_config.get('phase_bw', 0.02)

    # Initialize demodulator (consider sps=8 for easier lock if your data supports it)
    demodulator = BPSKDemodulator(
        samples_per_symbol=phase_config['samples_per_symbol'],
        timing_bw=timing_loop_bw,
        phase_bw=phase_loop_bw
    )

    # Find sample directories
    sample_dirs = sorted({p.parent for p in dataset_path.glob('**/rx.npy')})
    if not sample_dirs:
        print(f"No valid sample data (rx.npy) found in {dataset_path}")
        return

    phase_results = []

    # ---------------------------
    # Diagnostic run on first sample
    # ---------------------------
    diag_dir = sorted(sample_dirs)[0]
    print(f"=== Diagnostic run on {diag_dir.relative_to(dataset_path)} ===")
    try:
        sample_data = load_dataset(diag_dir)
        if 'rx_samples' not in sample_data:
            print("rx.npy missing, skipping diagnostics.")
        else:
            # Use acquire→track and longer preamble for the diagnostic too
            bits, corrected = demodulator.process(
                sample_data,
                preamble_len=PREAMBLE_LEN,
                acquire_phase_bw=ACQUIRE_BW,
                track_phase_bw=TRACK_BW,
                verbose=True
            )

            gt = np.array(sample_data.get("ground_truth_bits", [])).ravel().astype(np.uint8)
            print("len(bits)=", len(bits), "len(gt)=", len(gt))
            if gt.size > PREAMBLE_LEN and len(bits) > 0:
                gt_data = gt[PREAMBLE_LEN:]
                gt_window = gt_data[:len(bits) + 32]  # allow margin

                best_ber, best_shift, inverted = best_align_ber(bits, gt_window, max_shift=16)
                print(f"ALIGN: shift={best_shift:+d}, inverted={inverted}, BER={best_ber:.4e}")
            else:
                print("Not enough ground truth bits for aligned BER comparison")


            # Constellation plot (first 200 corrected symbols)
            if SHOW_CONSTELLATION_FOR_FIRST:
                try:
                    import matplotlib.pyplot as plt
                    cs = corrected[:200]
                    plt.scatter(np.real(cs), np.imag(cs), s=6)
                    plt.axhline(0, color='k'); plt.axvline(0, color='k')
                    plt.title(f"Constellation (first 200 syms) - {diag_dir.name}")
                    plt.show()
                except Exception as e:
                    print(f"(Constellation plot skipped: {e})")
    except Exception as e:
        print(f"Error in diagnostic run: {e}")

    # ---------------------------
    # Full phase processing (acquire→track, aligned BER, save outputs)
    # ---------------------------
    for i, sample_dir in enumerate(sorted(sample_dirs)):
        print(f"Processing {sample_dir.relative_to(dataset_path)}...")
        try:
            sample_data = load_dataset(sample_dir)
            if 'rx_samples' not in sample_data:
                print(f"  Skipping {sample_dir.name}: 'rx.npy' file not loaded properly.")
                continue

            bits, corrected_symbols = demodulator.process(
                sample_data,
                preamble_len=PREAMBLE_LEN,
                acquire_phase_bw=ACQUIRE_BW,
                track_phase_bw=TRACK_BW,
                verbose=VERBOSE_FIRST and i == 0  # verbose only for the first one
            )

            # --- Aligned BER (skip preamble)   ---
            ber = None
            if 'ground_truth_bits' in sample_data and bits.size > 0:
                gt_all = np.array(sample_data['ground_truth_bits']).ravel().astype(np.uint8)
                if gt_all.size > PREAMBLE_LEN:
                    gt_data = gt_all[PREAMBLE_LEN:]
                    gt_window = gt_data[:bits.size + 32]

                    best_ber, best_shift, inverted = best_align_ber(bits, gt_window, max_shift=16)
                    print(f"  ALIGN: shift={best_shift:+d}, inverted={inverted}, BER={best_ber:.2e}")
                    ber = best_ber


            if ber is not None:
                print(f"  BER: {ber:.2e}")
                phase_results.append({
                    'sample': sample_dir.name,
                    'ber': ber,
                    'snr': sample_data.get('snr', None)
                })
            else:
                print("  BER: (ground truth unavailable or not long enough)")

            # Save decoded bits (data portion only; preamble already skipped inside demod)
            save_decoded_bits(sample_dir, bits)

            # Optional: plot constellation for the first processed sample
            if i == 0 and SHOW_CONSTELLATION_FOR_FIRST:
                try:
                    plot_constellation(corrected_symbols, f"Constellation for {sample_dir.name}")
                except Exception as e:
                    print(f"(Constellation plot skipped: {e})")

        except Exception as e:
            print(f"  Error processing {sample_dir.name}: {e}")

    if phase_results:
        generate_phase_report(phase_num, phase_results, phase_config)


def save_decoded_bits(sample_dir, decoded_bits):
    """Save decoded bits as required by challenge."""
    output_path = sample_dir / 'decoded_bits.npy'
    np.save(output_path, decoded_bits.astype(np.uint8))


def generate_phase_report(phase_num, results, config):
    """Generate performance report for phase."""
    print(f"\n--- Phase {phase_num} Summary ---")
    bers = [r['ber'] for r in results if r['ber'] is not None]
    if not bers:
        print("No BERs computed for this phase.")
        return
    avg_ber = np.mean(bers)
    print(f"Average BER: {avg_ber:.2e}")

    threshold = config.get('performance_threshold', 1e-2)  # default if not in config
    print(f"Threshold: {threshold:.2e}")
    if avg_ber <= threshold:
        print("✓ PHASE PASSED")
    else:
        print("✗ Phase failed - needs improvement")


if __name__ == "__main__":
    main()
