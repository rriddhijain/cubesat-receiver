#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import json

import numpy as np

# Project imports (assumed to exist)
from config import Config
from receiver.demodulator import BPSKDemodulator
from evaluation.metrics import calculate_ber
from evaluation.plotting import plot_constellation


# -------------------------
# Alignment helpers
# -------------------------
def best_align_ber(decoded_bits: np.ndarray,
                   gt_bits: np.ndarray,
                   max_shift: int = 16):
    """
    Find the best small alignment (±max_shift) and optional inversion that
    minimizes BER between decoded_bits and gt_bits.

    Returns: (best_ber or None, best_shift, inverted_flag)

    If no overlap found or either array empty -> returns (None, 0, False)
    """
    if decoded_bits is None or gt_bits is None:
        return (None, 0, False)

    if decoded_bits.size == 0 or gt_bits.size == 0:
        return (None, 0, False)

    best = (1.0, 0, False)  # (ber, shift, inverted)
    found = False

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            L = min(decoded_bits.size - shift, gt_bits.size)
            if L <= 0:
                continue
            d = decoded_bits[shift:shift + L]
            g = gt_bits[:L]
        else:
            L = min(decoded_bits.size, gt_bits.size + shift)  # shift < 0 so gt.size+shift < gt.size
            if L <= 0:
                continue
            d = decoded_bits[:L]
            g = gt_bits[-shift:-shift + L]

        found = True

        # Prefer calculate_ber if available (keeps behavior consistent)
        try:
            ber = float(calculate_ber(d, g))
            ber_flip = float(calculate_ber(1 - d, g))
        except Exception:
            # Fallback: simple mean for boolean mismatch
            ber = float(np.mean(d != g))
            ber_flip = float(np.mean((1 - d) != g))

        if ber < best[0]:
            best = (ber, shift, False)
        if ber_flip < best[0]:
            best = (ber_flip, shift, True)

    if not found:
        return (None, 0, False)
    return best


def apply_alignment(bits: np.ndarray, shift: int, inverted: bool):
    """
    Apply a simple alignment (shift + optional inversion) to 'bits'.
    The aligned output will have the same length as the input 'bits'.
    - shift > 0: drop the first 'shift' samples (i.e., shift left)
    - shift < 0: pad with zeros at beginning (-shift)
    This choice keeps the array length fixed and is suitable for saving/debugging.
    """
    if bits is None:
        return None
    if shift == 0:
        aligned = bits.copy()
    elif shift > 0:
        if shift >= bits.size:
            aligned = np.zeros_like(bits)
        else:
            aligned = bits[shift:]
            # pad end to keep same length
            pad_len = bits.size - aligned.size
            if pad_len > 0:
                aligned = np.concatenate([aligned, np.zeros(pad_len, dtype=aligned.dtype)])
    else:  # shift < 0
        pad_left = -shift
        if pad_left >= bits.size:
            aligned = np.zeros_like(bits)
        else:
            aligned = np.concatenate([np.zeros(pad_left, dtype=bits.dtype), bits])[:bits.size]

    if inverted:
        aligned = 1 - aligned

    return aligned.astype(np.uint8)


# -------------------------
# Dataset loading
# -------------------------
def load_dataset(sample_dir: Path):
    """Loads a single sample directory, explicitly finding rx.npy and a .json file."""
    sample_data = {}

    # Explicit path to the rx.npy file (avoid loading other .npy files)
    npy_path = sample_dir / 'rx.npy'
    if npy_path.exists():
        sample_data['rx_samples'] = np.load(npy_path)

    # Load corresponding metadata JSON (if any)
    json_file = next(sample_dir.glob('*.json'), None)
    if json_file:
        with open(json_file, 'r') as f:
            sample_data.update(json.load(f))

    return sample_data


# -------------------------
# Main pipeline
# -------------------------
def process_phase(phase_num,
                  config,
                  preamble_len=800,
                  acquire_bw=0.06,
                  track_bw=0.003,
                  verbose_first=True,
                  show_constellation_for_first=True,
                  max_shift=16,
                  save_aligned=False,
                  margin=32):
    """
    Process a specific challenge phase with acquire→track Costas loop and aligned BER.
    """
    logger = logging.getLogger(__name__)
    phase_config = config.get_phase_config(phase_num)
    dataset_path = Path(phase_config['dataset_path'])
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return

    # Loop bandwidth defaults (used to construct the demod; per-call we override for acquire→track)
    timing_loop_bw = phase_config.get('timing_bw', 0.002)
    phase_loop_bw = phase_config.get('phase_bw', 0.02)

    demodulator = BPSKDemodulator(
        samples_per_symbol=phase_config['samples_per_symbol'],
        timing_bw=timing_loop_bw,
        phase_bw=phase_loop_bw
    )

    # Find sample directories containing rx.npy
    sample_dirs = sorted({p.parent for p in dataset_path.glob('**/rx.npy')})
    if not sample_dirs:
        logger.warning("No valid sample data (rx.npy) found in %s", dataset_path)
        return

    phase_results = []

    # Diagnostic run on first sample
    diag_dir = sorted(sample_dirs)[0]
    logger.info("=== Diagnostic run on %s ===", diag_dir.relative_to(dataset_path))
    try:
        sample_data = load_dataset(diag_dir)
        if 'rx_samples' not in sample_data:
            logger.warning("rx.npy missing, skipping diagnostics.")
        else:
            bits, corrected = demodulator.process(
                sample_data,
                preamble_len=preamble_len,
                acquire_phase_bw=acquire_bw,
                track_phase_bw=track_bw,
                verbose=True
            )

            gt = np.array(sample_data.get("ground_truth_bits", [])).ravel().astype(np.uint8)
            logger.info("len(bits)=%d, len(gt)=%d", len(bits), len(gt))
            if gt.size > preamble_len and len(bits) > 0:
                gt_data = gt[preamble_len:]
                # safe slicing of window
                gt_window = gt_data[:min(len(bits) + margin, gt_data.size)]

                best_ber, best_shift, inverted = best_align_ber(bits, gt_window, max_shift=max_shift)
                if best_ber is not None:
                    logger.info("ALIGN: shift=%+d, inverted=%s, BER=%.4e", best_shift, inverted, best_ber)
                else:
                    logger.info("No valid overlap found for alignment in diagnostic sample")
            else:
                logger.info("Not enough ground truth bits for aligned BER comparison")

            # Constellation plot (normalized)
            if show_constellation_for_first:
                try:
                    import matplotlib.pyplot as plt
                    cs = corrected[:200]
                    if cs.size > 0:
                        cs = cs / np.sqrt(np.mean(np.abs(cs) ** 2))
                        plt.scatter(np.real(cs), np.imag(cs), s=6)
                        plt.axhline(0, color='k'); plt.axvline(0, color='k')
                        plt.title(f"Constellation (first 200 syms) - {diag_dir.name}")
                        plt.show()
                except Exception as e:
                    logger.debug("Constellation plot skipped: %s", e)
    except Exception as e:
        logger.exception("Error in diagnostic run: %s", e)

    # Full phase processing
    for i, sample_dir in enumerate(sorted(sample_dirs)):
        logger.info("Processing %s...", sample_dir.relative_to(dataset_path))
        try:
            sample_data = load_dataset(sample_dir)
            if 'rx_samples' not in sample_data:
                logger.warning("  Skipping %s: 'rx.npy' file not loaded properly.", sample_dir.name)
                continue

            bits, corrected_symbols = demodulator.process(
                sample_data,
                preamble_len=preamble_len,
                acquire_phase_bw=acquire_bw,
                track_phase_bw=track_bw,
                verbose=verbose_first and i == 0
            )

            # Aligned BER
            ber = None
            best_shift = 0
            inverted = False
            if 'ground_truth_bits' in sample_data and bits.size > 0:
                gt_all = np.array(sample_data['ground_truth_bits']).ravel().astype(np.uint8)
                if gt_all.size > preamble_len:
                    gt_data = gt_all[preamble_len:]
                    gt_window = gt_data[:min(bits.size + margin, gt_data.size)]

                    best_ber, best_shift, inverted = best_align_ber(bits, gt_window, max_shift=max_shift)
                    if best_ber is not None:
                        logger.info("  ALIGN: shift=%+d, inverted=%s, BER=%.2e", best_shift, inverted, best_ber)
                        ber = best_ber
                else:
                    logger.info("  (GT present but too short after eval preamble)")
            else:
                logger.info("  (GT unavailable; skipping BER for this sample)")

            if ber is not None:
                logger.info("  BER: %.2e", ber)
                phase_results.append({
                    'sample': sample_dir.name,
                    'ber': ber,
                    'snr': sample_data.get('snr', None)
                })
            else:
                logger.info("  BER: (ground truth unavailable or not long enough)")

            # Save decoded bits (unaligned by default)
            save_decoded_bits(sample_dir, bits)

            # Optionally save aligned bits for debugging/analysis
            if save_aligned and (best_shift != 0 or inverted):
                aligned_bits = apply_alignment(bits, best_shift, inverted)
                np.save(sample_dir / 'decoded_bits_aligned.npy', aligned_bits.astype(np.uint8))
                logger.debug("  Saved aligned bits to %s", sample_dir / 'decoded_bits_aligned.npy')

            # Optional: plot constellation for the first processed sample (normalized)
            if i == 0 and show_constellation_for_first:
                try:
                    plot_constellation(corrected_symbols, f"Constellation for {sample_dir.name}")
                except Exception as e:
                    logger.debug("Constellation plot skipped: %s", e)

        except Exception as e:
            logger.exception("  Error processing %s: %s", sample_dir.name, e)

    if phase_results:
        generate_phase_report(phase_num, phase_results, phase_config)


def save_decoded_bits(sample_dir: Path, decoded_bits: np.ndarray):
    """Save decoded bits as required by challenge (unaligned)."""
    output_path = sample_dir / 'decoded_bits.npy'
    np.save(output_path, decoded_bits.astype(np.uint8))


def generate_phase_report(phase_num, results, config_dict):
    """Generate performance report for phase."""
    logger = logging.getLogger(__name__)
    logger.info("\n--- Phase %d Summary ---", phase_num)
    bers = [r['ber'] for r in results if r.get('ber') is not None]
    if not bers:
        logger.info("No BERs computed for this phase.")
        return
    avg_ber = np.mean(bers)
    logger.info("Average BER: %.2e", avg_ber)

    threshold = config_dict.get('performance_threshold', 1e-2)  # default if not in config
    logger.info("Threshold: %.2e", threshold)
    if avg_ber <= threshold:
        logger.info("✓ PHASE PASSED")
    else:
        logger.info("✗ Phase failed - needs improvement")


# -------------------------
# CLI Entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CubeSat receiver processing pipeline")
    p.add_argument("--preamble-len", type=int, default=800, help="Preamble length (samples) to skip for BER")
    p.add_argument("--max-shift", type=int, default=16, help="Max +/- bit shift to search when aligning BER")
    p.add_argument("--save-aligned", action="store_true", help="Also save aligned decoded bits (decoded_bits_aligned.npy)")
    p.add_argument("--verbose-all", action="store_true", help="Enable verbose output for all samples (not just first)")
    p.add_argument("--margin", type=int, default=32, help="Extra margin when building GT window")
    p.add_argument("--acquire-bw", type=float, default=0.06, help="Acquire phase loop bandwidth")
    p.add_argument("--track-bw", type=float, default=0.003, help="Track phase loop bandwidth")
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None, help="Process a single phase (1-4). Default: all phases")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting CubeSat receiver pipeline")

    config = Config()

    phases = [args.phase] if args.phase is not None else [1, 2, 3, 4]
    for phase in phases:
        logger.info("\n=== Processing Phase %d ===", phase)
        process_phase(
            phase_num=phase,
            config=config,
            preamble_len=args.preamble_len,
            acquire_bw=args.acquire_bw,
            track_bw=args.track_bw,
            verbose_first=not args.verbose_all,
            show_constellation_for_first=True,
            max_shift=args.max_shift,
            save_aligned=args.save_aligned,
            margin=args.margin
        )

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()

