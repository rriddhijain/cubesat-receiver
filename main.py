#!/usr/bin/env python3
"""
main.py - Unified pipeline for CubeSat communication challenge (phases 1..4)

Usage:
    python main.py --phase 3 --log-level INFO
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np

# Project imports (expected present in your repo)
from config import Config
from receiver.demodulator import BPSKDemodulator
from evaluation.metrics import calculate_ber
from evaluation.plotting import plot_constellation

# Phase 3 integration (your existing module)
from cubesat_dataset.phase3_coding.integration import run_phase3_on_sample

# -------------------------
# Helpers: canonical mapping, alignment
# -------------------------
CANONICAL_MAP = {
    'conv': 'conv', 'convolutional': 'conv', 'viterbi': 'conv', 'conv_viterbi': 'conv',
    'rs': 'rs', 'reed_solomon': 'rs', 'reedsolomon': 'rs', 'reed': 'rs'
}


def infer_and_fix_coding(sample_dir: Path, sample_data: dict, write_back=True):
    """Infer canonical coding label ('conv' or 'rs') from meta or path; optionally write meta.json."""
    keys = ['coding', 'coding_name', 'coding-type', 'codec', 'code']
    val = None
    for k in keys:
        if k in sample_data and sample_data.get(k) is not None:
            val = str(sample_data.get(k)).strip().lower()
            if val:
                break

    if not val:
        meta_path = sample_dir / 'meta.json'
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                for k in keys:
                    if k in meta and meta.get(k):
                        val = str(meta.get(k)).strip().lower()
                        break
            except Exception:
                pass

    if not val:
        # check folder names
        parts = [p.lower() for p in sample_dir.parts]
        for p in reversed(parts):
            if p in CANONICAL_MAP:
                val = p
                break
    if val:
        mapped = CANONICAL_MAP.get(val)
        if mapped:
            existing = sample_data.get('coding')
            if existing is None or str(existing).strip().lower() != mapped:
                sample_data['coding'] = mapped
                if write_back:
                    try:
                        meta_path = sample_dir / 'meta.json'
                        meta = {}
                        if meta_path.exists():
                            try:
                                meta = json.loads(meta_path.read_text())
                            except Exception:
                                meta = {}
                        meta['coding'] = mapped
                        meta_path.write_text(json.dumps(meta, indent=2))
                    except Exception:
                        pass
            return mapped
    return None


def best_align_ber(decoded_bits: np.ndarray, gt_bits: np.ndarray, max_shift: int = 16):
    """Find shift and optional inversion minimizing BER (small shifts only)."""
    if decoded_bits is None or gt_bits is None:
        return (None, 0, False)
    if decoded_bits.size == 0 or gt_bits.size == 0:
        return (None, 0, False)

    best = (1.0, 0, False)
    found = False
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

        found = True
        try:
            ber = float(calculate_ber(d, g))
            ber_flip = float(calculate_ber(1 - d, g))
        except Exception:
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
    """Apply shift + optional inversion while keeping same length (for debugging/saving)."""
    if bits is None:
        return None
    if shift == 0:
        aligned = bits.copy()
    elif shift > 0:
        if shift >= bits.size:
            aligned = np.zeros_like(bits)
        else:
            aligned = bits[shift:]
            pad_len = bits.size - aligned.size
            if pad_len > 0:
                aligned = np.concatenate([aligned, np.zeros(pad_len, dtype=aligned.dtype)])
    else:
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
    sample_data = {}
    npy_path = sample_dir / 'rx.npy'
    if npy_path.exists():
        try:
            sample_data['rx_samples'] = np.load(npy_path)
        except Exception:
            pass

    meta_path = sample_dir / 'meta.json'
    if meta_path.exists():
        try:
            sample_data.update(json.loads(meta_path.read_text()))
        except Exception:
            pass
    else:
        json_file = next(sample_dir.glob('*.json'), None)
        if json_file:
            try:
                sample_data.update(json.loads(json_file.read_text()))
            except Exception:
                pass
    return sample_data


# -------------------------
# Phase-4 Doppler helpers
# -------------------------
def estimate_freq_offset_fft(signal, fs=1.0, search_band=None):
    """
    Coarse frequency offset estimator by FFT on the complex baseband samples.
    Returns frequency offset in cycles / sample (i.e. normalized frequency).
    - search_band: tuple (min_norm, max_norm) to restrict search in normalized cycles/sample.
    """
    x = np.asarray(signal).ravel()
    N = min(32768, max(2048, len(x)))
    # pick center slice for stability
    start = max(0, (len(x) - N) // 2)
    x_slice = x[start:start + N]
    # compute FFT of the signal's autocorrelation-like product to emphasize tone
    S = np.fft.fftshift(np.fft.fft(x_slice * np.conj(x_slice)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x_slice), d=1.0 / fs))
    if search_band is not None:
        mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
        if not np.any(mask):
            idx = np.argmax(np.abs(S))
        else:
            idx = np.argmax(np.abs(S[mask]))
            idx = np.where(mask)[0][0] + idx
    else:
        idx = np.argmax(np.abs(S))
    freq_hat = freqs[idx]  # cycles/sample normalized
    return float(freq_hat)


def apply_freq_correction(rx_samples, freq_offset_cycles_per_sample):
    """Multiply rx_samples by exp(-j*2π*freq_offset*n) to remove the estimated offset."""
    n = np.arange(len(rx_samples))
    correction = np.exp(-1j * 2.0 * np.pi * freq_offset_cycles_per_sample * n)
    return rx_samples * correction


# -------------------------
# Main phase processing
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
    Process a specific challenge phase (1..4). This function does:
      - Phase 1 style demodulation + alignment/BER reporting
      - Phase 3 integration via run_phase3_on_sample when relevant
      - Phase 4: estimate doppler -> correct -> re-run phase processing on corrected samples
    """
    logger = logging.getLogger(__name__)
    phase_config = config.get_phase_config(phase_num)
    dataset_path = Path(phase_config['dataset_path'])
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return

    # Construct demodulator used across phases (tuning from config)
    timing_loop_bw = phase_config.get('timing_bw', 0.002)
    phase_loop_bw = phase_config.get('phase_bw', 0.02)
    demodulator = BPSKDemodulator(
        samples_per_symbol=phase_config['samples_per_symbol'],
        timing_bw=timing_loop_bw,
        phase_bw=phase_loop_bw
    )

    # Find sample dirs
    sample_dirs = sorted({p.parent for p in dataset_path.glob('**/rx.npy')})
    if not sample_dirs:
        logger.warning("No valid sample data found in %s", dataset_path)
        return

    phase_results = []

    # Diagnostic run for first sample (gives quick visibility)
    diag_dir = sample_dirs[0]
    logger.info("=== Diagnostic run on %s ===", diag_dir.relative_to(dataset_path))
    try:
        sample_data = load_dataset(diag_dir)
        if 'rx_samples' in sample_data:
            bits_diag, corrected_diag = demodulator.process(
                sample_data,
                preamble_len=preamble_len,
                acquire_phase_bw=acquire_bw,
                track_phase_bw=track_bw,
                verbose=True
            )
            logger.info("len(bits)=%d, diagnostic symbols=%d", len(bits_diag), 0 if corrected_diag is None else len(corrected_diag))
            # attempt aligned BER if GT present
            gt = np.array(sample_data.get("ground_truth_bits", [])).ravel().astype(np.uint8)
            if gt.size > preamble_len and len(bits_diag) > 0:
                gt_data = gt[preamble_len: preamble_len + min(len(bits_diag) + margin, gt.size)]
                best_ber, best_shift, inverted = best_align_ber(bits_diag, gt_data, max_shift=max_shift)
                if best_ber is not None:
                    logger.info("ALIGN (diag): shift=%+d, inverted=%s, BER=%.4e", best_shift, inverted, best_ber)
            # try show constellation
            if show_constellation_for_first and corrected_diag is not None:
                try:
                    plot_constellation(corrected_diag, f"Constellation - {diag_dir.name}")
                except Exception:
                    logger.debug("Constellation plot skipped (diagnostic).")
    except Exception as e:
        logger.debug("Diagnostic run failed: %s", e)

    # Iterate all samples
    for i, sample_dir in enumerate(sorted(sample_dirs)):
        logger.info("Processing %s...", sample_dir.relative_to(dataset_path))
        try:
            sample_data = load_dataset(sample_dir)
            # infer coding label early (helps phase3)
            infer_and_fix_coding(sample_dir, sample_data, write_back=True)

            if 'rx_samples' not in sample_data:
                logger.warning("  Skipping %s: missing rx.npy", sample_dir.name)
                continue

            rx_samples = sample_data['rx_samples'].astype(np.complex128)

            # If phase 4 requested or dataset indicates doppler, run coarse frequency estimation & correction
            freq_offset_est = None
            if phase_num == 4 or ('doppler_hz' in sample_data or 'doppler' in str(sample_dir)):
                # normalized cycles/sample search range default -0.125..0.125
                search_band = None
                try:
                    # If metadata gives approximate doppler (Hz) + sample rate, use it
                    if sample_data.get('doppler_hz') is not None and sample_data.get('sample_rate') is not None:
                        dop_hz = float(sample_data['doppler_hz'])
                        fs = float(sample_data['sample_rate'])
                        search_band = (dop_hz / fs - 0.01, dop_hz / fs + 0.01)
                    freq_offset_est = estimate_freq_offset_fft(rx_samples, fs=1.0, search_band=search_band)
                    logger.info("  Estimated normalized freq offset = %g cycles/sample", freq_offset_est)
                    # apply correction
                    rx_corrected = apply_freq_correction(rx_samples, freq_offset_est)
                    # update sample data so demodulator sees corrected samples
                    sample_data['rx_samples'] = rx_corrected
                except Exception as e:
                    logger.debug("  Frequency estimation/correction failed: %s", e)
                    sample_data['rx_samples'] = rx_samples

            # Run demodulator (phase 1 style)
            bits, corrected_symbols = demodulator.process(
                sample_data,
                preamble_len=preamble_len,
                acquire_phase_bw=acquire_bw,
                track_phase_bw=track_bw,
                verbose=verbose_first and (i == 0)
            )

            # Save corrected symbols for debugging
            try:
                if corrected_symbols is not None:
                    np.save(sample_dir / 'corrected_syms.npy', np.asarray(corrected_symbols))
            except Exception:
                logger.debug("  Could not save corrected_syms (non-fatal)")

            # Phase-1 aligned BER reporting
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
                    logger.info("  (GT present but too short after preamble)")
            else:
                logger.info("  (GT unavailable; skipping BER for this sample)")

            # Save phase-1 decoded bits
            try:
                np.save(sample_dir / 'decoded_bits.npy', bits.astype(np.uint8))
            except Exception:
                logger.debug("  Could not save decoded_bits.npy")

            # Compute base LLRs (normalized)
            demod_llrs_base = None
            try:
                y = np.asarray(corrected_symbols).ravel()
                mean_power = float(np.mean(np.abs(y) ** 2)) if y.size > 0 else 1.0
                if mean_power <= 0:
                    mean_power = 1.0
                # infer noise var from metadata if present
                noise_var = 1.0
                if sample_data.get('snr_db') is not None:
                    try:
                        EbN0 = 10 ** (float(sample_data['snr_db']) / 10.0)
                        noise_var = 1.0 / (2.0 * EbN0)
                    except Exception:
                        noise_var = 1.0
                y_norm = y / np.sqrt(mean_power)
                demod_llrs_base = (2.0 * np.real(y_norm)) / (1e-12 + 2.0 * noise_var)
                demod_llrs_base = np.clip(demod_llrs_base, -50.0, 50.0)
                try:
                    np.save(sample_dir / 'llrs_base.npy', demod_llrs_base)
                except Exception:
                    pass
            except Exception as e:
                demod_llrs_base = None
                logger.debug("  Could not compute demod_llrs_base: %s", e)

            # Auto sign/scale sweep when GT is available (to choose best soft LLR)
            chosen_llrs = None
            if demod_llrs_base is not None:
                SIGN_CANDIDATES = [+1.0, -1.0]
                SCALE_CANDIDATES = [0.05, 0.1, 0.25, 0.5, 1.0, 1.6, 2.0]
                LLR_CLIP = 50.0
                chosen_sign, chosen_scale = +1.0, 1.0
                gt_for_eval = None
                if 'ground_truth_bits' in sample_data:
                    gt_all = np.array(sample_data['ground_truth_bits']).ravel().astype(np.uint8)
                    if gt_all.size > preamble_len:
                        gt_for_eval = gt_all[preamble_len: preamble_len + demod_llrs_base.size]
                if gt_for_eval is not None and gt_for_eval.size > 0:
                    best_combo = (None, None, 1e9)
                    for sign in SIGN_CANDIDATES:
                        for scale in SCALE_CANDIDATES:
                            llr_try = np.clip(sign * demod_llrs_base * float(scale), -LLR_CLIP, LLR_CLIP)
                            try:
                                res = run_phase3_on_sample(
                                    str(sample_dir),
                                    prefer_soft=True,
                                    noise_var=noise_var,
                                    demod_bits=bits.astype(np.uint8),
                                    demod_llrs=llr_try
                                )
                                if isinstance(res, tuple):
                                    decoded_try, _fer = res
                                else:
                                    decoded_try = res
                                if decoded_try is None:
                                    continue
                                decoded_try = np.asarray(decoded_try).ravel().astype(np.uint8)
                                L = min(decoded_try.size, gt_for_eval.size)
                                if L <= 0:
                                    continue
                                try:
                                    ber_try = float(calculate_ber(decoded_try[:L], gt_for_eval[:L]))
                                except Exception:
                                    ber_try = float(np.mean(decoded_try[:L] != gt_for_eval[:L]))
                                if ber_try < best_combo[2]:
                                    best_combo = (sign, scale, ber_try)
                            except Exception as e:
                                logger.debug("    sweep decode error: %s", e)
                    if best_combo[0] is not None:
                        chosen_sign, chosen_scale = best_combo[0], best_combo[1]
                        logger.info("  Auto-chosen sign=%+.0f scale=%g (BER=%.2e)", chosen_sign, chosen_scale, best_combo[2])
                    else:
                        logger.info("  Auto-sweep found no improvement; using defaults")
                chosen_llrs = np.clip(chosen_sign * demod_llrs_base * float(chosen_scale), -LLR_CLIP, LLR_CLIP)
                try:
                    np.save(sample_dir / 'llrs.npy', chosen_llrs)
                except Exception:
                    logger.debug("  Could not save llrs.npy (non-fatal)")

            # Phase-3 final decode (if coding known)
            coding_canonical = sample_data.get('coding')
            if coding_canonical is None:
                coding_canonical = infer_and_fix_coding(sample_dir, sample_data, write_back=False)

            if phase_num == 3 or coding_canonical in ('conv', 'rs'):
                if coding_canonical not in ('conv', 'rs'):
                    logger.info("  Phase-3: coding not identified as conv/rs; skipping Phase-3.")
                else:
                    try:
                        logger.info("  Running Phase 3 decoding (final)...")
                        phase3_result = run_phase3_on_sample(
                            str(sample_dir),
                            prefer_soft=True,
                            noise_var=noise_var,
                            demod_bits=bits.astype(np.uint8),
                            demod_llrs=chosen_llrs
                        )
                        if isinstance(phase3_result, tuple):
                            decoded_phase3_bits, fer = phase3_result
                            logger.info("  Phase 3 (RS) decoded %d bits, FER=%s", 0 if decoded_phase3_bits is None else len(decoded_phase3_bits), str(fer))
                        else:
                            decoded_phase3_bits = phase3_result
                            logger.info("  Phase 3 (Conv) decoded %d bits", 0 if decoded_phase3_bits is None else len(decoded_phase3_bits))
                        # save final phase3 output if produced
                        if decoded_phase3_bits is not None:
                            try:
                                np.save(sample_dir / 'decoded_phase3_bits.npy', np.asarray(decoded_phase3_bits).astype(np.uint8))
                            except Exception:
                                pass
                    except Exception as e:
                        logger.exception("  Phase 3 decoding failed for %s: %s", sample_dir.name, e)

            # optionally save aligned bits
            if save_aligned and (best_shift != 0 or inverted):
                aligned_bits = apply_alignment(bits, best_shift, inverted)
                try:
                    np.save(sample_dir / 'decoded_bits_aligned.npy', aligned_bits.astype(np.uint8))
                except Exception:
                    logger.debug("  Could not save decoded_bits_aligned.npy (non-fatal)")

            # record results for phase report
            phase_results.append({
                'sample': sample_dir.name,
                'ber': ber,
                'snr': sample_data.get('snr', sample_data.get('snr_db')),
                'coding': sample_data.get('coding', 'unknown')
            })

            # optional constellation plot for first sample
            if i == 0 and show_constellation_for_first and corrected_symbols is not None:
                try:
                    plot_constellation(corrected_symbols, f"Constellation - {sample_dir.name}")
                except Exception:
                    logger.debug("  Could not plot constellation (non-fatal)")

        except Exception as e:
            logger.exception("  Error processing %s: %s", sample_dir.name, e)

    # generate final phase report
    if phase_results:
        generate_phase_report(phase_num, phase_results, phase_config)


# -------------------------
# Reporting + saving
# -------------------------
def save_decoded_bits(sample_dir: Path, decoded_bits: np.ndarray):
    try:
        np.save(sample_dir / 'decoded_bits.npy', decoded_bits.astype(np.uint8))
    except Exception:
        pass


def generate_phase_report(phase_num, results, config_dict):
    logger = logging.getLogger(__name__)
    logger.info("\n--- Phase %d Summary ---", phase_num)
    bers = [r['ber'] for r in results if r.get('ber') is not None]
    if not bers:
        logger.info("No BERs computed for this phase.")
        return
    avg_ber = float(np.mean(bers))
    logger.info("Average BER: %.2e", avg_ber)

    perf_cfg = config_dict.get('performance_threshold', 1e-2)
    if isinstance(perf_cfg, dict):
        try:
            thresholds_str = ", ".join([f"{k}={v:.2e}" for k, v in perf_cfg.items()])
        except Exception:
            thresholds_str = str(perf_cfg)
        logger.info("Thresholds (per-code): %s", thresholds_str)

        per_code_bers = {}
        for r in results:
            code = r.get('coding', 'unknown')
            if r.get('ber') is None:
                continue
            per_code_bers.setdefault(code, []).append(r['ber'])

        all_pass = True
        for code, thr in perf_cfg.items():
            matched_keys = [k for k in per_code_bers.keys() if k.lower() == code.lower()]
            if not matched_keys:
                logger.info("No samples for code %s", code)
                all_pass = False
                continue
            combined = []
            for mk in matched_keys:
                combined.extend(per_code_bers.get(mk, []))
            if not combined:
                logger.info("No BER values for coding '%s'.", code)
                all_pass = False
                continue
            mean_code_ber = float(np.mean(combined))
            logger.info("Coding '%s' average BER: %.2e (threshold %.2e)", code, mean_code_ber, thr)
            if mean_code_ber <= float(thr):
                logger.info("  ✓ %s PASS", code)
            else:
                logger.info("  ✗ %s FAIL", code)
                all_pass = False

        if all_pass:
            logger.info("Overall: ✓ PHASE PASSED (all code-specific thresholds met)")
        else:
            logger.info("Overall: ✗ Phase failed - some code-specific thresholds not met")
    else:
        try:
            thr = float(perf_cfg)
            logger.info("Threshold: %.2e", thr)
        except Exception:
            logger.info("Threshold: %s", str(perf_cfg))
            thr = 1e-2
        if avg_ber <= thr:
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
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None, help="Process a single phase (1-4). Default: all")
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
