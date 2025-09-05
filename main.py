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

# Phase 3 integration
from cubesat_dataset.phase3_coding.integration import run_phase3_on_sample


# -------------------------
# Helpers: inference / alignment
# -------------------------
CANONICAL_MAP = {
    'conv': 'conv',
    'convolutional': 'conv',
    'convolutional_viterbi': 'conv',
    'viterbi': 'conv',
    'convolutional-viterbi': 'conv',
    'conv_viterbi': 'conv',
    'rs': 'rs',
    'reed_solomon': 'rs',
    'reed-solomon': 'rs',
    'reedsolomon': 'rs',
    'reed': 'rs'
}


def infer_and_fix_coding(sample_dir: Path, sample_data: dict, write_back=True):
    keys_to_check = ['coding', 'coding_name', 'coding-type', 'codec', 'code']
    val = None
    for k in keys_to_check:
        if k in sample_data and sample_data.get(k) is not None:
            val = str(sample_data.get(k)).strip().lower()
            if val:
                break

    if not val:
        meta_path = sample_dir / 'meta.json'
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                for k in keys_to_check:
                    if k in meta and meta.get(k):
                        val = str(meta.get(k)).strip().lower()
                        break
            except Exception:
                pass

    if not val:
        parts = [p.lower() for p in sample_dir.parts]
        for p in parts[::-1]:
            if p in CANONICAL_MAP:
                val = p
                break
        if val is None and sample_dir.parent.name.lower() in CANONICAL_MAP:
            val = sample_dir.parent.name.lower()

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


def best_align_ber(decoded_bits: np.ndarray,
                   gt_bits: np.ndarray,
                   max_shift: int = 16):
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
    logger = logging.getLogger(__name__)
    phase_config = config.get_phase_config(phase_num)
    dataset_path = Path(phase_config['dataset_path'])
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return

    timing_loop_bw = phase_config.get('timing_bw', 0.002)
    phase_loop_bw = phase_config.get('phase_bw', 0.02)

    demodulator = BPSKDemodulator(
        samples_per_symbol=phase_config['samples_per_symbol'],
        timing_bw=timing_loop_bw,
        phase_bw=phase_loop_bw
    )

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
        if 'rx_samples' in sample_data:
            bits_diag, corrected_diag = demodulator.process(
                sample_data,
                preamble_len=preamble_len,
                acquire_phase_bw=acquire_bw,
                track_phase_bw=track_bw,
                verbose=True
            )
            gt = np.array(sample_data.get("ground_truth_bits", [])).ravel().astype(np.uint8)
            logger.info("len(bits)=%d, len(gt)=%d", len(bits_diag), len(gt))
            if gt.size > preamble_len and len(bits_diag) > 0:
                gt_data = gt[preamble_len: preamble_len + min(len(bits_diag) + margin, gt.size)]
                best_ber, best_shift, inverted = best_align_ber(bits_diag, gt_data, max_shift=max_shift)
                if best_ber is not None:
                    logger.info("ALIGN: shift=%+d, inverted=%s, BER=%.4e", best_shift, inverted, best_ber)
    except Exception as e:
        logger.debug("Diagnostic run skipped/failed: %s", e)

    # Full phase processing
    for i, sample_dir in enumerate(sorted(sample_dirs)):
        logger.info("Processing %s...", sample_dir.relative_to(dataset_path))
        try:
            sample_data = load_dataset(sample_dir)

            # Try to infer and fix coding early so Phase-3 integration can read it
            coding = infer_and_fix_coding(sample_dir, sample_data, write_back=True)
            if coding is None:
                logger.debug("  Could not infer coding for %s yet (will still attempt run_phase3 if integration infers)", sample_dir.name)

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

            # ---------------------------
            # Save corrected symbols for offline inspection (D)
            # ---------------------------
            try:
                np.save(sample_dir / 'corrected_syms.npy', np.asarray(corrected_symbols))
            except Exception:
                logger.debug("  Could not save corrected_syms (non-fatal)")

            # Aligned BER (phase-1)
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

            # record phase_results entry (include coding if present)
            coding_name = str(sample_data.get('coding', sample_data.get('coding_name', 'unknown'))).lower()
            phase_results.append({
                'sample': sample_dir.name,
                'ber': ber,
                'snr': sample_data.get('snr', sample_data.get('snr_db')),
                'coding': coding_name
            })

            # Save Phase-1 decoded bits and bits.npy
            save_decoded_bits(sample_dir, bits)
            try:
                np.save(sample_dir / 'bits.npy', bits.astype(np.uint8))
            except Exception:
                logger.debug("  Could not save bits.npy (non-fatal)")

            # ---------------------------
            # Robust LLR computation (normalize base LLRs)
            # ---------------------------
            LLR_CLIP = 20.0
            SCALE_CANDIDATES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
            SIGN_CANDIDATES = [+1.0, -1.0]
            # phase candidates every 5 degrees
            PHASE_CANDIDATES = np.deg2rad(np.arange(0, 360, 5.0))
            DEFAULT_SIGN = +1.0
            DEFAULT_SCALE = 1.0

            # infer noise_var
            noise_var = 1.0
            if sample_data.get('snr_db') is not None:
                try:
                    EbN0 = 10 ** (float(sample_data['snr_db']) / 10.0)
                    noise_var = 1.0 / (2.0 * EbN0)
                except Exception:
                    noise_var = 1.0
            elif sample_data.get('snr') is not None:
                try:
                    EbN0 = float(sample_data['snr'])
                    noise_var = 1.0 / (2.0 * EbN0)
                except Exception:
                    noise_var = 1.0

            demod_llrs_base = None
            y_norm = None
            try:
                y = np.asarray(corrected_symbols).ravel()
                if y.size == 0:
                    raise ValueError("corrected_symbols empty")
                mean_power = float(np.mean(np.abs(y) ** 2))
                if mean_power <= 0:
                    mean_power = 1.0
                y_norm = y / np.sqrt(mean_power)  # normalized complex symbols
                # base LLR computed from real(y_norm) assuming 0 phase
                demod_llrs_base = (2.0 * np.real(y_norm)) / (1e-12 + 2.0 * noise_var)
                demod_llrs_base = np.clip(demod_llrs_base, -LLR_CLIP, LLR_CLIP)
                try:
                    np.save(sample_dir / 'llrs_base.npy', demod_llrs_base)
                except Exception:
                    pass
            except Exception as e:
                demod_llrs_base = None
                y_norm = None
                logger.debug("  Could not compute demod_llrs_base: %s", e)

            # ---------------------------
            # Auto sign+scale+phase detection (if GT available) -> pick chosen_llrs
            # ---------------------------
            chosen_llrs = None
            chosen_sign = DEFAULT_SIGN
            chosen_scale = DEFAULT_SCALE
            chosen_phase = 0.0
            if y_norm is not None:
                gt_for_eval = None
                if 'ground_truth_bits' in sample_data:
                    gt_all = np.array(sample_data['ground_truth_bits']).ravel().astype(np.uint8)
                    if gt_all.size > preamble_len:
                        gt_for_eval = gt_all[preamble_len: preamble_len + y_norm.size]

                def _decode_with_llrs(llrs_arr):
                    try:
                        res = run_phase3_on_sample(
                            str(sample_dir),
                            prefer_soft=True,
                            noise_var=noise_var,
                            demod_bits=bits.astype(np.uint8),
                            demod_llrs=llrs_arr
                        )
                        if isinstance(res, tuple):
                            decoded_bits, _fer = res
                        else:
                            decoded_bits = res
                        if decoded_bits is None:
                            return None
                        return np.asarray(decoded_bits).ravel().astype(np.uint8)
                    except Exception as e:
                        logger.debug("    decode error during sweep: %s", e)
                        return None

                if gt_for_eval is not None and gt_for_eval.size > 0:
                    best_combo = (None, None, None, None, 1e9)  # sign, scale, phase, decoded, ber
                    for sign in SIGN_CANDIDATES:
                        for scale in SCALE_CANDIDATES:
                            for phase in PHASE_CANDIDATES:
                                # rotate complex symbols by -phase then take real part
                                y_rot = y_norm * np.exp(-1j * float(phase))
                                llrs_try = (2.0 * np.real(y_rot)) / (1e-12 + 2.0 * noise_var)
                                llrs_try = np.clip(sign * llrs_try * float(scale), -LLR_CLIP, LLR_CLIP)
                                decoded_try = _decode_with_llrs(llrs_try)
                                if decoded_try is None or decoded_try.size == 0:
                                    continue
                                L = min(decoded_try.size, gt_for_eval.size)
                                if L <= 0:
                                    continue
                                try:
                                    ber_try = float(calculate_ber(decoded_try[:L], gt_for_eval[:L]))
                                except Exception:
                                    ber_try = float(np.mean(decoded_try[:L] != gt_for_eval[:L]))
                                # keep best
                                if ber_try < best_combo[4]:
                                    best_combo = (sign, scale, float(phase), decoded_try, ber_try)
                                # small optimization: early stop if near zero BER
                                if ber_try <= 1e-6:
                                    break
                            else:
                                continue
                            break
                    if best_combo[0] is not None:
                        chosen_sign, chosen_scale, chosen_phase, chosen_decoded, chosen_ber = best_combo[0], best_combo[1], best_combo[2], best_combo[3], best_combo[4]
                        # compute chosen llrs based on chosen_phase/sign/scale
                        y_rot = y_norm * np.exp(-1j * float(chosen_phase))
                        chosen_llrs = np.clip(chosen_sign * (2.0 * np.real(y_rot)) / (1e-12 + 2.0 * noise_var) * float(chosen_scale), -LLR_CLIP, LLR_CLIP)
                        logger.info("  Auto-chosen sign=%+.0f scale=%g phase=%.2f (BER=%.2e)", chosen_sign, chosen_scale, chosen_phase, chosen_ber)
                    else:
                        # fallback: default sign/scale, phase=0
                        chosen_sign = DEFAULT_SIGN
                        chosen_scale = DEFAULT_SCALE
                        chosen_phase = 0.0
                        y_rot = y_norm * np.exp(-1j * float(chosen_phase))
                        chosen_llrs = np.clip(chosen_sign * (2.0 * np.real(y_rot)) / (1e-12 + 2.0 * noise_var) * float(chosen_scale), -LLR_CLIP, LLR_CLIP)
                        logger.info("  Auto-sweep found no valid decode; using default sign=%+.0f scale=%g phase=%.2f", chosen_sign, chosen_scale, chosen_phase)
                else:
                    # No GT -> pick safe defaults; rotate by 0
                    chosen_sign = DEFAULT_SIGN
                    chosen_scale = DEFAULT_SCALE
                    chosen_phase = 0.0
                    y_rot = y_norm * np.exp(-1j * float(chosen_phase))
                    chosen_llrs = np.clip(chosen_sign * (2.0 * np.real(y_rot)) / (1e-12 + 2.0 * noise_var) * float(chosen_scale), -LLR_CLIP, LLR_CLIP)
                    logger.debug("  No GT available; using default sign=%+.0f scale=%g phase=%.2f", chosen_sign, chosen_scale, chosen_phase)

                try:
                    np.save(sample_dir / 'llrs.npy', chosen_llrs)
                except Exception:
                    logger.debug("  Could not save chosen llrs.npy (non-fatal)")
            else:
                logger.debug("  No demod_llrs_base available; Phase-3 will be called with demod_bits only if supported")

            # ---------------------------
            # Final Phase-3 decode (only if we can identify coding)
            # ---------------------------
            coding_canonical = sample_data.get('coding')
            if coding_canonical is None:
                coding_canonical = infer_and_fix_coding(sample_dir, sample_data, write_back=False)

            if coding_canonical not in ('conv', 'rs'):
                logger.error("  Skipping Phase-3 for %s: unknown coding '%s'. Put 'conv' or 'rs' in meta.json.", sample_dir.name, str(coding_canonical))
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
                except Exception as e:
                    logger.exception("  Phase 3 decoding failed for %s: %s", sample_dir.name, e)

            # optionally save aligned bits
            if save_aligned and (best_shift != 0 or inverted):
                aligned_bits = apply_alignment(bits, best_shift, inverted)
                try:
                    np.save(sample_dir / 'decoded_bits_aligned.npy', aligned_bits.astype(np.uint8))
                except Exception:
                    logger.debug("  Could not save decoded_bits_aligned.npy (non-fatal)")

            # optional plot
            if i == 0 and show_constellation_for_first:
                try:
                    plot_constellation(corrected_symbols, f"Constellation for {sample_dir.name}")
                except Exception:
                    logger.debug("  constellation plotting failed (non-fatal)")

        except Exception as e:
            logger.exception("  Error processing %s: %s", sample_dir.name, e)

    if phase_results:
        generate_phase_report(phase_num, phase_results, phase_config)


# -------------------------
# Helpers: saving & reporting
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
        logger.info("Decision threshold used: %.2e", thr)
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

