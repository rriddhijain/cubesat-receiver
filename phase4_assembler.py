#!/usr/bin/env python3
"""
phase4_assembler.py

Phase-4 assembly & CRC-sweep helper.

Usage:
    python phase4_assembler.py --root cubesat_dataset/phase3_coding --min-bytes 8 --max-bytes 256

Defaults:
  - Root: cubesat_dataset/phase3_coding
  - CRCs tried: crc32, crc16-ccitt (init 0xFFFF), crc8 (poly 0x07)
  - Byte-aligns windows (step = 1 byte), window sizes default 8..256 bytes
  - Tries shifts in [-2..2] and inversion if --try-align supplied
Outputs:
  - extracted/ files in each sample dir for matches
  - phase4_report.json at root with summary
"""
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import binascii
from collections import defaultdict

# -----------------------
# CRC helpers
# -----------------------
def crc32_bytes(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF

def crc16_ccitt(data: bytes, init=0xFFFF) -> int:
    crc = init & 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

def crc8(data: bytes, poly=0x07, init=0x00) -> int:
    crc = init & 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ (poly << 0)) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF

# -----------------------
# Bit -> Bytes packing
# -----------------------
def bits_to_bytes(bits: np.ndarray, pad_to_byte=True):
    """
    bits: 1D numpy array of 0/1
    pack MSB-first (bitorder='big') so bits[0] -> highest bit of first byte
    If length not multiple of 8 and pad_to_byte=True, pad with zeros at end (LSBs).
    """
    if bits is None or bits.size == 0:
        return b''
    bits = np.asarray(bits).ravel().astype(np.uint8)
    rem = bits.size % 8
    if rem != 0:
        if pad_to_byte:
            pad_len = 8 - rem
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        else:
            bits = bits[: bits.size - rem]
    # np.packbits with bitorder='big' available in numpy >= 1.17; safe for most installs
    try:
        packed = np.packbits(bits, bitorder='big')
    except TypeError:
        # older numpy fallback (default is 'big' historically)
        packed = np.packbits(bits)
    return packed.tobytes()

# -----------------------
# Frame detection
# -----------------------
def scan_for_frames_by_crc(byte_stream: bytes, min_payload_b: int, max_payload_b: int,
                           crc_candidates=('crc32', 'crc16', 'crc8'),
                           max_windows=50000):
    """
    Slide a byte-window through byte_stream and test CRCs.
    Expects last N bytes of window are CRC field (4/2/1 depending).
    Returns list of matches: dicts with start_byte, total_len, payload_bytes, crc_name.
    """
    matches = []
    n = len(byte_stream)
    checked = 0
    # For efficiency limit windows tested
    for total_len in range(min_payload_b + 1, max_payload_b + 1):  # +1 to leave at least 1 byte for CRC
        # For each CRC type, compute crc_len
        for crc_name in crc_candidates:
            crc_len = 4 if crc_name == 'crc32' else (2 if crc_name == 'crc16' else 1)
            payload_len = total_len - crc_len
            if payload_len <= 0:
                continue
            # Slide start position (byte aligned)
            for start in range(0, n - total_len + 1):
                checked += 1
                if checked > max_windows:
                    return matches, checked
                window = byte_stream[start:start + total_len]
                payload = window[:payload_len]
                crc_field = window[payload_len:]
                crc_val = int.from_bytes(crc_field, byteorder='big', signed=False)
                # compute candidate CRC
                if crc_name == 'crc32':
                    calc = crc32_bytes(payload)
                    if calc == crc_val:
                        matches.append({'start_byte': start, 'total_len': total_len, 'crc': 'crc32',
                                        'payload': payload})
                elif crc_name == 'crc16':
                    calc = crc16_ccitt(payload, init=0xFFFF)
                    if calc == crc_val:
                        matches.append({'start_byte': start, 'total_len': total_len, 'crc': 'crc16-ccitt',
                                        'payload': payload})
                elif crc_name == 'crc8':
                    calc = crc8(payload, poly=0x07, init=0x00)
                    if calc == crc_val:
                        matches.append({'start_byte': start, 'total_len': total_len, 'crc': 'crc8',
                                        'payload': payload})
    return matches, checked

# -----------------------
# Main per-sample logic
# -----------------------
def process_sample(sample_dir: Path, args, logger):
    """
    Return dict summary for this sample.
    """
    sample_dir = Path(sample_dir)
    # prefer aligned file if present
    candidates = ['decoded_bits_aligned.npy', 'decoded_bits.npy', 'bits.npy']
    bits_path = None
    for fn in candidates:
        p = sample_dir / fn
        if p.exists():
            bits_path = p
            break
    if bits_path is None:
        logger.warning("  No decoded bits found in %s (tried %s)", sample_dir, candidates)
        return {'sample': str(sample_dir), 'found_bits': False}

    try:
        bits = np.load(bits_path).ravel().astype(np.uint8)
    except Exception as e:
        logger.exception("  Failed to load bits from %s: %s", bits_path, e)
        return {'sample': str(sample_dir), 'found_bits': False}

    logger.info("  Loaded bits from %s (len=%d)", bits_path.name, bits.size)
    # optionally try small shifts and inversion
    shifts = [0]
    if args.try_align:
        L = args.try_align_range
        shifts = list(range(-L, L + 1))
    inversions = [False, True] if args.try_invert else [False]

    sample_matches = []
    tried = 0
    for shift in shifts:
        # apply shift (left positive -> drop first shift samples; negative -> pad left zeros)
        if shift == 0:
            b_shifted = bits
        elif shift > 0:
            if shift >= bits.size:
                b_shifted = np.zeros_like(bits)
            else:
                tmp = bits[shift:]
                pad_len = bits.size - tmp.size
                if pad_len > 0:
                    tmp = np.concatenate([tmp, np.zeros(pad_len, dtype=tmp.dtype)])
                b_shifted = tmp
        else:
            pad_left = -shift
            if pad_left >= bits.size:
                b_shifted = np.zeros_like(bits)
            else:
                tmp = np.concatenate([np.zeros(pad_left, dtype=bits.dtype), bits])[:bits.size]
                b_shifted = tmp

        for invert in inversions:
            b_try = (1 - b_shifted) if invert else b_shifted
            # pack to bytes
            byte_stream = bits_to_bytes(b_try, pad_to_byte=True)
            matches, checked = scan_for_frames_by_crc(
                byte_stream,
                min_payload_b=args.min_bytes,
                max_payload_b=args.max_bytes,
                crc_candidates=args.crc_set,
                max_windows=args.max_windows_per_sample
            )
            tried += checked
            if matches:
                logger.info("  Found %d CRC matches (shift=%+d invert=%s)", len(matches), shift, invert)
                # persist matches
                out_dir = sample_dir / 'extracted'
                out_dir.mkdir(exist_ok=True)
                meta_list = []
                for mi, m in enumerate(matches):
                    fname = f"frame_s{shift:+d}_i{int(invert)}_pos{m['start_byte']}_len{m['total_len']}_{m['crc']}.bin"
                    out_path = out_dir / fname
                    with open(out_path, 'wb') as f:
                        f.write(m['payload'])
                    meta_list.append({
                        'file': str(out_path.relative_to(sample_dir)),
                        'start_byte': m['start_byte'],
                        'total_len': m['total_len'],
                        'crc': m['crc'],
                        'shift': shift,
                        'inverted': bool(invert)
                    })
                # write sample-level metadata
                with open(sample_dir / 'extracted_meta.json', 'w') as f:
                    json.dump({'matches': meta_list}, f, indent=2)
                sample_matches.extend(meta_list)
            else:
                logger.debug("  no matches (shift=%+d invert=%s) after checking %d windows", shift, invert, checked)

            # If user wanted single-match-per-sample behavior, we could stop here; keep scanning by default
            if args.stop_on_first and sample_matches:
                break
        if args.stop_on_first and sample_matches:
            break

    return {
        'sample': str(sample_dir),
        'found_bits': True,
        'bits_path': str(bits_path.name),
        'bits_len': int(bits.size),
        'tried_windows': int(tried),
        'matches': sample_matches
    }

# -----------------------
# CLI / run
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase-4 assembler / CRC-sweep helper")
    p.add_argument("--root", default="cubesat_dataset/phase3_coding", help="Samples root folder")
    p.add_argument("--min-bytes", type=int, default=8, help="Minimum payload bytes to try")
    p.add_argument("--max-bytes", type=int, default=256, help="Maximum payload bytes to try")
    p.add_argument("--crc", nargs="+", default=["crc32", "crc16", "crc8"],
                   help="CRC set to try (choose from crc32 crc16 crc8)")
    p.add_argument("--try-align", action="store_true", help="Try small bit shifts when scanning (slower)")
    p.add_argument("--try-align-range", type=int, default=2, help="Shift range +/- when --try-align used")
    p.add_argument("--try-invert", action="store_true", help="Also try bit inversion sweep")
    p.add_argument("--stop-on-first", action="store_true", help="Stop searching a sample after first match")
    p.add_argument("--max-windows-per-sample", type=int, default=200000,
                   help="Cap on number of sliding windows checked per sample to avoid runaway times")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("phase4")

    crc_map = []
    for c in args.crc:
        c = c.lower()
        if c in ("crc32", "crc16", "crc8"):
            crc_map.append(c if c != "crc16" else "crc16")
        else:
            logger.warning("Ignoring unknown crc name: %s", c)
    args.crc_set = crc_map

    root = Path(args.root)
    if not root.exists():
        logger.error("Root folder not found: %s", root)
        return

    sample_dirs = sorted([p.parent for p in root.glob("**/rx.npy")])  # prefer sample dirs known earlier
    # also include any directories that have decoded_bits.npy even if rx.npy absent
    for p in root.glob("**/decoded_bits.npy"):
        if p.parent not in sample_dirs:
            sample_dirs.append(p.parent)
    sample_dirs = sorted(set(sample_dirs))

    logger.info("Found %d candidate sample directories", len(sample_dirs))
    overall = {'total_samples': 0, 'processed': 0, 'matches_total': 0, 'samples': []}

    for sd in sample_dirs:
        overall['total_samples'] += 1
        logger.info("Processing sample: %s", sd)
        res = process_sample(sd, args, logger)
        if res.get('found_bits'):
            overall['processed'] += 1
        mcount = len(res.get('matches', []))
        overall['matches_total'] += mcount
        overall['samples'].append(res)

    # write summary
    out_report = root / 'phase4_report.json'
    with open(out_report, 'w') as f:
        json.dump(overall, f, indent=2)
    logger.info("Done. Processed %d samples, total matches found %d. Report saved to %s",
                overall['processed'], overall['matches_total'], out_report)

if __name__ == "__main__":
    main()
