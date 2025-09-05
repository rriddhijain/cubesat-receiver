#!/usr/bin/env python3
"""
phase4_assembler.py

Scan decoded bitstreams in sample directories, try packing bits->bytes with MSB/LSB,
slide a window over bytes, check CRCs (crc32, crc16-ccitt, crc16-ibm, crc8),
and save extracted payloads that pass CRC.

Also: optional small bit-shift & inversion searches, and a helper to
create decoded_bits_aligned.npy from ground truth.

Usage:
    python phase4_assembler.py --root cubesat_dataset/phase3_coding --min-bytes 8 --max-bytes 128 --try-align --try-invert --debug
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import os
import zlib
import binascii
from typing import Optional, Sequence, Tuple, Dict, Any

logger = logging.getLogger("phase4")

# -------------------------
# CRC helpers
# -------------------------
def crc32_be(data: bytes) -> int:
    # zlib.crc32 returns unsigned 32-bit
    return zlib.crc32(data) & 0xFFFFFFFF

def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    # binascii.crc_hqx implements CRC-CCITT with initial value
    return binascii.crc_hqx(data, init) & 0xFFFF

def crc16_ibm(data: bytes, poly: int = 0x8005, init: int = 0x0000) -> int:
    # Direct bitwise implementation of CRC16-IBM/ARC (poly 0x8005) reflected
    reg = init
    for b in data:
        reg ^= b
        for _ in range(8):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
    return reg & 0xFFFF

def crc8_poly7(data: bytes, init: int = 0x00, poly: int = 0x07) -> int:
    reg = init
    for b in data:
        reg ^= b
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ poly) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg & 0xFF

CRC_TABLE = {
    'crc32': {'len': 4, 'fn': crc32_be},
    'crc16': {'len': 2, 'fn': crc16_ccitt},     # default to CCITT
    'crc16_ibm': {'len': 2, 'fn': crc16_ibm},
    'crc8':  {'len': 1, 'fn': crc8_poly7},
}

# -------------------------
# Packing helpers
# -------------------------
def bits_to_bytes(bits: np.ndarray, bitorder: str = 'msb') -> bytes:
    """
    Pack array of 0/1 bits into bytes.
    bitorder: 'msb' => first bit is MSB of first byte (bit 7)
              'lsb' => first bit is LSB of first byte (bit 0)
    Pads with zeros to next byte boundary.
    """
    bits = np.asarray(bits).astype(np.uint8).ravel()
    if bits.size == 0:
        return b''

    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits = bits.reshape(-1, 8)

    if bitorder == 'msb':
        byte_vals = (bits * (1 << np.arange(7, -1, -1))).sum(axis=1).astype(np.uint8)
    else:  # lsb
        byte_vals = (bits * (1 << np.arange(0, 8))).sum(axis=1).astype(np.uint8)
    return bytes(byte_vals.tolist())

# -------------------------
# Alignment helper (makes decoded_bits_aligned.npy using GT)
# -------------------------
def best_align_simple(decoded_bits: np.ndarray, gt_bits: np.ndarray, max_shift:int=16) -> Tuple[Optional[int], Optional[bool], Optional[float]]:
    """
    Find best shift (-max_shift..+max_shift) and inversion flag minimizing BER.
    Returns (best_shift, inverted, best_ber) or (None,None,None) if no overlap.
    """
    if decoded_bits is None or gt_bits is None:
        return (None, None, None)
    if decoded_bits.size == 0 or gt_bits.size == 0:
        return (None, None, None)
    best = (None, None, 1.0)
    for shift in range(-max_shift, max_shift+1):
        if shift >= 0:
            L = min(decoded_bits.size - shift, gt_bits.size)
            if L <= 0: continue
            d = decoded_bits[shift:shift+L]
            g = gt_bits[:L]
        else:
            L = min(decoded_bits.size, gt_bits.size + shift)
            if L <= 0: continue
            d = decoded_bits[:L]
            g = gt_bits[-shift:-shift+L]
        ber = float(np.mean(d != g))
        ber_inv = float(np.mean((1 - d) != g))
        if ber < best[2]:
            best = (shift, False, ber)
        if ber_inv < best[2]:
            best = (shift, True, ber_inv)
    return best

def apply_alignment(bits: np.ndarray, shift: int, inverted: bool) -> np.ndarray:
    if bits is None:
        return None
    bits = np.asarray(bits).astype(np.uint8).ravel()
    if shift == 0:
        out = bits.copy()
    elif shift > 0:
        if shift >= bits.size:
            out = np.zeros_like(bits)
        else:
            out = bits[shift:]
            pad_len = bits.size - out.size
            if pad_len > 0:
                out = np.concatenate([out, np.zeros(pad_len, dtype=bits.dtype)])
    else:
        pad_left = -shift
        if pad_left >= bits.size:
            out = np.zeros_like(bits)
        else:
            out = np.concatenate([np.zeros(pad_left, dtype=bits.dtype), bits])[:bits.size]
    if inverted:
        out = 1 - out
    return out.astype(np.uint8)

# -------------------------
# Core assembler
# -------------------------
def inspect_and_extract_sample(
    sample_dir: Path,
    min_bytes: int,
    max_bytes: int,
    crc_list: Sequence[str],
    try_shifts: bool,
    max_shift: int,
    try_invert: bool,
    bitorder_modes: Sequence[str],
    whitelist_len: Optional[int] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    For one sample: load decoded bits (try aligned first),
    try packing to bytes under bitorder_modes, optionally test shifts and inversion,
    slide windows of bytes and test CRCs.
    Returns dict summary and saves extracted payloads under extracted/<sample_dir.name>/
    """
    summary = {'sample': str(sample_dir), 'matches': []}
    # Candidate input bit files
    cand_files = [
        sample_dir / 'decoded_bits_aligned.npy',
        sample_dir / 'decoded_bits.npy',
        sample_dir / 'bits.npy'
    ]
    bits = None
    for f in cand_files:
        if f.exists():
            try:
                bits = np.load(f)
                if isinstance(bits, np.ndarray):
                    bits = bits.ravel().astype(np.uint8)
                    break
            except Exception:
                continue
    if bits is None:
        summary['error'] = "no_bits"
        if debug:
            logger.debug("[%s] no decoded bit file found.", sample_dir)
        return summary

    # directory for extracted payloads
    extracted_dir = sample_dir / 'extracted'
    extracted_dir.mkdir(exist_ok=True)

    found_any = False
    # Pre-generate candidate shifts & inversion options
    shift_range = [0]
    if try_shifts:
        shift_range = list(range(-max_shift, max_shift+1))
    invert_opts = [False, True] if try_invert else [False]

    # Loop over bitorder modes - we will try each and keep matches
    for bitorder in bitorder_modes:
        if debug:
            logger.debug("[%s] trying bitorder=%s", sample_dir.name, bitorder)
        # optionally, we may try shifts/inversions in bit domain
        for shift in shift_range:
            for inverted in invert_opts:
                bits_try = apply_alignment(bits, shift, inverted)
                # convert to bytes
                b = bits_to_bytes(bits_try, bitorder=bitorder)
                if len(b) == 0:
                    continue
                # Now slide over bytes windows
                L_bytes = len(b)
                # if whitelist_len provided — only test that exact payload len (payload excludes CRC)
                if whitelist_len is not None:
                    payload_lengths = [whitelist_len]
                else:
                    payload_lengths = list(range(min_bytes, min(max_bytes, L_bytes) + 1))

                for payload_len in payload_lengths:
                    # total window must include CRC length; we'll test each CRC type
                    for crc_name in crc_list:
                        crc_info = CRC_TABLE.get(crc_name)
                        if crc_info is None:
                            continue
                        crc_len = crc_info['len']
                        window_len = payload_len + crc_len
                        if window_len > L_bytes:
                            continue
                        # Slide window across bytes
                        for start in range(0, L_bytes - window_len + 1):
                            window = b[start:start + window_len]
                            payload = window[:payload_len]
                            crc_bytes = window[payload_len:payload_len + crc_len]
                            # compute crc over payload
                            try:
                                val = crc_info['fn'](payload)
                            except Exception:
                                # fallback to binascii for crc16
                                val = crc_info['fn'](payload)
                            # integer in CRC bytes (big-endian)
                            crc_int = int.from_bytes(crc_bytes, 'big')
                            if val == crc_int:
                                # Save extracted payload
                                found_any = True
                                out_name = f"payload_{bitorder}_shift{shift}_{'inv' if inverted else 'ninv'}_{crc_name}_start{start}_plen{payload_len}.bin"
                                out_path = extracted_dir / out_name
                                with open(out_path, 'wb') as of:
                                    of.write(payload)
                                # record
                                match = {
                                    'bitorder': bitorder,
                                    'shift': int(shift),
                                    'inverted': bool(inverted),
                                    'crc': crc_name,
                                    'start_byte': int(start),
                                    'payload_len': int(payload_len),
                                    'crc_len': int(crc_len),
                                    'out_file': str(out_path.relative_to(sample_dir))
                                }
                                summary['matches'].append(match)
                                if debug:
                                    logger.debug("[%s] MATCH %s", sample_dir.name, json.dumps(match))
                                # avoid duplicates: continue scanning but keep recording
    if not found_any:
        summary['matches_count'] = 0
    else:
        summary['matches_count'] = len(summary['matches'])
    return summary

# -------------------------
# High-level run over dataset
# -------------------------
def run_phase4_on_root(
    root: Path,
    min_bytes: int = 8,
    max_bytes: int = 128,
    crc_list: Sequence[str] = ('crc32','crc16','crc16_ibm','crc8'),
    try_shifts: bool = True,
    max_shift: int = 8,
    try_invert: bool = True,
    bitorder_modes: Sequence[str] = ('msb','lsb'),
    whitelist_len: Optional[int] = None,
    align_from_gt: bool = False,
    alignment_max_shift: int = 16,
    dump_report: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    root = Path(root)
    report = {'root': str(root), 'samples': [], 'summary': {}}
    samples = sorted([p.parent for p in root.glob('**/rx.npy')])
    if not samples:
        logger.error("No samples found under %s", root)
        return report

    for s in samples:
        logger.info("Phase-4: processing %s", s.relative_to(root))
        # optionally build aligned version from GT
        if align_from_gt:
            # try load decoded_bits and ground_truth_bits
            decoded_f = None
            for cand in (s / 'decoded_bits.npy', s / 'bits.npy', s / 'decoded_bits_aligned.npy'):
                if cand.exists():
                    decoded_f = cand
                    break
            meta_f = s / 'meta.json'
            if decoded_f and meta_f.exists():
                try:
                    decoded = np.load(decoded_f).ravel().astype(np.uint8)
                    meta = json.loads(meta_f.read_text())
                    gt = np.array(meta.get('ground_truth_bits', [])).ravel().astype(np.uint8)
                    if gt.size > 0 and decoded.size > 0:
                        # evaluate only portion after preamble if available in meta
                        preamble_len = int(meta.get('preamble_len', 800))
                        if gt.size > preamble_len:
                            gt_window = gt[preamble_len: preamble_len + decoded.size]
                        else:
                            gt_window = gt[:decoded.size]
                        shift, inv, best_ber = best_align_simple(decoded, gt_window, max_shift=alignment_max_shift)
                        if shift is not None:
                            aligned = apply_alignment(decoded, shift, inv)
                            np.save(s / 'decoded_bits_aligned.npy', aligned.astype(np.uint8))
                            logger.info("  Wrote decoded_bits_aligned.npy (shift=%d inv=%s ber=%.3e)", shift, inv, best_ber)
                except Exception as e:
                    logger.debug("  align-from-gt failed for %s: %s", s, e)

        res = inspect_and_extract_sample(
            s,
            min_bytes=min_bytes,
            max_bytes=max_bytes,
            crc_list=crc_list,
            try_shifts=try_shifts,
            max_shift=max_shift,
            try_invert=try_invert,
            bitorder_modes=bitorder_modes,
            whitelist_len=whitelist_len,
            debug=debug
        )
        report['samples'].append(res)

    # simple summary
    total_matches = sum(len(r.get('matches', [])) for r in report['samples'])
    report['summary']['total_samples'] = len(report['samples'])
    report['summary']['total_matches'] = total_matches

    if dump_report:
        out_path = Path.cwd() / 'phase4_report.json'
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote report to %s (total matches=%d)", out_path, total_matches)

    return report

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase-4 assembler & CRC extractor")
    p.add_argument("--root", required=True, help="Root dataset directory (e.g., cubesat_dataset/phase3_coding)")
    p.add_argument("--min-bytes", type=int, default=8, help="Minimum payload bytes to try")
    p.add_argument("--max-bytes", type=int, default=128, help="Maximum payload bytes to try")
    p.add_argument("--crc", action='append', default=None, help="CRC to try (can be repeated). Options: crc32, crc16, crc16_ibm, crc8. Default=all")
    p.add_argument("--no-shift", action="store_true", help="Do NOT try bit shifts (faster)")
    p.add_argument("--max-shift", type=int, default=8, help="Max +/- bit shift for search")
    p.add_argument("--no-invert", action="store_true", help="Do NOT try bit inversion")
    p.add_argument("--bitorder", choices=['msb','lsb','both','auto'], default='both', help="Bit ordering to try. 'auto' tries msb then lsb and stops on first match")
    p.add_argument("--whitelist-len", type=int, default=None, help="If provided, only test that payload length (bytes)")
    p.add_argument("--align-from-gt", action="store_true", help="Attempt generation of decoded_bits_aligned.npy using ground_truth_bits in meta.json")
    p.add_argument("--align-max-shift", type=int, default=16, help="Max shift for align-from-gt")
    p.add_argument("--debug", action="store_true", help="Verbose debug logging")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.crc:
        crc_list = args.crc
    else:
        crc_list = list(CRC_TABLE.keys())
    if args.bitorder == 'both':
        bitorder_modes = ('msb','lsb')
    elif args.bitorder == 'msb':
        bitorder_modes = ('msb',)
    elif args.bitorder == 'lsb':
        bitorder_modes = ('lsb',)
    else:  # auto
        bitorder_modes = ('msb','lsb')  # inspector will try msb then lsb; this script doesn't short-circuit, but you can inspect report

    run_phase4_on_root(
        root=Path(args.root),
        min_bytes=args.min_bytes,
        max_bytes=args.max_bytes,
        crc_list=crc_list,
        try_shifts=(not args.no_shift),
        max_shift=args.max_shift,
        try_invert=(not args.no_invert),
        bitorder_modes=bitorder_modes,
        whitelist_len=args.whitelist_len,
        align_from_gt=args.align_from_gt,
        alignment_max_shift=args.align_max_shift,
        dump_report=True,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
