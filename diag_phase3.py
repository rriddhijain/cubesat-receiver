#!/usr/bin/env python3
# diagnose_phase3.py
import os, json, argparse
from pathlib import Path
import numpy as np
from cubesat_dataset.phase3_coding.integration import run_phase3_on_sample

def best_align_ber_simple(decoded_bits, gt_bits, max_shift=16):
    if decoded_bits is None or gt_bits is None: return (None,0,False)
    if decoded_bits.size==0 or gt_bits.size==0: return (None,0,False)
    best = (1.0,0,False)
    for shift in range(-max_shift, max_shift+1):
        if shift>=0:
            L = min(decoded_bits.size-shift, gt_bits.size)
            if L<=0: continue
            d = decoded_bits[shift:shift+L]; g = gt_bits[:L]
        else:
            L = min(decoded_bits.size, gt_bits.size+shift)
            if L<=0: continue
            d = decoded_bits[:L]; g = gt_bits[-shift:-shift+L]
        ber = float(np.mean(d != g))
        ber_flip = float(np.mean((1-d) != g))
        if ber < best[0]: best = (ber, shift, False)
        if ber_flip < best[0]: best = (ber_flip, shift, True)
    return best

def load_meta(sample_dir):
    p = Path(sample_dir) / "meta.json"
    if p.exists():
        return json.load(open(p))
    # fallback
    j = next(Path(sample_dir).glob("*.json"), None)
    if j: return json.load(open(j))
    return {}

def try_phase3(sample_dir, demod_bits=None, demod_llrs=None, noise_var=1.0):
    try:
        res = run_phase3_on_sample(str(sample_dir), prefer_soft=True, noise_var=noise_var,
                                   demod_bits=demod_bits, demod_llrs=demod_llrs)
        return res
    except Exception as e:
        return ("error", str(e))

def inspect_sample(sample_dir, noise_var=1.0):
    sd = Path(sample_dir)
    meta = load_meta(sd)
    coding = str(meta.get("coding","unknown"))
    print(f"\n--- {sd}  coding={coding} ---")
    # load ground-truth
    gt = None
    if "ground_truth_bits" in meta:
        gt = np.array(meta["ground_truth_bits"]).ravel().astype(np.uint8)
    else:
        # maybe saved as file
        gtp = sd / "ground_truth_bits.npy"
        if gtp.exists():
            gt = np.load(gtp).ravel().astype(np.uint8)

    bits = None
    llrs = None
    for n in ["decoded_bits.npy","decoded_conv_bits.npy","decoded_rs_bits.npy","bits.npy"]:
        p = sd / n
        if p.exists():
            try:
                arr = np.load(p)
                print("found", n, "len", arr.size)
            except Exception:
                arr = None
            if n == "bits.npy":
                bits = arr.astype(np.uint8) if arr is not None else None

    lp = sd / "llrs.npy"
    if lp.exists():
        try:
            llrs = np.load(lp)
            print("found llrs.npy len", llrs.size)
        except Exception:
            llrs = None

    # quick BER of saved decoded_bits vs GT if both present
    if gt is not None:
        # try unaligned compare
        candidates = []
        for cand_name in ["decoded_conv_bits.npy","decoded_rs_bits.npy","decoded_bits.npy"]:
            p = sd / cand_name
            if p.exists():
                try:
                    d = np.load(p).ravel().astype(np.uint8)
                    L = min(d.size, max(0, gt.size - 800))  # try skipping preamble if needed
                    if L>0:
                        # align small shifts
                        ber, shift, inv = best_align_ber_simple(d, gt[800:800+L+32])
                        candidates.append((cand_name, ber, shift, inv, d.size))
                except Exception:
                    pass
        for c in candidates:
            print("saved decode:", c)

    # Try Phase3 decoding attempts
    print("Attempt Phase3 decode attempts (this uses integration.run_phase3_on_sample)")
    # If llrs not present, try create pseudo-llr from bits if bits available
    if llrs is None and bits is not None:
        pseudo = (4.0 * (1.0 - 2.0 * bits.astype(np.float64))).ravel()
        llrs = pseudo
        print("  created pseudo llrs from bits")
    # attempt with original llrs
    if llrs is not None:
        r1 = try_phase3(sd, demod_bits=None, demod_llrs=llrs, noise_var=noise_var)
        print("  result original-llrs:", type(r1), (r1 if isinstance(r1, tuple) else r1))
        # flipped sign
        r2 = try_phase3(sd, demod_bits=None, demod_llrs=-llrs, noise_var=noise_var)
        print("  result flipped-llrs:", type(r2), (r2 if isinstance(r2, tuple) else r2))
        # scaled
        for A in [0.5, 2.0, 5.0, 10.0]:
            rA = try_phase3(sd, demod_bits=None, demod_llrs=llrs * A, noise_var=noise_var)
            print(f"  result scaled-llr x{A}:", type(rA), (rA if isinstance(rA, tuple) else rA))
    else:
        print("  no llrs/bits available to try Phase3 decodes")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="cubesat_dataset/phase3_coding")
    p.add_argument("--limit", type=int, default=0, help="limit number of samples to inspect (0=all)")
    args = p.parse_args()
    root = Path(args.root)
    sample_dirs = sorted({p.parent for p in root.glob("**/rx.npy")})
    if args.limit>0:
        sample_dirs = sample_dirs[:args.limit]
    for sd in sample_dirs:
        inspect_sample(sd)
