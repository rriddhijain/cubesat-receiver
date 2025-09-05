# integration.py
import os, json, numpy as np
from .conv_viterbi import viterbi_decode_llr
from .rs1511 import rs1511_decode_bits

def load_meta(sample_dir):
    with open(os.path.join(sample_dir, "meta.json"), "r") as f:
        return json.load(f)

def save_decoded(sample_dir, bits, tag="decoded_bits"):
    path = os.path.join(sample_dir, f"{tag}.npy")
    np.save(path, bits.astype(np.uint8))
    return path

def llr_from_waveform_samples(samples, noise_var):
    return 2.0 * np.asarray(samples, dtype=np.float64) / (noise_var + 1e-12)

def run_phase3_on_sample(sample_dir, prefer_soft=True, noise_var=1.0, demod_bits=None, demod_llrs=None):
    meta = load_meta(sample_dir)
    coding = meta.get("coding", "").lower()

    # load demod outputs if not provided
    if demod_bits is None and demod_llrs is None:
        bits_path = os.path.join(sample_dir, "bits.npy")
        llrs_path = os.path.join(sample_dir, "llrs.npy")
        if os.path.exists(llrs_path):
            demod_llrs = np.load(llrs_path)
        elif os.path.exists(bits_path):
            demod_bits = np.load(bits_path)
        else:
            raise FileNotFoundError("No demod bits/llrs found. Provide demod_bits or demod_llrs, or save them to sample dir.")

    # Conv case
    if "conv" in coding or "viterbi" in coding:
        if prefer_soft and demod_llrs is not None:
            llr = np.asarray(demod_llrs).ravel()
        elif demod_bits is not None:
            A = 4.0
            llr = A * (1 - 2 * np.asarray(demod_bits).astype(np.float64))
        else:
            raise ValueError("Need demod_llrs or demod_bits for convolutional decoding.")
        decoded_bits = viterbi_decode_llr(llr)
        save_decoded(sample_dir, decoded_bits, tag="decoded_conv_bits")
        return decoded_bits

    # RS case
    elif "rs" in coding or "1511" in coding:
        if demod_bits is None:
            if demod_llrs is not None:
                demod_bits = (np.asarray(demod_llrs) < 0).astype(np.uint8)
            else:
                raise ValueError("Need demod_bits or demod_llrs for RS decoding.")
        decoded_bits, fer, pad = rs1511_decode_bits(demod_bits)
        save_decoded(sample_dir, decoded_bits, tag="decoded_rs_bits")
        return decoded_bits, fer

    else:
        raise ValueError(f"Unknown coding: {coding}. Put 'conv' or 'rs' in meta.json.")

