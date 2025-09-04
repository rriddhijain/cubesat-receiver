import numpy as np
import json
import os

def load_dataset(path):
    """
    Load rx.npy (complex baseband samples) and meta.json from a dataset folder.
    Returns:
        rx   : numpy array of complex samples
        meta : dictionary with metadata (SNR, impairments, etc.)
    """
    rx_path = os.path.join(path, "rx.npy")
    meta_path = os.path.join(path, "meta.json")

    rx = np.load(rx_path)  # complex samples
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return rx, meta


def save_bits(path, bits):
    """
    Save decoded bits as decoded_bits.npy in the same folder.
    Bits should be a numpy array of 0/1 integers.
    """
    out_path = os.path.join(path, "decoded_bits.npy")
    np.save(out_path, bits.astype(np.int8))
    print(f"[INFO] Saved {len(bits)} bits to {out_path}")
