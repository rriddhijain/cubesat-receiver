# tests/quick_run_phase3.py
from cubesat_dataset.phase3_coding.integration import run_phase3_on_sample
import numpy as np, os, sys

if __name__ == "__main__":
    sample_dir = "cubesat_dataset/phase3_coding/sample000"  # change to your sample dir
    # Ensure bits.npy or llrs.npy present in sample_dir before running
    print("Running Phase3 on:", sample_dir)
    res = run_phase3_on_sample(sample_dir, prefer_soft=True, noise_var=1.0)
    print("Result:", res)
