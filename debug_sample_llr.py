#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "cubesat_dataset/phase3_coding"  # adjust if needed


def inspect_llrs(sample_dir):
    """Inspect LLRs + constellation in a single sample directory."""
    files = {
        "I-axis": os.path.join(sample_dir, "llrs_i.npy"),
        "Q-axis": os.path.join(sample_dir, "llrs_q.npy"),
        "Chosen": os.path.join(sample_dir, "llrs_chosen.npy"),
        "Corrected": os.path.join(sample_dir, "corrected_syms.npy"),
    }

    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            arr = np.load(path)
            data[key] = arr
            if arr.ndim > 1:  # flatten if matrix
                arr = arr.ravel()
            print(f"[{sample_dir}] {key}: min={arr.min():.3f}, mean={arr.mean():.3f}, max={arr.max():.3f}, len={len(arr)}")
        else:
            print(f"[{sample_dir}] {key}: (missing)")

    # Plot histogram of LLR distributions
    if any(k in data for k in ["I-axis", "Q-axis", "Chosen"]):
        plt.figure(figsize=(8, 4))
        for key in ["I-axis", "Q-axis", "Chosen"]:
            if key in data:
                arr = data[key].ravel()
                plt.hist(arr, bins=100, alpha=0.5, label=key, density=True)
        plt.title(f"LLR distributions - {os.path.basename(sample_dir)}")
        plt.xlabel("LLR value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    # Scatter constellation with LLR coloring
    if "Corrected" in data:
        syms = data["Corrected"].ravel()

        plt.figure(figsize=(6, 6))
        if "Chosen" in data:
            llrs = data["Chosen"].ravel()
            sc = plt.scatter(np.real(syms), np.imag(syms), c=llrs, cmap="coolwarm", s=8)
            plt.colorbar(sc, label="Chosen LLR")
        else:
            plt.scatter(np.real(syms), np.imag(syms), s=8, c="blue")

        plt.axhline(0, color="k", lw=0.5)
        plt.axvline(0, color="k", lw=0.5)
        plt.title(f"Constellation - {os.path.basename(sample_dir)}")
        plt.xlabel("I (real)")
        plt.ylabel("Q (imag)")
        plt.axis("equal")
        plt.show()


def main():
    for root, dirs, files in os.walk(BASE_PATH):
        if any(f in files for f in ["llrs_chosen.npy", "llrs_i.npy", "llrs_q.npy", "corrected_syms.npy"]):
            print("\n--- Inspecting:", root)
            inspect_llrs(root)


if __name__ == "__main__":
    main()
