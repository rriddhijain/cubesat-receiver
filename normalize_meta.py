#!/usr/bin/env python3
# normalize_meta.py
import json
from pathlib import Path
import argparse

# Mapping of common meta coding names -> canonical expected values
CANONICAL = {
    'conv': 'conv',
    'convolutional': 'conv',
    'convolutional_viterbi': 'conv',
    'viterbi': 'conv',
    'rs': 'rs',
    'reed_solomon': 'rs',
    'reed-solomon': 'rs',
    'reedsolomon': 'rs'
}

def normalize_one(meta_path: Path, dry=False):
    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"Skipping {meta_path} (read error): {e}")
        return False
    orig = data.get('coding') or data.get('coding_name') or data.get('coding-type') or data.get('codec')
    if orig is None:
        print(f"No coding field in {meta_path}, leaving unchanged.")
        return False
    key = str(orig).strip().lower()
    if key in CANONICAL:
        canon = CANONICAL[key]
        if data.get('coding') == canon:
            print(f"{meta_path}: already canonical='{canon}'")
            return False
        else:
            data['coding'] = canon
            if not dry:
                meta_path.write_text(json.dumps(data, indent=2))
            print(f"{meta_path}: set coding -> '{canon}' (from '{orig}')")
            return True
    else:
        print(f"{meta_path}: unknown coding value '{orig}' — no change.")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="cubesat_dataset/phase3_coding")
    p.add_argument("--dry", action="store_true", help="Do not write, just show")
    args = p.parse_args()
    root = Path(args.root)
    metas = list(root.glob("**/meta.json"))
    if not metas:
        print("No meta.json files found under", root)
        return
    changed = 0
    for m in metas:
        if normalize_one(m, dry=args.dry):
            changed += 1
    print(f"Done. Changed {changed}/{len(metas)} files.")

if __name__ == "__main__":
    main()
