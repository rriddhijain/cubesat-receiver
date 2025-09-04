# CubeSat BPSK Receiver

A BPSK receiver for the CubeSat challenge featuring:
- RRC matched filter + AGC
- Gardner timing recovery
- Blind coarse CFO + Costas loop (acquire → track)
- BER evaluation with alignment

## Run
```bash
python main.py

Datasets (.npy/.json) are ignored by .gitignore
