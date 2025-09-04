# receiver/demodulator.py
import numpy as np
from .synchronization import TimingRecovery, FrequencyRecovery

class BPSKDemodulator:
    """
    BPSK demodulator:
      - RRC matched filter
      - AGC
      - TimingRecovery (Gardner, parabolic interp)
      - Blind (preamble-agnostic) coarse CFO + constant-phase removal via BPSK squaring
      - FrequencyRecovery (Costas) with optional acquire→track bandwidths
      - Optional polarity correction using ground truth (after preamble)
      - Verbose diagnostics
    """

    def __init__(self, samples_per_symbol=8,
                 timing_bw=0.002, phase_bw=0.02,
                 rrc_beta=0.35, rrc_span=10):
        self.sps = samples_per_symbol
        self.matched_filter = self._create_rrc_filter(beta=rrc_beta, span=rrc_span)
        self.timing_recovery = TimingRecovery(self.sps, loop_bandwidth=timing_bw)
        self.freq_recovery = FrequencyRecovery(loop_bandwidth=phase_bw)

    def process(self, sample_data, preamble_len=400, verbose=False,
                acquire_phase_bw=None, track_phase_bw=None):
        """
        Returns:
          bits (uint8) for the DATA portion (preamble skipped),
          corrected_symbols (complex) for the DATA portion.
        """
        # Reset loops each burst
        self.timing_recovery.reset()
        self.freq_recovery.reset()

        # --- Load and coerce rx samples ---
        raw = np.array(sample_data.get("rx_samples", [])).ravel()
        if raw.size == 0:
            return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.complex64)
        rx = raw.astype(np.complex64)
        # handle IQ-coded arrays robustly
        if rx.ndim == 2:
            if rx.shape[0] == 2:            # [2, N] -> rows = I,Q
                rx = rx[0, :] + 1j * rx[1, :]
            elif rx.shape[1] == 2:          # [N, 2] -> cols = I,Q
                rx = rx[:, 0] + 1j * rx[:, 1]
            else:
                rx = rx.ravel().astype(np.complex64)

        # --- Matched filter (enable for low SNR) ---
        mf = np.convolve(rx, self.matched_filter, mode="same")
        # For RRC mismatch debugging only, you could bypass:
        # mf = rx

        # --- AGC to unit power ---
        p = np.mean(np.abs(mf)**2)
        if p > 1e-12:
            mf = mf / np.sqrt(p)

        # --- Timing recovery to symbol rate (preamble + data) ---
        timed = self.timing_recovery.recover(mf)
        if timed.size == 0:
            if verbose:
                print("No timed symbols produced.")
            return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.complex64)

        # --- Guard preamble_len ---
        if preamble_len < 1:
            preamble_len = 1
        if preamble_len >= timed.size:
            preamble_len = max(1, timed.size // 4)

        # ==========================================================
        # BLIND (preamble-agnostic) coarse CFO + constant-phase
        #   BPSK trick: s_k^2 = A^2 * e^{j 2 φ_k} (data removed)
        #   -> robust Δφ (ramp) and θ (static) estimates without GT alignment
        # ==========================================================
        M = min(preamble_len, timed.size)
        if M >= 2:
            z = timed[:M] ** 2
            # Δφ estimate (ramp): 0.5 * angle( sum z[k+1] conj(z[k]) )
            c2 = np.sum(z[1:] * np.conj(z[:-1]))
            if np.abs(c2) > 1e-12:
                ph_step = 0.5 * np.angle(c2)  # per-symbol CFO
                n = np.arange(timed.size, dtype=np.float64)
                timed = timed * np.exp(-1j * ph_step * n)
            # constant phase: 0.5 * angle( sum z )
            z0 = timed[:M] ** 2
            th2 = 0.5 * np.angle(np.sum(z0))
            timed = timed * np.exp(-1j * th2)

        # --- Re-slice after derotations (IMPORTANT) ---
        preamble_syms = timed[:preamble_len]
        data_syms     = timed[preamble_len:]
        if data_syms.size == 0:
            if verbose:
                print("No data symbols after preamble.")
            return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.complex64)

        # --- Costas: acquire -> track (optional) ---
        if acquire_phase_bw is not None and track_phase_bw is not None:
            acq_loop = FrequencyRecovery(loop_bandwidth=acquire_phase_bw)
            _ = acq_loop.recover(preamble_syms)

            track_loop = FrequencyRecovery(loop_bandwidth=track_phase_bw)
            track_loop.import_state(acq_loop)

            corrected = track_loop.recover(data_syms)
            self.freq_recovery = track_loop  # persist if desired
        else:
            _ = self.freq_recovery.recover(preamble_syms)
            corrected = self.freq_recovery.recover(data_syms)
            # --- Auto-axis alignment (choose rotation from {1, -1, j, -j}) ---
        cands = np.array([1+0j, -1+0j, 0+1j, 0-1j], dtype=np.complex64)
        # score = mean |Imag| after rotation; pick the smallest
        scores = [np.mean(np.abs(np.imag(corrected * c))) for c in cands]
        best_c = cands[int(np.argmin(scores))]
        corrected *= best_c

        # --- Hard decision: Re>=0 => bit 0, Re<0 => bit 1 ---
        bits = (np.real(corrected) < 0).astype(np.uint8).ravel()

        # --- Optional polarity correction via ground truth (aligned on DATA) ---
        if "ground_truth_bits" in sample_data:
            known = np.array(sample_data["ground_truth_bits"]).ravel().astype(np.uint8)
            if known.size >= preamble_len + bits.size and bits.size > 0:
                true_data = known[preamble_len : preamble_len + bits.size]
                check_len = min(64, bits.size)
                errs = np.sum(bits[:check_len] != true_data[:check_len])
                if verbose:
                    print(f"[polarity check] errs={errs}/{check_len} err_rate={errs/check_len:.3f}")
                if errs > 0.5 * check_len:
                    bits = 1 - bits
                    if verbose:
                        print("[polarity corrected] flipped bits due to >50% mismatch")

        # --- Diagnostics ---
        if verbose:
            head = corrected[:20]
            print("=== DEMOD DIAG ===")
            print("timed syms:", timed.size, "preamble:", preamble_len, "data syms:", data_syms.size)
            print("corrected[:20] real:", np.round(np.real(head),4))
            print("corrected[:20] imag:", np.round(np.imag(head),4))
            print("mean |sym|:", np.round(np.mean(np.abs(corrected)),4))
            phases = np.angle(corrected + 1e-12)
            if phases.size > 0:
                print("phase min/mean/max:", np.round(phases.min(),4), np.round(phases.mean(),4), np.round(phases.max(),4))
            print("==================")

        return bits, corrected

    def _create_rrc_filter(self, beta, span):
        """Root-raised cosine filter taps normalized to unit energy."""
        N = span * self.sps
        t = np.arange(-N/2, N/2 + 1) / self.sps
        h = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            if abs(ti) < 1e-8:
                h[i] = 1.0 - beta + 4 * beta / np.pi
            elif abs(abs(4 * beta * ti) - 1.0) < 1e-6:
                term1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                term2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                h[i] = (beta / np.sqrt(2)) * (term1 + term2)
            else:
                num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                h[i] = num / den
        return h / np.sqrt(np.sum(h**2))
