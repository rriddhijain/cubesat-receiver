import numpy as np
from .synchronization import TimingRecovery, FrequencyRecovery

class BPSKDemodulator:
    """
    A BPSK demodulator that uses stateful feedback loops for synchronization.
    """

    def __init__(self, samples_per_symbol=4, phase=1):
        self.sps = samples_per_symbol
        self.matched_filter = self._create_rrc_filter(beta=0.35, span=10)
        
        # Initialize the stateful recovery loops
        self.timing_recovery = TimingRecovery(samples_per_symbol)
        self.freq_recovery = FrequencyRecovery()

    def process(self, sample_data):
        """
        Main demodulation pipeline using feedback loops.
        """
        # Reset the state of the loops for each new signal
        self.timing_recovery.reset()
        self.freq_recovery.reset()
        
        rx_signal = sample_data['rx_samples'].astype(np.complex64)

        # 1. Matched filter
        mf_signal = np.convolve(rx_signal, self.matched_filter, mode="same")

        # 2. Timing Recovery (produces symbol-rate samples)
        timed_symbols = self.timing_recovery.recover(mf_signal)

        # 3. Frequency and Phase Recovery
        corrected_symbols = self.freq_recovery.recover(timed_symbols)

        # 4. Symbol-to-bit hard decisions
        bits = self._make_decision(corrected_symbols)
        return bits

    def _create_rrc_filter(self, beta, span):
        """Root-raised cosine filter taps."""
        N = span * self.sps
        t = np.arange(-N / 2, N / 2 + 1) / self.sps
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

        return h / np.sqrt(np.sum(h ** 2))  # normalize energy

    def _make_decision(self, symbols):
        """Map BPSK symbols to bits."""
        return (np.real(symbols) < 0).astype(np.uint8)