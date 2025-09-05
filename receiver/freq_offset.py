# receiver/freq_offset.py
import numpy as np

class FreqOffsetEstimator:
    """
    Estimate and correct carrier frequency offset due to Doppler.
    Uses FFT-based coarse estimation + fine PLL tracking.
    """
    def __init__(self, sps, search_width=0.1, nfft=4096):
        """
        Args:
            sps: samples per symbol
            search_width: fraction of Fs to search for coarse offset (e.g., 0.1 = ±10%)
            nfft: FFT size for coarse estimation
        """
        self.sps = sps
        self.search_width = search_width
        self.nfft = nfft

    def estimate(self, samples, fs=1.0):
        """
        Estimate frequency offset in normalized frequency (cycles/sample).
        Args:
            samples: complex baseband array
            fs: sample rate (Hz). If left 1.0, output is in normalized units.
        Returns: offset (Hz if fs provided)
        """
        N = min(len(samples), self.nfft)
        spectrum = np.fft.fftshift(np.fft.fft(samples[:N], self.nfft))
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1.0/fs))
        # limit search range
        half_bw = self.search_width * fs / 2
        mask = np.abs(freqs) <= half_bw
        idx = np.argmax(np.abs(spectrum[mask]))
        est_freq = freqs[mask][idx]
        return est_freq

    def correct(self, samples, freq_offset, fs=1.0):
        """
        Apply frequency correction.
        Args:
            samples: complex baseband array
            freq_offset: frequency offset to correct (Hz)
            fs: sample rate
        Returns: corrected samples
        """
        n = np.arange(len(samples))
        correction = np.exp(-1j * 2 * np.pi * freq_offset * n / fs)
        return samples * correction
