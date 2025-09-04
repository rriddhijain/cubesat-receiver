import matplotlib.pyplot as plt
import numpy as np

def plot_constellation(signal, title="Constellation Diagram"):
    """Plot constellation diagram."""
    
    plt.figure(figsize=(8, 6))
    plt.scatter(np.real(signal), np.imag(signal), alpha=0.6)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_ber_curve(snr_range, ber_measured, ber_theoretical=None, title="BER vs SNR"):
    """Plot BER performance curve."""
    
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(snr_range, ber_measured, 'bo-', label='Measured')
    
    if ber_theoretical is not None:
        plt.semilogy(snr_range, ber_theoretical, 'r--', label='Theoretical')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_signal_spectrum(signal, sample_rate, title="Signal Spectrum"):
    """Plot signal spectrum."""
    
    plt.figure(figsize=(10, 6))
    
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    spectrum = np.fft.fft(signal)
    
    plt.plot(freqs, 20*np.log10(np.abs(spectrum)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title(title)
    plt.grid(True)
    plt.show()