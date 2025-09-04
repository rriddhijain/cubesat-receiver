import numpy as np
from scipy.signal import lfilter

class TimingRecovery:
    """
    Timing synchronization using a Gardner (phase-independent) algorithm.
    This implementation is robust to carrier phase errors.
    """
    def __init__(self, samples_per_symbol, loop_bandwidth=0.001):
        self.sps = samples_per_symbol
        
        # Loop gains for a second-order loop filter
        damping_factor = 1.0 / np.sqrt(2)
        normalized_bw = loop_bandwidth / self.sps
        
        # Proportional and integral gains
        self.kp = (4 * damping_factor * normalized_bw) / (1 + 2 * damping_factor * normalized_bw + normalized_bw**2)
        self.ki = (4 * normalized_bw**2) / (1 + 2 * damping_factor * normalized_bw + normalized_bw**2)
        
        self.reset()

    def reset(self):
        """Reset the internal state of the loop for a new signal."""
        self.strobe_index = 0.0
        self.integrator = 0.0
        self.error = 0.0
        self.prev_sample = 0.0 + 0.0j

    def recover(self, signal):
        """Process an entire signal to recover symbol timing."""
        num_symbols_to_process = len(signal) // self.sps - 2
        output_symbols = np.zeros(num_symbols_to_process, dtype=np.complex64)
        out_idx = 0
        
        while self.strobe_index < len(signal) - self.sps and out_idx < num_symbols_to_process:
            idx_int = int(np.floor(self.strobe_index))
            frac = self.strobe_index - idx_int
            
            current_symbol = signal[idx_int] + frac * (signal[idx_int + 1] - signal[idx_int])
            
            mid_point_index = self.strobe_index - self.sps / 2.0
            idx_int_mid = int(np.floor(mid_point_index))
            frac_mid = mid_point_index - idx_int_mid
            mid_point_sample = signal[idx_int_mid] + frac_mid * (signal[idx_int_mid + 1] - signal[idx_int_mid])
            
            error = np.real(mid_point_sample) * (np.real(self.prev_sample) - np.real(current_symbol))
            
            output_symbols[out_idx] = current_symbol
            out_idx += 1
            self.prev_sample = current_symbol
            
            self.integrator += self.ki * error
            proportional = self.kp * error
            
            self.strobe_index += self.sps + proportional + self.integrator
            
        return output_symbols[:out_idx]

class FrequencyRecovery:
    """
    Frequency and phase synchronization using a Costas Loop with a PI controller.
    """
    def __init__(self, loop_bandwidth=0.001):
        # PI Controller Gains
        damping_factor = 1.0 / np.sqrt(2)
        normalized_bw = loop_bandwidth
        
        self.alpha = (4 * damping_factor * normalized_bw) / (1 + 2 * damping_factor * normalized_bw + normalized_bw**2)
        self.beta = (4 * normalized_bw**2) / (1 + 2 * damping_factor * normalized_bw + normalized_bw**2)
        self.reset()

    def reset(self):
        """Reset the internal state of the loop for a new signal."""
        self.current_phase = 0.0
        self.integrator = 0.0

    def recover(self, signal):
        """Process signal to correct for phase/frequency offset."""
        output_signal = np.zeros_like(signal, dtype=np.complex64)
        
        for i in range(len(signal)):
            # De-rotate the sample by the current phase estimate
            corrected_sample = signal[i] * np.exp(-1j * self.current_phase)
            output_signal[i] = corrected_sample
            
            # Costas loop error detector for BPSK (simplified and more stable)
            error = -(np.sign(np.real(corrected_sample)) * np.imag(corrected_sample))
            
            # Update the loop filter (PI controller)
            proportional = self.alpha * error
            self.integrator += self.beta * error
            
            # Update the phase for the next sample
            self.current_phase += proportional + self.integrator
        
        return output_signal
