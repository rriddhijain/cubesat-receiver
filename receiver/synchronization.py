# receiver/synchronization.py
import numpy as np

class TimingRecovery:
    """
    Complex Gardner timing recovery with parabolic interpolation,
    integrator clipping and safe indexing.
    """
    def __init__(self, samples_per_symbol, loop_bandwidth=0.002):
        self.sps = samples_per_symbol
        damping = 1.0 / np.sqrt(2.0)
        norm_bw = loop_bandwidth / max(1.0, self.sps)
        self.kp = (4 * damping * norm_bw) / (1 + 2*damping*norm_bw + norm_bw**2)
        self.ki = (4 * norm_bw**2) / (1 + 2*damping*norm_bw + norm_bw**2)
        self.reset()

    def reset(self):
        # Safe starting strobe (allow interpolation window)
        self.strobe_index = float(self.sps + 1)
        self.integrator = 0.0
        self.prev_symbol = 0+0j

    def _parabolic_interp(self, signal, index):
        """
        3-point parabolic interpolation (cheap + robust). Returns complex sample.
        """
        idx = int(np.floor(index))
        if idx < 1 or idx >= len(signal) - 1:
            return 0+0j
        frac = index - idx
        y0 = signal[idx - 1]
        y1 = signal[idx]
        y2 = signal[idx + 1]
        # Parabolic polynomial centered at y1
        return y1 + frac * (0.5 * (y2 - y0) + frac * 0.5 * (y2 + y0 - 2 * y1))

    def recover(self, signal):
        n_out = max(0, len(signal) // self.sps - 4)
        out = np.zeros(n_out, dtype=np.complex64)
        out_idx = 0

        while (self.strobe_index < len(signal) - self.sps - 1) and (out_idx < n_out):
            # Clamp strobe to safe region
            self.strobe_index = np.clip(self.strobe_index, 2.0, len(signal) - 3.0)

            s_cur = self._parabolic_interp(signal, self.strobe_index)
            s_mid = self._parabolic_interp(signal, self.strobe_index - self.sps / 2.0)

            # Complex Gardner error (I & Q)
            err = np.real(s_mid * np.conj(self.prev_symbol - s_cur))

            # Clip error to prevent runaway in noisy bursts
            err = np.clip(err, -0.2, 0.2)

            out[out_idx] = s_cur
            out_idx += 1
            self.prev_symbol = s_cur

            # Loop filter update with integrator clipping
            self.integrator += self.ki * err
            self.integrator = np.clip(self.integrator, -1.0, 1.0)

            self.strobe_index += self.sps + self.kp * err + self.integrator

        return out[:out_idx]


class FrequencyRecovery:
    """
    Costas loop (BPSK) with soft error detector, integrator clipping, phase wrap.
    Allows copying NCO state from another instance for acquire->track flow.
    """
    def __init__(self, loop_bandwidth=0.02):
        damping = 1.0 / np.sqrt(2.0)
        norm_bw = float(loop_bandwidth)
        self.alpha = (4 * damping * norm_bw) / (1 + 2 * damping * norm_bw + norm_bw**2)
        self.beta  = (4 * norm_bw**2) / (1 + 2 * damping * norm_bw + norm_bw**2)
        self.reset()

    def reset(self):
        self.phase = 0.0       # NCO phase
        self.integrator = 0.0  # PI integrator

    def import_state(self, other):
        """Copy NCO state from another FrequencyRecovery instance (acquire->track)."""
        if isinstance(other, FrequencyRecovery):
            self.phase = float(other.phase)
            self.integrator = float(other.integrator)

    def recover(self, signal):
        out = np.zeros_like(signal, dtype=np.complex64)
        for i, x in enumerate(signal):
            # De-rotate by current NCO phase
            y = x * np.exp(-1j * self.phase)
            out[i] = y

            # ---- Costas error (soft BPSK) ----
            err = np.real(y) * np.imag(y)
            err = np.clip(err, -0.2, 0.2)

            # ---- PI controller ----
            prop = self.alpha * err
            self.integrator += self.beta * err
            self.integrator = np.clip(self.integrator, -1.0, 1.0)

            # ---- NCO update (variant B: minus sign) ----
            self.phase -= (prop + self.integrator)

            # ---- Phase wrap to [-pi, pi] ----
            if self.phase > np.pi:
                self.phase -= 2 * np.pi
            elif self.phase < -np.pi:
                self.phase += 2 * np.pi

        return out

