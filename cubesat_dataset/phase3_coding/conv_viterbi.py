# conv_viterbi.py  -- safer, normalized LLR Viterbi wrapper
import numpy as np

# Example generator polynomials for constraint length 7 (change if your code differs)
# Represented here as tuples of output bits for the two output branches.
G = (0o171, 0o133)  # common (171,133) in octal (constraint length 7)



def _bits_from_symbol_pair(x):
    # map integer 0..3 to two-bit output symbols (msb, lsb)
    return ((x >> 1) & 1, x & 1)

def viterbi_decode_llr(llr):
    """
    Decode LLR sequence `llr` expected shape (N,) for rate-1/2 (two LLRs per input bit).
    Returns hard decoded bits (numpy array uint8).
    This routine normalizes/ clips the llr values to avoid huge intermediate costs.
    """
    # Ensure numpy array
    llr = np.asarray(llr, dtype=np.float64).ravel()
    if llr.size == 0:
        return np.zeros(0, dtype=np.uint8)

    # Expect llr length to be even (2 LLRs per bit)
    if llr.size % 2 != 0:
        # drop the last symbol if odd
        llr = llr[:-1]

    # normalize to mitigate overflow (scale to RMS ~ 1.0)
    rms = np.sqrt(np.mean(llr**2)) if llr.size else 1.0
    if rms <= 0:
        rms = 1.0
    llr_norm = llr / (rms + 1e-12)

    # clip values to avoid huge exponentials/overflow in cost
    llr_norm = np.clip(llr_norm, -50.0, 50.0)

    # reshape into pairs: shape (Nsym, 2)
    s_llr = llr_norm.reshape((-1, 2))

    # Viterbi parameters
    K = 7  # constraint length; adjust if needed
    n_states = 1 << (K - 1)

    # Precompute next states and output bits for each state/input bit (0/1)
    next_state = np.zeros((n_states, 2), dtype=np.int32)
    out_sym = np.zeros((n_states, 2), dtype=np.int32)  # encoded two-bit symbols as int 0..3
    for st in range(n_states):
        for bit in (0, 1):
            shift_reg = ((st << 1) | bit) & ((1 << K) - 1)
            # compute output bits from generator polynomials
            out0 = bin(shift_reg & G[0]).count("1") & 1
            out1 = bin(shift_reg & G[1]).count("1") & 1
            out_sym[st, bit] = (out0 << 1) | out1
            next_state[st, bit] = shift_reg >> 1  # next state (drop LSB)

    # Initialize path metrics
    INF = 1e9
    pm_prev = np.full(n_states, INF, dtype=np.float64)
    pm_prev[0] = 0.0  # assume zero state start
    # store predecessor bit and previous state for traceback
    n_steps = s_llr.shape[0]
    prev_state = np.zeros((n_steps, n_states), dtype=np.int32)
    prev_input = np.zeros((n_steps, n_states), dtype=np.uint8)

    # Run Viterbi forward
    for t in range(n_steps):
        metric0 = s_llr[t, 0]
        metric1 = s_llr[t, 1]
        pm_curr = np.full(n_states, INF, dtype=np.float64)
        for st in range(n_states):
            # branch from st with input 0
            ns0 = next_state[st, 0]
            sym0 = out_sym[st, 0]
            ob0, ob1 = _bits_from_symbol_pair(sym0)
            # cost: use negative correlation: cost = -0.5 * sum( llr_i * (1 - 2*bit) )
            # implement stable: compute dot product directly and clamp
            cost0 = -0.5 * (metric0 * (1 - 2*ob0) + metric1 * (1 - 2*ob1))
            cand0 = pm_prev[st] + cost0
            if cand0 < pm_curr[ns0]:
                pm_curr[ns0] = cand0
                prev_state[t, ns0] = st
                prev_input[t, ns0] = 0

            # branch from st with input 1
            ns1 = next_state[st, 1]
            sym1 = out_sym[st, 1]
            ob0, ob1 = _bits_from_symbol_pair(sym1)
            cost1 = -0.5 * (metric0 * (1 - 2*ob0) + metric1 * (1 - 2*ob1))
            cand1 = pm_prev[st] + cost1
            if cand1 < pm_curr[ns1]:
                pm_curr[ns1] = cand1
                prev_state[t, ns1] = st
                prev_input[t, ns1] = 1

        pm_prev = pm_curr

    # Traceback: pick minimal metric state at end
    end_state = int(np.argmin(pm_prev))
    decoded_bits = np.zeros(n_steps, dtype=np.uint8)
    st = end_state
    for t in range(n_steps - 1, -1, -1):
        ibit = prev_input[t, st]
        decoded_bits[t] = ibit
        st = prev_state[t, st]

    return decoded_bits


