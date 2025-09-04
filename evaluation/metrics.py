import numpy as np

def calculate_ber(tx_bits, rx_bits):
    """Calculate Bit Error Rate."""
    
    # Ensure inputs are numpy arrays
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)

    # Ensure same length
    min_len = min(len(tx_bits), len(rx_bits))
    
    # If there are no bits to compare, return 0.5 (random guess)
    if min_len == 0:
        return 0.5
        
    tx_bits = tx_bits[:min_len]
    rx_bits = rx_bits[:min_len]
    
    # Count errors
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / min_len
    
    return ber

def calculate_fer(tx_frames, rx_frames):
    """Calculate Frame Error Rate."""
    
    frame_errors = 0
    for tx_frame, rx_frame in zip(tx_frames, rx_frames):
        if not np.array_equal(tx_frame, rx_frame):
            frame_errors += 1
    
    fer = frame_errors / len(tx_frames)
    return fer

def theoretical_ber_bpsk(snr_db):
    """Calculate theoretical BER for BPSK in AWGN."""
    from scipy.special import erfc
    
    snr_linear = 10**(snr_db/10)
    ber = 0.5 * erfc(np.sqrt(snr_linear))
    return ber