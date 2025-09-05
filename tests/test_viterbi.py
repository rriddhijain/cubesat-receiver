# tests/test_viterbi.py
import numpy as np
from cubesat_dataset.phase3_coding.conv_viterbi import viterbi_decode_llr

def conv_encode(bits):
    G = [0o133, 0o171]
    K = 7
    NS = 1 << (K-1)
    state = 0
    out = []
    for b in bits:
        sr = ((state << 1) | b)
        y0 = bin(sr & G[0]).count("1") & 1
        y1 = bin(sr & G[1]).count("1") & 1
        out.append(y0); out.append(y1)
        state = sr & (NS-1)
    return np.array(out, dtype=np.uint8)

def awgn(samples, snr_db):
    EbN0 = 10**(snr_db/10)
    sigma2 = 1.0/(2*EbN0)
    noise = np.sqrt(sigma2) * np.random.randn(*samples.shape)
    return samples + noise, sigma2

def test_viterbi_basic():
    np.random.seed(0)
    K = 200
    bits = np.random.randint(0,2,K)
    coded = conv_encode(bits)
    bpsk = 1 - 2 * coded  # 0->+1,1->-1
    rx, sigma2 = awgn(bpsk, snr_db=6.0)
    llr = 2.0 * rx / sigma2
    dec = viterbi_decode_llr(llr)
    ber = np.mean(dec != bits[:len(dec)])
    print("BER:", ber)
    assert ber < 0.02

if __name__ == "__main__":
    test_viterbi_basic()
    print("Viterbi unit test passed.")
