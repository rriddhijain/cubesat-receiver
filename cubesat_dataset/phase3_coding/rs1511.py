# rs1511.py
import numpy as np

# RS(15,11) over GF(16) with primitive poly x^4 + x + 1 (0x13)
PRIM_POLY = 0x13
M = 4
FIELD_SIZE = 1 << M  # 16
ALOG = np.zeros(2*(FIELD_SIZE-1), dtype=np.int16)
LOG = np.full(FIELD_SIZE, -1, dtype=np.int16)

# Build log/antilog tables (alpha = 2)
x = 1
for i in range(FIELD_SIZE - 1):
    ALOG[i] = x
    LOG[x] = i
    x <<= 1
    if x & FIELD_SIZE:
        x ^= PRIM_POLY
# duplicate
ALOG[FIELD_SIZE-1:2*(FIELD_SIZE-1)] = ALOG[:FIELD_SIZE-1]

def gf_add(a, b):
    return int(a ^ b)

def gf_sub(a, b):
    return int(a ^ b)  # same as add in characteristic 2

def gf_mul(a, b):
    a = int(a); b = int(b)
    if a == 0 or b == 0:
        return 0
    return int(ALOG[ (LOG[a] + LOG[b]) % (FIELD_SIZE-1) ])

def gf_inv(a):
    a = int(a)
    if a == 0:
        raise ZeroDivisionError("Inverse of zero")
    return int(ALOG[ (FIELD_SIZE-1 - LOG[a]) % (FIELD_SIZE-1) ])

def gf_pow(a, e):
    a = int(a)
    if a == 0:
        return 0
    return int(ALOG[ (LOG[a] * e) % (FIELD_SIZE-1) ])

# RS parameters
n = 15
k = 11
r = n - k  # 4
t = r // 2
alpha = 2  # primitive element

def poly_scale(p, s):
    return [gf_mul(c, s) for c in p]

def poly_add(p, q):
    # add polynomials (coeffs lowest-first)
    m = max(len(p), len(q))
    res = [0]*m
    for i in range(len(p)):
        res[i] ^= p[i]
    for i in range(len(q)):
        res[i] ^= q[i]
    return res

def poly_mul(p, q):
    res = [0]*(len(p)+len(q)-1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            res[i+j] ^= gf_mul(a, b)
    return res

def poly_eval(poly, x_):
    # Horner evaluate (poly coeffs lowest-first)
    y = 0
    for c in reversed(poly):
        y = gf_mul(y, x_) ^ c
    return y

# build generator polynomial g(x) = (x - alpha^1)(x - alpha^2)...(x - alpha^r)
# Represent polynomials as lists of coefficients in ascending order [c0, c1, c2, ...] (c0 + c1 x + ...)
g = [1]
for i in range(1, r+1):
    root = gf_pow(alpha, i)
    # (x - root) -> coefficients: [root, 1] because (root) + (1)*x in ascending order (c0=root, c1=1)
    g = poly_mul(g, [root, 1])

def rs_encode(msg_symbols):
    """
    Systematic encoding: msg_symbols length k -> returns codeword length n.
    msg_symbols: list/array of ints 0..15 (length k)
    """
    # message polynomial m(x) with degree < k: coeffs lowest-first
    m = list(map(int, msg_symbols))
    # multiply m(x) by x^r -> shift by r zeros
    padded = [0]*(r) + m[:]  # (coeffs lowest-first)
    # compute remainder of padded / g
    # polynomial division long division
    rem = padded[:]  # mutated copy
    for i in range(len(m)-1, -1, -1):
        # index in rem corresponding to x^{i + r}
        coef = rem[i + r]
        if coef != 0:
            for j, gj in enumerate(g):
                rem[i + j] ^= gf_mul(coef, gj)
    parity = rem[:r]
    cw = np.array(m + parity, dtype=np.uint8)
    return cw

def berlekamp_massey(S):
    """
    S: syndrome list S1..Sr (length r), S[i] is S_{i+1}
    returns Lambda polynomial coefficients (lowest-first)
    """
    N = len(S)
    C = [1] + [0]*N
    B = [1] + [0]*N
    L = 0
    m = 1
    b = 1
    for n_ in range(N):
        # compute discrepancy
        d = S[n_]
        for i in range(1, L+1):
            d ^= gf_mul(C[i], S[n_-i])
        if d == 0:
            m += 1
        else:
            T = C.copy()
            coef = gf_mul(d, gf_inv(b))
            # C = C - coef * x^m * B
            for i in range(0, N+1-m):
                if B[i]:
                    C[i+m] ^= gf_mul(coef, B[i])
            if 2*L <= n_:
                L_new = n_ + 1 - L
                B = T
                b = d
                L = L_new
                m = 1
            else:
                m += 1
    # trim to degree L
    C = C[:L+1]
    return C

def chien_search(Lambda):
    """
    Find error positions from Lambda (coeffs lowest-first).
    Return list of positions (0..n-1) where errors occurred (positions counted from 0 for leftmost codeword symbol?).
    We'll follow convention: codeword symbols cw[0] ... cw[n-1] correspond to evaluation points alpha^{n-1},...,alpha^{0}
    We'll search for i in 0..n-1 where Lambda(alpha^{-i}) == 0, and position = n-1 - i
    """
    err_pos = []
    for i in range(n):
        x = gf_pow(alpha, (FIELD_SIZE-1 - i) % (FIELD_SIZE-1))  # alpha^{-i}
        # evaluate Lambda at x
        val = 0
        xp = 1
        for coeff in Lambda:
            if coeff:
                val ^= gf_mul(coeff, xp)
            xp = gf_mul(xp, x)
        if val == 0:
            # error at position n-1 - i
            pos = n-1 - i
            err_pos.append(pos)
    return err_pos

def rs_decode(cw):
    """
    cw: iterable length n (15) with values 0..15
    returns: (message_symbols length k as ndarray, n_errors, ok_flag)
    """
    cw = list(map(int, cw))
    # compute syndromes S1..Sr
    S = []
    any_nonzero = False
    for i in range(1, r+1):
        s = poly_eval(cw[::-1], gf_pow(alpha, i))  # poly_eval expects lowest-first; cw reversed so index matches power
        S.append(s)
        if s != 0:
            any_nonzero = True
    if not any_nonzero:
        # no errors
        return np.array(cw[:k], dtype=np.uint8), 0, True

    # BM to get error locator polynomial Lambda
    Lambda = berlekamp_massey(S)  # lowest-first
    L = len(Lambda) - 1
    # Chien search
    err_pos = chien_search(Lambda)
    if len(err_pos) == 0 or len(err_pos) > t:
        return np.array(cw[:k], dtype=np.uint8), len(err_pos), False

    # Compute error evaluator Omega = S(x) * Lambda(x) mod x^r
    # Build syndrome polynomial S(x) where S(x) = S1 + S2 x + S3 x^2 + ...
    S_poly = S[:]  # length r
    Omega = poly_mul(S_poly, Lambda)
    Omega = Omega[:r]  # mod x^r

    # Lambda' (formal derivative) (only odd powers survive in characteristic 2)
    Lambda_der = []
    for j in range(1, len(Lambda)):
        if j % 2 == 1:
            Lambda_der.append(Lambda[j])
    # Forney to compute error magnitudes
    cw_corrected = cw[:]
    for pos in err_pos:
        # X = alpha^{pos}
        X = gf_pow(alpha, pos)
        # Evaluate Omega at X^{-1}
        Xinv = gf_inv(X)
        num = 0
        xp = 1
        for coef in Omega:
            if coef:
                num ^= gf_mul(coef, xp)
            xp = gf_mul(xp, Xinv)
        # Evaluate Lambda_der at X^{-1}
        den = 0
        xp = 1
        for coef in Lambda_der:
            if coef:
                den ^= gf_mul(coef, xp)
            xp = gf_mul(xp, Xinv)
        if den == 0:
            return np.array(cw[:k], dtype=np.uint8), len(err_pos), False
        err_val = gf_mul(num, gf_inv(den))
        cw_corrected[pos] ^= err_val
    # Return message (first k symbols)
    return np.array(cw_corrected[:k], dtype=np.uint8), len(err_pos), True

# Helper bit/symbol pack/unpack
def bits_to_symbols_4bit(bits):
    bits = np.asarray(bits).astype(np.uint8).ravel()
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits4 = bits.reshape(-1, 4)
    sym = (bits4[:,0].astype(np.uint8) << 3) | (bits4[:,1].astype(np.uint8) << 2) | (bits4[:,2].astype(np.uint8) << 1) | bits4[:,3].astype(np.uint8)
    return sym.astype(np.uint8), pad

def symbols_to_bits_4bit(sym, pad=0):
    sym = np.asarray(sym).astype(np.uint8).ravel()
    arr = np.zeros((len(sym), 4), dtype=np.uint8)
    for i, s in enumerate(sym):
        arr[i,0] = (s >> 3) & 1
        arr[i,1] = (s >> 2) & 1
        arr[i,2] = (s >> 1) & 1
        arr[i,3] = s & 1
    bits = arr.reshape(-1)
    if pad:
        bits = bits[:-pad]
    return bits.astype(np.uint8)

def rs1511_decode_bits(coded_bits):
    """
    Top-level helper used by integrator.
    coded_bits: 1D array of 0/1 (concatenation of RS codewords; 60 bits per codeword)
    Returns:
      decoded_bits, fer_count, pad
    """
    sym, pad = bits_to_symbols_4bit(coded_bits)
    n_cw = len(sym) // n
    if n_cw == 0:
        return np.array([], dtype=np.uint8), 0, pad
    sym = sym[:n_cw*n].reshape(n_cw, n)
    msgs = []
    fer = 0
    for cw in sym:
        m, nerr, ok = rs_decode(cw)
        if not ok:
            fer += 1
        msgs.append(np.asarray(m, dtype=np.uint8))
    msgs = np.vstack(msgs).reshape(-1)
    bits_out = symbols_to_bits_4bit(msgs, pad=0)
    return bits_out, fer, pad
