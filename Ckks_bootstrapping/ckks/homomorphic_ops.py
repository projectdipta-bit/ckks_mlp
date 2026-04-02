import numpy as np
from ckks.poly_ring import poly_add, poly_mul, poly_sub_X, center_mod, canonical_norm
from ckks.encoding import encode

def add(C1_tag, C2_tag, N):
    if C1_tag is None: return C2_tag
    if C2_tag is None: return C1_tag
    (C1, ell1, nu1, B1) = C1_tag
    (C2, ell2, nu2, B2) = C2_tag
    C_sum = (poly_add(C1[0], C2[0]), poly_add(C1[1], C2[1]))
    return (C_sum, ell1, nu1 + nu2, B1 + B2)

def sub(C1_tag, C2_tag, N):
    if C2_tag is None: return C1_tag
    (C1, ell1, nu1, B1) = C1_tag
    (C2, ell2, nu2, B2) = C2_tag
    C_sub = (poly_add(C1[0], -np.array(C2[0])), poly_add(C1[1], -np.array(C2[1])))
    return (C_sub, ell1, nu1 + nu2, B1 + B2)

def add_constant(C_tag, m_const, N):
    (C, ell, nu, B) = C_tag
    C_new = (poly_add(C[0], m_const), C[1])
    return (C_new, ell, nu + canonical_norm(m_const, N), B)

def scalar_mul(C_tag, m_val, N, q_ell):
    (C, ell, nu, B) = C_tag
    m_val = np.array(m_val).astype(object) if isinstance(m_val, (list, np.ndarray)) else np.array([round(m_val)] + [0]*(N-1), dtype=object)
    C_new_1 = center_mod(poly_mul(C[0], m_val, N=N), q_ell)
    C_new_2 = center_mod(poly_mul(C[1], m_val, N=N), q_ell)
    return ((C_new_1, C_new_2), ell, nu * canonical_norm(m_val, N), B * canonical_norm(m_val, N))

def mul_plain(C_tag, m_const, N, q_ell):
    return scalar_mul(C_tag, m_const, N, q_ell)

def mod_decrease(C_tag, ell_new, q_ell_new, Delta, N, B_scale):
    (C, ell, nu, B) = C_tag
    steps = ell - ell_new
    C_new = (center_mod(C[0], q_ell_new), center_mod(C[1], q_ell_new))
    factor = Delta ** steps
    return (C_new, ell_new, nu / factor, B / factor + B_scale)

def key_switch(C, KSK, q_ell, P, q, N):
    KSK0, KSK1 = KSK
    val = C[1]
    raw0 = poly_mul(val, KSK0, N=N)
    raw1 = poly_mul(val, KSK1, N=N)
    ks0 = _round_div(raw0, P)
    ks1 = _round_div(raw1, P)
    C0_new = center_mod(poly_add(C[0], ks0), q_ell)
    C1_new = center_mod(ks1, q_ell)
    return (C0_new, C1_new)

def _round_div(arr, divisor):
    """Symmetric rounding of integer array divided by divisor (works with Python ints)."""
    result = np.empty(len(arr), dtype=object)
    half = divisor // 2
    for i, x in enumerate(arr):
        x = int(x)
        if x >= 0:
            result[i] = (x + half) // divisor
        else:
            result[i] = -((-x + half) // divisor)
    return result

def multiply(C1_tag, C2_tag, EK, Delta, N, q, P, B_mult, q0):
    if C1_tag is None or C2_tag is None:
        return None
    (Ca, ell, nu1, B1) = C1_tag
    (Cb, ell2, nu2, B2) = C2_tag

    # FIX (low severity): guard against silent level mismatch. Both ciphertexts
    # must be at the same level so that q_ell is computed consistently and the
    # rescaled output level (ell-1) is meaningful.  Mismatched levels produce
    # silently wrong results that are very hard to debug.
    if ell != ell2:
        raise ValueError(
            f"multiply(): level mismatch — C1 is at level {ell}, C2 is at level {ell2}. "
            "Mod-switch the higher-level ciphertext down before multiplying."
        )

    q_ell = q0 * (Delta ** ell)

    d0 = center_mod(poly_mul(Ca[0], Cb[0], N=N), q_ell)
    d1 = center_mod(poly_add(poly_mul(Ca[0], Cb[1], N=N), poly_mul(Ca[1], Cb[0], N=N)), q_ell)
    d2 = center_mod(poly_mul(Ca[1], Cb[1], N=N), q_ell)

    # Relinearization: round(d2 * EK / P)
    # d2 MUST be reduced mod q_ell first — otherwise ||d2|| ~ N*(q/2)^2
    # and relin error ~ N*||d2||*sigma/P overflows q
    ks0 = _round_div(poly_mul(d2, EK[0], N=N), P)
    ks1 = _round_div(poly_mul(d2, EK[1], N=N), P)
    C0 = center_mod(poly_add(d0, ks0), q_ell)
    C1_out = center_mod(poly_add(d1, ks1), q_ell)

    # Rescale by Delta
    q_new = q_ell // Delta
    C0_rs = center_mod(_round_div(C0, Delta), q_new)
    C1_rs = center_mod(_round_div(C1_out, Delta), q_new)

    new_B = (nu1 * B2 + nu2 * B1 + B1 * B2 + B_mult) / Delta
    return ((C0_rs, C1_rs), ell - 1, nu1 * nu2, new_B)

def square(C_tag, EK, Delta, N, q, P, B_mult, q0):
    return multiply(C_tag, C_tag, EK, Delta, N, q, P, B_mult, q0)

def rotate(C_tag, j, rot_keys, N, q_ell, P, q, B_ks):
    (C, ell, nu, B) = C_tag

    # FIX (low severity): guard against missing rotation key with a clear error
    # instead of an opaque KeyError deep inside key_switch.
    if j not in rot_keys:
        raise KeyError(
            f"rotate(): no rotation key for step j={j}. "
            f"Call gen_rotation_keys(..., max_j>={j}) to generate it."
        )

    exp = pow(5, j, 2*N)
    C_rot = (poly_sub_X(C[0], exp, N), poly_sub_X(C[1], exp, N))
    C_ks = key_switch(C_rot, rot_keys[j], q_ell, P, q, N)
    return (C_ks, ell, nu, B + B_ks)

def conjugate(C_tag, conj_key, N, q_ell, P, q, B_ks):
    (C, ell, nu, B) = C_tag
    C_conj = (poly_sub_X(C[0], 2*N-1, N), poly_sub_X(C[1], 2*N-1, N))
    C_ks = key_switch(C_conj, conj_key, q_ell, P, q, N)
    return (C_ks, ell, nu, B + B_ks)

def linear_transform(C_tag, A, rot_keys, params, hom_ops_wrapper):
    N = params.N
    Delta = params.Delta
    N2 = N // 2
    result = None
    for j in range(N2):
        a_j = np.array([A[i, (i + j) % N2] for i in range(N2)])
        if np.all(np.abs(a_j) < 1e-10):
            continue
        m_aj = encode(a_j, N, Delta)
        C_aj = (m_aj, np.zeros(N, int))
        q_ell = params.q0 * (params.Delta ** C_tag[1])
        if j > 0:
            C_rot = rotate(C_tag, j, rot_keys, N, q_ell, params.P, params.q, 0)
        else:
            C_rot = C_tag
        term = scalar_mul(C_rot, C_aj[0], N, q_ell)
        result = add(result, term, N)
    return result

def r_linear_transform(C_tag, A, B, rot_keys, conj_key, params, hom_ops_wrapper):
    part_A = linear_transform(C_tag, A, rot_keys, params, hom_ops_wrapper)
    q_ell = params.q0 * (params.Delta ** C_tag[1])
    C_conj = conjugate(C_tag, conj_key, params.N, q_ell, params.P, params.q, 0)
    part_B = linear_transform(C_conj, B, rot_keys, params, hom_ops_wrapper)
    return add(part_A, part_B, params.N)

class HomOpsWrapper:
    def __init__(self, params, EK=None):
        self.params = params
        self.EK = EK

    def add(self, c1, c2):
        return add(c1, c2, self.params.N)

    def sub(self, c1, c2):
        return sub(c1, c2, self.params.N)

    def multiply(self, c1, c2):
        return multiply(c1, c2, self.EK, self.params.Delta, self.params.N, self.params.q, self.params.P, 0, self.params.q0)

    def square(self, c):
        return square(c, self.EK, self.params.Delta, self.params.N, self.params.q, self.params.P, 0, self.params.q0)

    def scalar_mul(self, c, val):
        q_ell = self.params.q0 * (self.params.Delta ** c[1])
        return scalar_mul(c, val, self.params.N, q_ell)