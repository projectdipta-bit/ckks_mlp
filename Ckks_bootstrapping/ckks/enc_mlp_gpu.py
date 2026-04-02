"""
enc_mlp_gpu.py — GPU-accelerated version of enc_mlp.py.

What changed vs enc_mlp.py
---------------------------
1.  poly_mul / center_mod / _round_div
      All replaced with RNS equivalents from poly_ring_gpu.py.
      Ciphertext polynomial coefficients are stored as (k, N) int64 arrays
      (one row per RNS prime) instead of (N,) dtype=object bigint arrays.
      On a machine with an NVIDIA GPU and CuPy installed these arrays live
      in GPU memory and all arithmetic runs on-device.

2.  Parallel forward/backward loops
      The hid_dim matmul rows (W1 @ x) and out_dim rows (W2 @ h) are
      independent.  We dispatch them with concurrent.futures.ThreadPoolExecutor
      (threads, not processes — no pickling overhead, GIL released inside
      CuPy/NumPy C extensions).

3.  Everything else (level tracking, noise tags, key structure, privacy
    model) is identical to enc_mlp.py.

Installation
------------
  GPU path  (NVIDIA GPU required):
      pip install cupy-cuda12x          # match your CUDA version
      # cupy-cuda11x for CUDA 11, etc.

  CPU fallback (no GPU / no CuPy):
      poly_ring_gpu.py automatically falls back to NumPy.
      No code changes required here.

  Other deps (same as enc_mlp.py):
      pip install scikit-learn numpy

Running
-------
  cd .../Ckks_bootstrapping          # project root (parent of ckks/)
  python -m ckks.enc_mlp_gpu         # uses __main__ block below
"""

import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Import the GPU/CPU polynomial ring
# ---------------------------------------------------------------------------
from ckks.poly_ring_gpu import (
    RNSParams, rns_poly_mul, rns_poly_add, rns_poly_sub,
    rns_center_mod, rns_round_div, rns_poly_sub_X,
    bigint_to_rns, rns_to_bigint, rns_of_scalar, backend_name,
    _GPU_AVAILABLE, xp,
)

# ---------------------------------------------------------------------------
# Lazy RNSParams cache  (one set of precomputed tables per (N, q_bits))
# ---------------------------------------------------------------------------
_RNS_CACHE = {}

def get_rns_params(N, q, Delta, P):
    """Return a cached RNSParams allocating enough limbs for P * q."""
    max_modulus = int(P) * int(q)
    q_bits = max_modulus.bit_length() + 2   # +2 for headroom
    key = (N, q_bits)
    if key not in _RNS_CACHE:
        rp = RNSParams(N, q_bits=q_bits)
        rp.set_delta(int(Delta))   
        _RNS_CACHE[key] = rp
    return _RNS_CACHE[key]

# ---------------------------------------------------------------------------
# Drop-in replacements for poly_ring primitives, working on RNS ciphertexts
# ---------------------------------------------------------------------------

def _to_rns(poly_obj_or_rns, rp):
    """
    Accept either a dtype=object bigint array (legacy) or an already-converted
    (k, N) xp int64 array.  Always return (k, N) xp int64.
    """
    if isinstance(poly_obj_or_rns, np.ndarray) and poly_obj_or_rns.dtype == object:
        return bigint_to_rns(poly_obj_or_rns, rp)
    return poly_obj_or_rns


def _poly_mul_gpu(a, b, rp, q_rns):
    """poly_mul + center_mod on GPU."""
    a_r = _to_rns(a, rp)
    b_r = _to_rns(b, rp)
    c_r = rns_poly_mul(a_r, b_r, rp)
    return rns_center_mod(c_r, q_rns, rp)


def _poly_add_gpu(a, b, rp, q_rns=None):
    """poly_add (+optional center_mod) on GPU."""
    a_r = _to_rns(a, rp)
    b_r = _to_rns(b, rp)
    c_r = rns_poly_add(a_r, b_r, rp)
    if q_rns is not None:
        c_r = rns_center_mod(c_r, q_rns, rp)
    return c_r


def _center_mod_gpu(a, q_rns, rp):
    a_r = _to_rns(a, rp)
    return rns_center_mod(a_r, q_rns, rp)


def _round_div_gpu(a, d, rp):
    """round(a / d) on GPU (requires CPU round-trip for bigint division)."""
    a_r = _to_rns(a, rp)
    return rns_round_div(a_r, d, rp)


# ---------------------------------------------------------------------------
# Ciphertext type
#   A ciphertext is a tuple  (C, ell, nu, B)
#   where C = (C0, C1) and C0, C1 are (k, N) xp int64 RNS arrays.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cached Vandermonde inverse (for encode_fast)
# ---------------------------------------------------------------------------
_VINV_CACHE = {}

def _get_vinv(N):
    if N not in _VINV_CACHE:
        zeta = np.exp(1j * np.pi / N)
        slot_exp  = [pow(5, j, 2*N) for j in range(N // 2)]
        conj_exp  = [(2*N - e) % (2*N) for e in slot_exp]
        all_roots = [zeta**e for e in slot_exp + conj_exp]
        V = np.array([[r**j for j in range(N)] for r in all_roots], dtype=complex)
        _VINV_CACHE[N] = np.linalg.inv(V)
    return _VINV_CACHE[N]


def encode_fast(z, N, Delta):
    """Encode a complex vector to a bigint polynomial, then convert to RNS."""
    V_inv    = _get_vinv(N)
    all_vals = list(Delta * np.array(z)) + list(Delta * np.conj(z))
    p_coeffs = V_inv @ all_vals
    poly_obj = np.round(p_coeffs.real).astype(int).astype(object)
    return poly_obj   # returned as object array; callers call bigint_to_rns as needed


def decode(m_rns_or_obj, N, Delta, rp=None):
    """Decode an RNS or object-array polynomial back to complex slots."""
    if rp is not None and not (isinstance(m_rns_or_obj, np.ndarray) and m_rns_or_obj.dtype == object):
        m_obj = rns_to_bigint(m_rns_or_obj, rp)
    else:
        m_obj = m_rns_or_obj
    zeta     = np.exp(1j * np.pi / N)
    slot_exp = [pow(5, j, 2*N) for j in range(N // 2)]
    roots    = [zeta**e for e in slot_exp]
    evals    = [sum(int(m_obj[i]) * (r**i) for i in range(N)) for r in roots]
    return [e / Delta for e in evals]


# ---------------------------------------------------------------------------
# Key generation (GPU-aware)
#   Keys are stored as (k, N) RNS arrays on the target device.
# ---------------------------------------------------------------------------

import random as pyrandom

def _rand_poly_rns(q, N, rp):
    """Random polynomial in Z_q^N, represented in RNS."""
    poly_obj = np.array(
        [pyrandom.randrange(0, int(q)) for _ in range(N)], dtype=object
    )
    return bigint_to_rns(poly_obj, rp)


def _small_poly_rns(N, sigma, rp):
    """Gaussian small polynomial, represented in RNS."""
    poly_obj = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    return bigint_to_rns(poly_obj, rp)


def _ternary_poly_rns(N, rp):
    """Ternary small polynomial {-1,0,1}^N in RNS."""
    poly_obj = np.random.choice([-1, 0, 1], size=N).astype(object)
    return bigint_to_rns(poly_obj, rp)


def gen_secret_key_gpu(N, rp):
    s_obj = np.random.choice([-1, 0, 1], size=N).astype(object)
    s_rns = bigint_to_rns(s_obj, rp)
    return s_rns, s_obj   # return both; obj needed for legacy comparisons


def gen_public_key_gpu(s_rns, N, q, rp, sigma=3.2):
    q_rns = rns_of_scalar(q, rp)
    a     = _rand_poly_rns(q, N, rp)
    e     = _small_poly_rns(N, sigma, rp)
    
    # PK1 = [-a*s + e]_q
    PK1 = rns_center_mod(
        rns_poly_sub(e, rns_poly_mul(a, s_rns, rp), rp), q_rns, rp)
    
    # PK2 = a
    PK2 = rns_center_mod(a, q_rns, rp)
    return (PK1, PK2)


def gen_eval_key_gpu(s_rns, N, q, P, rp, sigma=3.2):
    Pq    = P * q
    Pq_rns = rns_of_scalar(Pq, rp)
    q_rns  = rns_of_scalar(q, rp)
    P_rns  = rns_of_scalar(P, rp)
    a_ek   = _rand_poly_rns(Pq, N, rp)
    e_ek   = _small_poly_rns(N, sigma, rp)
    s2     = rns_poly_mul(s_rns, s_rns, rp)
    Ps2    = xp.stack([s2[i] * int(P_rns[i]) % rp.primes[i] for i in range(rp.k)])
    # EK1 = [-a*s + e + P*s^2]_{Pq}
    neg_as = rns_poly_sub(e_ek, rns_poly_mul(a_ek, s_rns, rp), rp)
    EK1 = rns_center_mod(rns_poly_add(neg_as, Ps2, rp), Pq_rns, rp)
    EK2 = rns_center_mod(a_ek, Pq_rns, rp)
    return (EK1, EK2)


def gen_ksk_gpu(s_from_rns, s_to_rns, N, q, P, rp, sigma=3.2):
    Pq     = P * q
    Pq_rns = rns_of_scalar(Pq, rp)
    P_rns  = rns_of_scalar(P, rp)
    a      = _rand_poly_rns(Pq, N, rp)
    e      = _small_poly_rns(N, sigma, rp)
    Ps_from = xp.stack([s_from_rns[i] * int(P_rns[i]) % rp.primes[i] for i in range(rp.k)])
    neg_as  = rns_poly_sub(e, rns_poly_mul(a, s_to_rns, rp), rp)
    KSK1 = rns_center_mod(rns_poly_add(neg_as, Ps_from, rp), Pq_rns, rp)
    KSK2 = rns_center_mod(a, Pq_rns, rp)
    return (KSK1, KSK2)


def gen_rotation_keys_gpu(s_rns, N, q, P, rp, max_j=None):
    limit = max_j + 1 if max_j is not None else N // 2
    rot_keys = {}
    for j in range(1, limit):
        exp    = pow(5, j, 2 * N)
        s_auto = rns_poly_sub_X(s_rns, exp, rp)
        rot_keys[j] = gen_ksk_gpu(s_auto, s_rns, N, q, P, rp)
    return rot_keys


def gen_conj_key_gpu(s_rns, N, q, P, rp):
    s_conj = rns_poly_sub_X(s_rns, 2 * N - 1, rp)
    return gen_ksk_gpu(s_conj, s_rns, N, q, P, rp)


# ---------------------------------------------------------------------------
# Encrypt / Decrypt
# ---------------------------------------------------------------------------

def encrypt_gpu(m_obj, PK, N, q, rp, sigma=3.2):
    """Encrypt a dtype=object plaintext polynomial. Returns RNS ciphertext tuple."""
    q_rns = rns_of_scalar(q, rp)
    m_rns = bigint_to_rns(m_obj, rp)
    u     = _ternary_poly_rns(N, rp)
    e1    = _small_poly_rns(N, sigma, rp)
    e2    = _small_poly_rns(N, sigma, rp)
    # C1 = [u*PK0 + m + e1]_q
    C1 = rns_center_mod(
        rns_poly_add(rns_poly_add(rns_poly_mul(u, PK[0], rp), m_rns, rp), e1, rp),
        q_rns, rp)
    # C2 = [u*PK1 + e2]_q
    C2 = rns_center_mod(
        rns_poly_add(rns_poly_mul(u, PK[1], rp), e2, rp),
        q_rns, rp)
    return (C1, C2)


def decrypt_gpu(C, s_rns, q_ell, N, rp):
    """Decrypt an RNS ciphertext. Returns dtype=object polynomial."""
    q_rns = rns_of_scalar(q_ell, rp)
    val   = rns_center_mod(
        rns_poly_add(C[0], rns_poly_mul(C[1], s_rns, rp), rp),
        q_rns, rp)
    return rns_to_bigint(val, rp)


# ---------------------------------------------------------------------------
# Homomorphic operations on RNS ciphertexts
# ---------------------------------------------------------------------------

def _key_switch_gpu(C, KSK, q_ell, P, rp):
    """Key-switch C[1] using KSK; returns new (C0, C1) RNS pair."""
    q_rns  = rns_of_scalar(q_ell, rp)
    P_int  = int(P)
    val    = C[1]
    raw0   = rns_poly_mul(val, KSK[0], rp)
    raw1   = rns_poly_mul(val, KSK[1], rp)
    ks0    = rns_round_div(raw0, P_int, rp)
    ks1    = rns_round_div(raw1, P_int, rp)
    C0_new = rns_center_mod(rns_poly_add(C[0], ks0, rp), q_rns, rp)
    C1_new = rns_center_mod(ks1, q_rns, rp)
    return (C0_new, C1_new)


def add_gpu(C1_tag, C2_tag, rp):
    if C1_tag is None: return C2_tag
    if C2_tag is None: return C1_tag
    (C1, ell1, nu1, B1) = C1_tag
    (C2, ell2, nu2, B2) = C2_tag
    C_sum = (rns_poly_add(C1[0], C2[0], rp),
             rns_poly_add(C1[1], C2[1], rp))
    return (C_sum, ell1, nu1 + nu2, B1 + B2)


def multiply_gpu(C1_tag, C2_tag, EK, Delta, N, q, P, B_mult, q0, rp):
    if C1_tag is None or C2_tag is None: return None
    (Ca, ell, nu1, B1) = C1_tag
    (Cb, ell2, nu2, B2) = C2_tag
    if ell != ell2:
        raise ValueError(f"multiply_gpu: level mismatch {ell} vs {ell2}")
    q_ell  = q0 * (Delta ** ell)
    q_rns  = rns_of_scalar(q_ell, rp)
    d0 = rns_center_mod(rns_poly_mul(Ca[0], Cb[0], rp), q_rns, rp)
    d1 = rns_center_mod(rns_poly_add(
             rns_poly_mul(Ca[0], Cb[1], rp),
             rns_poly_mul(Ca[1], Cb[0], rp), rp), q_rns, rp)
    d2 = rns_center_mod(rns_poly_mul(Ca[1], Cb[1], rp), q_rns, rp)
    ks0 = rns_round_div(rns_poly_mul(d2, EK[0], rp), int(P), rp)
    ks1 = rns_round_div(rns_poly_mul(d2, EK[1], rp), int(P), rp)
    C0  = rns_center_mod(rns_poly_add(d0, ks0, rp), q_rns, rp)
    C1o = rns_center_mod(rns_poly_add(d1, ks1, rp), q_rns, rp)
    # Rescale
    q_new = q_ell // Delta
    q_new_rns = rns_of_scalar(q_new, rp)
    C0_rs = rns_center_mod(rns_round_div(C0, int(Delta), rp), q_new_rns, rp)
    C1_rs = rns_center_mod(rns_round_div(C1o, int(Delta), rp), q_new_rns, rp)
    new_B = (nu1 * B2 + nu2 * B1 + B1 * B2 + B_mult) / Delta
    return ((C0_rs, C1_rs), ell - 1, nu1 * nu2, new_B)


def square_gpu(C_tag, EK, Delta, N, q, P, B_mult, q0, rp):
    return multiply_gpu(C_tag, C_tag, EK, Delta, N, q, P, B_mult, q0, rp)


def rotate_gpu(C_tag, j, rot_keys, N, q_ell, P, q, rp):
    (C, ell, nu, B) = C_tag
    if j not in rot_keys:
        raise KeyError(f"rotate_gpu: no rotation key for j={j}")
    exp   = pow(5, j, 2 * N)
    C_rot = (rns_poly_sub_X(C[0], exp, rp), rns_poly_sub_X(C[1], exp, rp))
    C_ks  = _key_switch_gpu(C_rot, rot_keys[j], q_ell, P, rp)
    return (C_ks, ell, nu, B)


def mod_decrease_gpu(C_tag, ell_new, q_ell_new, Delta, rp):
    (C, ell, nu, B) = C_tag
    q_rns = rns_of_scalar(q_ell_new, rp)
    C_new = (rns_center_mod(C[0], q_rns, rp),
             rns_center_mod(C[1], q_rns, rp))
    steps  = ell - ell_new
    factor = Delta ** steps
    return (C_new, ell_new, nu / factor, B / factor)


# ---------------------------------------------------------------------------
# Helpers matching enc_mlp.py API
# ---------------------------------------------------------------------------

def make_enc_x_gpu(x_norm, PK, params, rp):
    padded = list(x_norm) + [0.0] * (params.N // 2 - len(x_norm))
    m_obj  = encode_fast(padded, params.N, params.Delta)
    C      = encrypt_gpu(m_obj, PK, params.N, params.q, rp)
    return (C, params.L, 1, 0)


def dec_scalar_gpu(enc_j, s_rns, params, rp):
    q_l   = params.q0 * (params.Delta ** enc_j[1])
    m_obj = decrypt_gpu(enc_j[0], s_rns, q_l, params.N, rp)
    vals  = decode(m_obj, params.N, params.Delta)
    return float(vals[0].real)


def _enc_elemwise_rescale_gpu(m_obj, enc_v, params, rp):
    """Plain-poly * ciphertext rescale — costs 1 level."""
    q_ell  = params.q0 * (params.Delta ** enc_v[1])
    q_rns  = rns_of_scalar(q_ell, rp)
    m_rns  = bigint_to_rns(m_obj, rp)
    C      = enc_v[0]
    C0     = rns_center_mod(rns_poly_mul(C[0], m_rns, rp), q_rns, rp)
    C1     = rns_center_mod(rns_poly_mul(C[1], m_rns, rp), q_rns, rp)
    q_new  = q_ell // params.Delta
    q_new_rns = rns_of_scalar(q_new, rp)
    C0_rs  = rns_center_mod(rns_round_div(C0, int(params.Delta), rp), q_new_rns, rp)
    C1_rs  = rns_center_mod(rns_round_div(C1, int(params.Delta), rp), q_new_rns, rp)
    return ((C0_rs, C1_rs), enc_v[1] - 1, enc_v[2], enc_v[3])


def _slot_sum_gpu(enc_v, n, rot_keys, params, rp):
    """Rotate-and-add tree — costs 0 levels."""
    result = enc_v
    step   = 1
    while step < n:
        q_ell   = params.q0 * (params.Delta ** result[1])
        if step not in rot_keys:
            raise RuntimeError(f"_slot_sum_gpu: missing rotation key for step={step}")
        rotated = rotate_gpu(result, step, rot_keys, params.N, q_ell, params.P, params.q, rp)
        result  = add_gpu(result, rotated, rp)
        step   *= 2
    return result


def enc_plain_scalar_mul_gpu(enc_v, w_float, params, rp):
    """Multiply all slots by a scalar float — costs 1 level."""
    coeff = int(round(w_float * params.Delta))
    m_obj = np.zeros(params.N, dtype=object)
    m_obj[0] = coeff
    return _enc_elemwise_rescale_gpu(m_obj, enc_v, params, rp)


def enc_sub_plain_scalar_gpu(enc_scalar, plain_val, params, rp):
    """Subtract a plaintext scalar from slot 0 — costs 0 levels."""
    coeff  = int(round(plain_val * params.Delta))
    neg    = np.zeros(params.N, dtype=object)
    neg[0] = coeff
    neg_rns = bigint_to_rns(neg, rp)
    q_ell  = params.q0 * (params.Delta ** enc_scalar[1])
    q_rns  = rns_of_scalar(q_ell, rp)
    C      = enc_scalar[0]
    C0_new = rns_center_mod(rns_poly_sub(C[0], neg_rns, rp), q_rns, rp)
    return ((C0_new, C[1]), enc_scalar[1], enc_scalar[2], enc_scalar[3])


def _mod_switch_down_gpu(enc_v, target_ell, params, rp):
    cur_ell = enc_v[1]
    if target_ell > cur_ell:
        raise ValueError(f"Cannot switch up from level {cur_ell} to {target_ell}")
    if target_ell == cur_ell:
        return enc_v
    q_target = params.q0 * (params.Delta ** target_ell)
    return mod_decrease_gpu(enc_v, target_ell, q_target, params.Delta, rp)


# ---------------------------------------------------------------------------
# EncMLP_GPU — drop-in replacement for EncMLP
# ---------------------------------------------------------------------------

class EncMLP_GPU:
    """
    GPU-accelerated 1-hidden-layer MLP with encrypted inputs.

    API is identical to EncMLP in enc_mlp.py; internally all polynomial
    arithmetic runs through poly_ring_gpu (CuPy on GPU, NumPy on CPU).

    Extra parameter
    ---------------
    rp : RNSParams instance.  Create with:
           rp = get_rns_params(params.N, params.q)
         Pass the same rp to all GPU key-gen and helper functions.

    n_workers : number of parallel threads for the matmul loops.
                Default = min(hid_dim, 8).  Set 1 to disable parallelism.
    """

    def __init__(self, in_dim, hid_dim, out_dim, params, EK, rot_keys,
                 PK, s_rns, rp, seed=42, grad_clip=3.0, n_workers=None):
        self.in_dim    = in_dim
        self.hid_dim   = hid_dim
        self.out_dim   = out_dim
        self.params    = params
        self.EK        = EK
        self.rot_keys  = rot_keys
        self.PK        = PK
        self.s         = s_rns
        self.rp        = rp
        self.grad_clip = grad_clip
        self.n_workers = n_workers or min(hid_dim, 8)

        rng      = np.random.default_rng(seed)
        self.W1  = rng.standard_normal((hid_dim, in_dim))  * np.sqrt(1.0 / in_dim)
        self.W2  = rng.standard_normal((out_dim, hid_dim)) * np.sqrt(1.0 / hid_dim)
        self._W1_enc = None
        self._W2_enc = None
        self._encode_weights()

    # ------------------------------------------------------------------
    # Weight encoding cache
    # ------------------------------------------------------------------

    def _encode_weights(self):
        p  = self.params
        n2 = p.N // 2
        def enc_rows(W, n_cols):
            return [encode_fast(list(row) + [0.0] * (n2 - n_cols), p.N, p.Delta)
                    for row in W]
        self._W1_enc = enc_rows(self.W1, self.in_dim)
        self._W2_enc = enc_rows(self.W2, self.hid_dim)

    # ------------------------------------------------------------------
    # Activation packing
    # ------------------------------------------------------------------

    def _pack_scalars(self, enc_list):
        p    = self.params
        vals = [dec_scalar_gpu(e, self.s, p, self.rp) for e in enc_list]
        padded = vals + [0.0] * (p.N // 2 - len(vals))
        m_obj  = encode_fast(padded, p.N, p.Delta)
        C      = encrypt_gpu(m_obj, self.PK, p.N, p.q, self.rp)
        return (C, p.L, 1, 0)

    # ------------------------------------------------------------------
    # Forward pass  (parallelised over weight rows)
    # ------------------------------------------------------------------

    def forward(self, enc_x):
        p   = self.params
        rp  = self.rp

        # --- Layer 1: h_pre = W1 @ x  (parallel over hid_dim rows) ---
        def _fwd_row1(m_obj):
            enc_dot = _enc_elemwise_rescale_gpu(m_obj, enc_x, p, rp)
            return _slot_sum_gpu(enc_dot, self.in_dim, self.rot_keys, p, rp)

        enc_h_pre = _parallel_map(_fwd_row1, self._W1_enc, self.n_workers)

        # --- Activation: h_act = h_pre^2 ---
        def _square_row(h):
            return square_gpu(h, self.EK, p.Delta, p.N, p.q, p.P, 0, p.q0, rp)

        enc_h_act = _parallel_map(_square_row, enc_h_pre, self.n_workers)

        # --- Pack scalars and re-encrypt at level L ---
        enc_h_packed = self._pack_scalars(enc_h_act)

        # --- Layer 2: out = W2 @ h  (parallel over out_dim rows) ---
        def _fwd_row2(m_obj):
            enc_dot = _enc_elemwise_rescale_gpu(m_obj, enc_h_packed, p, rp)
            return _slot_sum_gpu(enc_dot, self.hid_dim, self.rot_keys, p, rp)

        enc_out = _parallel_map(_fwd_row2, self._W2_enc, self.n_workers)

        cache = dict(enc_x=enc_x, enc_h_pre=enc_h_pre, enc_h_act=enc_h_act)
        return enc_out, cache

    # ------------------------------------------------------------------
    # Backward pass  (parallelised over gradient loops)
    # ------------------------------------------------------------------

    def backward(self, enc_out, label, cache, lr=0.01):
        p   = self.params
        rp  = self.rp
        enc_x     = cache['enc_x']
        enc_h_pre = cache['enc_h_pre']
        enc_h_act = cache['enc_h_act']

        # d_out = enc_out - label  (free)
        enc_d_out = [enc_sub_plain_scalar_gpu(enc_out[j], label[j], p, rp)
                     for j in range(self.out_dim)]

        # dW2[j,i] = d_out[j] * h_act[i]
        tgt_ell = enc_h_act[0][1]

        def _dW2_row(j):
            dout_ms = _mod_switch_down_gpu(enc_d_out[j], tgt_ell, p, rp)
            row = np.zeros(self.hid_dim)
            for i in range(self.hid_dim):
                gc = multiply_gpu(dout_ms, enc_h_act[i],
                                  self.EK, p.Delta, p.N, p.q, p.P, 0, p.q0, rp)
                row[i] = dec_scalar_gpu(gc, self.s, p, rp)
            return row

        dW2_rows = _parallel_map(_dW2_row, list(range(self.out_dim)), self.n_workers)
        d_W2 = np.stack(dW2_rows)

        # d_h_act = W2.T @ d_out
        def _d_h_act_row(i):
            acc = None
            for j in range(self.out_dim):
                term = enc_plain_scalar_mul_gpu(enc_d_out[j], self.W2[j, i], p, rp)
                acc  = add_gpu(acc, term, rp) if acc is not None else term
            return acc

        enc_d_h_act = _parallel_map(_d_h_act_row, list(range(self.hid_dim)), self.n_workers)

        # d_h_pre = d_h_act * 2 * h_pre
        def _d_h_pre_row(i):
            dha_x2  = enc_plain_scalar_mul_gpu(enc_d_h_act[i], 2.0, p, rp)
            tgt     = dha_x2[1]
            hp_ms   = _mod_switch_down_gpu(enc_h_pre[i], tgt, p, rp)
            return multiply_gpu(dha_x2, hp_ms,
                                self.EK, p.Delta, p.N, p.q, p.P, 0, p.q0, rp)

        enc_d_h_pre = _parallel_map(_d_h_pre_row, list(range(self.hid_dim)), self.n_workers)

        d_h_pre_vals = np.array([dec_scalar_gpu(e, self.s, p, rp) for e in enc_d_h_pre])

        # dW1[i,:] = d_h_pre_scalar[i] * decrypt(scalar * enc_x)
        def _dW1_row(i):
            enc_row = enc_plain_scalar_mul_gpu(enc_x, float(d_h_pre_vals[i]), p, rp)
            q_l = p.q0 * (p.Delta ** enc_row[1])
            m   = decrypt_gpu(enc_row[0], self.s, q_l, p.N, rp)
            vals = decode(m, p.N, p.Delta)
            return [vals[k].real for k in range(self.in_dim)]

        dW1_rows = _parallel_map(_dW1_row, list(range(self.hid_dim)), self.n_workers)
        d_W1 = np.array(dW1_rows, dtype=float)

        # Gradient clipping
        if self.grad_clip is not None:
            for g in [d_W1, d_W2]:
                norm = np.linalg.norm(g)
                if norm > self.grad_clip:
                    g *= self.grad_clip / norm

        self.W1 -= lr * d_W1
        self.W2 -= lr * d_W2
        self._encode_weights()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_step(self, enc_x, label, lr=0.01):
        enc_out, cache = self.forward(enc_x)
        out_vals = np.array([dec_scalar_gpu(e, self.s, self.params, self.rp)
                             for e in enc_out])
        loss = float(0.5 * np.sum((out_vals - label) ** 2))
        self.backward(enc_out, label, cache, lr=lr)
        return loss, out_vals

    def predict(self, enc_x):
        enc_out, _ = self.forward(enc_x)
        return np.array([dec_scalar_gpu(e, self.s, self.params, self.rp)
                         for e in enc_out])


# ---------------------------------------------------------------------------
# Thread-pool helper
# ---------------------------------------------------------------------------

def _parallel_map(fn, items, n_workers):
    """
    Apply fn to each item in items, in parallel using threads.
    Returns a list in the same order as items.

    Threads (not processes) — no pickling, and CuPy/NumPy C extensions
    release the GIL, so multiple threads genuinely run concurrently on
    different CUDA streams / CPU cores.
    """
    if n_workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(fn, items[i]): i for i in range(len(items))}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch_gpu(mlp, X_norm, Y, lr=0.003, shuffle=True):
    n      = len(X_norm)
    idx    = np.random.permutation(n) if shuffle else np.arange(n)
    n_out  = mlp.out_dim
    total_loss = 0.0
    n_correct  = 0
    for i in idx:
        x, y  = X_norm[i], int(Y[i])
        label = np.zeros(n_out); label[y] = 1.0
        enc_x = make_enc_x_gpu(x, mlp.PK, mlp.params, mlp.rp)
        loss, out_vals = mlp.train_step(enc_x, label, lr=lr)
        total_loss += loss
        if int(np.argmax(out_vals)) == y:
            n_correct += 1
    return total_loss / n, n_correct / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from ckks.params import ParameterSet

    print(f"Polynomial arithmetic backend: {backend_name()}")

    digits  = load_digits()
    X, Y    = digits.data[:300, :16], digits.target[:300]
    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, Y, test_size=0.2, random_state=42)

    params = ParameterSet(
        N=32, Delta=2**20, q0=2**20, L=10,
        P=(2**20)**11, sigma=3.2, d=5, k=2, K=4, nu=10
    )

# Build RNS parameter tables (one-time cost)
    print("Building RNS parameter tables...")
    t0 = time.time()
    rp = get_rns_params(params.N, params.q, params.Delta, params.P)
    print(f"  RNS ready: {rp.k} limbs, primes: {rp.primes}")
    print(f"  Setup time: {time.time()-t0:.1f}s")
    # Key generation
    print("Generating keys...")
    t0 = time.time()
    s_rns, _ = gen_secret_key_gpu(params.N, rp)
    PK       = gen_public_key_gpu(s_rns, params.N, params.q, rp)
    EK       = gen_eval_key_gpu(s_rns, params.N, params.q, params.P, rp)
    rot_keys = gen_rotation_keys_gpu(s_rns, params.N, params.q, params.P, rp, max_j=8)
    print(f"  Key gen time: {time.time()-t0:.1f}s")

    mlp = EncMLP_GPU(
        in_dim=16, hid_dim=8, out_dim=10,
        params=params, EK=EK, rot_keys=rot_keys,
        PK=PK, s_rns=s_rns, rp=rp,
        n_workers=4,
    )

    print(f"\nStarting GPU-accelerated encrypted training...")
    for epoch in range(15):
        t0 = time.time()
        avg_loss, acc = train_epoch_gpu(mlp, X_train, y_train, lr=0.01)
        n_correct = sum(
            int(np.argmax(mlp.predict(make_enc_x_gpu(X_test[i], PK, params, rp))))
            == int(y_test[i])
            for i in range(len(X_test))
        )
        test_acc = n_correct / len(X_test)
        print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | "
              f"train={acc*100:.1f}% | test={test_acc*100:.1f}% | "
              f"{time.time()-t0:.1f}s")