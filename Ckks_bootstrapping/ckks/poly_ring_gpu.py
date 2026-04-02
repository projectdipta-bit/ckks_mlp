"""
poly_ring_gpu.py  —  GPU-native polynomial ring for CKKS (RNS + NTT).

No CPU round-trips on the hot path.  Every function called inside
ciphertext multiply / rescale / rotate stays on the compute device.

Key algorithms
--------------

rns_round_div  (power-of-2 divisor Δ = 2^b)
  Uses the Shenoy-Kumaresan mixed-radix approach, adapted for a
  power-of-2 divisor.  Steps entirely on GPU:

  1. Compute ε = x mod Δ  via multi-limb mixed-radix conversion.
     We reconstruct x mod (p₀·p₁·…·p_{t-1}) for enough limbs t so
     that p₀·…·p_{t-1} > 2·q_ℓ  (≥ the full dynamic range).
     Then ε = that reconstruction mod Δ.
     Because we only need the result mod Δ (< 2^21), intermediate
     values fit in int64 and the computation is entirely vectorised
     on the device.

  2. Compute x' = x - ε in RNS  (exact, per-limb subtraction mod pᵢ).
     x' is now exactly divisible by Δ.

  3. Compute x'/Δ mod pᵢ = x' · Δ⁻¹ mod pᵢ  for each limb i.
     Δ⁻¹ mod pᵢ is precomputed once in RNSParams.

  The rounding correction is:  if ε > Δ/2, add 1 to every limb
  (i.e., add 1 mod pᵢ), because x' = x - ε underestimates by 1.

  This is exact for any x bounded by Q/2 (product of all limb primes),
  and requires no CPU transfer.

rns_poly_sub_X  (automorphism for rotation / conjugation)
  Permutation tables (dst, sign) are precomputed once per (exp, N),
  pushed to the device, and cached.  The scatter is xp.add.at on-device.
  No transfer after the first call per exponent.

rns_to_bigint / bigint_to_rns
  Still use CPU (necessary: bigint arithmetic).
  Called only at key-gen and decode time  [OFF-HOTPATH].
"""

import math
import numpy as np

# ── backend ──────────────────────────────────────────────────────────────────

USE_GPU = True

try:
    if USE_GPU:
        import cupy as xp
        _GPU_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    import numpy as xp
    _GPU_AVAILABLE = False


def backend_name():
    return "CuPy (GPU)" if _GPU_AVAILABLE else "NumPy (CPU)"


# ── NTT prime utilities ───────────────────────────────────────────────────────

def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _find_ntt_primes(N, n_primes, bits=30):
    """
    Primes p = c·2N + 1, p < 2^bits.
    p ≡ 1 (mod 2N)  ⟹  primitive 2N-th roots exist  ⟹  negacyclic NTT valid.
    """
    primes, c = [], (1 << (bits - 1)) // (2 * N)
    while len(primes) < n_primes and c > 0:
        p = c * 2 * N + 1
        if p.bit_length() <= bits and _is_prime(p):
            primes.append(p)
        c -= 1
    if len(primes) < n_primes:
        raise ValueError(f"Cannot find {n_primes} NTT primes for N={N}, bits={bits}")
    return primes


def _primitive_root(p):
    phi, n, factors = p - 1, p - 1, set()
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root mod {p}")


def _ntt_root(p, N):
    """Primitive 2N-th root of unity ψ mod p, with ψ^N ≡ -1 (mod p)."""
    psi = pow(_primitive_root(p), (p - 1) // (2 * N), p)
    assert pow(psi, N, p) == p - 1
    return psi


def _twiddle_table_np(N, root, p):
    tw = [1] * N
    for j in range(1, N):
        tw[j] = tw[j - 1] * root % p
    return tw


# ── RNSParams ─────────────────────────────────────────────────────────────────

class RNSParams:
    """
    One-time precomputed tables for RNS + negacyclic NTT.

    Hot-path tables (on device):
      _tw_fwd[i]  : xp int64 (N,)   forward NTT twiddle
      _tw_inv[i]  : xp int64 (N,)   inverse NTT twiddle
      _N_inv[i]   : int              N⁻¹ mod pᵢ
      _delta_inv  : list[int]        Δ⁻¹ mod pᵢ for each i  (set by set_delta)
      _mr_tables  : mixed-radix tables for rns_round_div     (set by set_delta)
      _subX_cache : dict exp → (dst_dev, sign_dev)

    Off-hot-path:
      crt_M, crt_y, Q, Q_half  for rns_to_bigint
    """

    def __init__(self, N, q_bits=220, n_limbs=None):
        assert N & (N - 1) == 0, "N must be a power of 2"
        self.N = N
        # Scale limbs up to match the 30-bit prime shift
        if n_limbs is None:
            n_limbs = math.ceil(q_bits / 29) + 2
        self.k      = n_limbs
        self.primes = _find_ntt_primes(N, n_limbs, bits=30)

        psi_f = [_ntt_root(p, N)       for p in self.primes]
        psi_i = [pow(ps, -1, p)        for ps, p in zip(psi_f, self.primes)]
        self._N_inv = [pow(N, -1, p)   for p in self.primes]

        self._tw_fwd = [xp.array(_twiddle_table_np(N, ps, p), dtype=xp.int64)
                        for ps, p in zip(psi_f, self.primes)]
        self._tw_inv = [xp.array(_twiddle_table_np(N, pi, p), dtype=xp.int64)
                        for pi, p in zip(psi_i, self.primes)]

        # CRT  [OFF-HOTPATH]
        self.Q      = math.prod(self.primes)
        self.Q_half = self.Q // 2
        self.crt_M  = [self.Q // p for p in self.primes]
        self.crt_y  = [pow(int(M % p), -1, p)
                       for M, p in zip(self.crt_M, self.primes)]

        # Delta-dependent tables — populated by set_delta()
        self._delta      = None
        self._delta_inv  = None   # list[int]: Δ⁻¹ mod pᵢ
        self._mr_tables  = None   # mixed-radix tables for round-div

        # Rotation cache
        self._subX_cache: dict = {}

    def set_delta(self, Delta):
        """
        Precompute Δ-dependent tables for rns_round_div.
        Call once after construction with your scheme's Delta.
        """
        if self._delta == Delta:
            return
        self._delta     = Delta
        self._delta_inv = [pow(int(Delta), -1, p) for p in self.primes]
        # Build mixed-radix tables: see _build_mr_tables
        self._mr_tables = _build_mr_tables(self.primes, Delta)

    def _get_subX_tables(self, exp):
        if exp in self._subX_cache:
            return self._subX_cache[exp]
        N = self.N
        dst_np  = np.empty(N, dtype=np.int32)
        sign_np = np.ones(N,  dtype=np.int32)
        for j in range(N):
            ne = (j * exp) % (2 * N)
            if ne < N:
                dst_np[j] = ne
            else:
                dst_np[j]  = ne - N
                sign_np[j] = -1
        dst_dev  = xp.array(dst_np,  dtype=xp.int32)
        sign_dev = xp.array(sign_np, dtype=xp.int32)
        self._subX_cache[exp] = (dst_dev, sign_dev)
        return dst_dev, sign_dev


# ── Mixed-radix tables for rns_round_div ─────────────────────────────────────

def _build_mr_tables(primes, Delta):
    """
    Precompute the mixed-radix conversion constants needed to extract
    ε = x mod Δ  from the RNS representation of x, entirely in int64.
    """
    k = len(primes)
    # inv_table[i][j] = p_j^{-1} mod p_i  for j < i
    inv_table = []
    for i in range(k):
        row = [pow(primes[j], -1, primes[i]) for j in range(i)]
        inv_table.append(row)

    # partial_prod_mod_delta[i] = p_0 * p_1 * … * p_{i-1}  mod Delta
    Delta = int(Delta)
    partial_prod_mod_delta = [None] * k
    partial_prod_mod_delta[0] = 1          # empty product
    prod = 1
    for i in range(1, k):
        prod = (prod * primes[i - 1]) % Delta
        partial_prod_mod_delta[i] = prod

    return {
        'inv_table':              inv_table,
        'partial_prod_mod_delta': partial_prod_mod_delta,
        'Delta':                  Delta,
    }


# ── rns_round_div — GPU-native, multi-limb, exact ────────────────────────────

def rns_round_div(a, d, rp):
    """
    Compute round(a / d) for each coefficient, entirely on the GPU.

    d must equal rp._delta (set via rp.set_delta(d) before calling).
    d must be a power of 2.
    """
    assert rp._delta is not None, \
        "Call rp.set_delta(Delta) before rns_round_div"
    assert int(d) == rp._delta, \
        f"rns_round_div: d={d} != rp._delta={rp._delta}; call set_delta first"

    k, N  = a.shape
    Delta = rp._delta
    half  = Delta >> 1
    mr    = rp._mr_tables
    inv_t = mr['inv_table']
    pp_md = mr['partial_prod_mod_delta']   # partial_prod_mod_delta[i]

    # Running "partial reconstruction" per limb: used to compute digits.
    partial = [xp.zeros(N, dtype=xp.int64) for _ in range(k)]

    eps_dev = xp.zeros(N, dtype=xp.int64)   # accumulator mod Δ

    for i in range(k):
        pi = rp.primes[i]
        diff = (a[i] - partial[i]) % pi      # shape (N,), values in [0,pᵢ)
        if i == 0:
            ci = diff
        else:
            prod_inv_mod_pi = 1
            for j in range(i):
                prod_inv_mod_pi = prod_inv_mod_pi * inv_t[i][j] % pi
            ci = diff * prod_inv_mod_pi % pi  # shape (N,)

        # FIX APPLIED: (ci % Delta) ensures strict 64-bit bounds on array multiply
        contrib = (ci % Delta) * pp_md[i] % Delta       # shape (N,), int64
        eps_dev = (eps_dev + contrib) % Delta

        for j in range(i + 1, k):
            pj = rp.primes[j]
            basis = 1
            for m in range(i):
                basis = basis * (rp.primes[m] % pj) % pj
            partial[j] = (partial[j] + ci * basis) % pj

    # round_up: where we should add 1 after dividing  (symmetric rounding)
    round_up = (eps_dev >= half).astype(xp.int64)   # 0 or 1, shape (N,)

    rows = []
    for i in range(k):
        pi       = int(rp.primes[i])
        dinv     = int(rp._delta_inv[i])         # Δ⁻¹ mod pᵢ, Python scalar
        eps_i    = eps_dev % pi                  # ε mod pᵢ, on device
        row      = (a[i] - eps_i) * dinv % pi    # (x-ε)/Δ mod pᵢ
        row      = (row + round_up) % pi         # symmetric rounding correction
        rows.append(row)

    return xp.stack(rows)


# ── NTT (Gentleman–Sande DIF forward, Cooley–Tukey DIT inverse) ──────────────

def _ntt_fwd_limb(a, tw, p):
    """
    Forward negacyclic NTT — Gentleman–Sande DIF with pre-twist.
    a, tw : 1-D xp int64, length N.
    """
    N   = len(a)
    out = (a * tw) % p          # pre-twist: a[j] ← a[j]·ψʲ
    m   = N
    while m > 1:
        half     = m >> 1
        step     = N // m
        n_groups = N // m
        g   = xp.arange(n_groups, dtype=xp.int32)
        h   = xp.arange(half,     dtype=xp.int32)
        i0  = g[:, None] * m + h[None, :]
        i1  = i0 + half
        twj = tw[h * step % N]
        u   = out[i0];  v = out[i1]
        out[i0] = (u + v) % p
        out[i1] = ((u - v) * twj) % p
        m = half
    return out


def _ntt_inv_limb(a, tw_inv, N_inv, p):
    """
    Inverse negacyclic NTT — Cooley–Tukey DIT + post-untwist + 1/N scale.
    """
    N   = len(a)
    out = a.copy()
    m   = 2
    while m <= N:
        half     = m >> 1
        step     = N // m
        n_groups = N // m
        g   = xp.arange(n_groups, dtype=xp.int32)
        h   = xp.arange(half,     dtype=xp.int32)
        i0  = g[:, None] * m + h[None, :]
        i1  = i0 + half
        twj = tw_inv[h * step % N]
        u   = out[i0];  v = (out[i1] * twj) % p
        out[i0] = (u + v) % p
        out[i1] = (u - v) % p
        m <<= 1
    out = (out * tw_inv) % p    # post-untwist
    out = (out * N_inv)  % p   # 1/N scale
    return out


def _ntt_fwd(polys, rp):
    return xp.stack([_ntt_fwd_limb(polys[i], rp._tw_fwd[i], rp.primes[i])
                     for i in range(rp.k)])


def _ntt_inv(polys, rp):
    return xp.stack([_ntt_inv_limb(polys[i], rp._tw_inv[i], rp._N_inv[i], rp.primes[i])
                     for i in range(rp.k)])


# ── Core RNS operations ───────────────────────────────────────────────────────

def rns_poly_mul(a, b, rp):
    A = _ntt_fwd(a, rp);  B = _ntt_fwd(b, rp)
    C = xp.stack([(A[i] * B[i]) % rp.primes[i] for i in range(rp.k)])
    return _ntt_inv(C, rp)


def rns_poly_add(a, b, rp):
    return xp.stack([(a[i] + b[i]) % rp.primes[i] for i in range(rp.k)])


def rns_poly_sub(a, b, rp):
    return xp.stack([(a[i] - b[i]) % rp.primes[i] for i in range(rp.k)])


def rns_center_mod(a, q_rns, rp):
    """Per-limb: reduce mod qᵢ then centre-lift to (-q/2, q/2]."""
    rows = []
    for i in range(rp.k):
        qi  = int(q_rns[i])
        row = a[i] % qi
        row = xp.where(row > qi // 2, row - qi, row)
        rows.append(row)
    return xp.stack(rows)


# ── rns_poly_sub_X — on-device, cached permutation ───────────────────────────

def rns_poly_sub_X(poly_rns, exp, rp):
    """
    Automorphism X → X^exp mod (X^N+1).

    Permutation (dst, sign) precomputed and cached on device per exp.
    Scatter via xp.add.at — no CPU transfer after first call.
    """
    dst, sign = rp._get_subX_tables(exp)
    k, N = poly_rns.shape
    signed = poly_rns * sign.astype(xp.int64)[None, :]
    result = xp.zeros((k, N), dtype=xp.int64)
    for i in range(k):
        xp.add.at(result[i], dst, signed[i])
        result[i] %= rp.primes[i]
    return result


# ── CRT conversion  [OFF-HOTPATH] ────────────────────────────────────────────

def bigint_to_rns(poly_obj, rp):
    """dtype=object → (k,N) int64 on device.  Key-gen / encode only."""
    N   = rp.N
    arr = np.empty((rp.k, N), dtype=np.int64)
    for i, p in enumerate(rp.primes):
        for j in range(N):
            arr[i, j] = int(poly_obj[j]) % p
    return xp.array(arr, dtype=xp.int64)


def rns_to_bigint(a_rns, rp):
    """(k,N) int64 → dtype=object via CRT.  Decode only."""
    a_cpu = xp.asnumpy(a_rns) if _GPU_AVAILABLE else np.asarray(a_rns)
    N     = rp.N
    out   = np.zeros(N, dtype=object)
    for j in range(N):
        x = sum(int(a_cpu[i, j]) * rp.crt_y[i] * rp.crt_M[i]
                for i in range(rp.k)) % rp.Q
        out[j] = x - rp.Q if x > rp.Q_half else x
    return out


def rns_of_scalar(q, rp):
    return xp.array([int(q) % p for p in rp.primes], dtype=xp.int64)