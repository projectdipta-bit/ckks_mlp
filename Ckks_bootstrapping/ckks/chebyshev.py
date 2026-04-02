import numpy as np
from ckks.encoding import encode

def _make_one_ciph(hom_ops):
    """Return a ciphertext-shaped tag encoding the constant 1 (c0=poly, c1=0)."""
    N = hom_ops.params.N
    Delta = hom_ops.params.Delta
    poly_one = encode([1.0] * (N // 2), N, Delta)
    zeros = np.zeros(N, dtype=object)
    return ((poly_one, zeros), hom_ops.params.L, 1, 0)

def baby_step_chebyshev(x_ciph, u, hom_ops):
    # u must be power of 2
    T = [None] * (u + 1)
    T[0] = _make_one_ciph(hom_ops)
    T[1] = x_ciph
    
    j = 1
    while 2 * j <= u:
        sq = hom_ops.square(T[j])
        mul2 = hom_ops.scalar_mul(sq, 2.0)
        T[2*j] = hom_ops.sub(mul2, T[0])
        
        if 2*j + 1 <= u:
            mulx = hom_ops.multiply(T[2*j], x_ciph)
            mulx2 = hom_ops.scalar_mul(mulx, 2.0)
            T[2*j+1] = hom_ops.sub(mulx2, T[2*j-1])
        j *= 2
    return T

def giant_step_chebyshev(T_u, v, hom_ops):
    G = [T_u]
    curr = T_u
    
    T_0 = _make_one_ciph(hom_ops)

    while len(G) < v.bit_length():
        sq = hom_ops.square(curr)
        mul2 = hom_ops.scalar_mul(sq, 2.0)
        curr = hom_ops.sub(mul2, T_0)
        G.append(curr)
    return G

def compute_T_ju(j, T_giant, hom_ops):
    bits = [k for k in range(j.bit_length()) if j & (1 << k)]
    result = None
    for k in bits:
        g = T_giant[k]   
        result = hom_ops.multiply(result, g) if result else g
        
    if not result:
        return _make_one_ciph(hom_ops)
    return result

def chebyshev_ps_eval(cheb_coeffs, x_ciph, u, v, hom_ops):
    T_baby = baby_step_chebyshev(x_ciph, u, hom_ops)
    T_giant = giant_step_chebyshev(T_baby[u], v, hom_ops)

    Q = []
    for j in range(v):
        # Guard against out-of-bounds when len(cheb_coeffs) < (j+1)*u
        if j * u >= len(cheb_coeffs):
            break
        coeffs_j = cheb_coeffs[j * u : min((j + 1) * u, len(cheb_coeffs))]
        Qj = None
        for i, c in enumerate(coeffs_j):
            if abs(c) < 1e-10:
                continue
            term = hom_ops.scalar_mul(T_baby[i], c)
            Qj = hom_ops.add(Qj, term) if Qj is not None else term
        Q.append(Qj)

    result = None
    for j in range(len(Q)):
        if Q[j] is None:
            continue
        if j == 0:
            # FIX: T_{0,u} = T_0 = 1, so multiplying by it wastes a level.
            # The j=0 term contributes Q[0] directly — no homomorphic multiply needed.
            term = Q[0]
        else:
            T_ju = compute_T_ju(j, T_giant, hom_ops)
            term = hom_ops.multiply(T_ju, Q[j])
        result = hom_ops.add(result, term) if result is not None else term
    return result

def chebyshev_nodes(n, a, b):
    # n nodes that minimize max interpolation error
    return [((a + b) / 2) + ((b - a) / 2) * np.cos((k + 0.5) * np.pi / n)
            for k in range(n)]

def chebyshev_coeffs(f, n, a, b):
    nodes = chebyshev_nodes(n, a, b)
    f_vals = np.array([f(x) for x in nodes])
    c = np.zeros(n)
    for k in range(n):
        c[k] = (2.0 / n) * sum(f_vals[j] * np.cos(k * np.pi * (j + 0.5) / n)
                                for j in range(n))
    c[0] /= 2   
    return c

def better_bootstrap_nodes(K, eps, k, n_total, n_per_interval=None):
    """
    Part 3 ‘better interpolation points’ from Section 10.

    Parameters
    ----------
    K   : integer range of t_i, i.e. |t_i| <= K
    eps : fractional half-width of each sub-interval, eps = nu / q0
          where nu = max|m_i| and q0 is the base modulus.
    k   : squaring depth (input is divided by 2^k before approximation)
    n_total : total number of interpolation nodes across all sub-intervals
    n_per_interval : optional dict {a: n_a}; if None, allocates uniformly.

    The PDF defines I_a = [2*pi*(a - eps) / 2^k,  2*pi*(a + eps) / 2^k]
    and places n_a Chebyshev nodes *inside I_a*, not in [-eps, eps].
    """
    num_intervals = 2 * K + 1
    n_a_default = n_total // num_intervals
    nodes = []
    scale = 2 * np.pi / (2 ** k)
    for a in range(-K, K + 1):
        n_a = (n_per_interval[a] if n_per_interval and a in n_per_interval
               else n_a_default)
        # I_a = [scale*(a - eps), scale*(a + eps)]
        # Chebyshev nodes on I_a (not on [-eps, eps])
        lo = scale * (a - eps)
        hi = scale * (a + eps)
        local = chebyshev_nodes(n_a, lo, hi)
        nodes += local
    return nodes