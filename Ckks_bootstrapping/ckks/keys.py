import numpy as np
import random as pyrandom
from ckks.poly_ring import poly_add, poly_mul, center_mod, poly_sub_X

def _rand_poly(q, N):
    """Generate a random polynomial with coefficients in [0, q) as object array."""
    return np.array([pyrandom.randrange(0, int(q)) for _ in range(N)], dtype=object)

def gen_secret_key(N):
    s = np.random.choice([-1, 0, 1], size=N)
    return s.astype(object)

def gen_public_key(s, N, q, sigma=3.2):
    a  = _rand_poly(q, N)
    e  = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    PK1 = center_mod(poly_add(poly_mul(-a, s, N=N), e), q)
    PK2 = center_mod(a, q)
    return (PK1, PK2)

def gen_eval_key(s, N, q, P, sigma=3.2):
    Pq   = P * q
    a_ek = _rand_poly(Pq, N)
    e_ek = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    s2   = poly_mul(s, s, N=N)
    EK1  = center_mod(poly_add(poly_mul(-a_ek, s, N=N), poly_add(e_ek, P * s2)), Pq)
    EK2  = center_mod(a_ek, Pq)
    return (EK1, EK2)

def gen_ksk(s, s_new, N, q, P, sigma=3.2):
    Pq = P * q
    a  = _rand_poly(Pq, N)
    e  = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    KSK1 = center_mod(poly_add(poly_mul(-a, s_new, N=N), poly_add(e, P * s)), Pq)
    KSK2 = center_mod(a, Pq)
    return (KSK1, KSK2)

def gen_rotation_keys(s, N, q, P, max_j=None):
    rot_keys = {}
    limit = max_j + 1 if max_j is not None else N // 2
    for j in range(1, limit):
        exp = pow(5, j, 2*N)
        s_auto = poly_sub_X(s, exp, N)  # s(X^{5^j}) — key after automorphism
        rot_keys[j] = gen_ksk(s_auto, s, N, q, P)  # switch FROM s_auto TO s
    return rot_keys

def gen_conj_key(s, N, q, P):
    s_conj = poly_sub_X(s, 2*N - 1, N)
    return gen_ksk(s_conj, s, N, q, P)  # switch FROM s_conj TO s
