import numpy as np
from ckks.poly_ring import center_mod, poly_add, poly_mul

def encrypt(m, PK, N, q, sigma=3.2):
    u  = np.random.choice([-1, 0, 1], size=N).astype(object)
    e1 = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    e2 = np.round(np.random.normal(0, sigma, N)).astype(int).astype(object)
    C1 = center_mod(poly_add(poly_mul(u, PK[0], N=N), poly_add(m, e1)), q)
    C2 = center_mod(poly_add(poly_mul(u, PK[1], N=N), e2), q)
    return (C1, C2)

def decrypt(C, s, q_ell, N):
    val = poly_add(C[0], poly_mul(C[1], s, N=N))
    return center_mod(val, q_ell)
