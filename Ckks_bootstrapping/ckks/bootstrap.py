import numpy as np
from ckks.poly_ring import center_mod
from ckks.homomorphic_ops import r_linear_transform
from ckks.chebyshev import chebyshev_coeffs, chebyshev_ps_eval

def raise_modulus(C_tag, L, q_L, N):
    (C, ell, nu, B) = C_tag
    C1_new = center_mod(C[0], q_L)
    C2_new = center_mod(C[1], q_L)
    return ((C1_new, C2_new), L, nu, B)

def compute_vandermonde_matrices(N, zeta):
    L1 = np.zeros((N//2, N//2), dtype=complex)
    L2 = np.zeros((N//2, N//2), dtype=complex)
    for i in range(N//2):
        for j in range(N//2):
            pow_val = pow(5, i, 2*N)
            L1[i][j] = zeta ** ((pow_val * j) % (2*N))
            L2[i][j] = zeta ** ((pow_val * (j + N//2)) % (2*N))
    return L1, L2

def coeffs_to_slots(C_raised_tag, rot_keys, conj_key, params, hom_ops):
    L1, L2 = compute_vandermonde_matrices(params.N, params.zeta)
    L1_inv = np.linalg.inv(L1)
    L2_inv = np.linalg.inv(L2)
    C_p1 = r_linear_transform(C_raised_tag, L1_inv.real, L1_inv.imag,
                               rot_keys, conj_key, params, hom_ops)
    C_p2 = r_linear_transform(C_raised_tag, L2_inv.real, L2_inv.imag,
                               rot_keys, conj_key, params, hom_ops)
    return C_p1, C_p2

def eval_sine_approx(C_slots_tag, params, hom_ops, conj_key):
    k = params.k
    K = params.K
    scale = 2 * np.pi * K / (2**k)
    f_real = lambda x: np.cos(x / (2**k))
    f_imag = lambda x: np.sin(x / (2**k))
    n = params.d + 1
    
    c_real = chebyshev_coeffs(f_real, n, -scale, scale)
    c_imag = chebyshev_coeffs(f_imag, n, -scale, scale)

    u = int(2 ** np.ceil(np.log2(np.sqrt(params.d)))) 
    v = params.d // u + 1
    
    C_real = chebyshev_ps_eval(c_real, C_slots_tag, u, v, hom_ops)
    C_imag = chebyshev_ps_eval(c_imag, C_slots_tag, u, v, hom_ops)

    for _ in range(k):
        C_real_sq = hom_ops.sub(hom_ops.square(C_real), hom_ops.square(C_imag))
        C_imag_sq = hom_ops.scalar_mul(hom_ops.multiply(C_real, C_imag), 2.0)
        C_real, C_imag = C_real_sq, C_imag_sq

    C_sin = hom_ops.scalar_mul(C_imag, params.q0 / (2 * np.pi))
    return C_sin

def slots_to_coeffs(C_p1, C_p2, rot_keys, conj_key, params, hom_ops):
    L1, L2 = compute_vandermonde_matrices(params.N, params.zeta)
    part1 = r_linear_transform(C_p1, L1.real, L1.imag, rot_keys, conj_key, params, hom_ops)
    part2 = r_linear_transform(C_p2, L2.real, L2.imag, rot_keys, conj_key, params, hom_ops)
    return hom_ops.add(part1, part2)

class AllKeys:
    def __init__(self, PK, EK, rot_keys, conj_key):
        self.PK = PK
        self.EK = EK
        self.rot_keys = rot_keys
        self.conj_key = conj_key

def bootstrap(C_level0_tag, params, all_keys, hom_ops):
    C_raised = raise_modulus(C_level0_tag, params.L, params.q_L, params.N)

    C_p1, C_p2 = coeffs_to_slots(C_raised, all_keys.rot_keys,
                                   all_keys.conj_key, params, hom_ops)

    C_p1_sin = eval_sine_approx(C_p1, params, hom_ops, all_keys.conj_key)
    C_p2_sin = eval_sine_approx(C_p2, params, hom_ops, all_keys.conj_key)

    C_boot = slots_to_coeffs(C_p1_sin, C_p2_sin,
                              all_keys.rot_keys, all_keys.conj_key, params, hom_ops)
                              
    return C_boot
