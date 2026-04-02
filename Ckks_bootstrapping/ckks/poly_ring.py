import numpy as np

def center_mod(a, q):
    if q is None: return a
    res = np.array([((int(x) + q // 2) % q) - q // 2 for x in a], dtype=object)
    return res

def poly_add(a, b, q=None, N=None):
    result = np.array(a) + np.array(b)
    if q is not None:
        result = center_mod(result, q)
    return result

def poly_mul(a, b, q=None, N=None):
    # C-accelerated np.convolve for object/bigint arrays (Claude's patch)
    a_obj = np.asarray(a, dtype=object)
    b_obj = np.asarray(b, dtype=object)
    raw   = np.convolve(a_obj, b_obj)
    
    if N is not None:
        result = raw[:N].copy()
        if len(raw) > N:
            result[:len(raw) - N] -= raw[N:]   # X^N = -1
    else:
        result = raw
        
    if q is not None:
        result = center_mod(result, q)
    return result

def poly_sub_X(poly, exp, N):
    result = np.zeros(N, dtype=object)
    for i, coef in enumerate(poly):
        new_exp = (i * exp) % (2 * N)
        if new_exp < N:
            result[new_exp] += coef
        else:
            result[new_exp - N] -= coef
    return result

def canonical_norm(poly, N):
    zeta  = np.exp(1j * np.pi / N)
    roots = [zeta ** (2*k + 1) for k in range(N)]
    evals = [sum(poly[i] * (r**i) for i in range(N)) for r in roots]
    return max(abs(e) for e in evals)