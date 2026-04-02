import numpy as np

def slot_roots(N):
    zeta = np.exp(1j * np.pi / N)
    exponents = [pow(5, j, 2*N) for j in range(N // 2)]
    return [zeta ** e for e in exponents]

def encode(z, N, Delta):
    zeta = np.exp(1j * np.pi / N)
    slot_exp = [pow(5, j, 2*N) for j in range(N // 2)]
    conj_exp = [(2*N - e) % (2*N) for e in slot_exp]
    all_exp = slot_exp + conj_exp                     # length N
    all_roots = [zeta**e for e in all_exp]
    all_vals = list(Delta * np.array(z)) + list(Delta * np.conj(z)) 

    V = np.array([[r**j for j in range(N)] for r in all_roots], dtype=complex)
    p_coeffs = np.linalg.solve(V, all_vals)

    imag_tol = max(1e-6 * Delta, 1e-6)
    assert np.max(np.abs(p_coeffs.imag)) < imag_tol, (
        f"Encoding produced non-real coefficients (max imag = {np.max(np.abs(p_coeffs.imag)):.2e}, "
        f"tol = {imag_tol:.2e}). Check that z satisfies the conjugate-symmetry condition."
    )
    return np.round(p_coeffs.real).astype(int).astype(object)

def decode(m, N, Delta):
    zeta = np.exp(1j * np.pi / N)
    slot_exp = [pow(5, j, 2*N) for j in range(N // 2)]
    roots = [zeta**e for e in slot_exp]
    evals = [sum(m[i] * (r**i) for i in range(N)) for r in roots]
    return [e / Delta for e in evals]
