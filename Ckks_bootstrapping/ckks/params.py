import numpy as np

class ParameterSet:
    def __init__(self, N, Delta, q0, L, P, sigma, d, k, K, nu):
        self.N = N
        self.Delta = Delta
        self.q0 = q0
        self.L = L
        # q = q0 * Delta^L
        self.q = q0 * (Delta ** L)
        self.P = P
        self.sigma = sigma
        self.d = d
        self.k = k
        self.K = K
        self.nu = nu
        # zeta is primitive 2N-th root of unity e^{i * pi / N}
        self.zeta = np.exp(1j * np.pi / N)
        self.q_L = self.q

def get_toy_params():
    return ParameterSet(
        N=16,
        Delta=2**20,
        q0=1,
        L=5,
        P=1 * (2**20)**5,  # P ≈ q
        sigma=3.2,
        d=5,
        k=2,
        K=4,
        nu=10
    )

def get_prod_params():
    # Production values
    return ParameterSet(
        N=2**14,      # 16384
        Delta=2**40,
        q0=2**30,     # approximation of a prime ~2^30
        L=20,
        P=(2**30) * (2**40)**20, # P ≈ q
        sigma=3.2,
        d=60,
        k=15,
        K=128,
        nu=20
    )
