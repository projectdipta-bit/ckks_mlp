import numpy as np
from ckks.params import ParameterSet
from ckks.encoding import encode, decode
from ckks.keys import gen_secret_key, gen_public_key, gen_eval_key
from ckks.encrypt import encrypt, decrypt
from ckks.poly_ring import center_mod, poly_mul
from ckks.homomorphic_ops import _round_div

params = ParameterSet(N=16, Delta=2**20, q0=2**20, L=10, P=(2**20)**11, sigma=3.2, d=5, k=2, K=4, nu=10)
s = gen_secret_key(params.N)
PK = gen_public_key(s, params.N, params.q)
EK = gen_eval_key(s, params.N, params.q, params.P)

IN, HID, OUT = 4, 4, 2
x_plain = [0.5, 0.3, 0.8, 0.1] + [0.0]*4
m_x = encode(x_plain, params.N, params.Delta)
C_x = encrypt(m_x, PK, params.N, params.q)
enc_x = (C_x, params.L, 1, 0)

np.random.seed(42)
W1 = np.random.randn(HID, IN) * 0.3
W2 = np.random.randn(OUT, HID) * 0.3
label = np.array([1.0, 0.0])

def enc_plain_dot(w_row, enc_v, n_in, params, s):
    w_padded = list(w_row) + [0.0]*(params.N//2 - len(w_row))
    m_w = encode(w_padded, params.N, params.Delta)
    q_ell = params.q0 * (params.Delta ** enc_v[1])
    C = enc_v[0]
    C0_new = center_mod(poly_mul(C[0], m_w, N=params.N), q_ell)
    C1_new = center_mod(poly_mul(C[1], m_w, N=params.N), q_ell)
    q_new = q_ell // params.Delta
    C0_rs = center_mod(_round_div(C0_new, params.Delta), q_new)
    C1_rs = center_mod(_round_div(C1_new, params.Delta), q_new)
    enc_prod = ((C0_rs, C1_rs), enc_v[1]-1, 1, 0)
    q_l = params.q0 * (params.Delta ** enc_prod[1])
    m_dec = decrypt(enc_prod[0], s, q_l, params.N)
    vals = decode(m_dec, params.N, params.Delta)
    return sum(v.real for v in vals[:n_in])

h_pre = np.array([enc_plain_dot(W1[j,:], enc_x, IN, params, s) for j in range(HID)])
h_pre_pt = W1 @ np.array(x_plain[:IN])
print(f'h_pre enc: {h_pre.round(5)}')
print(f'h_pre pt:  {h_pre_pt.round(5)}')
print(f'Match: {np.allclose(h_pre, h_pre_pt, rtol=0.01)}')

h_act = h_pre**2
h_act_pt = h_pre_pt**2
out = W2 @ h_act
out_pt = W2 @ h_act_pt
loss = 0.5*np.sum((out-label)**2)
loss_pt = 0.5*np.sum((out_pt-label)**2)
print(f'output enc: {out.round(5)}  loss={loss:.5f}')
print(f'output pt:  {out_pt.round(5)}  loss={loss_pt:.5f}')
print(f'Match output: {np.allclose(out, out_pt, rtol=0.01)}')

lr = 0.1
losses = []
for step in range(5):
    m_x = encode(x_plain, params.N, params.Delta)
    C_x = encrypt(m_x, PK, params.N, params.q)
    enc_x = (C_x, params.L, 1, 0)
    h_pre = np.array([enc_plain_dot(W1[j,:], enc_x, IN, params, s) for j in range(HID)])
    h_act = h_pre**2
    out = W2 @ h_act
    loss = 0.5*np.sum((out-label)**2)
    losses.append(loss)
    d_out = out - label
    d_W2 = np.outer(d_out, h_act)
    d_h_act = W2.T @ d_out
    d_h_pre = d_h_act * 2 * h_pre
    d_W1 = np.outer(d_h_pre, x_plain[:IN])
    W1 -= lr * d_W1
    W2 -= lr * d_W2

print(f'\nLoss over 5 steps: {[round(l,5) for l in losses]}')
print('Loss decreasing:', losses[-1] < losses[0])


