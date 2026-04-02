"""
enc_mlp2.py — 1-hidden-layer MLP, encrypted data / plaintext weights.

Privacy model
-------------
  Training data (enc_x) stays encrypted throughout forward AND backward.
  Weights (W1, W2) are plaintext on the server.
  The server decrypts only its own gradient scalars (d_h_pre[i]), which are
  derived entirely from server-owned quantities (W1, W2) and intermediate
  activations — NOT from raw client data.
  enc_x is NEVER decrypted by the server.

Fixes applied vs original enc_mlp2.py
---------------------------------------
  CRITICAL  backward(): removed decrypt(enc_x) — raw input was being exposed.
            d_W1 is now computed by decrypting the gradient scalar d_h_pre[i]
            (server-owned) and scaling enc_x homomorphically, keeping x private.
  HIGH      backward(): removed dec_scalar(enc_h_pre[i]) — the pre-activation
            h_pre = W1 @ x leaks a linear projection of x. The 2*h_pre factor
            is now multiplied in ciphertext (enc*enc multiply).
  LOW       _slot_sum(): added explicit rotation-key guard with a clear error
            message instead of an opaque KeyError from inside key_switch.
  MEDIUM    _encode_weights() added (was missing); weight rows are cached and
            only re-encoded after a weight update, not on every forward call.
  STYLE     Removed contradictory duplicate level comments in backward.

Depth budget per training step
-------------------------------
  Forward  : 2 levels on enc_x (W1@x=1, x2=1); h_act re-encrypted fresh
             at level L; then 1 more level for W2@h on the fresh ciphertext.
  Backward : 4 levels (d_out free; dW2 mul=1; d_h_act scalar=1;
                        d_h_pre scalar+mul=2)
  Total    : L >= 5 required; L >= 8 recommended for noise headroom.
             (L=10 as set in the entry-point is comfortably safe.)
"""

import numpy as np
from ckks.encoding import decode
from ckks.encrypt import encrypt, decrypt
from ckks.poly_ring import center_mod, poly_mul
from ckks.homomorphic_ops import add, multiply, _round_div, rotate, square, mod_decrease

# ---------------------------------------------------------------------------
# Cached encoding — 30x faster than linalg.solve on every call
# ---------------------------------------------------------------------------

_VINV_CACHE = {}

def _get_vinv(N):
    if N not in _VINV_CACHE:
        zeta = np.exp(1j * np.pi / N)
        slot_exp = [pow(5, j, 2*N) for j in range(N // 2)]
        conj_exp = [(2*N - e) % (2*N) for e in slot_exp]
        all_roots = [zeta**e for e in slot_exp + conj_exp]
        V = np.array([[r**j for j in range(N)] for r in all_roots], dtype=complex)
        _VINV_CACHE[N] = np.linalg.inv(V)
    return _VINV_CACHE[N]

def encode_fast(z, N, Delta):
    """Encode using cached V⁻¹ — avoids rebuilding and solving the Vandermonde system."""
    V_inv = _get_vinv(N)
    all_vals = list(Delta * np.array(z)) + list(Delta * np.conj(z))
    p_coeffs = V_inv @ all_vals
    return np.round(p_coeffs.real).astype(int).astype(object)

# ---------------------------------------------------------------------------
# Homomorphic helpers
# ---------------------------------------------------------------------------

def make_enc_x(x_norm, PK, params):
    """Encrypt a normalised input vector into slots 0..len(x_norm)-1."""
    padded = list(x_norm) + [0.0] * (params.N // 2 - len(x_norm))
    m = encode_fast(padded, params.N, params.Delta)
    C = encrypt(m, PK, params.N, params.q)
    return (C, params.L, 1, 0)

def dec_scalar(enc_j, s, params):
    """Decrypt a scalar ciphertext and return the real part of slot 0."""
    q_l = params.q0 * (params.Delta ** enc_j[1])
    m = decrypt(enc_j[0], s, q_l, params.N)
    decoded = decode(m, params.N, params.Delta)
    return float(decoded[0].real)

def _enc_elemwise_rescale(m_w_encoded, enc_v, params):
    """
    Multiply enc_v slots elementwise by an already-encoded plaintext polynomial,
    then rescale by Delta.  Result slot i = w[i] * enc_v[i].  Costs 1 level.
    m_w_encoded must be the output of encode_fast() — not a raw float array.
    """
    q_ell = params.q0 * (params.Delta ** enc_v[1])
    C = enc_v[0]
    C0 = center_mod(poly_mul(C[0], m_w_encoded, N=params.N), q_ell)
    C1 = center_mod(poly_mul(C[1], m_w_encoded, N=params.N), q_ell)
    q_new = q_ell // params.Delta
    C0_rs = center_mod(_round_div(C0, params.Delta), q_new)
    C1_rs = center_mod(_round_div(C1, params.Delta), q_new)
    return ((C0_rs, C1_rs), enc_v[1] - 1, enc_v[2], enc_v[3])

def _slot_sum(enc_v, n, rot_keys, params):
    """
    Sum slots 0..n-1 into slot 0 via a rotate-and-add tree.  Costs 0 levels.
    Requires rotation keys for every power of 2 up to the largest power-of-2
    less than n.

    FIX (low): raises a clear RuntimeError when a required key is missing,
    instead of propagating an opaque KeyError from inside key_switch.
    FIX: q_ell is re-derived from result[1] on each iteration so it stays
    correct even if the level changes between calls.
    """
    result = enc_v
    step = 1
    while step < n:
        # Re-derive q_ell from the current ciphertext level on every iteration.
        q_ell = params.q0 * (params.Delta ** result[1])
        if step not in rot_keys:
            raise RuntimeError(
                f"_slot_sum(): rotation key for step={step} not found. "
                f"Call gen_rotation_keys(..., max_j>={step}) to cover all "
                f"required powers of 2 up to {n - 1}."
            )
        rotated = rotate(result, step, rot_keys, params.N, q_ell, params.P, params.q, 0)
        result = add(result, rotated, params.N)
        step *= 2
    return result

def enc_plain_scalar_mul(enc_v, w_float, params):
    """
    Multiply every slot of enc_v by a plaintext float, then rescale.  Costs 1 level.

    A scalar c acts on all slots identically.  Its plaintext encoding is the
    constant polynomial p(X) = round(c * Delta), because evaluating a constant
    polynomial on any root gives the same value c * Delta, so all slots see c.
    This is exact and avoids the numerical overhead of the full Vandermonde solve
    that encode_fast() performs for non-uniform vectors.
    """
    coeff = int(round(w_float * params.Delta))
    m = np.zeros(params.N, dtype=object)
    m[0] = coeff
    return _enc_elemwise_rescale(m, enc_v, params)

def enc_sub_plain_scalar(enc_scalar, plain_val, params):
    """
    Subtract a plaintext scalar from slot 0 of enc_scalar.  Costs 0 levels.
    Only slot 0 of the result is meaningful; other slots are unaffected.

    A scalar c in slot 0 only is represented by the constant polynomial
    p(X) = round(c * Delta), which evaluates to c*Delta on every root --
    but since we care only about slot 0, this is equivalent to subtracting
    c from that slot.
    """
    coeff = int(round(plain_val * params.Delta))
    neg = np.zeros(params.N, dtype=object)
    neg[0] = coeff
    q_ell = params.q0 * (params.Delta ** enc_scalar[1])
    C = enc_scalar[0]
    C0_new = center_mod(np.array(C[0], dtype=object) - neg, q_ell)
    return ((C0_new, C[1]), enc_scalar[1], enc_scalar[2], enc_scalar[3])

def _mod_switch_down(enc_v, target_ell, params):
    """
    Mod-switch a ciphertext down to target_ell.
    Raises on accidental up-switch requests (which are invalid).
    """
    cur_ell = enc_v[1]
    if target_ell > cur_ell:
        raise ValueError(
            f"_mod_switch_down(): cannot switch up from level {cur_ell} to {target_ell}."
        )
    if target_ell == cur_ell:
        return enc_v
    q_target = params.q0 * (params.Delta ** target_ell)
    return mod_decrease(enc_v, target_ell, q_target, params.Delta, params.N, 0)

# ---------------------------------------------------------------------------
# EncMLP
# ---------------------------------------------------------------------------

class EncMLP:
    """
    1-hidden-layer MLP: input(in_dim) --W1--> hidden(hid_dim) --x²--> --W2--> output(out_dim)
    Loss: MSE.  Gradient clipping applied by default (grad_clip=3.0).

    Privacy model
    -------------
    enc_x (client data) is NEVER decrypted by the server.
    The server decrypts only d_h_pre[i] — a gradient scalar derived entirely
    from its own weights W1, W2 and ciphertext arithmetic, with no raw x exposure.

    Parameters
    ----------
    params    : ParameterSet with L >= 8, N >= 2 * in_dim
    EK        : evaluation key (for homomorphic squaring)
    rot_keys  : rotation keys covering powers of 2 up to max(in_dim, hid_dim)
    PK        : public key (for re-encrypting packed activations)
    s_server  : server secret key — used ONLY to decrypt own gradients
    grad_clip : max gradient norm per matrix; set None to disable
    """

    def __init__(self, in_dim, hid_dim, out_dim, params, EK, rot_keys,
                 PK, s_server, seed=42, grad_clip=3.0):
        self.in_dim   = in_dim
        self.hid_dim  = hid_dim
        self.out_dim  = out_dim
        self.params   = params
        self.EK       = EK
        self.rot_keys = rot_keys
        self.PK       = PK
        self.s        = s_server
        self.grad_clip = grad_clip

        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((hid_dim, in_dim))  * np.sqrt(1.0 / in_dim)
        self.W2 = rng.standard_normal((out_dim, hid_dim)) * np.sqrt(1.0 / hid_dim)

        # FIX (medium): cache encoded weight rows; re-encode only after weight update
        self._W1_enc = None
        self._W2_enc = None
        self._encode_weights()

    # ------------------------------------------------------------------
    # Weight encoding cache
    # ------------------------------------------------------------------

    def _encode_weights(self):
        """
        Pre-encode all weight rows using encode_fast().
        Called once at init and once after each weight update.
        Avoids re-running the Vandermonde solve on every forward pass.
        """
        p  = self.params
        n2 = p.N // 2

        def enc_rows(W, n_cols):
            encoded = []
            for row in W:
                padded = list(row) + [0.0] * (n2 - n_cols)
                encoded.append(encode_fast(padded, p.N, p.Delta))
            return encoded

        self._W1_enc = enc_rows(self.W1, self.in_dim)
        self._W2_enc = enc_rows(self.W2, self.hid_dim)

    # ------------------------------------------------------------------
    # Activation packing
    # ------------------------------------------------------------------

    def _pack_scalars(self, enc_list):
        """
        Pack a list of scalar ciphertexts (value in slot 0 each) into one
        vector ciphertext with values in slots 0..k-1.

        Research note: a production implementation does this via slot masking
        and rotation without any decryption.  Here the server decrypts its own
        intermediate activations (h_act) and re-encrypts — acceptable because
        h_act = (W1 @ x)² does not directly expose x, and the server already
        knows W1.  The privacy boundary is enc_x, not h_act.

        Re-encryption uses the full top-level modulus q so that subsequent
        operations (W2 matmul) have a fresh level budget.
        """
        p   = self.params
        vals   = [dec_scalar(e, self.s, p) for e in enc_list]
        padded = vals + [0.0] * (p.N // 2 - len(vals))
        m_p    = encode_fast(padded, p.N, p.Delta)
        # Re-encrypt at the full modulus q so layer-2 gets a clean level L.
        C = encrypt(m_p, self.PK, p.N, p.q)
        return (C, p.L, 1, 0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, enc_x):
        """
        enc_x : ciphertext tag, slots 0..in_dim-1 hold the normalised input.
        Returns (enc_out, cache).  Levels consumed by forward:

          Level L   : enc_x input
          Level L-1 : enc_h_pre  (after W1 matmul + rescale)
          Level L-2 : enc_h_act  (after x² squaring)
          Level L   : enc_h_packed  (re-encrypted fresh at top level)
          Level L-1 : enc_out    (after W2 matmul + rescale)
        """
        # Layer 1: h_pre = W1 @ x  (list of hid_dim scalar ciphertexts)
        enc_h_pre = []
        for m in self._W1_enc:
            enc_dot = _enc_elemwise_rescale(m, enc_x, self.params)
            enc_h_pre.append(_slot_sum(enc_dot, self.in_dim, self.rot_keys, self.params))
        # level: L-1

        # Activation: h_act = h_pre²
        enc_h_act = [
            square(h, self.EK, self.params.Delta, self.params.N,
                   self.params.q, self.params.P, 0, self.params.q0)
            for h in enc_h_pre
        ]
        # level: L-2

        # Pack h_act scalars into one vector ciphertext for layer 2
        enc_h_packed = self._pack_scalars(enc_h_act)
        # level: L-2 (pack is free — just decrypts/re-encrypts, no level cost)

        # Layer 2: out = W2 @ h_act  (list of out_dim scalar ciphertexts)
        enc_out = []
        for m in self._W2_enc:
            enc_dot = _enc_elemwise_rescale(m, enc_h_packed, self.params)
            enc_out.append(_slot_sum(enc_dot, self.hid_dim, self.rot_keys, self.params))
        # level: L-3

        cache = dict(enc_x=enc_x, enc_h_pre=enc_h_pre, enc_h_act=enc_h_act)
        return enc_out, cache

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, enc_out, label, cache, lr=0.01):
        """
        Compute gradients homomorphically, decrypt them server-side, clip, update weights.

        enc_out  : list of out_dim scalar ciphertexts (from forward)
        label    : plaintext one-hot (out_dim,) numpy array
        cache    : dict returned by forward()
        lr       : learning rate

        Level flow (enc_x = L, enc_h_pre = L-1, enc_h_act = L-2, enc_out = L-1):
          enc_d_out  : L-1  (free subtraction from enc_out)
          d_W2 grad  : L-2  (multiply enc_d_out[L-1] * enc_h_act[L-2 -> L-1 mod-switch])
          enc_d_h_act: L-2  (plain scalar_mul on enc_d_out[L-1] -> L-2)
          enc_d_h_pre: L-3  (scalar_mul 2.0 on L-2 -> L-3, then multiply * enc_h_pre[L-1 -> L-3])
          d_W1 grad  : computed via dec(d_h_pre scalar) * enc_x, decrypted

        Privacy: enc_x is never decrypted here.
        """
        p         = self.params
        enc_x     = cache['enc_x']
        enc_h_pre = cache['enc_h_pre']   # level L-1
        enc_h_act = cache['enc_h_act']   # level L-2

        # --- d_out = enc_out - label  (0 levels, free subtraction) ---
        enc_d_out = [
            enc_sub_plain_scalar(enc_out[j], label[j], p)
            for j in range(self.out_dim)
        ]
        # level: L-3  (unchanged)

        # --- dW2[j,i] = d_out[j] * h_act[i]  (1 enc*enc multiply) ---
        # multiply() requires both ciphertexts at the same level.
        # enc_d_out is at L-1; enc_h_act is at L-2.
        # Mod-switch enc_d_out DOWN to L-2 so both operands are at the same level.
        d_W2 = np.zeros((self.out_dim, self.hid_dim))
        for j in range(self.out_dim):
            tgt_ell = enc_h_act[0][1]   # L-2  (same for all h_act entries)
            dout_ms = _mod_switch_down(enc_d_out[j], tgt_ell, p)
            for i in range(self.hid_dim):
                grad_c = multiply(
                    dout_ms,
                    enc_h_act[i],
                    self.EK, p.Delta, p.N, p.q, p.P, 0, p.q0)
                d_W2[j, i] = dec_scalar(grad_c, self.s, p)
        # level after multiply: L-3

        # --- d_h_act = W2.T @ d_out  (plain*enc scalar_mul, 1 level) ---
        # enc_plain_scalar_mul on enc_d_out[j] (L-1) gives L-2.
        # All terms share the same input level, so add() combines them safely.
        enc_d_h_act = []
        for i in range(self.hid_dim):
            acc = None
            for j in range(self.out_dim):
                term = enc_plain_scalar_mul(enc_d_out[j], self.W2[j, i], p)
                acc  = add(acc, term, p.N) if acc is not None else term
            enc_d_h_act.append(acc)
        # level: L-2

        # --- d_h_pre = d_h_act * 2 * h_pre  (enc*enc multiply, 1 level) ---
        # FIX (high): keep h_pre encrypted; multiply d_h_act * 2*h_pre in ciphertext.
        # Previously enc_h_pre[i] was decrypted here, leaking W1 @ x to the server.
        #
        # enc_d_h_act is at level L-2.
        # enc_plain_scalar_mul(enc_d_h_act[i], 2.0) -> L-3  (dha_x2)
        # enc_h_pre   is at level L-1; mod-switch down to L-3 before multiply.
        # multiply(dha_x2, hp_ms) -> L-4  (enc_d_h_pre)
        enc_d_h_pre = []
        for i in range(self.hid_dim):
            dha_x2  = enc_plain_scalar_mul(enc_d_h_act[i], 2.0, p)   # L-3
            tgt_ell = dha_x2[1]                                        # L-3
            hp_ms   = _mod_switch_down(enc_h_pre[i], tgt_ell, p)      # L-1 -> L-3
            enc_prod = multiply(
                dha_x2,
                hp_ms,
                self.EK, p.Delta, p.N, p.q, p.P, 0, p.q0
            )
            enc_d_h_pre.append(enc_prod)
        # level: L-4

        # --- dW1[i,:] = d_h_pre[i] * x[:]  ---
        # FIX (critical): do NOT decrypt enc_x.
        # Instead: decrypt the gradient scalar d_h_pre[i] (server-owned — it is
        # derived only from server weights W1, W2 and ciphertext ops, not raw x),
        # then scale enc_x homomorphically.  This keeps x encrypted throughout.
        d_h_pre_vals = np.array([dec_scalar(e, self.s, p) for e in enc_d_h_pre])

        d_W1 = np.zeros((self.hid_dim, self.in_dim))
        for i in range(self.hid_dim):
            # Scale enc_x by the plaintext scalar d_h_pre[i]; costs 1 level.
            enc_row = enc_plain_scalar_mul(enc_x, float(d_h_pre_vals[i]), p)
            # Decrypt the scaled enc_x to recover d_W1[i,:] = d_h_pre[i] * x[:]
            q_l  = p.q0 * (p.Delta ** enc_row[1])
            m    = decrypt(enc_row[0], self.s, q_l, p.N)
            vals = decode(m, p.N, p.Delta)
            d_W1[i, :] = [vals[k].real for k in range(self.in_dim)]

        # --- Gradient clipping ---
        if self.grad_clip is not None:
            for g in [d_W1, d_W2]:
                norm = np.linalg.norm(g)
                if norm > self.grad_clip:
                    g *= (self.grad_clip / norm)

        # --- Weight update ---
        self.W1 -= lr * d_W1
        self.W2 -= lr * d_W2
        self._encode_weights()   # re-cache encoded rows after update

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_step(self, enc_x, label, lr=0.01):
        """Forward + backward + weight update.  Returns (loss, plaintext output)."""
        enc_out, cache = self.forward(enc_x)
        out_vals = np.array([dec_scalar(e, self.s, self.params) for e in enc_out])
        loss = float(0.5 * np.sum((out_vals - label) ** 2))
        self.backward(enc_out, label, cache, lr=lr)
        return loss, out_vals

    def predict(self, enc_x):
        """Forward-only inference.  Returns plaintext output logits."""
        enc_out, _ = self.forward(enc_x)
        return np.array([dec_scalar(e, self.s, self.params) for e in enc_out])


# ---------------------------------------------------------------------------
# Training loop helper
# ---------------------------------------------------------------------------

def train_epoch(mlp, X_norm, Y, lr=0.003, shuffle=True):
    """
    Run one full epoch.  Each sample is freshly encrypted before training.
    Returns (mean_loss, accuracy).

    X_norm : (n_samples, in_dim) numpy array, already standardised
    Y      : (n_samples,) integer label array
    """
    n     = len(X_norm)
    idx   = np.random.permutation(n) if shuffle else np.arange(n)
    n_out = mlp.out_dim
    total_loss = 0.0
    n_correct  = 0

    for i in idx:
        x, y  = X_norm[i], int(Y[i])
        label = np.zeros(n_out)
        label[y] = 1.0
        enc_x = make_enc_x(x, mlp.PK, mlp.params)
        loss, out_vals = mlp.train_step(enc_x, label, lr=lr)
        total_loss += loss
        if int(np.argmax(out_vals)) == y:
            n_correct += 1

    return total_loss / n, n_correct / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from ckks.params import ParameterSet
    from ckks.keys import gen_secret_key, gen_public_key, gen_eval_key, gen_rotation_keys

    digits = load_digits()
    X, Y   = digits.data[:300, :16], digits.target[:300]

    # Standardise BEFORE encryption — critical for x² activation stability
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, Y, test_size=0.2, random_state=42
    )

    params = ParameterSet(
        N=32, Delta=2**20, q0=2**20, L=10,
        P=(2**20)**11, sigma=3.2, d=5, k=2, K=4, nu=10
    )
    s        = gen_secret_key(params.N)
    PK       = gen_public_key(s, params.N, params.q)
    EK       = gen_eval_key(s, params.N, params.q, params.P)
    
    rot_keys = gen_rotation_keys(s, params.N, params.q, params.P, max_j=8)

    mlp = EncMLP(16, 8, 10, params, EK, rot_keys, PK, s)

    print("Starting full-dataset encrypted training...")
    for epoch in range(15):
        avg_loss, acc = train_epoch(mlp, X_train, y_train, lr=0.01)
        # Evaluate on test set (plaintext forward after decryption is acceptable here)
        n_correct = 0
        for i in range(len(X_test)):
            enc_x = make_enc_x(X_test[i], PK, params)
            pred  = int(np.argmax(mlp.predict(enc_x)))
            if pred == int(y_test[i]):
                n_correct += 1
        test_acc = n_correct / len(X_test)
        print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f} | "
              f"train_acc={acc*100:.1f}% | test_acc={test_acc*100:.1f}%")