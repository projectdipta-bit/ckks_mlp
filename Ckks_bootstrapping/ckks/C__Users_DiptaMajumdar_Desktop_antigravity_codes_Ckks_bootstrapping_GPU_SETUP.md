# GPU Setup Guide for enc_mlp_gpu.py

## Why not just "add GPU=True"?

The original enc_mlp.py stores every polynomial coefficient as a Python bigint
(dtype=object).  GPUs have no hardware bigint support — they only handle int32/int64.

The solution used here is **RNS (Residue Number System)**, the same technique
used by all production CKKS libraries (OpenFHE, SEAL, HEAAN, Lattigo).

Instead of one 220-bit coefficient mod q, we store k residues mod small 58-bit
primes p_1,...,p_k where p_1*...*p_k > q.  Each residue fits in int64.
Polynomial multiplication becomes k independent NTT convolutions over int64 —
massively parallelisable on a GPU.

---

## Step 1 — Find your CUDA version

```bash
nvidia-smi          # shows "CUDA Version: X.Y" in top-right corner
nvcc --version      # shows "release X.Y"
```

If neither command works you don't have a CUDA-capable GPU.
The code still runs on CPU — skip to the CPU section.

---

## Step 2 — Install CuPy (GPU) or use CPU fallback

### GPU (NVIDIA)

Pick the package matching your CUDA version:

| CUDA version | pip package          |
|-------------|----------------------|
| 12.x        | cupy-cuda12x         |
| 11.x        | cupy-cuda11x         |
| 10.2        | cupy-cuda102         |

```bash
pip install cupy-cuda12x        # adjust for your version
```

Verify:
```python
import cupy as cp
print(cp.cuda.runtime.runtimeGetVersion())   # should print e.g. 12020
a = cp.array([1, 2, 3])
print(a)                                      # should print on GPU
```

### CPU fallback (no GPU / no CuPy)

No action needed.  `poly_ring_gpu.py` automatically falls back to NumPy:

```python
# At top of poly_ring_gpu.py:
USE_GPU = True     # set False to force CPU even if CuPy is installed
```

---

## Step 3 — Install other dependencies

```bash
pip install numpy scikit-learn
```

---

## Step 4 — Run

Run from the **project root** (the folder that contains the `ckks/` package):

```bash
cd path/to/Ckks_bootstrapping
python -m ckks.enc_mlp_gpu
```

Expected startup output:
```
Polynomial arithmetic backend: CuPy (GPU)      # or NumPy (CPU)
Building RNS parameter tables...
  RNS ready: 5 limbs, primes: [...]
  Setup time: 2.3s
Generating keys...
  Key gen time: 4.1s
Starting GPU-accelerated encrypted training...
Epoch  1 | loss=2.3142 | train=11.2% | test=10.5% | 38.2s
...
```

The RNS table build and key generation are one-time costs per run.

---

## What was changed vs enc_mlp.py

| Component         | enc_mlp.py (CPU)              | enc_mlp_gpu.py (GPU)           |
|-------------------|-------------------------------|--------------------------------|
| Poly coefficients | Python bigints (dtype=object) | int64 RNS residues (k, N)      |
| poly_mul          | np.convolve on object arrays  | NTT over k int64 limbs on GPU  |
| center_mod        | Python loop, bigint mod       | Vectorised int64 mod on GPU    |
| _round_div        | Python loop, bigint division  | CRT lift → divide → re-encode  |
| poly_sub_X        | Python loop                   | Python loop (index ops, fast)  |
| Forward rows      | Sequential loop               | ThreadPoolExecutor (parallel)  |
| Backward rows     | Sequential loop               | ThreadPoolExecutor (parallel)  |

Everything else — level tracking, noise bounds, key structure, privacy model,
encode/decode, training loop — is identical to enc_mlp.py.

---

## Performance expectations

On CPU (NumPy fallback), enc_mlp_gpu.py will be **slower** than enc_mlp.py
for small N (e.g. N=32) because RNS conversion has overhead.  It becomes
faster than the bigint version for N >= 256.

On GPU (CuPy), NTT polynomial multiplication is 10-100x faster than the CPU
bigint version for N >= 256, and the parallel forward/backward loops give an
additional hid_dim / n_workers speedup.

For the toy params in this codebase (N=32, hid_dim=8), the speedup is modest
because the polynomial degree is very small.  To see real GPU benefit, increase
N to 1024 or 4096 in ParameterSet.

---

## Two-file summary

```
ckks/
  poly_ring_gpu.py   — RNS + NTT polynomial ring, CuPy/NumPy backend
  enc_mlp_gpu.py     — EncMLP_GPU class + training loop
```

`poly_ring_gpu.py` is self-contained and can also be used as a drop-in
accelerator for bootstrap.py and homomorphic_ops.py by replacing the
`poly_ring` import with `poly_ring_gpu`.
