# math_refs.py independent verification — 2026-08-28

**Item:** theory-audit flag #8 (highest priority). `math_refs.py` is the Gram
screen's ground truth: 27 float64 spec implementations. Its unit tests compare
against torch built-ins chosen by the same author under the same assumptions,
so a transcription error replicated into the test would be invisible.

**Verdict: NO DISCREPANCY FOUND.** All 27 operators agree with an
independently constructed reference to ≤ 1.4e-13 relative (worst case is the
deliberately ill-conditioned near-constant layernorm row; the median case is
≤ 3e-16, i.e. fp64 roundoff), across 51 cases including non-power-of-two
widths, padded attention tiles, small-scale eps-discriminating inputs, and
padding-active pooling. A 10-mutation power check confirms the comparison
would have detected every plausible transcription-error class at ≥ 3e-5
relative — five-plus orders above the 1e-10 pass threshold. Downstream
Gram-screen conclusions stand.

## 1. Method — what "independent" means here

Two legs, both independent of the existing tests AND of the math_refs torch
expressions:

**Leg A — fresh source derivation.** All 27 Triton reference kernel sources
were re-read line by line on 2026-08-28 (this session, context cleared — no
carryover from the 2026-08-27 transcription pass), extracting: eps values and
placements, masked-load sentinels, reduction denominators, scaling factors,
mask conventions, index arithmetic, and host-side wrapper logic (groupnorm's
gamma expansion, cross_entropy's host-side mean, frobenius' host-side sqrt).

**Leg B — from-scratch construction** (`probes/independent_refs.py`).
A pure-Python implementation of every operator — Python floats + the `math`
module, zero torch ops in any computation — written from Leg A's algorithm
notes, not from the math_refs expressions. For the attention family the
construction reproduces the kernel's ONLINE-softmax tiling (BLOCK_N=32,
running max/normalizer/accumulator, padded columns masked to −inf before the
running-max update), so its agreement with math_refs' closed form also
re-proves the streaming identity at fp64. Different transcendental
implementations (`math.erf`/`math.exp` = libm, vs torch's) agree to ~1e-16,
which is itself a two-implementation cross-check of those primitives.

**Input design for eps discrimination.** With O(1) inputs, sqrt(s+1e-12) vs
sqrt(s)+1e-12 differ below fp64 resolution — a naive comparison would pass
both placements. Inputs scaled to 1e-5 (norms) / 1e-3 (rmsnorm) and
near-constant rows (layernorm variance path) push each placement alternative
orders of magnitude above the tolerance (measured in §3).

## 2. Source-derivation results (Leg A)

Every math_refs function matches the kernel source. The load-bearing
convention facts, as freshly derived:

| operator(s) | spec facts verified | math_refs |
|---|---|---|
| softmax, log_softmax | rowwise, max-shifted, −inf pad sentinel | ✓ |
| gelu | exact erf form, INV_SQRT2 = 0.7071067811865476 | ✓ (same constant) |
| swish | x·σ(x), σ = 1/(1+e^−x) | ✓ |
| l1norm | row/(Σ\|x\|+eps), **eps outside**, eps=1e-12 | ✓ |
| l2norm | row/√(Σx²+eps), **eps inside sqrt**, 1e-12 | ✓ |
| frobenius_norm | x/(√S **+ eps outside**), global S, 1e-12 | ✓ |
| layernorm | biased var (÷n), masked diff, eps=1e-5 inside sqrt | ✓ |
| rmsnorm | √(mean(x²)+eps), eps=1e-5 inside | ✓ |
| groupnorm | per-(n,g) row = cpg·spatial contiguous block, biased var, 1e-5; per-channel affine | ✓ (reshape(n,G,−1) ≡ wrapper's view) |
| instancenorm | per-(n,c) spatial, biased var, 1e-5 | ✓ |
| batchnorm | inference mode, running stats, 1e-5 | ✓ |
| sum/mean/max/min | mean ÷ n_cols (not BLOCK); ∓inf sentinels for max/min | ✓ |
| cross_entropy | per-row −log_softmax[target], **host mean** over rows | ✓ |
| matmul | plain A@B, zero-padded tiles | ✓ |
| flash/sdpa/causal attention | scale 1/√D, D=q.shape[1]; padded keys −inf (post-fix); causal keeps j ≤ i | ✓ (triu(diagonal=1) ≡ j≤i) |
| avg_pool 1/2/3d | **count_include_pad** (÷k^d always), floor output size, stride defaults to k | ✓ (pad+unfold+mean) |
| max_pool 1/2/3d | −inf padding, floor sizes | ✓ (pad value=−inf) |

The eps-placement asymmetries the flag specifically worried about (l1/frobenius
outside vs l2/norm-family inside) are real properties of the kernel sources,
faithfully transcribed in both directions.

## 3. Numeric results (Leg B)

`data/independent_refs_results.json` — 51 cases, 0 failures, 27/27 registered
ops covered (coverage asserted in the probe; it exits nonzero on any gap).
Highlights: attention at N=33 (partial second tile, 31 padded columns) agrees
to 8.6e-16 — the pre-fix padded-denominator bug would have produced O(1)
deviation here, so this also pins that math_refs matches the FIXED kernels.
Worst case layernorm/near_const 1.393e-13 (catastrophic cancellation in
(x−mean) at var~1e-12 — expected, and still 10³ below threshold).

**Power check** (`probes/mutation_power.py`) — each plausible transcription
error injected into the pure-Python side; deviation it produces vs math_refs:

| mutation | rel_dev | vs 1e-10 |
|---|---|---|
| l2norm eps outside sqrt | 2.0e-04 | 2e6× |
| frobenius eps inside sqrt | 3.0e-05 | 3e5× |
| layernorm unbiased (n−1) variance | 1.5e-02 | 1e8× |
| layernorm eps 1e-6 | 9.6e-04 | 1e7× |
| rmsnorm eps outside sqrt | 1.8e+00 | — |
| attention scale 1/D | 7.3e-01 | — |
| causal j<i (excl. self) | 2.2e+00 | — |
| avg_pool count_include_pad=False | 2.8e-01 | — |
| mean ÷ BLOCK_SIZE instead of n_cols | 4.8e-01 | — |
| cross_entropy sum instead of mean | 8.0e+00 | — |

Every class detectable; minimum margin 3e5×. A clean pass is therefore
informative, not vacuous.

## 4. Independence scoping — verified vs cross-checked

**Verified independently (all 27):** the authority used is the Triton kernel
source itself — which IS the definition the Gram screen must model — via a
fresh derivation and an implementation path (pure Python/libm) sharing no code
and no torch built-in semantics with either math_refs or its tests. The
transcription path, the failure mode flag #8 names, is fully independently
re-walked for every operator.

**Residual shared assumptions (stated plainly):**
1. **Triton language semantics.** Both the original transcription and this
   round read the same kernel sources under the same understanding of
   `tl.load(mask, other=…)`, `tl.sum`, `tl.dot`, `tl.where`. If that shared
   reading of Triton itself were wrong, both would be wrong together. This is
   bounded, not eliminated, by GPU evidence: 854 banked corpus records measure
   the real compiled kernels against tolerances built from these definitions
   at paired Gram ratio ~1 (gram_screen_2026-08-27), and the two known cases
   where a kernel genuinely differed from its intended spec (attention padded
   columns, layernorm pad variance) were both *detected* by exactly this
   machinery — evidence the shared-reading channel is thin.
2. **Docstring-claimed torch equivalences** (gelu ≡ nn.GELU exact, sdpa ≡
   F.scaled_dot_product_attention defaults, pools ≡ F.*_pool floor-mode):
   these claims about *torch* remain cross-checked only through the existing
   tests plus PyTorch's documented semantics; they are not load-bearing for
   the Gram screen (which needs kernel-spec fidelity, not torch fidelity).
3. **CPU-only round.** Nothing here re-executes the compiled kernels; whether
   the GPU binaries match their source is the corpus arms' job, not this one.

## 5. Reproduce

```bash
.venv/bin/python verification_runs/math_refs_verification_2026-08-28/probes/independent_refs.py
.venv/bin/python verification_runs/math_refs_verification_2026-08-28/probes/mutation_power.py  # run with repo root + probes dir on sys.path
```
Deterministic (seeded Python `random`, no torch RNG, no GPU).
