"""
verification/adversarial_search/prompts/base.py

All LLM prompt content lives here — not in worker.py.

Separating prompts from agent logic means:
  - Prompt ablations (does operator context help?) don't touch agent code
  - Easy to diff prompt versions in git
  - Prompts can be loaded from YAML/JSON for config-driven experiments
  - System prompt and operator context evolve independently

Structure:
  SYSTEM_PROMPT        — fixed instruction block, operator-agnostic
  OPERATOR_CONTEXT     — per-operator background injected into first user turn
  BUG_PATTERN_HINTS    — per-operator bug pattern → input structure guidance
  format_first_turn()  — builds the cold-start user message
  format_refine_turn() — builds a refinement user message from WorkerFeedback
"""

from __future__ import annotations
from typing import Optional

from verification.adversarial_search.schemas import WorkerFeedback


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specialising in finding adversarial inputs \
that expose correctness bugs in Triton kernels.

Your task: propose input tensors that will expose a bug in a candidate kernel \
while being VALID inputs that any correct implementation of the same operator \
would handle correctly.

A valid adversarial input satisfies ALL of:
1. Passes a correct reference implementation through a three-layer checker
   (structural, numeric oracle, and algebraic invariants)
2. Fails a buggy candidate kernel — exposing the specific bug
3. Is NOT noise or an extreme value that crashes everything

You communicate ONLY via JSON. No markdown. No explanation outside the JSON object.

Output schema (respond with ONLY this JSON, nothing else):
{
  "rationale": "<one sentence: why this input will expose the target bug>",
  "predicted_failure_mode": "<which specific bug this targets>",
  "tensors": {
    "<arg_name>": {
      "shape": [<int>, ...],
      "dtype": "float32",
      "fill": "<randn|ones|zeros|arange|literal>",
      "scale": <float>,
      "shift": <float>,
      "patches": [
        {"indices": "<python slice like [:, -32:]>", "value": <float>}
      ]
    }
  }
}

Tensor construction rules:
- fill="randn"    →  torch.randn(*shape) * scale + shift
- fill="ones"     →  torch.ones(*shape)  * scale + shift
- fill="zeros"    →  torch.zeros(*shape) * scale + shift
- fill="arange"   →  torch.arange(n).reshape(shape) * scale + shift
- patches applied after: t[indices] = value (supports any Python slice)

Bug pattern → input structure cheat sheet:
- partial_tile        spike in last tile: patches [{"indices": "[:, -32:]", "value": 1e4}]
- wrong_reduction     non-power-of-two columns (e.g. shape=[512, 333])
- dtype_overflow      values near fp16 max: scale=6e4
- boundary_mask       shape not aligned to tile: e.g. [33, 33] for BLOCK_SIZE=32
- wrong_axis          constant rows: fill="ones", scale=10.0, then patches to vary rows
- numerical_instab    large mean shift: fill="randn", shift=1000.0
- ignore_affine_param non-unit scale parameter (gamma=2.0, beta=3.0)
- partial_reduction   large values in second half only:
                        fill="zeros", patches=[{"indices": "[:, 128:]", "value": 1e4}]"""


# ── Per-operator context ──────────────────────────────────────────────────────

OPERATOR_CONTEXT: dict[str, str] = {
    "softmax": """\
Operator: row-wise softmax over a 2D tensor x of shape (n_rows, n_cols).
Required tensors: {"x": shape (n_rows, n_cols)}
Typical shapes: (512, 2048), (256, 4096). Non-power-of-two cols (e.g. 333) expose tile bugs.

Reference formula: softmax(x)[i,j] = exp(x[i,j] - max_i) / sum_k exp(x[i,k] - max_i)

Common bugs in candidate implementations:
- first_tile: only processes the first BLOCK columns — spike in last tile catches it
- wrong_reduction: incomplete denominator sum — non-power-of-two columns catches it
- missing_max_shift: no numerical stability shift — large values cause overflow/NaN
- skip_output_mask: writes past column boundary — non-aligned shape catches it""",

    "layernorm": """\
Operator: layer normalisation over last dim, f(x, gamma, beta) → y.
Required tensors: {"x": (n_rows, n_cols), "gamma": (n_cols,), "beta": (n_cols,)}
Typical shapes: (512, 512), (256, 1024).

Reference formula: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

Common bugs:
- ignore_gamma_beta: normalises correctly but ignores affine params → use gamma=2, beta=3
- skip_mean_subtract: divides raw x by std instead of (x-mean) → use large mean shift
- wrong_variance: E[x^2] - mean^2 vs E[(x-mean)^2] → numerically diverges at large mean
- wrong_axis: reduces over rows instead of columns → constant rows catches it""",

    "matmul": """\
Operator: matrix multiplication C = A @ B.
Required tensors: {"A": (M, K), "B": (K, N)}
Typical shapes: (256, 256), (512, 256, 128). Non-aligned shapes expose boundary bugs.

Reference formula: C[i,j] = sum_k A[i,k] * B[k,j]

Common bugs:
- partial_k_reduct: accumulates only first K//2 — A=ones, B=ones exposes (output should be K)
- skip_boundary: missing output mask corrupts last tile — non-aligned shape (e.g. 65x65)
- swapped_strides: rectangular A, B so all strides differ (e.g. 64x128 @ 128x32)
- wrong_dtype: fp16 accumulator overflows — values near 1e2 with K=256 accumulations""",

    "flash_attention": """\
Operator: flash attention f(Q, K, V), all 2D (N, D) — single head, no masking.
Required tensors: {"Q": (N, D), "K": (N, D), "V": (N, D)}
  -- ALL THREE ARE RANK 2, shape (N, D): a SINGLE sequence with NO batch and
  NO head dimension. The reference does `N, D = Q.shape` and raises on any
  other rank.
Constraints:
  -- D MUST BE A POWER OF TWO (16, 32, 64, 128, ...). D is passed as a
     tl.constexpr into `tl.arange(0, D)`, and Triton requires an arange range
     to be a power of two, so D = 48 or D = 100 FAILS TO COMPILE. This is a
     hard rejection, not a preference.
  -- D >= 16 (tl.dot requires every participating dimension to be at least 16).
  -- N has NO minimum and NO power-of-two requirement. It is masked, and never
     enters a tl.dot shape. Non-power-of-two N is legal and USEFUL -- it is
     exactly what leaves a partial tile at the end of the loop.
Typical shapes: (128, 64), (256, 64), (96, 64).

Reference formula: O = softmax(Q @ K.T / sqrt(D)) @ V  (online, tiled)

Common bugs:
- drop_last_tile: loop stops at N-BLOCK — spike K in last 32 rows, or N not multiple of 32
- skip_rescaling: missing exp(m-m_new) rescale between tiles — large K shifts the max score
- approx_denom: incomplete normaliser accumulation — large Q values amplify the error
- wrong_mask: off-by-one causal mask — large K values in masked positions should be -inf""",

    "rmsnorm": """\
Operator: RMS normalisation f(x, gamma) → y. No beta — bias-free.
Required tensors: {"x": (n_rows, n_cols), "gamma": (n_cols,)}
Typical shapes: (512, 512), (1000, 333).

Reference formula: y = x / sqrt(mean(x^2) + eps) * gamma

Common bugs:
- ignore_gamma: loads gamma but multiplies by 1 instead — use gamma=2 or gamma=0.5
- wrong_norm: mean(|x|) instead of sqrt(mean(x^2)) — large_variance input catches it
- partial_reduction: only reduces first half of cols — zeros in first half, 1e4 in second
- missing_eps: omits eps in denominator — near-zero input triggers div-by-zero""",

    # ── Added after the causal_flash_attention post-mortem ────────────────
    # OPERATOR_CONTEXT previously held 5 entries against 21 wired operators,
    # and format_first_turn's fallback is `f"Operator: {operator}"` -- a bare
    # label with no tensor keys, no rank, no formula. Measured consequence:
    # operators WITH context averaged 10.8 proposals-to-hit and hit 100% of
    # the time; operators WITHOUT averaged 20.0 and produced the only non-hit.
    # causal_flash_attention spent 27% of its 120 proposals on (B, H, N, D)
    # batched tensors against a 2-D-only reference. Shape convention and rank
    # are stated explicitly below for exactly that reason.
    # See adversarial_results/CFA_NONHIT_ROOTCAUSE.md.

    "log_softmax": """\
Operator: row-wise log-softmax over a 2D tensor x of shape (n_rows, n_cols).
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Typical shapes: (512, 2048), (256, 4096). Non-power-of-two cols (e.g. 333) are
fine and expose tile bugs -- the kernel pads via next_power_of_2(n_cols).

Reference formula: log_softmax(x)[i,j] = x[i,j] - max_i - log(sum_k exp(x[i,k] - max_i))

Common bugs:
- skip_max_subtraction: omits the max shift, so exp() overflows -- large
  positive values (scale 1e2+) diverge while small ones agree""",

    "swish": """\
Operator: elementwise Swish/SiLU, f(x) -> y. Shape-preserving.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Typical shapes: (512, 512), (1024, 1024). BLOCK_SIZE=1024, flat elementwise.

Reference formula: y = x * sigmoid(x) = x / (1 + exp(-x))

Common bugs:
- linear_sigmoid_approx: replaces sigmoid with a piecewise-linear approximation
  -- the two agree near 0 and diverge most around |x| in [2, 6]""",

    "gelu": """\
Operator: elementwise GELU, f(x) -> y. Shape-preserving.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Typical shapes: (512, 512), (1024, 1024). BLOCK_SIZE=1024, flat elementwise.

Reference formula (tanh approximation):
  y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

Common bugs:
- sigmoid_approx: uses x*sigmoid(1.702*x) instead -- agrees near 0, diverges
  most for x in [2, 4]; put mass in that band""",

    "sum_reduction": """\
Operator: sum over the LAST dimension of a 2D tensor. f(x) -> (n_rows,).
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Output is 1-D, one value per row. Kernel pads to next_power_of_2(n_cols).

Reference formula: out[i] = sum_j x[i,j]

Common bugs:
- partial_reduction: sums only the first half of the columns -- put ~zero in
  the first half and large values in the second (patches on [:, n/2:])""",

    "mean_reduction": """\
Operator: mean over the LAST dimension of a 2D tensor. f(x) -> (n_rows,).
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: out[i] = (1/n_cols) * sum_j x[i,j]

Common bugs:
- partial_reduction: averages only the first half of the columns -- zeros in
  the first half, large values in the second""",

    "max_reduction": """\
Operator: max over the LAST dimension of a 2D tensor. f(x) -> (n_rows,).
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: out[i] = max_j x[i,j]

Common bugs:
- wrong_padding: pads the tail of the last tile with 0 instead of -inf, so an
  all-negative row returns 0. Use a NEGATIVE-only fill with a non-power-of-two
  n_cols (e.g. 333) so a real padded tile exists""",

    "min_reduction": """\
Operator: min over the LAST dimension of a 2D tensor. f(x) -> (n_rows,).
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: out[i] = min_j x[i,j]

Common bugs:
- wrong_padding: pads with 0 instead of +inf, so an all-positive row returns 0.
  Use a POSITIVE-only fill with a non-power-of-two n_cols (e.g. 333)""",

    "l1norm": """\
Operator: row-wise L1 normalisation. f(x) -> y, shape-preserving.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: y[i,j] = x[i,j] / (sum_k |x[i,k]| + eps)

Common bugs:
- partial_reduction: the denominator sums only the first half of the columns --
  zeros in the first half, large values in the second""",

    "l2norm": """\
Operator: row-wise L2 normalisation. f(x) -> y, shape-preserving.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: y[i,j] = x[i,j] / (sqrt(sum_k x[i,k]^2) + eps)

Common bugs:
- wrong_norm: uses sum|x| (L1) instead of sqrt(sum x^2) -- the two coincide
  when one element dominates the row, and diverge most when the row's mass is
  spread evenly, so use a flat-magnitude row with varied signs""",

    "frobenius_norm": """\
Operator: GLOBAL Frobenius normalisation -- x / ||x||_F over ALL elements,
not per row. f(x) -> y, shape-preserving.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.

Reference formula: y = x / (sqrt(sum over ALL i,j of x[i,j]^2) + eps)

NOTE: the kernel accumulates via atomic_add across blocks, so its reduction
order is not deterministic in floating point. Tiny run-to-run differences are
legitimate, not a bug.

Common bugs:
- wrong_norm: uses a per-row norm instead of the global one -- make the rows
  have very DIFFERENT magnitudes so per-row and global normalisation diverge""",

    "argmax": """\
Operator: argmax over the LAST dimension. f(x) -> (n_rows,) of int64 INDICES.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Output is integer indices, compared with EXACT equality -- there is no
tolerance, so near-misses do not count.

Reference: first-occurrence tie-break on duplicate max values.

Common bugs:
- tiebreak: uses >= instead of >, returning the LAST occurrence instead of the
  first. Trigger with EXACT duplicate maxima in a row (fill='literal' or a
  patch writing the same value at two positions)""",

    "argmin": """\
Operator: argmin over the LAST dimension. f(x) -> (n_rows,) of int64 INDICES.
Required tensors: {"x": (n_rows, n_cols)}   -- RANK 2, exactly one tensor.
Output is integer indices, compared with EXACT equality.

Reference: first-occurrence tie-break on duplicate min values.

Common bugs:
- tiebreak: returns the LAST occurrence instead of the first. Trigger with
  EXACT duplicate minima in a row""",

    "instancenorm": """\
Operator: instance normalisation, f(x, weight, bias) -> y.
Required tensors: {"x": (N, C, H, W), "weight": (C,), "bias": (C,)}
  -- x is RANK 4 (N, C, *spatial); weight/bias are RANK 1 of length C.
  The channel count C must match across all three. This is the most common
  mistake: a rank-3 x, or weight/bias sized to something other than C.
Typical shapes: x (2, 4, 4, 4) with weight/bias (4,).

Reference formula: normalise each (n, c) instance over its spatial dims to
zero mean / unit variance, then y = x_hat * weight[c] + bias[c]

Common bugs:
- skip_eps: computes rsqrt(var) instead of rsqrt(var + eps) -- only diverges
  when the variance is near zero, so use a near-CONSTANT spatial plane
  (fill='ones' with a tiny scale, or fill='zeros' plus a 1e-8 perturbation)""",

    "batchnorm": """\
Operator: batch normalisation, INFERENCE MODE (uses running statistics).
f(x, running_mean, running_var, weight, bias) -> y.
Required tensors: {"x": (N, C, H, W), "running_mean": (C,),
                   "running_var": (C,), "weight": (C,), "bias": (C,)}
  -- x is RANK 4; the other FOUR are RANK 1 of length C. All five are
  required; C must agree across all of them.
running_var must be POSITIVE (it is a variance) -- use fill='ones' or a
positive scale, never randn, or the reference itself will produce NaN.

Reference formula: y = (x - running_mean[c]) / sqrt(running_var[c] + eps)
                       * weight[c] + bias[c]

Common bugs:
- wrong_running_stats_broadcast: broadcasts the per-channel statistics along
  the wrong axis -- make C differ from the spatial dims so a mis-broadcast
  cannot coincidentally line up, and give each channel a DIFFERENT
  running_mean/running_var""",

    "scaled_dot_product_attention": """\
Operator: scaled dot-product attention, f(Q, K, V) -> O. NO masking, NO dropout.
Required tensors: {"Q": (N, D), "K": (N, D), "V": (N, D)}
  -- ALL THREE ARE RANK 2, shape (N, D): a SINGLE sequence with NO batch and
  NO head dimension. Do NOT propose (B, H, N, D) or (H, N, D); the reference
  does `N, D = Q.shape` and will raise on any other rank.
Constraints:
  -- D MUST BE A POWER OF TWO (16, 32, 64, 128, ...). D is passed as a
     tl.constexpr into `tl.arange(0, D)`, and Triton requires an arange range
     to be a power of two, so D = 48 or D = 100 FAILS TO COMPILE. This is a
     hard rejection, not a preference.
  -- D >= 16 (tl.dot requires every participating dimension to be at least 16).
  -- N has NO minimum and NO power-of-two requirement. It is masked, and never
     enters a tl.dot shape. Short and non-power-of-two N are legal and are
     often the interesting cases.
BLOCK_M = BLOCK_N = 32.
Typical shapes: (128, 64), (256, 32).

Reference formula: O = softmax(Q @ K.T / sqrt(D)) @ V

Common bugs:
- wrong_mask: applies a mask where none should exist -- large positive values
  in K at positions a spurious mask would suppress will expose it""",

    "causal_flash_attention": """\
Operator: CAUSAL flash attention, f(Q, K, V) -> O. Position i attends only to
positions <= i.
Required tensors: {"Q": (N, D), "K": (N, D), "V": (N, D)}
  -- ALL THREE ARE RANK 2, shape (N, D): a SINGLE sequence with NO batch and
  NO head dimension. Do NOT propose (B, H, N, D) or (H, N, D); the reference
  does `N, D = Q.shape` and will raise on any other rank. This was by far the
  most common failure on this operator historically.
Constraints:
  -- D MUST BE A POWER OF TWO (16, 32, 64, 128, ...). D is passed as a
     tl.constexpr into `tl.arange(0, D)`, and Triton requires an arange range
     to be a power of two, so D = 48 or D = 33 FAILS TO COMPILE. This is a
     hard rejection, not a preference, and it was the single most common
     wasted proposal on this operator.
  -- D >= 16 (tl.dot requires every participating dimension to be at least 16).
  -- N has NO minimum and NO power-of-two requirement. It is masked, and never
     enters a tl.dot shape. Non-power-of-two N is legal and USEFUL here: it is
     what puts the causal boundary inside a partial tile.
BLOCK_M = BLOCK_N = 32, so N spanning more than one 32-row block is what makes
the causal boundary interesting.
Typical shapes: (128, 64), (64, 32), (33, 64), (65, 32).

Reference formula: O = softmax(mask(Q @ K.T / sqrt(D))) @ V, where the mask
sets score[i,j] = -inf for j > i.

Common bugs:
- wrong_causal_mask: off-by-one in the mask boundary, so position i can attend
  to i+1 (or is wrongly blocked from i). It is invisible unless the wrongly
  (un)masked position carries weight, so put LARGE positive K values just past
  the diagonal, and make V differ sharply between adjacent positions so the
  leaked contribution changes the output. Use N > 32 so the mask crosses a
  tile boundary. Q/K/V must all VARY -- constant fills make the output
  independent of Q and trip a structural check on the reference itself.""",
}


# ── Bug pattern → seed hint ───────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# GENERIC seed patterns -- NOT mutant ids, and deliberately NOT orphans.
#
# `seed_bug_pattern` is free-form: a caller may seed a worker with a real
# mutant id (targeting one known bug) OR with one of these generic bug
# *classes* when exploring an operator whose mutants are not yet written.
# Both resolve through the same BUG_PATTERN_HINTS lookup.
#
# A coverage audit against _MUTANT_MAP will show these matching nothing. That
# is EXPECTED. DO NOT DELETE THEM as dead keys -- they are the vocabulary for
# seeding a search on a new operator, and validate_bug_pattern_hints()
# deliberately does not require them to correspond to a mutant.
# ---------------------------------------------------------------------------
GENERIC_SEED_HINTS: dict[str, str] = {
    "partial_tile":       "spike in the last BLOCK columns: patches=[{'indices': '[:, -32:]', 'value': 1e4}]",
    "dtype_overflow":     "values near fp16 max: scale=6e4 on randn",
    "boundary_mask":      "non-aligned shape: use shape like [33, 33] or [65, 65]",
    "wrong_axis":         "constant rows: fill='ones', scale=10, vary per-row via patches",
    "numerical_instab":   "large mean shift: fill='randn', shift=1000.0",
    "missing_eps":        "near-zero input: fill='randn', scale=1e-8",
}


# ---------------------------------------------------------------------------
# PER-MUTANT hints, keyed by the EXACT mutant id used in _MUTANT_MAP
# (scripts/run_adversarial_search.py).
#
# The lookup is exact and silent (`.get(key, "")`), so a key that does not
# match a real mutant id yields an empty hint and a prompt that never names the
# bug, with no error anywhere. That has happened twice in this table already:
# "wrong_mask" did not cover "wrong_causal_mask", and "partial_tile" did not
# cover "first_tile". validate_bug_pattern_hints() below makes that class of
# miss a hard startup failure -- keep every id here in sync with _MUTANT_MAP.
#
# Mutant ids are NOT unique across operators (see the "wrong_mask" note), so a
# duplicate key here silently overwrites rather than colliding. Adding an entry
# means checking it is not already present.
# ---------------------------------------------------------------------------
MUTANT_HINTS: dict[str, str] = {
    "wrong_reduction":    "non-power-of-two column count: shape[-1]=333 or 777",
    "ignore_gamma_beta":  "non-unit affine params: gamma scale != 1.0, beta shift != 0.0",
    "ignore_gamma":       "non-unit gamma: e.g. gamma fill='ones', scale=2.0",
    "partial_reduction":  "zeros in first half, large values in second: fill='zeros', patches=[{'indices': '[:, 128:]', 'value': 1e4}]",
    "skip_mean_subtract": "large mean: fill='randn', shift=500.0",
    "wrong_variance":     "very large mean relative to variance: fill='ones', scale=0.01, shift=1000.0",
    "drop_last_tile":     "N not multiple of BLOCK: shape[0]=130 for BLOCK=32, or spike in last 32 rows",
    "skip_rescaling":     "dramatically different per-tile max scores: spike in second half rows",
    "approx_denom":       "large Q values that amplify small normaliser error: Q scale=1e3",
    # KEY COLLISION: two operators have a mutant named `wrong_mask` and they
    # need opposite advice -- flash_attention's is an off-by-one causal mask
    # (a mask that should exist is misplaced), scaled_dot_product_attention's
    # applies a mask where there should be none. The table is keyed by mutant
    # id ALONE, not (operator, mutant), so one entry must serve both. Adding a
    # second "wrong_mask" key silently overwrites the first -- Python keeps the
    # last -- which is exactly what happened while writing these hints and was
    # caught only by a duplicate-key audit. If this table grows, key it by
    # (operator, mutant_id) instead; the lookup already has `operator` in hand.
    "wrong_mask":         "large positive K values at positions where the mask boundary is in doubt -- for a causal mask, just past the diagonal; for an unmasked operator, anywhere a spurious mask would suppress them",
    "wrong_norm":         "large variance in second half only: fill='zeros', patches=[{'indices': '[:, 256:]', 'value': 1e4}]",
    "partial_k_reduct":   "A=ones, B=ones so output should be K (catches partial accumulation)",
    "skip_boundary":      "non-aligned shape: M or N not multiple of BLOCK_SIZE",
    "swapped_strides":    "tall thin A and wide flat B: A=(64,256), B=(256,32)",
    "wrong_dtype":        "values near 100 with large K so K accumulations overflow fp16",
    # Added after the causal_flash_attention post-mortem. The lookup is exact
    # (BUG_PATTERN_HINTS.get(seed_bug_pattern, "")), so "wrong_causal_mask"
    # silently resolved to "" against the existing "wrong_mask" key and the
    # worker was never told what bug it was hunting. validate_bug_pattern_hints
    # below now makes that class of silent miss a hard startup error.
    "wrong_causal_mask":  "large positive K values JUST PAST the diagonal (position i+1 for query i), with V differing sharply between adjacent positions so a leaked contribution changes the output; use N > 32 so the mask crosses a tile boundary",
    "skip_max_subtraction": "large positive values (scale 1e2+) so the un-shifted exp() overflows",
    "linear_sigmoid_approx": "mass in |x| within [2, 6], where the linear approximation diverges most from the true sigmoid",
    "sigmoid_approx":     "mass in x within [2, 4], the widest divergence band for x*sigmoid(1.702x) vs exact GELU",
    "wrong_padding":      "non-power-of-two n_cols (e.g. 333) so a real padded tile exists, plus an all-negative fill for max (all-positive for min) so 0-padding wins",
    "tiebreak":           "EXACT duplicate extrema in a row (fill='literal', or a patch writing the same value at two positions)",
    "skip_eps":           "near-zero variance: a near-CONSTANT spatial plane (fill='ones' with tiny scale, or 'zeros' plus a 1e-8 perturbation)",
    "wrong_running_stats_broadcast": "C different from the spatial dims so a mis-broadcast cannot coincidentally align, with a DIFFERENT running_mean/running_var per channel",
    # `first_tile` is a LIVE softmax mutant id in _MUTANT_MAP with no entry --
    # "partial_tile" above was evidently meant to serve it but the lookup is by
    # exact mutant id, so it never matched. Same key-mismatch class as
    # wrong_causal_mask/wrong_mask; found by the startup validator below, which
    # is the whole reason that validator exists.
    "first_tile":         "spike in the LAST tile so a first-tile-only kernel never sees the true max: patches=[{'indices': '[:, -32:]', 'value': 1e4}]",
}

# Merged view -- the single table every lookup goes through. Kept as one dict
# so no call site changes; the split above is what documents intent.
BUG_PATTERN_HINTS: dict[str, str] = {**GENERIC_SEED_HINTS, **MUTANT_HINTS}


def validate_bug_pattern_hints(mutant_ids) -> list:
    """
    Return the mutant ids that have no BUG_PATTERN_HINTS entry.

    Exists because the hint lookup is a silent `.get(key, "")`: a missing key
    produces an empty hint and a prompt that never names the bug, with no
    error anywhere. That is what happened to "wrong_causal_mask" -- the table
    had "wrong_mask" but not "wrong_causal_mask", so across a 120-proposal
    run the worker was given no bug-specific guidance at all and nothing
    reported a problem. Callers should treat a non-empty return as fatal at
    startup rather than discovering it from a wasted search budget.
    """
    return sorted(m for m in mutant_ids if m not in BUG_PATTERN_HINTS)


# ── Message formatters ────────────────────────────────────────────────────────

def format_first_turn(
    operator: str,
    seed_bug_pattern: Optional[str] = None,
) -> str:
    context = OPERATOR_CONTEXT.get(operator, f"Operator: {operator}")
    seed_hint = ""
    if seed_bug_pattern:
        pattern_hint = BUG_PATTERN_HINTS.get(seed_bug_pattern, "")
        seed_hint = (
            f"\nStart by targeting the '{seed_bug_pattern}' bug pattern. "
            f"Suggested input structure: {pattern_hint}"
            if pattern_hint
            else f"\nStart by targeting the '{seed_bug_pattern}' bug pattern."
        )

    return (
        f"Operator context:\n{context}\n\n"
        f"This is your first proposal. Generate an adversarial input for "
        f"the '{operator}' operator.{seed_hint}\n\n"
        f"Respond with ONLY the JSON proposal schema."
    )


def format_rejection_turn(error: Exception, operator: str) -> str:
    """
    Correction prompt after a proposal failed to parse or violated the shape
    contract, used by worker._call_and_parse's retry loop.

    Why this is not one fixed string: the retry message used to be hardcoded in
    worker.py as "Respond with ONLY the JSON proposal schema. No markdown, no
    text outside the JSON object." That is right for a JSON parse failure and
    actively WRONG for a shape-contract rejection, where the JSON was already
    perfectly well-formed -- telling the model to fix its formatting teaches it
    nothing about the actual problem, and this string is the only thing telling
    it what to change. Since shape constraints landed (2026-08-21) the semantic
    case is the common one, so the two are separated here.

    format_refine_turn cannot serve this purpose: it requires a populated
    ProposalVerdict, and a proposal rejected before execution has none.
    """
    detail = str(error)
    is_shape_violation = "must be" in detail or "must have" in detail

    if is_shape_violation:
        context = OPERATOR_CONTEXT.get(operator, f"Operator: {operator}")
        return (
            f"Your proposal was REJECTED before it ran, because the input does "
            f"not satisfy the operator's shape contract:\n\n"
            f"    {detail}\n\n"
            f"The JSON itself was fine -- do not change the format. Change the "
            f"SHAPE so it satisfies the constraint above, then resend.\n\n"
            f"The operator's requirements again:\n{context}\n\n"
            f"Keep the adversarial IDEA you were going for; only adjust the "
            f"dimensions so the kernel can actually run on it. An input the "
            f"kernel cannot execute tells us nothing about correctness."
        )

    return (
        f"Your response could not be parsed: {detail}\n"
        f"Respond with ONLY the JSON proposal schema. "
        f"No markdown, no text outside the JSON object."
    )


def format_refine_turn(feedback: WorkerFeedback) -> str:
    verdict = feedback.verdict
    status = (
        "HIT (stop condition met)" if verdict.is_hit
        else "MISS — reference failed checker (input is invalid)" if not verdict.reference_passed
        else "MISS — no mutants caught"
    )
    hint_block = "\n".join(f"  - {h}" for h in feedback.hints)

    memory_block = ""
    if feedback.memory_items:
        memory_block = (
            "\nRelevant memory from past runs:\n"
            + "\n".join(f"  [{m}]" for m in feedback.memory_items)
            + "\n"
        )

    return (
        f"Result of last proposal (id={feedback.proposal_id[:8]}):\n"
        f"  Status:           {status}\n"
        f"  Reference passed: {verdict.reference_passed}\n"
        f"  Caught, gap confirmed: {verdict.hit_mutants or 'none'}\n"
        # Reported separately from "not caught" because they call for opposite
        # responses: caught_no_gap means the checker DID find the bug and only
        # the allclose gap is missing, so the input is close and should be
        # refined; not_caught means the bug was not exposed at all and the
        # input needs rethinking. Collapsing them into "Mutants missed" told
        # the worker to abandon inputs that were nearly there.
        f"  Caught, no allclose gap: {verdict.caught_no_gap or 'none'}\n"
        f"  NOT caught:            {verdict.not_caught or 'none'}\n"
        f"  Gap confirmed:    {verdict.gap_confirmed}\n"
        f"  Summary:          {verdict.failure_summary}\n\n"
        f"Hints from checker output:\n{hint_block}\n"
        f"{memory_block}\n"
        # The trailing "If the reference failed, reduce magnitudes" was the
        # SAME hardcoded misdiagnosis fixed in executor.build_feedback_hints,
        # duplicated into the turn template -- so it re-asserted the wrong
        # advice on every refine turn regardless of what the per-failure hint
        # above had just said. Measured over 122 recorded reference failures,
        # reducing magnitude was the right response to at most 9 of them.
        # The hint block now carries a failure-specific diagnosis; this text
        # must not contradict it.
        f"Generate your next proposal. Vary the input structure — "
        f"if large values failed, try structural patterns (patches, non-aligned shapes). "
        f"If the reference failed, follow the specific diagnosis in the hints "
        f"above rather than assuming magnitude is the problem — it usually is "
        f"not.\n\n"
        f"Respond with ONLY the JSON proposal schema."
    )