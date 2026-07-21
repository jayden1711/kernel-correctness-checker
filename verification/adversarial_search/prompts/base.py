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
Typical shapes: (128, 64), (256, 64).

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
}


# ── Bug pattern → seed hint ───────────────────────────────────────────────────

BUG_PATTERN_HINTS: dict[str, str] = {
    "partial_tile":       "spike in the last BLOCK columns: patches=[{'indices': '[:, -32:]', 'value': 1e4}]",
    "wrong_reduction":    "non-power-of-two column count: shape[-1]=333 or 777",
    "dtype_overflow":     "values near fp16 max: scale=6e4 on randn",
    "boundary_mask":      "non-aligned shape: use shape like [33, 33] or [65, 65]",
    "wrong_axis":         "constant rows: fill='ones', scale=10, vary per-row via patches",
    "numerical_instab":   "large mean shift: fill='randn', shift=1000.0",
    "ignore_gamma_beta":  "non-unit affine params: gamma scale != 1.0, beta shift != 0.0",
    "ignore_gamma":       "non-unit gamma: e.g. gamma fill='ones', scale=2.0",
    "partial_reduction":  "zeros in first half, large values in second: fill='zeros', patches=[{'indices': '[:, 128:]', 'value': 1e4}]",
    "skip_mean_subtract": "large mean: fill='randn', shift=500.0",
    "wrong_variance":     "very large mean relative to variance: fill='ones', scale=0.01, shift=1000.0",
    "drop_last_tile":     "N not multiple of BLOCK: shape[0]=130 for BLOCK=32, or spike in last 32 rows",
    "skip_rescaling":     "dramatically different per-tile max scores: spike in second half rows",
    "approx_denom":       "large Q values that amplify small normaliser error: Q scale=1e3",
    "wrong_mask":         "large positive K values in positions that should be masked",
    "missing_eps":        "near-zero input: fill='randn', scale=1e-8",
    "wrong_norm":         "large variance in second half only: fill='zeros', patches=[{'indices': '[:, 256:]', 'value': 1e4}]",
    "partial_k_reduct":   "A=ones, B=ones so output should be K (catches partial accumulation)",
    "skip_boundary":      "non-aligned shape: M or N not multiple of BLOCK_SIZE",
    "swapped_strides":    "tall thin A and wide flat B: A=(64,256), B=(256,32)",
    "wrong_dtype":        "values near 100 with large K so K accumulations overflow fp16",
}


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
        f"  Mutants caught:   {verdict.hit_mutants or 'none'}\n"
        f"  Mutants missed:   {verdict.missed_mutants or 'none'}\n"
        f"  Gap confirmed:    {verdict.gap_confirmed}\n"
        f"  Summary:          {verdict.failure_summary}\n\n"
        f"Hints from checker output:\n{hint_block}\n"
        f"{memory_block}\n"
        f"Generate your next proposal. Vary the input structure — "
        f"if large values failed, try structural patterns (patches, non-aligned shapes). "
        f"If the reference failed, reduce magnitudes.\n\n"
        f"Respond with ONLY the JSON proposal schema."
    )