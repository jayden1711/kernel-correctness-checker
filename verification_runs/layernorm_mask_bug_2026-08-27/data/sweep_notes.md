# Unmasked-lane sweep — source extracts (2026-08-27)

Commands: per kernel file,
`grep -nE "other=|tl\.sum|tl\.max|tl\.min|tl\.where|tl\.cumsum|diff|var|mean" <file>`
over `TritonBench/reference/*.py`,
`verification_runs/phase1_derivations_2026-08-27/kernels/phase1_kernels.py`,
`verification_runs/phase2_convolution_2026-08-27/kernels/conv_kernels.py`.

Key lines (the decisive ones only; full context in the sources):

- layernorm.py: `row = tl.load(..., other=0.0)`; `diff = row - mean`;
  `variance = tl.sum(diff * diff) / n_cols`  ← UNMASKED (the bug)
- groupnorm.py: `diff = tl.where(mask, row - mean, 0.0)`  ← masked
- instancenorm.py: `diff = tl.where(mask, row - mean, 0.0)`  ← masked
- rmsnorm.py: `sq = row * row` over `other=0.0`  ← safe by construction
- softmax/log_softmax/cross_entropy: `other=-float('inf')` before
  `exp(row - max)`  ← safe
- sum/mean_reduction, l1norm (`tl.abs`), l2norm/frobenius (`row*row`):
  `other=0.0`  ← safe
- max/min_reduction, argmax/argmin, max_pool1/2/3d: `other=∓inf`  ← safe
- avg_pool1/2/3d: `other=0.0`, divisor = kernel size (count_include_pad
  semantics)  ← safe
- mat_mult + phase1 matmul variants + conv_kernels.py: zero-padded
  multiply-accumulate  ← safe
- phase1_kernels.py: losses `acc += tl.sum(tl.where(m, v, 0.0))`;
  logsumexp `tl.sum(tl.where(m, tl.exp(x - mx), 0.0))`;
  std/var `d = tl.where(m, x - mu, 0.0)`  ← all explicitly masked
- scans: `tl.cumsum` over `other=0.0` loads, masked stores  ← safe
- flash_attention/scaled_dot_product_attention: unmasked `S` (fixed
  2026-08-27); causal masked via `q_idx >= kv_idx`  ← fixed / safe

Layernorm mutants (for the blast-radius logic):
- ignore_gamma_beta, skip_mean_subtract: `diff = row - mean` unmasked —
  SHARE the reference bug.
- wrong_variance_estimate: `tl.sum(row*row)/n − mean²` over 0-pads — does
  NOT share it (pads contribute 0 to both sums).
