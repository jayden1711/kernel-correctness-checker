"""
Layer 2 -- CLOSED-FORM adaptive tolerance (opt-in, default OFF).

An alternate path for `check_perturbation_tolerance` that computes
`adaptive_tol` from each operator's Jacobian in closed form instead of by
Monte-Carlo probing the reference kernel `n_samples` times.

  KCC_STRUCTURAL_L=1              use this path where a formula exists
  KCC_STRUCTURAL_NSIM=<int>       M3 simulation draws (default 3000)

DEFAULT OFF. Nothing about the shipped verdict changes unless the flag is set.
This module exists to be MEASURED against the Monte-Carlo path, not to replace
it; see verification_runs/structural_l_2026-08-26/FINDINGS.md.

---------------------------------------------------------------------------
WHAT IS ACTUALLY CLOSED-FORM HERE, AND WHAT IS NOT
---------------------------------------------------------------------------
Read this before assuming this path is a drop-in for the probe. The derivation
in verification_runs/adaptive_tol_theory_2026-08-25/generalization/FINDINGS.md
established TWO different things, and only the first is a closed form:

  (A) `L = max_i ||J_i||_2` is structural, for all 27 in-scope operators.
      Closed form matches a converged probe to 0.994-1.018x at K=20000; 9 of
      the 27 need only the output shape. `row_norms()` below is that result,
      transcribed from the validated probe `gen_native.py` unchanged.

  (B) `adaptive_tol` is NOT a closed form in `L`. The same round tested the
      theorem's own leading term -- tol = 3 sigma L sqrt(2 ln 2m) -- and got
      R^2 = -0.34, WORSE than predicting the mean, with 3.53x residual spread.
      `sqrt(2 ln 2m)` is a correct upper BOUND and does not double as an
      estimator. What does predict tol (R^2 = 0.958, no fitted constants) is
      model M3: a Monte-Carlo simulation over the whole closed-form row-norm
      PROFILE, `y_profile()` below.

So this path removes the probe of the KERNEL but replaces it with a simulation
over the profile. It trades `n_samples` reference launches for
`nsim * n_samples * len(profile)` Gaussian draws. Whether that is cheaper is an
empirical question and is exactly what the accompanying run measures -- it is
NOT self-evidently a saving, and the arithmetic suggests it is not one.

---------------------------------------------------------------------------
SCOPE -- WHERE THIS PATH DECLINES TO ANSWER
---------------------------------------------------------------------------
`structural_adaptive_tol` returns None (caller falls back to the probe) when:

  * the operator has no derived formula -- argmax/argmin route through exact
    equality anyway (checker.py), and nothing was derived for them;
  * the companion tensors needed by the formula are absent or the wrong shape;
  * `quantile != 0.95` -- M3 simulates the q95 order statistic specifically;
  * any non-finite value appears in the profile.

It does NOT decline on adversarial inputs, and that is a KNOWN GAP rather than
a claim of validity. The generalization round states plainly that the closed
forms were verified on the corpus's ordinary inputs, and that the saturating
and fp-floor adversarial inputs "are outside the linear regime, so a
Jacobian-based prediction is not expected to hold there and was not tested."
75% of this checker's perturbation-routed calls are exactly those adversarial
variants. Anyone enabling this flag is extrapolating on three quarters of the
call volume. The gate is left open so the cost can be measured on the real
call mix; it must not be read as evidence the formula holds there.
"""

import math
import os
import torch

_STRUCTURAL = os.environ.get("KCC_STRUCTURAL_L") == "1"
_NSIM = int(os.environ.get("KCC_STRUCTURAL_NSIM", "3000"))

# ---------------------------------------------------------------------------
# ESTIMATOR MODE (2026-08-28). Default "m3" keeps the validated simulation
# bit-for-bit. "direct" computes the SAME estimand -- E[q95_n] of M3's
# structural parent -- as a DETERMINISTIC grid integral of the exact parent
# CDF F(t) = prod_i (2 Phi(t/w_i) - 1), with no simulation anywhere:
#
#     E[q95_n] = (1-frac) E[X_(lo+1:n)] + frac E[X_(hi+1:n)],
#     E[X_(k:n)] = int (1 - G_k(t)) dt,
#     G_k(t) = sum_{j>=k} C(n,j) F(t)^j (1-F(t))^{n-j}
#
# (torch.quantile's linear interpolation, by linearity of expectation).
# This is the B.3 DIRECT route validated end-to-end in
# verification_runs/theory_closure_2026-08-28/ (228/228 native invocations,
# n-curve within 0.1% at n=20) and, as an absolute tolerance, in
# verification_runs/direct_tol_2026-08-28/ (vs the M3 simulation: worst
# 0.09% over synthetic profiles; vs the banked measured q95_20:
# R^2(log) = 0.997, median ratio 1.006). It removes the structural round's
# cost objection: ~0.5-1.2 ms per call (dev machine; windowed 512-pt grid,
# 192 log-bins, worst transcription deviation vs the reference
# implementation 0.13%) against the simulation's 27-4200 ms.
#
#   KCC_STRUCTURAL_MODE=direct    (only read when KCC_STRUCTURAL_L=1)
#
# SCOPE under "direct" -- beyond structural_adaptive_tol's own gates, the
# direct mode DECLINES (falls back to the probe) for:
#   * the attention family: the onset law
#     (verification_runs/attn_onset_2026-08-28/) derives that the measured
#     response there is not Jacobian-generated beyond saturation onset, and
#     the gram round measured per-delta ratios spanning six decades on
#     multi_tile_rescaling -- a linearized parent must not set tolerances it
#     cannot see;
#   * the scan family: H1 (theory_audit) proves the independent-rows parent
#     under-disperses scans (exact Brownian parent required); the LEVEL is
#     over-predicted +24.7% (phase1), which is the unsafe direction.
# Known accepted bias, stated not hidden: at non-C^1 points the parent
# misses |delta|-terms and understates the response by at most the kink
# bound g(p) <= 1.44 at the corpus's p = 1/2 (theory_closure §2) -- a
# TIGHTER tolerance, i.e. safe for catches, bounded FP exposure.
#
# MEASURED OUTCOME (T4 A/B, 2026-08-28, direct_tol round): verdicts and
# failing-check sets are IDENTICAL to the probe arm (40/40, 0/200, zero
# diffs; E/draw tol ratio p50 = 1.008 over 605 direct-taken records) --
# and the arm is SLOWER: +18% checker wall. The probe's 20 launches
# pipeline to ~free on the GPU; its large "share" in KCC_CHECK_TIMING
# arms is a serialisation artifact, and host-side closed-form math is a
# de-pipelining step. This mode is an INSTRUMENT (deterministic
# tolerance, no draw noise -- useful for boundary experiments), not a
# latency lever. Default stays the probe.
# ---------------------------------------------------------------------------
_MODE = os.environ.get("KCC_STRUCTURAL_MODE", "m3")

DIRECT_EXCLUDED = frozenset({
    "flash_attention", "causal_flash_attention",
    "scaled_dot_product_attention",
    "cumsum", "cumsum_reverse", "cumsum_exclusive", "masked_cumsum",
})

_SQRT2INV = 0.7071067811865476

# The 9 operators whose row-norm profile follows from the OUTPUT SHAPE alone --
# no pass over the input at all. Kept as a named set because it is the only
# subset for which the structural path is unambiguously cheaper than probing.
STATIC_OPS = frozenset({
    "sum_reduction", "mean_reduction", "max_reduction", "min_reduction",
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    # --- added 2026-08-27, Phase 1 -----------------------------------------
    # The three unmasked scans and std. Each row norm follows from the shape
    # alone, exactly like the ten above -- no pass over the input at all.
    #   cumsum            sqrt(i+1)      (i contributing inputs, inclusive)
    #   cumsum_reverse    sqrt(n-i)
    #   cumsum_exclusive  sqrt(i)
    #   std_reduction     1/sqrt(n-1)
    # masked_cumsum is NOT here -- it reads the mask.
    # std is here but VAR IS NOT: var's row norm is 2||x-m||/(n-1), which is
    # input-dependent. The two differ and are not interchangeable.
    "cumsum", "cumsum_reverse", "cumsum_exclusive",
    "std_reduction",
})

SUPPORTED_OPS = STATIC_OPS | frozenset({
    "matmul", "gelu", "swish", "softmax", "log_softmax",
    "l1norm", "l2norm", "frobenius_norm",
    "layernorm", "rmsnorm", "groupnorm", "instancenorm", "batchnorm",
    "cross_entropy",
    "flash_attention", "causal_flash_attention",
    "scaled_dot_product_attention",
    # --- added 2026-08-27, Phase 1 -----------------------------------------
    # Every form below was derived here and checked against an autograd-exact
    # Jacobian over 5 seeds x 3 input regimes (380 invocations, worst relative
    # error 2.98e-08). Artifacts:
    #   verification_runs/phase1_derivations_2026-08-27/
    #
    # PROBE-VERIFIED ON REAL TRITON KERNELS, 2026-08-27 (Colab T4). 27 purpose-
    # written @triton.jit kernels, 162 native invocations: sandwich 162/162 both
    # sides, and closed-form L vs a converged probe at K=20000 is
    # 1.000-1.023x (median 1.012) -- the same convergence the original 27 showed
    # (1.081 -> 1.025 -> 1.010 across K = 400/4k/20k). The K=400 gap is estimator
    # bias, not formula error.
    #   verification_runs/phase1_derivations_2026-08-27/GPU_NATIVE.md
    #
    # M3 SCOPE, measured in the same round and worth knowing before trusting a
    # tolerance from this path: over the full 54-operator corpus M3's R^2 is
    # 0.8567, down from 0.9579 on the original 27. The entire drop is the SCAN
    # family (+24.7% over-prediction); excluding it, R^2 is 0.9635. M3 assumes
    # orthogonal Jacobian rows and a prefix scan is the maximally correlated
    # case -- row i's support is a strict subset of row j's for i<j. The
    # residual is the orthogonality assumption being paid for, and it is signed
    # in the direction the assumption predicts.
    "relu", "leaky_relu", "sigmoid", "tanh", "selu", "elu",
    "softplus", "hardsigmoid", "new_gelu",
    "masked_cumsum",
    "matvec", "batched_matmul", "diagonal_matmul", "triangular_matmul",
    "mse_loss", "huber_loss", "kldiv_loss", "bce_loss", "nll_loss",
    "rope", "swiglu", "logsumexp", "var_reduction",
    # --- Phase 2, convolution (2026-08-27) ---------------------------------
    # Derived and autograd-verified (19 configs, max rel err 3.8e-16), then
    # probe-verified natively on real Triton kernels. See
    # verification_runs/phase2_convolution_2026-08-27/FINDINGS.md
    "conv1d", "conv2d", "conv3d", "depthwise_conv2d", "pointwise_conv2d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
})

# argmax/argmin are absent DELIBERATELY, not by oversight. checker.py sends
# index-returning operators to _check_exact_match, and the derivation excluded
# them (EXCLUDE = {"argmax", "argmin"} in gen_native.py) because an index has
# no meaningful Jacobian. Adding them here would be forcing a formula where
# none was derived.


def _qlin(a, q, dim=-1):
    """Linear-interpolation quantile, matching torch.quantile's convention.

    Written out rather than calling torch.quantile because torch.quantile caps
    its input at 2**24 elements, which the M3 simulation exceeds.
    """
    s, _ = torch.sort(a, dim=dim)
    n = s.shape[dim]
    h = q * (n - 1)
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return s.select(dim, lo) + (h - lo) * (s.select(dim, hi) - s.select(dim, lo))


def _out_numel(op, x, rest):
    import torch.nn.functional as F
    k, s, p = rest[0], rest[1], rest[2]
    fn = {"avg_pool1d": F.avg_pool1d, "avg_pool2d": F.avg_pool2d,
          "avg_pool3d": F.avg_pool3d, "max_pool1d": F.max_pool1d,
          "max_pool2d": F.max_pool2d, "max_pool3d": F.max_pool3d}[op]
    return fn(x, k, s, p).numel()


def row_norms(op, x, rest):
    """Closed-form `{||J_i||_2}` for one operator on one input.

    TRANSCRIBED UNCHANGED from the validated probe
    verification_runs/adaptive_tol_theory_2026-08-25/generalization/gen_native.py.
    Only the device handling differs: that probe hardcoded `device="cuda"`;
    here every constructed tensor takes its device from `x`, so the same code
    runs on CPU for offline validation.

    Deliberately NOT re-derived. These forms are the measured result -- each
    agrees with a K=20000 probe to 0.994-1.018x -- and rewriting them from the
    report's table would risk introducing a transcription error into the one
    artefact that has actually been checked against a GPU.
    """
    dev = x.device
    if op == "sum_reduction":
        return torch.full((x.shape[0],), math.sqrt(x.shape[-1]), device=dev)
    if op == "mean_reduction":
        return torch.full((x.shape[0],), math.sqrt(x.shape[-1]) / x.shape[-1], device=dev)
    if op in ("max_reduction", "min_reduction"):
        return torch.ones(x.shape[0], device=dev)
    if op.startswith("max_pool"):
        return torch.ones(_out_numel(op, x, rest), device=dev)
    if op.startswith("avg_pool"):
        W = rest[0] ** int(op[-2])
        return torch.full((_out_numel(op, x, rest),), math.sqrt(W) / W, device=dev)
    if op == "matmul":
        return rest[0].norm(dim=0).repeat(x.shape[0])
    if op == "gelu":
        pdf = torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        return (0.5 * (1 + torch.erf(x * _SQRT2INV)) + x * pdf).abs().flatten()
    if op == "swish":
        s = torch.sigmoid(x)
        return (s * (1 + x * (1 - s))).abs().flatten()
    if op in ("softmax", "log_softmax"):
        p = torch.softmax(x, -1)
        s2 = (p * p).sum(-1, keepdim=True)
        base = torch.sqrt((1 - 2 * p + s2).clamp_min(0))
        return (p * base).flatten() if op == "softmax" else base.flatten()
    if op == "l2norm":
        nrm = torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)
        u = x / nrm
        return (torch.sqrt((1 - u * u).clamp_min(0)) / nrm).flatten()
    if op == "l1norm":
        S = x.abs().sum(-1, keepdim=True) + 1e-12
        f = x / S
        n = x.shape[-1]
        return (torch.sqrt((1 - 2 * f.abs() + n * f * f).clamp_min(0)) / S).flatten()
    if op == "frobenius_norm":
        nrm = torch.sqrt((x * x).sum()) + 1e-12
        u = x / nrm
        return (torch.sqrt((1 - u * u).clamp_min(0)) / nrm).flatten()
    if op == "layernorm":
        g = rest[0]
        n = x.shape[-1]
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        z = (x - m) * torch.rsqrt(v + 1e-5)
        return (g.abs() * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "rmsnorm":
        g = rest[0]
        n = x.shape[-1]
        r = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        a = x * x / (n * r * r)
        c = (x * x).mean(-1, keepdim=True) / (r * r)
        return ((g.abs() / r) * torch.sqrt((1 - 2 * a + a * c).clamp_min(0))).flatten()
    if op == "batchnorm":
        rv, w = rest[1], rest[2]
        sh = (1, -1) + (1,) * (x.dim() - 2)
        return (w.view(sh).abs() * torch.rsqrt(rv.view(sh) + 1e-5)
                ).expand_as(x).flatten().contiguous()
    if op == "instancenorm":
        w = rest[0]
        N, C = x.shape[0], x.shape[1]
        x2 = x.contiguous().view(N * C, -1)
        n = x2.shape[-1]
        m = x2.mean(-1, keepdim=True)
        v = ((x2 - m) ** 2).mean(-1, keepdim=True)
        z = (x2 - m) * torch.rsqrt(v + 1e-5)
        ch = torch.arange(N * C, device=dev) % C
        return (w[ch].abs().unsqueeze(-1) * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "groupnorm":
        ng, w = rest[0], rest[1]
        N, C = x.shape[0], x.shape[1]
        sp = x.shape[2:]
        ssz = 1
        for dd in sp:
            ssz *= dd
        cpg = C // ng
        gsz = cpg * ssz
        x2 = x.contiguous().view(N * ng, gsz)
        n = gsz
        m = x2.mean(-1, keepdim=True)
        v = ((x2 - m) ** 2).mean(-1, keepdim=True)
        z = (x2 - m) * torch.rsqrt(v + 1e-5)
        g2 = w.view(ng, cpg).unsqueeze(-1).expand(ng, cpg, ssz).reshape(ng, gsz)
        g2 = g2.unsqueeze(0).expand(N, ng, gsz).reshape(N * ng, gsz)
        return (g2.abs() * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "cross_entropy":
        t = rest[0]
        p = torch.softmax(x, -1).clone()
        p[torch.arange(x.shape[0], device=dev), t] -= 1.0
        return torch.tensor([p.norm().item() / x.shape[0]], device=dev)
    # ======================================================================
    # PHASE 1 (added 2026-08-27). Derived and autograd-verified in
    # verification_runs/phase1_derivations_2026-08-27/. Unlike the block
    # above, these were NOT transcribed from a GPU-validated probe -- see the
    # note on SUPPORTED_OPS. Written here in the same style so the file stays
    # one table rather than two.
    # ======================================================================

    # --- elementwise: diagonal Jacobian, ||J_i|| = |phi'(x_i)| -------------
    if op == "relu":
        return (x > 0).to(x.dtype).flatten()
    if op == "leaky_relu":
        s_ = rest[0] if rest else 0.01
        return torch.where(x > 0, torch.ones_like(x),
                           torch.full_like(x, float(s_))).abs().flatten()
    if op == "sigmoid":
        g = torch.sigmoid(x)
        return (g * (1 - g)).abs().flatten()
    if op == "tanh":
        t = torch.tanh(x)
        return (1 - t * t).abs().flatten()
    if op == "selu":
        a, sc = 1.6732632423543772, 1.0507009873554805
        return (sc * torch.where(x > 0, torch.ones_like(x),
                                 a * torch.exp(x))).abs().flatten()
    if op == "elu":
        a = rest[0] if rest else 1.0
        return torch.where(x > 0, torch.ones_like(x),
                           float(a) * torch.exp(x)).abs().flatten()
    if op == "softplus":
        b = rest[0] if rest else 1.0
        return torch.sigmoid(float(b) * x).abs().flatten()
    if op == "hardsigmoid":
        # MEASURED DEGENERATE on saturating input: 96.7% of rows are exactly
        # zero (only |x| < 3 is live). This is the one Phase-1 operator whose
        # profile collapses, and M3's max-of-|z| simulation over an
        # all-but-empty profile is not a validated regime. Flagged, not fixed.
        return (((x > -3) & (x < 3)).to(x.dtype) / 6.0).flatten()
    if op == "new_gelu":
        c = math.sqrt(2.0 / math.pi)
        k = 0.044715
        t = torch.tanh(c * (x + k * x ** 3))
        return (0.5 * (1 + t)
                + 0.5 * x * (1 - t * t) * c * (1 + 3 * k * x * x)).abs().flatten()

    # --- scans: J is a (masked) triangular block of ones ------------------
    if op in ("cumsum", "cumsum_reverse", "cumsum_exclusive"):
        n = x.shape[-1]
        lead = x.shape[:-1].numel()
        i = torch.arange(n, device=dev, dtype=x.dtype)
        r = {"cumsum": lambda: torch.sqrt(i + 1),
             "cumsum_reverse": lambda: torch.sqrt(n - i),
             "cumsum_exclusive": lambda: torch.sqrt(i)}[op]()
        return r.repeat(lead)
    if op == "masked_cumsum":
        mask = rest[0].to(x.dtype)
        return torch.sqrt(torch.cumsum(mask * mask, dim=-1)).flatten()

    # --- matmul variants ---------------------------------------------------
    if op == "matvec":
        # y = A @ v, primary A. dy_i/dA_pq = v_q delta_ip -> ||J_i|| = ||v||
        return rest[0].norm().repeat(x.shape[0])
    if op == "batched_matmul":
        B_ = rest[0]
        cn = B_.norm(dim=1)                                   # (batch, N)
        return cn.unsqueeze(1).expand(B_.shape[0], x.shape[1],
                                      B_.shape[2]).flatten()
    if op == "diagonal_matmul":
        # C = diag(d) @ B, primary d. dC_ij/dd_p = delta_ip B_ij
        return rest[0].abs().flatten()
    if op == "triangular_matmul":
        # C = tril(A @ B), primary A. Rows above the diagonal are a STRUCTURAL
        # zero, not a small number -- 41.7% of the profile.
        B_ = rest[0]
        M_, N_ = x.shape[0], B_.shape[1]
        keep = torch.tril(torch.ones(M_, N_, device=dev, dtype=x.dtype))
        return (keep * B_.norm(dim=0).unsqueeze(0)).flatten()

    # --- losses: scalar output, m = 1 -------------------------------------
    # All five join cross_entropy at m = 1. CORRECTED 2026-08-28: an earlier
    # version of this comment called m = 1 "M3's known-worst regime (+121%
    # over-prediction)" and predicted these ops would drag the fit down. The
    # +121% belongs to M1' (the sqrt(2 ln 2m) leading term, R^2 = -0.34), not
    # to M3; under M3 cross_entropy measures -1.8%, and the five losses came
    # in at +9.1/+7.3/+0.8/-8.2/-11.9% -- unbiased as a group. m = 1 is not
    # M3's problem and never was. See phase1_derivations_2026-08-27/
    # GPU_NATIVE.md "The m=1 prediction was based on a misreading".
    if op == "mse_loss":
        return (2.0 * (x - rest[0]) / x.numel()).norm().reshape(1)
    if op == "huber_loss":
        t_, beta = rest[0], (rest[1] if len(rest) > 1 else 1.0)
        d = x - t_
        return (torch.where(d.abs() < beta, d / beta, torch.sign(d))
                / x.numel()).norm().reshape(1)
    if op == "kldiv_loss":
        # F.kl_div(input=log q, target=p), reduction='batchmean'
        return (-rest[0] / x.shape[0]).norm().reshape(1)
    if op == "bce_loss":
        t_ = rest[0]
        return ((-(t_ / x) + (1 - t_) / (1 - x)) / x.numel()).norm().reshape(1)
    if op == "nll_loss":
        # grad is -1/N at each (row, target) cell, 0 elsewhere: ||J|| = 1/sqrt(N)
        N_ = x.shape[0]
        return torch.tensor([math.sqrt(N_) / N_], device=dev, dtype=x.dtype)

    # --- convolution (Phase 2, 2026-08-27) ---------------------------------
    # ONE identity covers all eight forms. Conv is linear in x and, for a fixed
    # output element o, the map tap -> input position is injective, so
    #     ||J_o||_2^2 = sum over the IN-BOUNDS taps reaching o of W[tap]^2.
    # That right-hand side is itself the SAME convolution, run with W^2 on an
    # all-ones input: in-bounds taps contribute W^2, out-of-bounds taps
    # contribute 0 exactly as zero-padding does. Hence
    #     ||J_o||_2 = sqrt( F(ones_like(x), W^2, same hyperparameters)[o] )
    # and stride, padding, dilation, groups and asymmetric kernels need no
    # special cases -- they are already encoded in F.
    #
    # Verified against an autograd-exact Jacobian on 19 configurations spanning
    # every combination: max relative error 3.8e-16.
    #
    # INPUT-INDEPENDENT (confirmed, not assumed -- recomputing on a completely
    # different x is bitwise identical), which puts conv in matmul/batchnorm's
    # class. NOT shape-only: it needs W, and padding makes the profile
    # genuinely non-constant across o because border outputs tap fewer weights.
    if op in ("conv1d", "conv2d", "conv3d", "depthwise_conv2d",
              "pointwise_conv2d", "conv_transpose1d", "conv_transpose2d",
              "conv_transpose3d"):
        import torch.nn.functional as _F
        W = rest[0]
        st = rest[1] if len(rest) > 1 else 1
        pd = rest[2] if len(rest) > 2 else 0
        dl = rest[3] if len(rest) > 3 else 1
        gr = rest[4] if len(rest) > 4 else 1
        if op == "depthwise_conv2d":
            gr = x.shape[1]
        elif op == "pointwise_conv2d":
            st, pd, dl, gr = 1, 0, 1, 1
        nd = x.dim() - 2
        fwd = {1: _F.conv1d, 2: _F.conv2d, 3: _F.conv3d}
        rev = {1: _F.conv_transpose1d, 2: _F.conv_transpose2d,
               3: _F.conv_transpose3d}
        ones = torch.ones_like(x)
        W2 = W * W
        if op.startswith("conv_transpose"):
            sq = rev[nd](ones, W2, None, st, pd, 0, gr, dl)
        else:
            sq = fwd[nd](ones, W2, None, st, pd, dl, gr)
        return sq.clamp_min(0).sqrt().flatten()

    # --- the rest ----------------------------------------------------------
    if op == "rope":
        # Rows of J are (cos_k, -sin_k) and (sin_k, cos_k), so
        # ||J_i|| = sqrt(cos^2 + sin^2) -- EXACTLY 1 for a genuine rotation.
        #
        # The general form is kept rather than hardcoding 1.0, deliberately. A
        # cos/sin cache that is not a unit rotation is a real kernel bug, and
        # this reports its true row norm instead of assuming orthogonality.
        # Verified against autograd BOTH ways: unit table -> exactly 1, and a
        # deliberately scaled non-rotation table -> still exact.
        cos, sin = rest[0], rest[1]
        r = torch.sqrt(cos * cos + sin * sin)
        h = x.shape[-1] // 2
        if r.dim() < x.dim():
            r = r.expand(x.shape[:-1] + (h,))
        return torch.cat([r, r], dim=-1).flatten()
    if op == "swiglu":
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        g = torch.sigmoid(a)
        return torch.sqrt((g * (1 + a * (1 - g)) * b) ** 2 + (a * g) ** 2).flatten()
    if op == "logsumexp":
        # dy_r/dx_rj = softmax_rj -> ||J_r|| = ||p_r||_2
        return torch.softmax(x, -1).norm(dim=-1).flatten()
    if op == "var_reduction":
        n = x.shape[-1]
        m = x.mean(-1, keepdim=True)
        return (2.0 * (x - m).norm(dim=-1) / (n - 1)).flatten()
    if op == "std_reduction":
        # 1/sqrt(n-1), shape-only. NOT the same as var_reduction above.
        n = x.shape[-1]
        return torch.full((x.shape[:-1].numel(),), 1.0 / math.sqrt(n - 1),
                          device=dev, dtype=x.dtype)

    if "attention" in op:
        Kk, V = rest[0], rest[1]
        N, D = x.shape
        S = x @ Kk.transpose(-2, -1) * (1.0 / math.sqrt(D))
        if op == "causal_flash_attention":
            ii = torch.arange(N, device=dev).unsqueeze(1)
            jj = torch.arange(N, device=dev).unsqueeze(0)
            S = S.masked_fill(jj > ii, float("-inf"))
        p = torch.softmax(S, -1)
        f = p @ V
        outs = []
        for i in range(N):
            W = p[i].unsqueeze(1) * (V - f[i].unsqueeze(0))
            G = (Kk.transpose(0, 1) @ W) / math.sqrt(D)
            outs.append(G.norm(dim=0))
        return torch.cat(outs)
    return None


def y_profile(rn, n_samples, nsim=None, seed=0):
    """Model M3: E[ q95_n( max_i (||J_i||/L) |z_i| ) ] under orthogonal rows.

    This is the estimator the derivation validated (R^2 = 0.958, 36/38 within
    +-10%) -- and it is a SIMULATION, not an equation. That is the honest
    shape of the result: collapsing the profile to (L, m) and using the
    theorem's `sqrt(2 ln 2m)` leading term scores R^2 = -0.34, so the profile
    cannot be summarised away.

    The row-count cap below is transcribed from the probe: profiles longer than
    30000 keep every row above 1e-3 of the max and subsample the negligible
    tail, because rows far below L cannot win a max.

    NOTE ON `n_samples`. The derivation originally validated this at
    n_samples=40 only, and this docstring flagged the shipped default of 20
    as an unquantified extrapolation. RESOLVED 2026-08-28
    (verification_runs/theory_closure_2026-08-28/, B.3 chaining): the
    structural parent predicts the measured tol_n/tol_40 prefix curve at
    EVERY n in {2..40} on all 228 replayed native invocations — at n=20 the
    aggregate is measured 0.9852 vs predicted 0.9841, and the per-invocation
    deviations are z ~ N(0,1) against the parent's own sampling noise (mean
    z +0.026, sd 0.938). The n=20 default is inside, not outside, the
    validated regime. Caveat inherited from H1: for the SCAN family the
    independent-rows parent under-disperses (CV 0.075 vs measured ~0.14) and
    the exact Brownian/Gram parent is the correct one; the scans are not
    served by this estimator anyway.
    """
    if nsim is None:
        nsim = _NSIM
    rn = rn[rn > 0]
    if rn.numel() == 0:
        return None
    L = rn.max()
    w = rn / L
    if w.numel() > 30000:
        top = w[w > 1e-3]
        rest_ = w[w <= 1e-3]
        if rest_.numel():
            top = torch.cat([top, rest_[:: max(1, rest_.numel() // 8000)]])
        w = top
    g = torch.Generator(device=rn.device).manual_seed(seed)
    acc = 0.0

    # CHUNK SIZE IS ADAPTIVE (fixed 2026-08-27). It was a flat 200, which
    # allocates 200 * n_samples * w.numel() floats in one go.
    #
    # The row-count trim above only subsamples the tail BELOW 1e-3 of the max.
    # For a profile that is dense near its maximum -- |B| for a random B, which
    # is exactly diagonal_matmul's -- almost nothing is trimmed, so w.numel()
    # stays at the full output size. At m = 262144 that is
    # 200 * 40 * 262144 * 4 B = 8.4 GB and the call dies with a CUDA OOM.
    # MEASURED on a T4: diagonal_matmul tried to allocate 7.81 GiB and
    # triangular_matmul 3.91 GiB; both returned None, which the caller reads as
    # "this path declines to answer" -- so the failure was SILENT, presenting
    # as missing coverage rather than as an error.
    #
    # This did not surface on the original 27 because their largest m was 8192.
    # It is a scale bug, not a math bug: the estimator below is UNCHANGED, only
    # the batching of it is. Capping rows instead would have altered the
    # estimator, which is why that was not the fix.
    ELEM_BUDGET = 32_000_000                       # ~128 MB of float32
    CH = max(1, min(200, ELEM_BUDGET // max(1, n_samples * w.numel())))
    done = 0
    while done < nsim:
        b = min(CH, nsim - done)
        z = torch.randn(b, n_samples, w.numel(), generator=g,
                        device=rn.device).abs()
        s = (z * w).max(dim=2).values
        acc += _qlin(s, 0.95, dim=1).sum().item()
        done += b
    return acc / nsim


def _order_stat_weights(n, q=0.95):
    """torch.quantile's interpolated q-quantile as weighted 1-based order
    statistics: q_n = sum_k wt_k * X_(k:n)."""
    h = q * (n - 1)
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    frac = h - lo
    out = [(lo + 1, 1.0 - frac)]
    if frac > 0:
        out.append((hi + 1, frac))
    return out


# Cache for shape-only profiles (STATIC_OPS): E depends only on the
# normalized profile, which for those ops is a function of the output shape
# alone. Bounded; cleared wholesale when full.
_DIRECT_CACHE = {}
_DIRECT_CACHE_MAX = 256

# Truncation threshold, in units of w_max. Rows below it cannot influence
# the parent over the integration region at any measurable level: at the
# relevant t >= ~w_max (where F leaves 0), a dropped row's factor is
# 2 Phi(t/(0.25 w_max)) - 1 >= 2 Phi(4) - 1 = 1 - 6.3e-5, and the region
# where the order-statistic means accumulate mass sits at
# t ~ w_max sqrt(2 ln 2m), pushing that to ~1 - 1e-30. Validated
# numerically against untruncated profiles in
# verification_runs/direct_tol_2026-08-28/ (worst 0.09% incl. MC error).
_DIRECT_W_CUT = 0.25
_DIRECT_GRID = 512
_DIRECT_NBINS = 192


def e_q95_direct(rn, n_samples, quantile=0.95):
    """Deterministic E[q_n] of the structural parent, normalized units
    (multiply by sigma * L for the tolerance scale). None if degenerate.

    Host-side math throughout: the profile is moved to CPU float64 up
    front (grid, binning and erf all live on CPU; a CUDA profile indexed
    against CPU grid tensors is a device-mismatch error, found the hard
    way on the first GPU arm)."""
    rn = rn.detach().to(device="cpu", dtype=torch.float64)
    rn = rn[rn > 0]
    if rn.numel() == 0:
        return None
    L = rn.max()
    w = rn / L
    m_full = int(w.numel())
    w = w[w >= _DIRECT_W_CUT]
    # log-binning: relative width ln(1/0.25)/512 ~ 0.27%, brackets the
    # statistic multiplicatively at +-0.14% -- inside every validation
    # margin this path is used under.
    logw = torch.log(w.clamp(_DIRECT_W_CUT, 1.0))
    lo_edge = math.log(_DIRECT_W_CUT)
    idx = ((logw - lo_edge) / (-lo_edge) * (_DIRECT_NBINS - 1)).round().long()
    cnts = torch.bincount(idx, minlength=_DIRECT_NBINS).double()
    nz = cnts > 0
    centers = torch.exp(torch.linspace(lo_edge, 0.0, _DIRECT_NBINS,
                                       dtype=torch.float64))[nz]
    cnts = cnts[nz]

    t_hi = math.sqrt(2 * math.log(max(2 * m_full, 4))) + 6.0

    def parent_F(t):
        a = t.unsqueeze(0) / (centers.unsqueeze(1) * math.sqrt(2.0))
        logF = (cnts.unsqueeze(1)
                * torch.log(torch.erf(a).clamp_min(1e-300))).sum(dim=0)
        return torch.exp(logF)

    # Two-pass grid: a coarse pass locates the window where F leaves 0 and
    # reaches 1; below it the order-statistic integrand is exactly 1
    # (contributing t_lo as a closed rectangle), above it exactly 0. The
    # fine grid is spent only inside the window -- pure cost, no accuracy
    # change (F is monotone).
    tc = torch.linspace(0.0, t_hi, 96, dtype=torch.float64)
    Fc = parent_F(tc)
    below = (Fc < 1e-12).nonzero()
    above = (Fc > 1.0 - 1e-12).nonzero()
    t_lo = float(tc[below[-1]]) if below.numel() else 0.0
    t_up = float(tc[above[0]]) if above.numel() else t_hi
    t = torch.linspace(t_lo, t_up, _DIRECT_GRID, dtype=torch.float64)
    F = parent_F(t)

    total = 0.0
    dt = t[1] - t[0]
    for k, wt in _order_stat_weights(n_samples, quantile):
        G = torch.zeros_like(F)
        for j in range(k, n_samples + 1):
            G = G + math.comb(n_samples, j) * F**j * (1 - F)**(n_samples - j)
        one_minus = 1.0 - G
        ek = t_lo + float((one_minus[:-1] + one_minus[1:]).sum()) * 0.5 * float(dt)
        total += wt * ek
    return total


def structural_adaptive_tol(op, x, rest, n_samples, quantile, scale,
                            delta_scale, nsim=None):
    """Closed-form `adaptive_tol`, or None if this path declines to answer.

    Returns the same quantity the Monte-Carlo path returns:
        tol = scale * sigma * L * y,    sigma = delta_scale * std(x)
    with `L` and the profile from `row_norms` and `y` from M3.

    Every `return None` below hands the call back to the probe. Declining is
    always safe; guessing is not.
    """
    if op not in SUPPORTED_OPS:
        return None
    if quantile != 0.95:
        # M3 simulates the q95 order statistic specifically. A different
        # quantile is a different estimator and was never validated.
        return None
    try:
        rn = row_norms(op, x, list(rest))
    except Exception:
        # Missing or mis-shaped companion (gamma, B, targets, K/V). Fall back
        # rather than raise: this path must never turn a checkable candidate
        # into an error.
        return None
    if rn is None or rn.numel() == 0 or not torch.isfinite(rn).all():
        return None

    x_std = x.float().std().item()
    if x_std == 0:
        x_std = 1.0
    sigma = delta_scale * x_std

    if _MODE == "direct":
        # Taxonomy-derived exclusions -- see the KCC_STRUCTURAL_MODE block
        # at the top of this file. Declining falls back to the probe.
        if op in DIRECT_EXCLUDED:
            return None
        cache_key = None
        if op in STATIC_OPS:
            # shape-only profile: E is a pure function of (op, shape, n, q)
            cache_key = (op, tuple(x.shape), n_samples, quantile)
            if cache_key in _DIRECT_CACHE:
                y = _DIRECT_CACHE[cache_key]
                tol = scale * sigma * rn.max().item() * y
                return max(tol, 1e-6)
        try:
            y = e_q95_direct(rn, n_samples, quantile)
        except Exception:
            # Fail CLOSED into the probe, never into a check failure: an
            # exception here must not turn a checkable candidate into a
            # verdict (the first GPU arm demonstrated exactly that mode via
            # a device mismatch -- 160/200 reference fails from one bug).
            return None
        if y is None or not math.isfinite(y):
            return None
        if cache_key is not None:
            if len(_DIRECT_CACHE) >= _DIRECT_CACHE_MAX:
                _DIRECT_CACHE.clear()
            _DIRECT_CACHE[cache_key] = y
        tol = scale * sigma * rn.max().item() * y
        return max(tol, 1e-6)

    y = y_profile(rn.float(), n_samples, nsim=nsim)
    if y is None or not math.isfinite(y):
        return None

    tol = scale * sigma * rn.max().item() * y
    # Same floor as the probe path (perturbation.py). Without it, operators
    # whose reference is genuinely perturbation-insensitive would get a
    # tolerance of ~0 and the check would become exact-match by accident.
    return max(tol, 1e-6)
