"""Correct + mutant implementations for the 27 Phase-1 operators, at the TORCH
level.

WHY TORCH AND NOT CUDA/TRITON. The shipped corpus
(benchmarks/kernelbench_corpus/candidates/) is CUDA via load_inline and needs a
GPU to compile. These are the same BUGS expressed in torch, which run here and
therefore answer the question that actually matters now: do the Phase-1 specs'
adversarial batteries and algebraic properties SEPARATE a correct kernel from a
buggy one? Porting a mutant to CUDA once its semantics are pinned is mechanical;
guessing whether an unrunnable CUDA mutant is caught is not.

Each mutant encodes a REAL failure mode named in its spec's adversarial
comments -- not a random perturbation.
"""
import math
import torch
import torch.nn.functional as F

# name -> (reference, correct, {mutant_name: fn}, spec_key)
OPS = {}

def reg(key, ref, correct, mutants):
    OPS[key] = dict(ref=ref, correct=correct, mutants=mutants)

# --- elementwise activations ----------------------------------------------
reg("relu", lambda x: F.relu(x), lambda x: torch.where(x > 0, x, torch.zeros_like(x)), {
    "leaky_slope":   lambda x: torch.where(x > 0, x, 0.1 * x),      # not idempotent
    # a real relu bug, not a 1e-7 tickle: zeroes the boundary instead of passing it
    "zero_at_boundary": lambda x: torch.where(x > 1e-3, x, torch.zeros_like(x)),
})
reg("leaky_relu", lambda x, s: F.leaky_relu(x, s), lambda x, s: torch.where(x > 0, x, s * x), {
    "hardcoded_slope": lambda x, s: torch.where(x > 0, x, 0.01 * x),  # ignores s
    "dead_negative":   lambda x, s: F.relu(x),
})
reg("sigmoid", lambda x: torch.sigmoid(x), lambda x: torch.sigmoid(x), {
    # exp(x)/(1+exp(x)): inf/inf -> NaN for large positive x. NOTE 1/(1+exp(-x))
    # is NOT a valid mutant here -- exp(inf) makes it 1/(1+inf) = 0, the CORRECT
    # answer, so it is numerically fine in fp32 and was silently a no-op.
    "unstable_exp":  lambda x: torch.exp(x) / (1.0 + torch.exp(x)),
    "missing_one":   lambda x: torch.exp(x) / (torch.exp(x) + 0.9),
})
reg("tanh", lambda x: torch.tanh(x), lambda x: torch.tanh(x), {
    "hard_approx":   lambda x: x.clamp(-1, 1),
    "wrong_coeff":   lambda x: torch.tanh(0.9 * x),
})
reg("selu", lambda x: F.selu(x), lambda x: 1.0507009873554805 * torch.where(
        x > 0, x, 1.6732632423543772 * (torch.exp(x) - 1)), {
    "missing_scale": lambda x: torch.where(x > 0, x, 1.6732632423543772 * (torch.exp(x) - 1)),
    "swapped_consts":lambda x: 1.6732632423543772 * torch.where(
        x > 0, x, 1.0507009873554805 * (torch.exp(x) - 1)),
})
reg("elu", lambda x, a: F.elu(x, a), lambda x, a: torch.where(x > 0, x, a * (torch.exp(x) - 1)), {
    "missing_minus_one": lambda x, a: torch.where(x > 0, x, a * torch.exp(x)),
    "ignores_alpha":     lambda x, a: torch.where(x > 0, x, torch.exp(x) - 1),
})
reg("softplus", lambda x, b: F.softplus(x, beta=b), lambda x, b: F.softplus(x, beta=b), {
    "naive_overflow": lambda x, b: torch.log1p(torch.exp(b * x)) / b,   # overflows at x=40
    "linear_approx":  lambda x, b: F.relu(x) + math.log(2) / b,
})
reg("hardsigmoid", lambda x: F.hardsigmoid(x), lambda x: ((x + 3) / 6).clamp(0, 1), {
    "wrong_divisor": lambda x: ((x + 3) / 5).clamp(0, 1),
    "wrong_bounds":  lambda x: ((x + 2) / 6).clamp(0, 1),
})
reg("new_gelu", lambda x: F.gelu(x, approximate="tanh"),
    lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3))), {
    "sigmoid_approx": lambda x: x * torch.sigmoid(1.702 * x),
    "dropped_cubic":  lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * x)),
})

# --- scans -----------------------------------------------------------------
reg("cumsum", lambda x: torch.cumsum(x, -1), lambda x: torch.cumsum(x, -1), {
    "exclusive_off_by_one": lambda x: torch.cumsum(x, -1) - x,
    "first_tile_only":      lambda x: torch.cat(
        [torch.cumsum(x[..., :64], -1),
         torch.cumsum(x[..., :64], -1)[..., -1:].expand(*x.shape[:-1], max(0, x.shape[-1]-64))], -1)
        if x.shape[-1] > 64 else torch.cumsum(x, -1),
})
reg("cumsum_reverse", lambda x: torch.cumsum(x.flip(-1), -1).flip(-1),
    lambda x: torch.cumsum(x.flip(-1), -1).flip(-1), {
    "forward_instead": lambda x: torch.cumsum(x, -1),
    "off_by_one":      lambda x: torch.cumsum(x.flip(-1), -1).flip(-1) - x,
})
reg("cumsum_exclusive", lambda x: torch.cumsum(x, -1) - x, lambda x: torch.cumsum(x, -1) - x, {
    "inclusive_instead": lambda x: torch.cumsum(x, -1),
    "shifted_wrong":     lambda x: torch.cat([torch.zeros_like(x[..., :1]),
                                              torch.cumsum(x, -1)[..., :-1]], -1) + 1e-3,
})
reg("masked_cumsum", lambda x, m: torch.cumsum(x * m, -1), lambda x, m: torch.cumsum(x * m, -1), {
    "ignores_mask":  lambda x, m: torch.cumsum(x, -1),
    "mask_after":    lambda x, m: torch.cumsum(x, -1) * m,
})

# --- matmul variants -------------------------------------------------------
reg("matvec", lambda A, v: A @ v, lambda A, v: A @ v, {
    "partial_k":      lambda A, v: A[:, :A.shape[1]//2] @ v[:A.shape[1]//2],
    "uninit_accum":   lambda A, v: A @ v + 1e-3,
})
reg("batched_matmul", lambda A, B: torch.bmm(A, B), lambda A, B: torch.bmm(A, B), {
    "ignores_batch_stride": lambda A, B: torch.matmul(A, B[0:1]).expand_as(torch.bmm(A, B)).contiguous(),
    "partial_k":            lambda A, B: torch.bmm(A[:, :, :A.shape[2]//2], B[:, :B.shape[1]//2, :]),
})
reg("diagonal_matmul", lambda d, B: torch.diag(d) @ B, lambda d, B: d.unsqueeze(1) * B, {
    "wrong_axis":  lambda d, B: d.unsqueeze(0).expand(B.shape[1], -1).T.flip(0) * B,
    "no_scaling":  lambda d, B: B * d.mean(),
})
reg("triangular_matmul", lambda A, B: torch.tril(A @ B), lambda A, B: torch.tril(A @ B), {
    "forgot_mask":  lambda A, B: A @ B,
    "wrong_side":   lambda A, B: torch.triu(A @ B),
})

# --- losses ----------------------------------------------------------------
reg("mse_loss", lambda x, t: F.mse_loss(x, t), lambda x, t: ((x - t) ** 2).mean(), {
    "sum_not_mean": lambda x, t: ((x - t) ** 2).sum() / max(1, x.shape[0]),
    "abs_not_sq":   lambda x, t: (x - t).abs().mean(),
})
reg("huber_loss", lambda x, t, b=1.0: F.smooth_l1_loss(x, t, beta=b),
    lambda x, t, b=1.0: F.smooth_l1_loss(x, t, beta=b), {
    "quadratic_only": lambda x, t, b=1.0: (0.5 * (x - t) ** 2 / b).mean(),
    "wrong_boundary": lambda x, t, b=1.0: F.smooth_l1_loss(x, t, beta=b * 2),
})
reg("bce_loss", lambda p, t: F.binary_cross_entropy(p, t),
    lambda p, t: -(t * torch.log(p).clamp_min(-100)
                   + (1 - t) * torch.log(1 - p).clamp_min(-100)).mean(), {
    # BCE without the log-argument floor: p=1e-7 is fine, but p=0 or p=1 give
    # -inf. torch's own binary_cross_entropy clamps log to -100.
    "no_clamp":      lambda p, t: -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean(),
    "missing_term":  lambda p, t: -(t * torch.log(p)).mean(),
})
reg("kldiv_loss", lambda lq, p: F.kl_div(lq, p, reduction="batchmean"),
    lambda lq, p: (p * (torch.log(p.clamp_min(1e-12)) - lq)).sum() / lq.shape[0], {
    "forgot_log_target": lambda lq, p: (-p * lq).sum() / lq.shape[0],
    "mean_not_batchmean":lambda lq, p: (p * (torch.log(p.clamp_min(1e-12)) - lq)).mean(),
})
reg("nll_loss", lambda lp, t: F.nll_loss(lp, t),
    lambda lp, t: -lp.gather(1, t.unsqueeze(1)).mean(), {
    "wrong_index":  lambda lp, t: -lp.gather(1, ((t + 1) % lp.shape[1]).unsqueeze(1)).mean(),
    "no_negation":  lambda lp, t: lp.gather(1, t.unsqueeze(1)).mean(),
})

# --- the rest --------------------------------------------------------------
def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)
reg("rope", _rope, _rope, {
    "half_swap":  lambda x, c, s: torch.cat(
        [x[..., x.shape[-1]//2:] * c - x[..., :x.shape[-1]//2] * s,
         x[..., x.shape[-1]//2:] * s + x[..., :x.shape[-1]//2] * c], -1),
    "sign_error": lambda x, c, s: torch.cat(
        [x[..., :x.shape[-1]//2] * c + x[..., x.shape[-1]//2:] * s,
         x[..., :x.shape[-1]//2] * s + x[..., x.shape[-1]//2:] * c], -1),
})
_sw = lambda x: F.silu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:]
reg("swiglu", _sw, _sw, {
    "halves_swapped": lambda x: F.silu(x[..., x.shape[-1]//2:]) * x[..., :x.shape[-1]//2],
    "gelu_gate":      lambda x: F.gelu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:],
})
reg("logsumexp", lambda x: torch.logsumexp(x, -1), lambda x: torch.logsumexp(x, -1), {
    "no_max_subtract": lambda x: torch.log(torch.exp(x).sum(-1)),
    "first_tile_max":  lambda x: (lambda m: m + torch.log(torch.exp(x - m.unsqueeze(-1)).sum(-1)))(
        x[..., :max(1, x.shape[-1]//8)].max(-1).values),
})
reg("std_reduction", lambda x: x.std(-1), lambda x: x.std(-1), {
    "one_pass_cancel": lambda x: torch.sqrt(
        ((x ** 2).mean(-1) - x.mean(-1) ** 2).clamp_min(0) * x.shape[-1] / (x.shape[-1] - 1)),
    "biased_divisor":  lambda x: x.std(-1, unbiased=False),
})
reg("var_reduction", lambda x: x.var(-1), lambda x: x.var(-1), {
    "biased_divisor":  lambda x: x.var(-1, unbiased=False),
    "one_pass_cancel": lambda x: ((x ** 2).mean(-1) - x.mean(-1) ** 2) * x.shape[-1] / (x.shape[-1] - 1),
})
