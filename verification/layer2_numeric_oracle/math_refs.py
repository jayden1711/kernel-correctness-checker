"""
Float64 MATH DEFINITIONS of the corpus operators, for the Gram screen.

One differentiable PyTorch function per operator, matching the TritonBench
reference kernel's *specification* -- same eps values, same eps placement,
same reduction dims, same padding conventions -- verified against the kernel
sources on 2026-08-27 (see verification_runs/gram_screen_2026-08-27/). These
are what `scope_detect.measure_gram` takes exact directional derivatives of
(torch.func.jvp, float64, CPU).

THE FUNCTIONS ENCODE THE SPEC, NOT THE KERNEL'S BUGS. Where a reference
kernel deviates from its spec, the Gram screen will show a systematic
measured/predicted offset -- that is a feature (it is how the attention
padded-column bug was found), and any such fire must be adjudicated as
"kernel computes a different function" rather than "screen false alarm".
Known instance at the time of writing: the layernorm reference left its
pad lanes unmasked in the variance (`diff = row - mean` with no tl.where),
inflating the variance by (BLOCK-n_cols)*mean^2/n_cols when n_cols is not a
power of two -- FIXED 2026-08-28
(verification_runs/layernorm_mask_fix_2026-08-28/), so the kernel now
matches this spec at every width. No such deviation is currently known to
be live anywhere in the reference set.

Signature convention: f(primary, *companions) with companions in the exact
order the spec's input tuple carries them (checker.py passes
`inputs[1:]` / `adv_inputs[1:]` through unchanged). Scalar companions
(num_groups, kernel_size, stride, padding) arrive as plain ints.

Two corpus operators are deliberately absent: argmax and argmin are
index-valued (J = 0 almost everywhere) and are structurally excluded before
any Gram measurement -- see scope_detect.classify.

Pooling is implemented with pad + unfold + mean/amax rather than
torch.nn.functional pooling so that every registered function is built purely
from ops with well-defined forward-mode AD, and so the count_include_pad /
-inf-padding conventions are visible in the arithmetic instead of hidden in
a backend flag.

Width-adaptive companions: the layernorm/rmsnorm functions slice gamma/beta
to x's width. HISTORY (2026-08-28): this slicing was written for the pre-fix
`non_power_of_two` variants, which fed a width-333 x with full-length
captured companions -- valid when the companions were LONGER than x (kernels
read the first n_cols entries), and silently masking the case where they
were SHORTER, in which the kernels read out of bounds
(verification_runs/oob_adjudication_2026-08-28/). The specs now slice
companions to each variant's width themselves, so this slicing is inert on
contract-satisfying inputs; it is kept because it is harmless there and
kernel-faithful for any longer-companion caller.
"""

import math

import torch

_REGISTRY = {}


def _register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name):
    """The registered float64 math definition for `name`, or None. None means
    the Gram screen declines (fail-open), matching structural_l's contract."""
    return _REGISTRY.get(name)


def registered_ops():
    return sorted(_REGISTRY)


# --- elementwise ----------------------------------------------------------

@_register("softmax")
def _softmax(x):
    return torch.softmax(x, dim=-1)


@_register("log_softmax")
def _log_softmax(x):
    return torch.log_softmax(x, dim=-1)


@_register("gelu")
def _gelu(x):
    # Exact erf variant -- the kernel uses tl.math.erf(x / sqrt(2)).
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


@_register("swish")
def _swish(x):
    return x * torch.sigmoid(x)


# --- rowwise / whole-tensor norms -----------------------------------------

@_register("l1norm")
def _l1norm(x):
    return x / (x.abs().sum(dim=-1, keepdim=True) + 1e-12)


@_register("l2norm")
def _l2norm(x):
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-12)


@_register("frobenius_norm")
def _frobenius_norm(x):
    return x / (torch.sqrt((x * x).sum()) + 1e-12)


@_register("layernorm")
def _layernorm(x, gamma, beta):
    g = gamma[: x.shape[-1]]
    b = beta[: x.shape[-1]]
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)   # biased, masked
    return (x - mean) / torch.sqrt(var + 1e-5) * g + b


@_register("rmsnorm")
def _rmsnorm(x, gamma):
    g = gamma[: x.shape[-1]]
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(ms + 1e-5) * g


@_register("groupnorm")
def _groupnorm(x, num_groups, weight, bias):
    n, c = x.shape[0], x.shape[1]
    xg = x.reshape(n, num_groups, -1)
    mean = xg.mean(dim=-1, keepdim=True)
    var = ((xg - mean) ** 2).mean(dim=-1, keepdim=True)  # biased
    y = ((xg - mean) / torch.sqrt(var + 1e-5)).reshape_as(x)
    shape = (1, c) + (1,) * (x.dim() - 2)
    return y * weight.reshape(shape) + bias.reshape(shape)


@_register("instancenorm")
def _instancenorm(x, weight, bias):
    dims = tuple(range(2, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # biased
    y = (x - mean) / torch.sqrt(var + 1e-5)
    shape = (1, x.shape[1]) + (1,) * (x.dim() - 2)
    return y * weight.reshape(shape) + bias.reshape(shape)


@_register("batchnorm")
def _batchnorm(x, running_mean, running_var, weight, bias):
    # Inference mode: supplied running stats, nothing reduced.
    shape = (1, x.shape[1]) + (1,) * (x.dim() - 2)
    return ((x - running_mean.reshape(shape))
            / torch.sqrt(running_var.reshape(shape) + 1e-5)
            * weight.reshape(shape) + bias.reshape(shape))


# --- reductions -----------------------------------------------------------

@_register("sum_reduction")
def _sum_reduction(x):
    return x.sum(dim=-1)


@_register("mean_reduction")
def _mean_reduction(x):
    return x.mean(dim=-1)


@_register("max_reduction")
def _max_reduction(x):
    return x.amax(dim=-1)


@_register("min_reduction")
def _min_reduction(x):
    return x.amin(dim=-1)


@_register("cross_entropy")
def _cross_entropy(logits, targets):
    ls = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -ls.gather(1, targets.reshape(-1, 1).to(torch.int64)).mean()


# --- matmul ---------------------------------------------------------------

@_register("matmul")
def _matmul(a, b):
    return a @ b


# --- attention (N, D), no batch/head dim ----------------------------------

@_register("flash_attention")
def _flash_attention(q, k, v):
    s = (q @ k.T) / math.sqrt(q.shape[1])
    return torch.softmax(s, dim=-1) @ v


@_register("scaled_dot_product_attention")
def _sdpa(q, k, v):
    s = (q @ k.T) / math.sqrt(q.shape[1])
    return torch.softmax(s, dim=-1) @ v


@_register("causal_flash_attention")
def _causal_flash_attention(q, k, v):
    n = q.shape[0]
    s = (q @ k.T) / math.sqrt(q.shape[1])
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device),
                      diagonal=1)
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


# --- pooling --------------------------------------------------------------
# pad + unfold + reduce: count_include_pad=True for avg (divide by the full
# window via .mean), -inf padding for max, floor output sizes -- exactly the
# reference kernels' conventions.

def _norm_stride(kernel_size, stride):
    return kernel_size if stride is None else stride


@_register("avg_pool1d")
def _avg_pool1d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding, padding))
    return xp.unfold(-1, kernel_size, s).mean(dim=-1)


@_register("avg_pool2d")
def _avg_pool2d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding,) * 4)
    return (xp.unfold(-2, kernel_size, s).unfold(-2, kernel_size, s)
            .mean(dim=(-1, -2)))


@_register("avg_pool3d")
def _avg_pool3d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding,) * 6)
    return (xp.unfold(-3, kernel_size, s).unfold(-3, kernel_size, s)
            .unfold(-3, kernel_size, s).mean(dim=(-1, -2, -3)))


@_register("max_pool1d")
def _max_pool1d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding, padding), value=float("-inf"))
    return xp.unfold(-1, kernel_size, s).amax(dim=-1)


@_register("max_pool2d")
def _max_pool2d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding,) * 4, value=float("-inf"))
    return (xp.unfold(-2, kernel_size, s).unfold(-2, kernel_size, s)
            .amax(dim=(-1, -2)))


@_register("max_pool3d")
def _max_pool3d(x, kernel_size, stride=None, padding=0):
    s = _norm_stride(kernel_size, stride)
    xp = torch.nn.functional.pad(x, (padding,) * 6, value=float("-inf"))
    return (xp.unfold(-3, kernel_size, s).unfold(-3, kernel_size, s)
            .unfold(-3, kernel_size, s).amax(dim=(-1, -2, -3)))
