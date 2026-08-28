"""
Base KernelSpec classes.

Each subclass defines how to run the candidate/reference, generate
adversarial inputs, list algebraic properties, and build inputs for
shape-generalisation tests.

Concrete classes:
  SingleTensorSpec    f(x) -> Tensor            (softmax)
  LayernormSpec       f(x, gamma, beta) -> Tensor
  MatmulSpec          f(A, B) -> Tensor
  AttentionSpec       f(Q, K, V) -> Tensor       (2D: NxD)
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable, Any
import torch


@dataclass
class KernelSpec:
    name: str
    requires_backward: bool = True
    # None means "output dtype must match input dtype" -- true for every
    # operator except index-returning ones (argmax/argmin), which set this
    # explicitly to torch.int64. See runtime_guards.check_dtype_preserved.
    output_dtype: Any = None

    # May the 20 perturbation samples be stacked into ONE kernel call?
    #
    # DEFAULT OFF, and that default is the safety property: an operator added
    # later must not silently inherit batching nobody checked it for. Opt in
    # per family, never blanket-enable.
    #
    # Safe only when BOTH hold:
    #   (a) dim 0 carries independent samples, so 20 stacked samples of shape
    #       (R, C) are just a (20R, C) tensor and the per-row math is unchanged;
    #   (b) no COMPANION tensor is per-sample. checker.py's _ref/_cand
    #       substitute only the primary (`(x,) + inputs[1:]`), so companions
    #       keep their original size -- fine for per-feature weights, which
    #       broadcast, and fine for scalars, but not for anything indexed by
    #       sample.
    #
    # Known exclusions, each verified rather than assumed:
    #   frobenius_norm -- reduces across the WHOLE tensor, not per row. Its own
    #       kernel says so. Stacking yields ONE norm over 20x the data, and it
    #       would return a plausible wrong number rather than raising.
    #   batchnorm      -- normalises ACROSS the batch dim; stacking pollutes
    #       the statistics. Same silent-wrongness shape.
    #   cross_entropy  -- companion `targets` is per-sample (N,); a (20N, C)
    #       logits tensor against an (N,) target is wrong by (b).
    #   matmul, the 3 attention ops -- need a genuinely batched kernel variant.
    #
    # A PROPERTY, deliberately, not a dataclass field. As a field it is silently
    # unoverridable: a subclass that is not ITSELF decorated with @dataclass
    # (most spec files are not) gets its `batch_samples = False` treated as a
    # plain class attribute, while the inherited dataclass __init__ assigns the
    # PARENT's default to the instance and shadows it. frobenius_norm hit
    # exactly this -- the class attribute read False while every instance read
    # True, which would have batched the one operator that must never be
    # batched. A property is resolved on the class and cannot be overwritten by
    # __init__, so overriding it works whether or not the subclass is a
    # dataclass.
    @property
    def batch_samples(self) -> bool:
        return False

    def run_candidate(self, candidate_fn: Callable, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def run_reference(self, reference_fn: Callable, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def primary_input(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def get_adversarial_inputs(self, inputs: Any) -> List[Tuple[str, Any]]:
        raise NotImplementedError

    @property
    def algebraic_properties(self) -> List[Tuple[str, Callable]]:
        return []

    @property
    def valid_shapes(self) -> List[Any]:
        return []

    def make_inputs(self, shape: Any, device: str, dtype: torch.dtype) -> Any:
        raise NotImplementedError


@dataclass
class SingleTensorSpec(KernelSpec):
    # Single tensor in, no companions at all -- condition (b) is vacuous.
    # frobenius_norm subclasses this and overrides back to False.
    @property
    def batch_samples(self) -> bool:
        return True

    def run_candidate(self, candidate_fn, inputs):
        return candidate_fn(inputs)

    def run_reference(self, reference_fn, inputs):
        return reference_fn(inputs)

    def primary_input(self, inputs):
        return inputs

    def make_inputs(self, shape, device, dtype):
        return torch.randn(*shape, device=device, dtype=dtype)


@dataclass
class LayernormKernelSpec(KernelSpec):
    def run_candidate(self, candidate_fn, inputs):
        x, gamma, beta = inputs
        return candidate_fn(x, gamma, beta)

    def run_reference(self, reference_fn, inputs):
        x, gamma, beta = inputs
        return reference_fn(x, gamma, beta)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        n_rows, n_cols = shape
        x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        gamma = torch.ones(n_cols, device=device, dtype=dtype)
        beta = torch.zeros(n_cols, device=device, dtype=dtype)
        return x, gamma, beta


@dataclass
class MatmulKernelSpec(KernelSpec):
    def run_candidate(self, candidate_fn, inputs):
        A, B = inputs
        return candidate_fn(A, B)

    def run_reference(self, reference_fn, inputs):
        A, B = inputs
        return reference_fn(A, B)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        M, K, N = shape
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        return A, B


@dataclass
class AttentionKernelSpec(KernelSpec):
    def run_candidate(self, candidate_fn, inputs):
        Q, K, V = inputs
        return candidate_fn(Q, K, V)

    def run_reference(self, reference_fn, inputs):
        Q, K, V = inputs
        return reference_fn(Q, K, V)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        N, D = shape
        Q = torch.randn(N, D, device=device, dtype=dtype)
        K = torch.randn(N, D, device=device, dtype=dtype)
        V = torch.randn(N, D, device=device, dtype=dtype)
        return Q, K, V


@dataclass
class RMSNormKernelSpec(KernelSpec):
    """f(x, gamma) -> Tensor.  No beta — RMSNorm is bias-free."""

    def run_candidate(self, candidate_fn, inputs):
        x, gamma = inputs
        return candidate_fn(x, gamma)

    def run_reference(self, reference_fn, inputs):
        x, gamma = inputs
        return reference_fn(x, gamma)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        n_rows, n_cols = shape
        x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        gamma = torch.ones(n_cols, device=device, dtype=dtype)
        return x, gamma


@dataclass
class GroupNormKernelSpec(KernelSpec):
    """f(x, num_groups, weight, bias) -> Tensor. num_groups is a fixed
    python int hyperparameter, not a tensor -- held fixed automatically
    since checker.run() only ever replaces inputs[0] (primary_input)."""

    def run_candidate(self, candidate_fn, inputs):
        x, num_groups, weight, bias = inputs
        return candidate_fn(x, num_groups, weight, bias)

    def run_reference(self, reference_fn, inputs):
        x, num_groups, weight, bias = inputs
        return reference_fn(x, num_groups, weight, bias)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        # shape: (N, C, H, W, num_groups)
        N, C, H, W, num_groups = shape
        x = torch.randn(N, C, H, W, device=device, dtype=dtype)
        weight = torch.ones(C, device=device, dtype=dtype)
        bias = torch.zeros(C, device=device, dtype=dtype)
        return x, num_groups, weight, bias


@dataclass
class BatchNormKernelSpec(KernelSpec):
    """f(x, running_mean, running_var, weight, bias) -> Tensor.
    INFERENCE MODE ONLY -- see batchnorm.py's own docstring for why."""

    def run_candidate(self, candidate_fn, inputs):
        x, running_mean, running_var, weight, bias = inputs
        return candidate_fn(x, running_mean, running_var, weight, bias)

    def run_reference(self, reference_fn, inputs):
        x, running_mean, running_var, weight, bias = inputs
        return reference_fn(x, running_mean, running_var, weight, bias)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        N, C, H, W = shape
        x = torch.randn(N, C, H, W, device=device, dtype=dtype)
        running_mean = torch.zeros(C, device=device, dtype=dtype)
        running_var = torch.ones(C, device=device, dtype=dtype)
        weight = torch.ones(C, device=device, dtype=dtype)
        bias = torch.zeros(C, device=device, dtype=dtype)
        return x, running_mean, running_var, weight, bias


@dataclass
class CrossEntropyKernelSpec(KernelSpec):
    """f(logits, targets) -> scalar Tensor. targets is int64 class
    indices -- held fixed automatically while logits (primary_input) is
    perturbed, via the same inputs[1:] mechanism every multi-arg spec
    uses. No special-casing needed for the non-float dtype."""

    def run_candidate(self, candidate_fn, inputs):
        logits, targets = inputs
        return candidate_fn(logits, targets)

    def run_reference(self, reference_fn, inputs):
        logits, targets = inputs
        return reference_fn(logits, targets)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        n_rows, n_cols = shape
        logits = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        targets = torch.randint(0, n_cols, (n_rows,), device=device)
        return logits, targets


@dataclass
class PoolKernelSpec(KernelSpec):
    """f(x, kernel_size, stride, padding) -> Tensor. kernel_size/stride/
    padding are fixed python ints, held fixed automatically same as
    GroupNorm's num_groups above."""

    # N is already a batch dim and the companions are scalars, so
    # stacking to (20N, C, L) leaves the per-sample math untouched.
    @property
    def batch_samples(self) -> bool:
        return True

    def run_candidate(self, candidate_fn, inputs):
        x, kernel_size, stride, padding = inputs
        return candidate_fn(x, kernel_size=kernel_size, stride=stride, padding=padding)

    def run_reference(self, reference_fn, inputs):
        x, kernel_size, stride, padding = inputs
        return reference_fn(x, kernel_size=kernel_size, stride=stride, padding=padding)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        # shape: (*tensor_shape, kernel_size, stride, padding)
        *tensor_shape, kernel_size, stride, padding = shape
        x = torch.randn(*tensor_shape, device=device, dtype=dtype)
        return x, kernel_size, stride, padding


# ===========================================================================
# PHASE 1 SPEC CLASSES (added 2026-08-27)
#
# Each follows the established contract: primary_input is inputs[0], every
# companion rides along untouched via checker.run()'s `(x,) + inputs[1:]`
# substitution, and batch_samples is opted into ONLY where both conditions in
# KernelSpec.batch_samples' note hold. Where a companion is indexed by sample
# (a per-row mask, a per-row target) batching is left OFF -- that is the same
# reasoning that excluded cross_entropy.
# ===========================================================================


@dataclass
class MaskedScanKernelSpec(KernelSpec):
    """f(x, mask) -> Tensor.  mask is per-ELEMENT and shaped like x.

    batch_samples stays OFF: the mask is indexed by sample, so stacking 20
    samples of x against an unstacked mask violates condition (b). Exactly
    cross_entropy's exclusion, for exactly the same reason.
    """

    def run_candidate(self, candidate_fn, inputs):
        x, mask = inputs
        return candidate_fn(x, mask)

    def run_reference(self, reference_fn, inputs):
        x, mask = inputs
        return reference_fn(x, mask)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        x = torch.randn(*shape, device=device, dtype=dtype)
        mask = torch.randint(0, 2, shape, device=device).to(dtype)
        return x, mask


@dataclass
class MatvecKernelSpec(KernelSpec):
    """f(A, v) -> Tensor.  A is (M, K), v is (K,)."""

    def run_candidate(self, candidate_fn, inputs):
        A, v = inputs
        return candidate_fn(A, v)

    def run_reference(self, reference_fn, inputs):
        A, v = inputs
        return reference_fn(A, v)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        M, K = shape
        return (torch.randn(M, K, device=device, dtype=dtype),
                torch.randn(K, device=device, dtype=dtype))


@dataclass
class BatchedMatmulKernelSpec(KernelSpec):
    """f(A, B) -> Tensor.  A is (Bt, M, K), B is (Bt, K, N).

    batch_samples stays OFF. dim 0 is already the BATCH dim, not a sample dim,
    and B is indexed by it -- stacking 20 perturbation samples on dim 0 would
    pair sample s's A against batch s's B. Same class of silent wrongness the
    base note describes for frobenius_norm.
    """

    def run_candidate(self, candidate_fn, inputs):
        A, B = inputs
        return candidate_fn(A, B)

    def run_reference(self, reference_fn, inputs):
        A, B = inputs
        return reference_fn(A, B)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        Bt, M, K, N = shape
        return (torch.randn(Bt, M, K, device=device, dtype=dtype),
                torch.randn(Bt, K, N, device=device, dtype=dtype))


@dataclass
class DiagonalMatmulKernelSpec(KernelSpec):
    """f(d, B) -> Tensor.  C = diag(d) @ B; d is (N,), B is (N, M).

    The PRIMARY is the diagonal VECTOR, not a matrix -- that is what makes the
    row norm |B_ij| rather than a column norm.
    """

    def run_candidate(self, candidate_fn, inputs):
        d, B = inputs
        return candidate_fn(d, B)

    def run_reference(self, reference_fn, inputs):
        d, B = inputs
        return reference_fn(d, B)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        N, M = shape
        return (torch.randn(N, device=device, dtype=dtype),
                torch.randn(N, M, device=device, dtype=dtype))


@dataclass
class TriangularMatmulKernelSpec(KernelSpec):
    """f(A, B) -> Tensor.  C = tril(A @ B), both square (N, N)."""

    def run_candidate(self, candidate_fn, inputs):
        A, B = inputs
        return candidate_fn(A, B)

    def run_reference(self, reference_fn, inputs):
        A, B = inputs
        return reference_fn(A, B)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        N = shape[0]
        return (torch.randn(N, N, device=device, dtype=dtype),
                torch.randn(N, N, device=device, dtype=dtype))


@dataclass
class TargetLossKernelSpec(KernelSpec):
    """f(x, target) -> scalar Tensor, with target shaped like x.

    batch_samples OFF -- target is per-sample.
    """

    def run_candidate(self, candidate_fn, inputs):
        x, t = inputs
        return candidate_fn(x, t)

    def run_reference(self, reference_fn, inputs):
        x, t = inputs
        return reference_fn(x, t)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        return (torch.randn(*shape, device=device, dtype=dtype),
                torch.randn(*shape, device=device, dtype=dtype))


@dataclass
class RopeKernelSpec(KernelSpec):
    """f(x, cos, sin) -> Tensor.  x is (rows, 2h); cos/sin are (rows, h).

    The cos/sin cache is a COMPANION, held fixed while x is perturbed -- which
    is what makes the closed-form row norm sqrt(cos^2+sin^2) constant across
    the perturbation battery.
    """

    def run_candidate(self, candidate_fn, inputs):
        x, cos, sin = inputs
        return candidate_fn(x, cos, sin)

    def run_reference(self, reference_fn, inputs):
        x, cos, sin = inputs
        return reference_fn(x, cos, sin)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        rows, width = shape
        h = width // 2
        theta = torch.randn(rows, h, device=device, dtype=dtype)
        return (torch.randn(rows, width, device=device, dtype=dtype),
                torch.cos(theta), torch.sin(theta))


@dataclass
class ConvKernelSpec(KernelSpec):
    """f(x, W, stride, padding, dilation, groups) -> Tensor.

    `valid_shapes` entries are structured configs, not flat shape tuples:

        (N, C_in, C_out, spatial_tuple, kernel_tuple, stride, padding,
         dilation, groups)

    CORPUS_EXPANSION_PLAN.md §3.2 change 6 flagged that flat tuples stop
    scaling here and recommended a config dataclass *before* the expansion.
    That refactor would touch all 56 existing specs, so it is deliberately NOT
    done inside this phase -- a structured tuple gets the readability without
    a repo-wide change riding along with a corpus addition. The refactor is
    still the right move and is still open.

    batch_samples stays OFF: W is indexed by channel and every conv
    hyperparameter is positional, so stacking 20 perturbation samples on dim 0
    would pair sample s against a weight tensor it does not belong to.

    TRANSPOSED variants set `transposed = True`, which only changes how
    make_inputs shapes W -- (C_in, C_out/g, *k) instead of (C_out, C_in/g, *k).
    """

    transposed: bool = False
    depthwise: bool = False

    def run_candidate(self, candidate_fn, inputs):
        x, W, s, p, d, g = inputs
        if self.depthwise:
            return candidate_fn(x, W, s, p, d)
        if self.name == "pointwise_conv2d":
            return candidate_fn(x, W)
        return candidate_fn(x, W, s, p, d, g)

    def run_reference(self, reference_fn, inputs):
        return self.run_candidate(reference_fn, inputs)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        N, Cin, Cout, sp, k, s, p, d, g = shape
        x = torch.randn(N, Cin, *sp, device=device, dtype=dtype)
        wsh = ((Cin, Cout // g) + tuple(k)) if self.transposed \
              else ((Cout, Cin // g) + tuple(k))
        W = torch.randn(*wsh, device=device, dtype=dtype)
        return x, W, s, p, d, g
