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
