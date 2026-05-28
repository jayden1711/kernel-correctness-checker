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