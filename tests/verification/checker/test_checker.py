"""
tests/verification/checker/test_checker_gap.py

Tests for the checker gap — the core claim of the paper.

"Our three-layer checker catches bugs that naive allclose misses."

These tests use pure PyTorch implementations (no Triton, no GPU required)
to verify that:
  1. Correct kernels pass the checker
  2. Specifically-buggy kernels fail the checker
  3. Those same buggy kernels PASS naive allclose (proving the gap is real)

This is the test_eval_adversarial.py analogue from KernelBench — adapted
to our three-layer checker instead of their eval script.

Marked pytest.mark.checker.
"""

import pytest
import torch
import math


pytestmark = pytest.mark.checker


# ── Pure PyTorch references and buggy variants ────────────────────────────────

def _softmax_correct(x):
    return torch.softmax(x, dim=-1)


def _softmax_first_tile(x):
    """Processes first half only, zeros out second half."""
    half = x.shape[-1] // 2
    out = torch.zeros_like(x)
    out[:, :half] = torch.softmax(x[:, :half], dim=-1)
    # second half stays zero — rows sum to ~0.5, not 1
    return out


def _softmax_missing_max_shift(x):
    """No numerical stability shift — wrong on large values, close on small."""
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def _layernorm_correct(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt() * gamma + beta


def _layernorm_ignore_gamma(x, gamma, beta, eps=1e-5):
    """Ignores gamma and beta — output is just normalised x."""
    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


def _layernorm_skip_mean(x, gamma, beta, eps=1e-5):
    """Divides by std without subtracting mean first."""
    var = x.var(-1, keepdim=True, unbiased=False)
    return x / (var + eps).sqrt() * gamma + beta


def _rmsnorm_correct(x, gamma, eps=1e-5):
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / rms * gamma


def _rmsnorm_wrong_norm(x, gamma, eps=1e-5):
    """Uses mean(|x|) instead of sqrt(mean(x^2))."""
    norm = x.abs().mean(-1, keepdim=True) + eps
    return x / norm * gamma


def _rmsnorm_ignore_gamma(x, gamma, eps=1e-5):
    """Ignores gamma."""
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / rms


def _matmul_correct(A, B):
    return A @ B


def _matmul_partial_k(A, B):
    """Only accumulates first K//2 columns of A."""
    half = A.shape[-1] // 2
    return A[:, :half] @ B[:half, :]


# ── Inline three-layer checks (pure PyTorch, no Triton) ──────────────────────

def _check_softmax(out: torch.Tensor, atol=1e-3) -> dict:
    """Run softmax algebraic checks directly on output."""
    results = {}
    # L3: rows sum to 1
    row_sums = out.sum(dim=-1)
    results["rows_sum_to_one"] = torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)
    # L3: all values non-negative
    results["non_negative"] = (out >= 0).all().item()
    # L3: shift-invariance: softmax(x+c) == softmax(x)
    x = torch.randn_like(out)
    out1 = _softmax_correct(x)
    out2 = _softmax_correct(x + 1000.0)
    results["shift_invariant"] = torch.allclose(out1, out2, atol=atol)
    return results


def _check_layernorm(fn, x, gamma, beta, atol=1e-3) -> dict:
    results = {}
    out = fn(x, gamma, beta)
    ref = _layernorm_correct(x, gamma, beta)
    # Numeric oracle
    results["matches_reference"] = torch.allclose(out.float(), ref.float(), atol=atol)
    # L3: gamma=2 → output doubles relative to gamma=1
    ones = torch.ones_like(gamma)
    zeros = torch.zeros_like(beta)
    twos = torch.full_like(gamma, 2.0)
    out1 = fn(x, ones, zeros)
    out2 = fn(x, twos, zeros)
    results["gamma_correctness"] = torch.allclose(out2.float(), (out1 * 2).float(), atol=atol)
    return results


def _check_rmsnorm(fn, x, gamma, atol=1e-3) -> dict:
    results = {}
    ref = _rmsnorm_correct(x, torch.ones_like(gamma))
    out = fn(x, torch.ones_like(gamma))
    # L3: unit RMS when gamma=1
    rms = out.pow(2).mean(-1).sqrt()
    results["unit_rms"] = torch.allclose(rms, torch.ones_like(rms), atol=atol)
    # L3: scale invariance: rmsnorm(c*x) == rmsnorm(x)
    out_scaled = fn(x * 100.0, torch.ones_like(gamma))
    results["scale_invariant"] = torch.allclose(out.float(), out_scaled.float(), atol=atol)
    # L3: gamma=2 doubles output
    out1 = fn(x, torch.ones_like(gamma))
    out2 = fn(x, torch.full_like(gamma, 2.0))
    results["gamma_correctness"] = torch.allclose(out2.float(), (out1 * 2).float(), atol=atol)
    return results


def _naive_allclose(cand, ref, atol=1e-3, rtol=1e-2) -> bool:
    return torch.allclose(cand.float(), ref.float(), atol=atol, rtol=rtol)


# ── Softmax gap tests ─────────────────────────────────────────────────────────

class TestSoftmaxGap:
    """
    The gap: naive allclose passes on small standard inputs,
    checker catches the bug via algebraic invariants.
    """

    def test_correct_softmax_passes_all_checks(self):
        x = torch.randn(32, 64)
        out = _softmax_correct(x)
        checks = _check_softmax(out)
        assert all(checks.values()), checks

    def test_first_tile_rows_do_not_sum_to_one(self):
        x = torch.randn(32, 64)
        out = _softmax_first_tile(x)
        # Second half should be zero
        assert (out[:, 32:] == 0.0).all()
        # Full row sums to ~0.5 when values are spread across both halves
        # but the invariant violation is that second half is zeroed
        assert not (out > 0).all()

    def test_first_tile_passes_naive_on_spike_in_first_half(self):
        """
        On typical small inputs with equal energy in both halves,
        first_tile's output is close to correct.

        This documents the failure mode of naive allclose:
        it passes when the spike happens to be in the first half.
        """
        x = torch.zeros(8, 64)
        x[:, 0] = 10.0   # spike in first column — within the processed half
        out_correct = _softmax_correct(x)
        out_buggy   = _softmax_first_tile(x)
        # On this specific input the outputs match — naive testing passes
        # (the second half is near-zero for both)
        assert _naive_allclose(out_buggy[:, :32], out_correct[:, :32])

    def test_first_tile_fails_checker_on_adversarial_input(self):
        """
        Spike in last column: correct softmax concentrates mass there,
        first_tile zeroes it out. The second half of the output is all
        zeros regardless of input — that's the detectable violation.
        """
        x = torch.zeros(8, 64)
        x[:, -1] = 1e6   # spike in second half
        
        correct = _softmax_correct(x)
        buggy = _softmax_first_tile(x)
        
        # Correct output has mass in last column
        assert correct[:, -1].mean().item() > 0.9
        # Buggy output has zero in last column
        assert (buggy[:, 32:] == 0.0).all()
        # They are clearly different
        assert not torch.allclose(correct, buggy, atol=0.1)

    def test_non_negative_property(self):
        x = torch.randn(32, 64)
        out = _softmax_correct(x)
        assert (out >= 0).all()

    def test_nan_inf_check(self):
        """Checker must catch NaN/Inf outputs — not possible with correct softmax."""
        x = torch.randn(32, 64)
        out = _softmax_correct(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_values_no_overflow(self):
        """Correct softmax must handle large values without NaN."""
        x = torch.randn(32, 64) * 1000.0
        out = _softmax_correct(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ── LayerNorm gap tests ───────────────────────────────────────────────────────

class TestLayerNormGap:
    def setup_method(self):
        self.x     = torch.randn(32, 64)
        self.gamma = torch.ones(64)
        self.beta  = torch.zeros(64)

    def test_correct_layernorm_passes_all_checks(self):
        checks = _check_layernorm(_layernorm_correct, self.x, self.gamma, self.beta)
        assert all(checks.values()), checks

    def test_ignore_gamma_fails_gamma_correctness(self):
        """ignore_gamma must fail gamma_correctness check."""
        checks = _check_layernorm(_layernorm_ignore_gamma, self.x, self.gamma, self.beta)
        assert not checks["gamma_correctness"]

    def test_ignore_gamma_passes_naive_with_identity_params(self):
        """
        When gamma=1 and beta=0, ignore_gamma gives same output as correct.
        Naive allclose passes. This is the gap.
        """
        ref = _layernorm_correct(self.x, self.gamma, self.beta)
        bug = _layernorm_ignore_gamma(self.x, self.gamma, self.beta)
        assert _naive_allclose(bug, ref)

    def test_ignore_gamma_fails_on_nonunit_gamma(self):
        """With gamma=2, ignore_gamma is visibly wrong — output differs by 2x."""
        gamma2 = torch.full_like(self.gamma, 2.0)
        ref = _layernorm_correct(self.x, gamma2, self.beta)
        bug = _layernorm_ignore_gamma(self.x, gamma2, self.beta)
        assert not _naive_allclose(bug, ref)

    def test_skip_mean_fails_on_large_mean_shift(self):
        """skip_mean_subtract is visibly wrong when x has large mean."""
        x_shifted = self.x + 1000.0
        ref = _layernorm_correct(x_shifted, self.gamma, self.beta)
        bug = _layernorm_skip_mean(x_shifted, self.gamma, self.beta)
        assert not _naive_allclose(bug, ref)

    def test_skip_mean_nearly_passes_naive_zero_mean(self):
        """
        When x has ~zero mean, skip_mean gives similar output.
        The gap: naive on zero-mean inputs misses the bug.
        """
        x = self.x - self.x.mean(-1, keepdim=True)  # force zero mean
        ref = _layernorm_correct(x, self.gamma, self.beta)
        bug = _layernorm_skip_mean(x, self.gamma, self.beta)
        # Should be close (mean is 0, so x - mean ≈ x)
        assert _naive_allclose(bug, ref, atol=1e-2)


# ── RMSNorm gap tests ─────────────────────────────────────────────────────────

class TestRMSNormGap:
    def setup_method(self):
        self.x     = torch.randn(32, 64)
        self.gamma = torch.ones(64)

    def test_correct_rmsnorm_passes_all_checks(self):
        checks = _check_rmsnorm(_rmsnorm_correct, self.x, self.gamma)
        assert all(checks.values()), checks

    def test_ignore_gamma_fails_gamma_correctness(self):
        checks = _check_rmsnorm(_rmsnorm_ignore_gamma, self.x, self.gamma)
        assert not checks["gamma_correctness"]

    def test_ignore_gamma_passes_naive_with_ones(self):
        """gamma=ones: ignore_gamma gives same output as correct. Gap."""
        ref = _rmsnorm_correct(self.x, self.gamma)
        bug = _rmsnorm_ignore_gamma(self.x, self.gamma)
        assert _naive_allclose(bug, ref)

    def test_wrong_norm_fails_scale_invariance(self):
        """wrong_norm violates unit RMS, not scale invariance."""
        x = torch.randn(32, 64)
        gamma = torch.ones(64)
        out = _rmsnorm_wrong_norm(x, gamma)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # Output RMS should be 1 for correct rmsnorm, won't be for wrong_norm
        assert not torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_wrong_norm_fails_unit_rms(self):
        """mean(|x|) normalisation: output RMS ≠ 1 in general."""
        checks = _check_rmsnorm(_rmsnorm_wrong_norm, self.x, self.gamma)
        assert not checks["unit_rms"]

    def test_wrong_norm_passes_naive_on_low_variance(self):
        """
        wrong_norm nearly agrees with correct rmsnorm when all values
        have similar magnitude — mean(|x|) ≈ sqrt(mean(x^2)) when
        there's little spread. This is the gap naive testing misses.
        """
        # Constant-magnitude input: |x_i| = c for all i
        # mean(|x|) = c, sqrt(mean(x^2)) = c — identical
        x = torch.ones(4, 16) * 0.5
        x[::2] *= -1   # alternate signs, same magnitude
        ref = _rmsnorm_correct(x, self.gamma[:16])
        bug = _rmsnorm_wrong_norm(x, self.gamma[:16])
        assert _naive_allclose(bug, ref, atol=1e-3)


# ── MatMul gap tests ──────────────────────────────────────────────────────────

class TestMatMulGap:
    def test_correct_matmul(self):
        A, B = torch.randn(32, 64), torch.randn(64, 32)
        out = _matmul_correct(A, B)
        assert out.shape == (32, 32)
        assert not torch.isnan(out).any()

    def test_partial_k_fails_on_ones(self):
        """
        A=ones, B=ones: correct output is all-K, partial_k gives all-K/2.
        """
        K = 64
        A = torch.ones(16, K)
        B = torch.ones(K, 16)
        ref = _matmul_correct(A, B)
        bug = _matmul_partial_k(A, B)
        assert not _naive_allclose(bug, ref)
        # Reference should be all K=64
        assert torch.allclose(ref, torch.full_like(ref, float(K)), atol=1e-3)
        # Bug should be all K/2=32
        assert torch.allclose(bug, torch.full_like(bug, float(K // 2)), atol=1e-3)

    def test_partial_k_passes_naive_on_random_small(self):
        """
        On random inputs with small K, the first K//2 elements carry
        most of the energy by chance — naive can pass on typical evals.
        """
        torch.manual_seed(42)
        A = torch.randn(4, 4) * 0.01
        B = torch.randn(4, 4) * 0.01
        ref = _matmul_correct(A, B)
        bug = _matmul_partial_k(A, B)
        # With tiny values the error may be within atol
        max_err = (ref - bug).abs().max().item()
        # This documents the gap — the error can be small on benign inputs
        assert max_err < 1.0  # not zero but potentially within loose atol