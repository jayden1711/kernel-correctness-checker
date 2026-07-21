"""
tests/conftest.py

Shared fixtures and pytest configuration for the KernelChecker test suite.

Markers:
    gpu        — requires a real CUDA device; skipped in CPU-only CI
    llm        — makes a live LLM API call; skipped without CHECKER_LLM_TESTS=1
    slow       — integration tests > 10s; skipped without CHECKER_SLOW_TESTS=1
    checker    — tests of the three-layer KernelChecker pipeline
    adversarial — tests of the adversarial search system

Run subsets:
    pytest tests/ -m "not gpu and not llm"   # pure unit tests, no hardware
    pytest tests/ -m checker                  # checker pipeline only
    pytest tests/ -m adversarial              # search system only
    CHECKER_SLOW_TESTS=1 pytest tests/ -m slow
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Callable

import pytest
import torch

# Ensure project root on sys.path
CHECKER_ROOT = os.environ.get("CHECKER_ROOT", str(Path(__file__).parent.parent))
if CHECKER_ROOT not in sys.path:
    sys.path.insert(0, CHECKER_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA device")
    config.addinivalue_line("markers", "llm: requires live LLM API key")
    config.addinivalue_line("markers", "slow: slow integration tests (>10s)")
    config.addinivalue_line("markers", "checker: three-layer checker pipeline tests")
    config.addinivalue_line("markers", "adversarial: adversarial search system tests")


# ── Skip conditions ───────────────────────────────────────────────────────────

def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="requires CUDA device (run with a GPU)")
    skip_llm = pytest.mark.skip(reason="set CHECKER_LLM_TESTS=1 to run live LLM tests")
    skip_slow = pytest.mark.skip(reason="set CHECKER_SLOW_TESTS=1 to run slow tests")

    has_cuda = torch.cuda.is_available()
    run_llm  = os.environ.get("CHECKER_LLM_TESTS", "0") == "1"
    run_slow = os.environ.get("CHECKER_SLOW_TESTS", "0") == "1"

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "llm" in item.keywords and not run_llm:
            item.add_marker(skip_llm)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


# ── Core fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def checker_root() -> Path:
    return Path(CHECKER_ROOT)


@pytest.fixture(scope="session")
def tritonbench_available(checker_root) -> bool:
    return (checker_root / "TritonBench" / "reference" / "softmax.py").exists()


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    d = tmp_path / "adversarial_results"
    d.mkdir()
    return d


# ── Tensor helpers ────────────────────────────────────────────────────────────

@pytest.fixture
def make_softmax_input():
    def _make(n_rows=64, n_cols=128, device="cpu", dtype=torch.float32):
        return torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    return _make


@pytest.fixture
def make_layernorm_inputs():
    def _make(n_rows=64, n_cols=128, device="cpu", dtype=torch.float32):
        x     = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        gamma = torch.ones(n_cols, device=device, dtype=dtype)
        beta  = torch.zeros(n_cols, device=device, dtype=dtype)
        return x, gamma, beta
    return _make


@pytest.fixture
def make_rmsnorm_inputs():
    def _make(n_rows=64, n_cols=128, device="cpu", dtype=torch.float32):
        x     = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        gamma = torch.ones(n_cols, device=device, dtype=dtype)
        return x, gamma
    return _make


@pytest.fixture
def make_matmul_inputs():
    def _make(M=64, K=64, N=64, device="cpu", dtype=torch.float32):
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        return A, B
    return _make


# ── Reference PyTorch implementations (no Triton, used in non-GPU tests) ──────

@pytest.fixture(scope="session")
def pt_softmax():
    """Pure PyTorch row-wise softmax."""
    def _fn(x):
        return torch.softmax(x, dim=-1)
    return _fn


@pytest.fixture(scope="session")
def pt_layernorm():
    """Pure PyTorch layer norm."""
    def _fn(x, gamma, beta, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + eps).sqrt() * gamma + beta
    return _fn


@pytest.fixture(scope="session")
def pt_rmsnorm():
    """Pure PyTorch RMS norm."""
    def _fn(x, gamma, eps=1e-5):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return x / rms * gamma
    return _fn


@pytest.fixture(scope="session")
def pt_matmul():
    def _fn(A, B):
        return A @ B
    return _fn


# ── Buggy reference implementations (for checker gap tests) ───────────────────

@pytest.fixture(scope="session")
def buggy_softmax_first_tile():
    """Softmax that only processes the first half of columns."""
    def _fn(x):
        half = x.shape[-1] // 2
        out = torch.zeros_like(x)
        out[:, :half] = torch.softmax(x[:, :half], dim=-1)
        return out
    return _fn


@pytest.fixture(scope="session")
def buggy_layernorm_ignore_gamma():
    """LayerNorm that ignores gamma and beta."""
    def _fn(x, gamma, beta, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + eps).sqrt()   # no affine transform
    return _fn


@pytest.fixture(scope="session")
def buggy_rmsnorm_wrong_norm():
    """RMSNorm that uses mean(|x|) instead of sqrt(mean(x^2))."""
    def _fn(x, gamma, eps=1e-5):
        norm = x.abs().mean(dim=-1, keepdim=True) + eps
        return x / norm * gamma
    return _fn


@pytest.fixture(scope="session")
def buggy_matmul_partial_k():
    """Matmul that only accumulates first K//2 elements."""
    def _fn(A, B):
        half = A.shape[-1] // 2
        return A[:, :half] @ B[:half, :]
    return _fn