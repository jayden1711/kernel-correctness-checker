"""
Data-driven registry mapping every TritonBench (reference, cheating-mutant)
pair to the harness's corpus contract, plus the extra torch-native handles
checker_adapter.py needs to run the REAL three-layer checker (not just a
numpy-facing allclose).

REQUIRES A REAL CUDA GPU + `pip install torch triton`. These are actual
`@triton.jit` kernels -- there is no CPU fallback. This file was built and
statically verified (function names, host-wrapper signatures, KernelSpec
base classes) from the source in TritonBench/ and verification/specs/ on a
machine with no GPU and no triton installed, so none of it has been
runtime-tested. Run `python corpus_contract.py my_corpus.py` on Colab
first and fix whatever it flags before trusting results.py.

One known duplicate, `TritonBench/cheating/max_reduction.py/` (stray
directory with a dot in the name, byte-identical to
`TritonBench/cheating/max_reduction/wrong_padding.py` bar a trailing
newline), was always excluded here and was deleted from disk 2026-08-28.

29 operators, 41 mutant files, one per registry entry.
"""

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Per-family numpy input generators + numpy<->torch conversion.
#
# "family" determines the positional-argument shape every op in that family
# shares (verified against each operator's real host-wrapper signature and
# its KernelSpec subclass in verification/specs/base_spec.py).
# ---------------------------------------------------------------------------

def _mk_single(rng):
    return (rng.normal(size=(64, 128)).astype(np.float32),)


def _to_torch_single(args):
    (x,) = args
    return torch.from_numpy(x).to(_DEVICE)


def _mk_triple(shape_x, shape_w):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        w1 = rng.normal(size=shape_w).astype(np.float32)
        w2 = rng.normal(size=shape_w).astype(np.float32)
        return (x, w1, w2)
    return _mk


def _to_torch_triple(args):
    x, w1, w2 = args
    return tuple(torch.from_numpy(a).to(_DEVICE) for a in (x, w1, w2))


def _mk_rmsnorm(rng):
    x = rng.normal(size=(64, 128)).astype(np.float32)
    gamma = rng.normal(size=(128,)).astype(np.float32)
    return (x, gamma)


def _to_torch_pair(args):
    a, b = args
    return tuple(torch.from_numpy(v).to(_DEVICE) for v in (a, b))


def _mk_matmul(rng):
    A = rng.normal(size=(32, 16)).astype(np.float32)
    B = rng.normal(size=(16, 32)).astype(np.float32)
    return (A, B)


def _mk_attention(rng):
    N, D = 64, 32
    Q = rng.normal(size=(N, D)).astype(np.float32)
    K = rng.normal(size=(N, D)).astype(np.float32)
    V = rng.normal(size=(N, D)).astype(np.float32)
    return (Q, K, V)


def _to_torch_attention(args):
    return tuple(torch.from_numpy(a).to(_DEVICE) for a in args)


def _mk_groupnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    num_groups = 2
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, num_groups, weight, bias)


def _to_torch_groupnorm(args):
    x, num_groups, weight, bias = args
    return (torch.from_numpy(x).to(_DEVICE), num_groups,
            torch.from_numpy(weight).to(_DEVICE), torch.from_numpy(bias).to(_DEVICE))


def _mk_batchnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    running_mean = rng.normal(size=(C,)).astype(np.float32)
    running_var = rng.uniform(0.5, 2.0, size=(C,)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, running_mean, running_var, weight, bias)


def _to_torch_batchnorm(args):
    return tuple(torch.from_numpy(a).to(_DEVICE) for a in args)


def _mk_cross_entropy(rng):
    n_rows, n_cols = 64, 32
    logits = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    targets = rng.integers(0, n_cols, size=(n_rows,)).astype(np.int64)
    return (logits, targets)


def _to_torch_cross_entropy(args):
    logits, targets = args
    return (torch.from_numpy(logits).to(_DEVICE), torch.from_numpy(targets).to(_DEVICE))


def _mk_pool(shape_x, kernel_size, stride, padding):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        return (x, kernel_size, stride, padding)
    return _mk


def _to_torch_pool(args):
    x, k, s, p = args
    return (torch.from_numpy(x).to(_DEVICE), k, s, p)


FAMILIES = {
    "single": (_mk_single, _to_torch_single),
    "layernorm": (_mk_triple((64, 128), (128,)), _to_torch_triple),
    "instancenorm": (_mk_triple((2, 4, 4, 4), (4,)), _to_torch_triple),
    "rmsnorm": (_mk_rmsnorm, _to_torch_pair),
    "matmul": (_mk_matmul, _to_torch_pair),
    "attention": (_mk_attention, _to_torch_attention),
    "groupnorm": (_mk_groupnorm, _to_torch_groupnorm),
    "batchnorm": (_mk_batchnorm, _to_torch_batchnorm),
    "cross_entropy": (_mk_cross_entropy, _to_torch_cross_entropy),
    "pool1d": (_mk_pool((2, 3, 32), 4, 4, 0), _to_torch_pool),
    "pool2d": (_mk_pool((2, 3, 16, 16), 4, 4, 0), _to_torch_pool),
    "pool3d": (_mk_pool((2, 3, 8, 8, 8), 2, 2, 0), _to_torch_pool),
}


# ---------------------------------------------------------------------------
# Operator table: (spec_key, ref_module_file, cheat_dir, family, [mutant filenames])
#
# spec_key    -- matches verification/specs/<spec_key>.py (get_spec()) and
#                is used as the harness's `op` label.
# ref_module_file -- matches TritonBench/reference/<ref_module_file>.py
# cheat_dir   -- matches TritonBench/cheating/<cheat_dir>/
#
# ref_func / cheat_func are NOT listed separately: verified that in every
# case the host-wrapper function name equals spec_key exactly (both in the
# reference module and in every cheating mutant module).
# ---------------------------------------------------------------------------

OPS = [
    ("argmax", "argmax", "argmax", "single", ["tiebreak"]),
    ("argmin", "argmin", "argmin", "single", ["tiebreak"]),
    ("avg_pool1d", "avg_pool1d", "avg_pool1d", "pool1d", ["wrong_divisor"]),
    ("avg_pool2d", "avg_pool2d", "avg_pool2d", "pool2d", ["wrong_divisor"]),
    ("avg_pool3d", "avg_pool3d", "avg_pool3d", "pool3d", ["wrong_divisor"]),
    ("batchnorm", "batchnorm", "batchnorm", "batchnorm", ["wrong_running_stats_broadcast"]),
    ("causal_flash_attention", "causal_flash_attention", "causal_flash_attention",
     "attention", ["wrong_causal_mask"]),
    ("cross_entropy", "cross_entropy", "cross_entropy", "cross_entropy",
     ["missing_max_subtraction"]),
    ("flash_attention", "flash_attention", "flash_attention", "attention",
     ["approx_denom", "drop_last_tile", "skip_rescaling", "wrong_mask"]),
    ("frobenius_norm", "frobenius_norm", "frobenius_norm", "single", ["wrong_norm"]),
    ("gelu", "gelu", "gelu", "single", ["sigmoid_approx"]),
    ("groupnorm", "groupnorm", "groupnorm", "groupnorm", ["ignore_affine"]),
    ("instancenorm", "instancenorm", "instancenorm", "instancenorm", ["skip_eps"]),
    ("l1norm", "l1norm", "l1norm", "single", ["partial_reduction"]),
    ("l2norm", "l2norm", "l2norm", "single", ["wrong_norm"]),
    ("layernorm", "layernorm", "layer_norm", "layernorm",
     ["ignore_gamma_beta", "skip_mean_subtract", "wrong_variance_estimate"]),
    ("log_softmax", "log_softmax", "log_softmax", "single", ["skip_max_subtraction"]),
    ("matmul", "mat_mult", "matmult", "matmul",
     ["partial_k_reduct", "skip_boundary_tiles", "swapped_strides", "wrong_dtype"]),
    ("max_pool1d", "max_pool1d", "max_pool1d", "pool1d", ["wrong_padding"]),
    ("max_pool2d", "max_pool2d", "max_pool2d", "pool2d", ["wrong_padding"]),
    ("max_pool3d", "max_pool3d", "max_pool3d", "pool3d", ["wrong_padding"]),
    ("max_reduction", "max_reduction", "max_reduction", "single", ["wrong_padding"]),
    ("mean_reduction", "mean_reduction", "mean_reduction", "single", ["partial_reduction"]),
    ("min_reduction", "min_reduction", "min_reduction", "single", ["wrong_padding"]),
    ("rmsnorm", "rmsnorm", "rmsnorm", "rmsnorm",
     ["ignore_gamma", "partial_reduction", "wrong_norm"]),
    ("scaled_dot_product_attention", "scaled_dot_product_attention",
     "scaled_dot_product_attention", "attention", ["wrong_mask"]),
    ("softmax", "softmax", "softmax", "single", ["first_tile", "wrong_reduction"]),
    ("sum_reduction", "sum_reduction", "sum_reduction", "single", ["partial_reduction"]),
    ("swish", "swish", "swish", "single", ["linear_sigmoid_approx"]),
]


def _find_raw_kernel(module, wrapper_name):
    """Return the @triton.jit kernel object in `module`, distinguished from
    the host wrapper by name (every kernel object's name contains
    'kernel'; the host wrapper's name never does -- it equals spec_key).
    Purely decorative for the current checker (see tile_coverage.py's
    check_all_tiles_visited: raw_kernel param is unused, kept for API
    compatibility only), so an approximate match is fine."""
    for name, obj in vars(module).items():
        if name != wrapper_name and "kernel" in name.lower():
            return obj
    return None


def build_corpus():
    corpus = []
    for spec_key, ref_file, cheat_dir, family, mutant_names in OPS:
        mk_fn, to_torch = FAMILIES[family]

        ref_module = importlib.import_module(f"TritonBench.reference.{ref_file}")
        ref_torch_fn = getattr(ref_module, spec_key)
        raw_kernel_ref = _find_raw_kernel(ref_module, spec_key)

        spec_module = importlib.import_module(f"verification.specs.{spec_key}")
        spec = spec_module.get_spec()

        for mutant_name in mutant_names:
            cheat_module = importlib.import_module(
                f"TritonBench.cheating.{cheat_dir}.{mutant_name}")
            mutant_torch_fn = getattr(cheat_module, spec_key)
            raw_kernel_mutant = _find_raw_kernel(cheat_module, spec_key)

            def _make_np_fn(torch_fn, to_torch=to_torch):
                def _fn(*np_args):
                    torch_inputs = to_torch(np_args)
                    if isinstance(torch_inputs, tuple):
                        out = torch_fn(*torch_inputs)
                    else:
                        out = torch_fn(torch_inputs)
                    return out.detach().cpu().numpy()
                return _fn

            corpus.append(dict(
                op=spec_key,
                mutant_name=mutant_name,
                ref_fn=_make_np_fn(ref_torch_fn),
                mutant_fn=_make_np_fn(mutant_torch_fn),
                input_fn=mk_fn,
                # extra keys, ignored by corpus_contract.py, used by
                # checker_adapter.py to run the real torch-native checker:
                family=family,
                to_torch=to_torch,
                spec=spec,
                torch_ref_fn=ref_torch_fn,
                torch_mutant_fn=mutant_torch_fn,
                raw_kernel_ref=raw_kernel_ref,
                raw_kernel_mutant=raw_kernel_mutant,
            ))

    return corpus
