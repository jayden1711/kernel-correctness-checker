"""
run_checker.py — run the KernelChecker against every cheating kernel
in TritonBench and print a summary table.

Run from project root:
    python run_checker.py

In Colab:
    !python run_checker.py
"""

import sys
import torch

# Add project root to path if needed
sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# Imports — specs
# ---------------------------------------------------------------------------
from verification.checker import KernelChecker
from verification.specs.softmax import get_spec as softmax_spec
from verification.specs.layernorm import get_spec as layernorm_spec
from verification.specs.matmul import get_spec as matmul_spec
from verification.specs.flash_attention import get_spec as flash_attention_spec
from verification.specs.rmsnorm import get_spec as rmsnorm_get_spec

from verification.specs.log_softmax import get_spec as log_softmax_spec
from verification.specs.swish import get_spec as swish_spec
from verification.specs.gelu import get_spec as gelu_spec
from verification.specs.sum_reduction import get_spec as sum_reduction_spec
from verification.specs.mean_reduction import get_spec as mean_reduction_spec
from verification.specs.max_reduction import get_spec as max_reduction_spec
from verification.specs.min_reduction import get_spec as min_reduction_spec
from verification.specs.l1norm import get_spec as l1norm_spec
from verification.specs.l2norm import get_spec as l2norm_spec
from verification.specs.frobenius_norm import get_spec as frobenius_norm_spec
from verification.specs.argmax import get_spec as argmax_spec
from verification.specs.argmin import get_spec as argmin_spec
from verification.specs.instancenorm import get_spec as instancenorm_spec
from verification.specs.groupnorm import get_spec as groupnorm_spec
from verification.specs.batchnorm import get_spec as batchnorm_spec
from verification.specs.cross_entropy import get_spec as cross_entropy_spec
from verification.specs.max_pool1d import get_spec as max_pool1d_spec
from verification.specs.max_pool2d import get_spec as max_pool2d_spec
from verification.specs.max_pool3d import get_spec as max_pool3d_spec
from verification.specs.avg_pool1d import get_spec as avg_pool1d_spec
from verification.specs.avg_pool2d import get_spec as avg_pool2d_spec
from verification.specs.avg_pool3d import get_spec as avg_pool3d_spec
from verification.specs.scaled_dot_product_attention import get_spec as sdpa_spec
from verification.specs.causal_flash_attention import get_spec as causal_flash_attention_spec

# ---------------------------------------------------------------------------
# Imports — reference kernels (original 5)
# ---------------------------------------------------------------------------
from TritonBench.reference.softmax import softmax as ref_softmax, softmax_kernel as ref_softmax_kernel
from TritonBench.reference.layernorm import layernorm as ref_layernorm, layernorm_kernel as ref_layernorm_kernel
from TritonBench.reference.mat_mult import matmul as ref_matmul, matmul_kernel as ref_matmul_kernel
from TritonBench.reference.flash_attention import flash_attention as ref_flash_attention, flash_attention_kernel as ref_flash_attention_kernel
from TritonBench.reference.rmsnorm import rmsnorm as ref_rmsnorm, rmsnorm_kernel as ref_rmsnorm_kernel

from TritonBench.cheating.softmax.first_tile import softmax as cheat_softmax_first_tile, softmax_kernel_cheat_first_tile
from TritonBench.cheating.softmax.wrong_reduction import softmax as cheat_softmax_wrong_reduction

from TritonBench.cheating.layer_norm.ignore_gamma_beta import layernorm as cheat_ln_ignore_gamma_beta, layernorm_kernel as cheat_ln_ignore_gamma_beta_kernel
from TritonBench.cheating.layer_norm.skip_mean_subtract import layernorm as cheat_ln_skip_mean, layernorm_kernel as cheat_ln_skip_mean_kernel
from TritonBench.cheating.layer_norm.wrong_variance_estimate import layernorm as cheat_ln_wrong_var, layernorm_kernel as cheat_ln_wrong_var_kernel

from TritonBench.cheating.matmult.partial_k_reduct import matmul as cheat_mm_partial_k, matmul_kernel as cheat_mm_partial_k_kernel
from TritonBench.cheating.matmult.skip_boundary_tiles import matmul as cheat_mm_skip_boundary, matmul_kernel as cheat_mm_skip_boundary_kernel
from TritonBench.cheating.matmult.swapped_strides import matmul as cheat_mm_swapped, matmul_kernel as cheat_mm_swapped_kernel
from TritonBench.cheating.matmult.wrong_dtype import matmul as cheat_mm_wrong_dtype, matmul_kernel as cheat_mm_wrong_dtype_kernel

from TritonBench.cheating.flash_attention.approx_denom import flash_attention as cheat_fa_approx, flash_attention_kernel_cheat_approx_denominator as cheat_fa_approx_kernel
from TritonBench.cheating.flash_attention.drop_last_tile import flash_attention as cheat_fa_drop, flash_attention_kernel_cheat_drop_last_tile as cheat_fa_drop_kernel
from TritonBench.cheating.flash_attention.skip_rescaling import flash_attention as cheat_fa_skip, flash_attention_kernel_cheat_skip_rescale as cheat_fa_skip_kernel
from TritonBench.cheating.flash_attention.wrong_mask import flash_attention as cheat_fa_mask, flash_attention_kernel_cheat_wrong_mask as cheat_fa_mask_kernel

from TritonBench.cheating.rmsnorm.ignore_gamma import rmsnorm as rmsnorm_ignore_gamma, rmsnorm_kernel as rmsnorm_ignore_gamma_kernel
from TritonBench.cheating.rmsnorm.wrong_norm import rmsnorm as rmsnorm_wrong_norm, rmsnorm_kernel as rmsnorm_wrong_norm_kernel
from TritonBench.cheating.rmsnorm.partial_reduction import rmsnorm as rmsnorm_partial_reduction, rmsnorm_kernel as rmsnorm_partial_reduction_kernel

# ---------------------------------------------------------------------------
# Imports — reference kernels (24 new)
# ---------------------------------------------------------------------------
from TritonBench.reference.log_softmax import log_softmax as ref_log_softmax, log_softmax_kernel as ref_log_softmax_kernel
from TritonBench.reference.swish import swish as ref_swish, swish_kernel as ref_swish_kernel
from TritonBench.reference.gelu import gelu as ref_gelu, gelu_kernel as ref_gelu_kernel
from TritonBench.reference.sum_reduction import sum_reduction as ref_sum_reduction, sum_reduce_kernel as ref_sum_reduction_kernel
from TritonBench.reference.mean_reduction import mean_reduction as ref_mean_reduction, mean_reduce_kernel as ref_mean_reduction_kernel
from TritonBench.reference.max_reduction import max_reduction as ref_max_reduction, max_reduce_kernel as ref_max_reduction_kernel
from TritonBench.reference.min_reduction import min_reduction as ref_min_reduction, min_reduce_kernel as ref_min_reduction_kernel
from TritonBench.reference.l1norm import l1norm as ref_l1norm, l1norm_kernel as ref_l1norm_kernel
from TritonBench.reference.l2norm import l2norm as ref_l2norm, l2norm_kernel as ref_l2norm_kernel
from TritonBench.reference.frobenius_norm import frobenius_norm as ref_frobenius_norm
from TritonBench.reference.argmax import argmax as ref_argmax, argmax_kernel as ref_argmax_kernel
from TritonBench.reference.argmin import argmin as ref_argmin, argmin_kernel as ref_argmin_kernel
from TritonBench.reference.instancenorm import instancenorm as ref_instancenorm, instancenorm_kernel as ref_instancenorm_kernel
from TritonBench.reference.groupnorm import groupnorm as ref_groupnorm, groupnorm_kernel as ref_groupnorm_kernel
from TritonBench.reference.batchnorm import batchnorm as ref_batchnorm, batchnorm_kernel as ref_batchnorm_kernel
from TritonBench.reference.cross_entropy import cross_entropy as ref_cross_entropy, cross_entropy_kernel as ref_cross_entropy_kernel
from TritonBench.reference.max_pool1d import max_pool1d as ref_max_pool1d, maxpool1d_kernel as ref_max_pool1d_kernel
from TritonBench.reference.max_pool2d import max_pool2d as ref_max_pool2d, maxpool2d_kernel as ref_max_pool2d_kernel
from TritonBench.reference.max_pool3d import max_pool3d as ref_max_pool3d, maxpool3d_kernel as ref_max_pool3d_kernel
from TritonBench.reference.avg_pool1d import avg_pool1d as ref_avg_pool1d, avgpool1d_kernel as ref_avg_pool1d_kernel
from TritonBench.reference.avg_pool2d import avg_pool2d as ref_avg_pool2d, avgpool2d_kernel as ref_avg_pool2d_kernel
from TritonBench.reference.avg_pool3d import avg_pool3d as ref_avg_pool3d, avgpool3d_kernel as ref_avg_pool3d_kernel
from TritonBench.reference.scaled_dot_product_attention import scaled_dot_product_attention as ref_sdpa, sdpa_kernel as ref_sdpa_kernel
from TritonBench.reference.causal_flash_attention import causal_flash_attention as ref_causal_fa, causal_flash_attention_kernel as ref_causal_fa_kernel

from TritonBench.cheating.log_softmax.skip_max_subtraction import log_softmax as cheat_log_softmax, log_softmax_kernel_cheat_skip_max as cheat_log_softmax_kernel
from TritonBench.cheating.swish.linear_sigmoid_approx import swish as cheat_swish, swish_kernel_cheat_linear_sigmoid as cheat_swish_kernel
from TritonBench.cheating.gelu.sigmoid_approx import gelu as cheat_gelu, gelu_kernel_cheat_sigmoid_approx as cheat_gelu_kernel
from TritonBench.cheating.sum_reduction.partial_reduction import sum_reduction as cheat_sum_reduction, sum_reduce_kernel_cheat_partial as cheat_sum_reduction_kernel
from TritonBench.cheating.mean_reduction.partial_reduction import mean_reduction as cheat_mean_reduction, mean_reduce_kernel_cheat_partial as cheat_mean_reduction_kernel
from TritonBench.cheating.max_reduction.wrong_padding import max_reduction as cheat_max_reduction, max_reduce_kernel_cheat_wrong_padding as cheat_max_reduction_kernel
from TritonBench.cheating.min_reduction.wrong_padding import min_reduction as cheat_min_reduction, min_reduce_kernel_cheat_wrong_padding as cheat_min_reduction_kernel
from TritonBench.cheating.l1norm.partial_reduction import l1norm as cheat_l1norm, l1norm_kernel_cheat_partial as cheat_l1norm_kernel
from TritonBench.cheating.l2norm.wrong_norm import l2norm as cheat_l2norm, l2norm_kernel_cheat_wrong_norm as cheat_l2norm_kernel
from TritonBench.cheating.frobenius_norm.wrong_norm import frobenius_norm as cheat_frobenius_norm
from TritonBench.cheating.argmax.tiebreak import argmax as cheat_argmax, argmax_kernel_cheat_tiebreak as cheat_argmax_kernel
from TritonBench.cheating.argmin.tiebreak import argmin as cheat_argmin, argmin_kernel_cheat_tiebreak as cheat_argmin_kernel
from TritonBench.cheating.instancenorm.skip_eps import instancenorm as cheat_instancenorm, instancenorm_kernel_cheat_skip_eps as cheat_instancenorm_kernel
from TritonBench.cheating.groupnorm.ignore_affine import groupnorm as cheat_groupnorm, groupnorm_kernel_cheat_ignore_affine as cheat_groupnorm_kernel
from TritonBench.cheating.batchnorm.wrong_running_stats_broadcast import batchnorm as cheat_batchnorm, batchnorm_kernel_cheat_wrong_broadcast as cheat_batchnorm_kernel
from TritonBench.cheating.cross_entropy.missing_max_subtraction import cross_entropy as cheat_cross_entropy, cross_entropy_kernel_cheat_skip_max as cheat_cross_entropy_kernel
from TritonBench.cheating.max_pool1d.wrong_padding import max_pool1d as cheat_max_pool1d, maxpool1d_kernel_cheat_wrong_padding as cheat_max_pool1d_kernel
from TritonBench.cheating.max_pool2d.wrong_padding import max_pool2d as cheat_max_pool2d, maxpool2d_kernel_cheat_wrong_padding as cheat_max_pool2d_kernel
from TritonBench.cheating.max_pool3d.wrong_padding import max_pool3d as cheat_max_pool3d, maxpool3d_kernel_cheat_wrong_padding as cheat_max_pool3d_kernel
from TritonBench.cheating.avg_pool1d.wrong_divisor import avg_pool1d as cheat_avg_pool1d, avgpool1d_kernel_cheat_wrong_divisor as cheat_avg_pool1d_kernel
from TritonBench.cheating.avg_pool2d.wrong_divisor import avg_pool2d as cheat_avg_pool2d, avgpool2d_kernel_cheat_wrong_divisor as cheat_avg_pool2d_kernel
from TritonBench.cheating.avg_pool3d.wrong_divisor import avg_pool3d as cheat_avg_pool3d, avgpool3d_kernel_cheat_wrong_divisor as cheat_avg_pool3d_kernel
from TritonBench.cheating.scaled_dot_product_attention.wrong_mask import scaled_dot_product_attention as cheat_sdpa, sdpa_kernel_cheat_wrong_mask as cheat_sdpa_kernel
from TritonBench.cheating.causal_flash_attention.wrong_causal_mask import causal_flash_attention as cheat_causal_fa, causal_flash_attention_kernel_cheat_wrong_boundary as cheat_causal_fa_kernel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test-input builders — original 5
# ---------------------------------------------------------------------------

def make_softmax_inputs():
    return torch.randn(512, 2048, device=DEVICE)

def make_layernorm_inputs():
    n_rows, n_cols = 512, 512
    x     = torch.randn(n_rows, n_cols, device=DEVICE)
    gamma = torch.ones(n_cols, device=DEVICE)
    beta  = torch.zeros(n_cols, device=DEVICE)
    return x, gamma, beta

def make_matmul_inputs():
    return torch.randn(256, 256, device=DEVICE), torch.randn(256, 256, device=DEVICE)

def make_flash_attention_inputs():
    N, D = 128, 64
    return (
        torch.randn(N, D, device=DEVICE),
        torch.randn(N, D, device=DEVICE),
        torch.randn(N, D, device=DEVICE),
    )

def make_rmsnorm_inputs():
    n_rows, n_cols = 512, 512
    x     = torch.randn(n_rows, n_cols, device=DEVICE)
    # Nonzero, non-uniform gamma -- torch.ones() would make ignore_gamma
    # numerically indistinguishable from the reference (scaling by 1 is a no-op),
    # guaranteeing a missed mutant regardless of checker quality.
    gamma = torch.randn(n_cols, device=DEVICE).abs() + 0.1
    return x, gamma


# ---------------------------------------------------------------------------
# Test-input builders — 24 new operators
# Reuses the EXACT trigger conditions already validated in
# scripts/test_stage2_mutant_catchability.py, not new guesses.
# ---------------------------------------------------------------------------

def make_log_softmax_inputs():
    return torch.full((8, 64), 200.0, device=DEVICE) + torch.randn(8, 64, device=DEVICE)

def make_swish_inputs():
    return torch.tensor([10.0, -10.0, 20.0, -20.0], device=DEVICE)

def make_gelu_inputs():
    return torch.tensor([-2.27, -2.0, -1.7, -1.5, -2.3, -2.1], device=DEVICE)

def make_sum_mean_reduction_inputs():
    x = torch.zeros(4, 64, device=DEVICE)
    x[:, 32:] = torch.randn(4, 32, device=DEVICE) * 10
    return x

def make_max_reduction_inputs():
    return -torch.rand(4, 100, device=DEVICE) - 0.1

def make_min_reduction_inputs():
    return torch.rand(4, 100, device=DEVICE) + 0.1

def make_l1norm_inputs():
    x = torch.zeros(4, 64, device=DEVICE)
    x[:, 32:] = torch.randn(4, 32, device=DEVICE) * 10
    return x

def make_l2norm_inputs():
    return torch.randn(4, 64, device=DEVICE) * torch.tensor([0.01, 100.0] * 32, device=DEVICE)

def make_frobenius_norm_inputs():
    x = torch.randn(20, 20, device=DEVICE)
    x[0, 0] = 500.0
    return x

def make_argmax_inputs():
    x = torch.zeros(4, 16, device=DEVICE)
    x[:, 2] = 1.0
    x[:, 11] = 1.0
    return x

def make_argmin_inputs():
    x = torch.zeros(4, 16, device=DEVICE)
    x[:, 2] = -1.0
    x[:, 11] = -1.0
    return x

def make_instancenorm_inputs():
    x = torch.full((2, 4, 8, 8), 3.0, device=DEVICE) + torch.randn(2, 4, 8, 8, device=DEVICE) * 1e-6
    weight = torch.randn(4, device=DEVICE)
    bias = torch.randn(4, device=DEVICE)
    return x, weight, bias

def make_groupnorm_inputs():
    x = torch.randn(2, 8, 4, 4, device=DEVICE)
    num_groups = 4
    gamma = torch.rand(8, device=DEVICE) * 3 + 0.5   # non-uniform, away from 1.0
    beta = torch.randn(8, device=DEVICE) * 2          # nonzero
    return x, num_groups, gamma, beta

def make_batchnorm_inputs():
    x = torch.randn(2, 8, 4, 4, device=DEVICE)   # spatial_size=16 > 1, required for wrong_running_stats_broadcast to be catchable
    running_mean = torch.randn(8, device=DEVICE)
    running_var = torch.rand(8, device=DEVICE) + 0.5
    weight = torch.randn(8, device=DEVICE)
    bias = torch.randn(8, device=DEVICE)
    return x, running_mean, running_var, weight, bias

def make_cross_entropy_inputs():
    logits = torch.full((8, 50), 150.0, device=DEVICE) + torch.randn(8, 50, device=DEVICE)
    targets = torch.randint(0, 50, (8,), device=DEVICE)
    return logits, targets

def make_max_pool_inputs(shape):
    x = -torch.rand(*shape, device=DEVICE) - 0.1
    return x, 3, 2, 1  # kernel_size, stride, padding

def make_avg_pool_inputs(shape):
    x = torch.randn(*shape, device=DEVICE)
    return x, 3, 2, 1  # padding=1 required for wrong_divisor to be catchable

def make_attention_inputs():
    N, D = 128, 64
    return (
        torch.randn(N, D, device=DEVICE),
        torch.randn(N, D, device=DEVICE),
        torch.randn(N, D, device=DEVICE),
    )


# ---------------------------------------------------------------------------
# Test cases: (kernel_name, candidate_fn, raw_kernel, reference_fn, inputs, spec)
# ---------------------------------------------------------------------------

def build_test_cases():
    softmax_inputs   = make_softmax_inputs()
    layernorm_inputs = make_layernorm_inputs()
    matmul_inputs    = make_matmul_inputs()
    fa_inputs        = make_flash_attention_inputs()
    rmsnorm_inputs   = make_rmsnorm_inputs()

    s_spec = softmax_spec()
    l_spec = layernorm_spec()
    m_spec = matmul_spec()
    f_spec = flash_attention_spec()
    r_spec = rmsnorm_get_spec()

    cases = [
        # Softmax
        ("softmax/first_tile", cheat_softmax_first_tile, softmax_kernel_cheat_first_tile, ref_softmax, softmax_inputs, s_spec),
        ("softmax/wrong_reduction", cheat_softmax_wrong_reduction, None, ref_softmax, softmax_inputs, s_spec),

        ("layernorm/ignore_gamma_beta", cheat_ln_ignore_gamma_beta, cheat_ln_ignore_gamma_beta_kernel, ref_layernorm, layernorm_inputs, l_spec),
        ("layernorm/skip_mean_subtract", cheat_ln_skip_mean,       cheat_ln_skip_mean_kernel,         ref_layernorm, layernorm_inputs, l_spec),
        ("layernorm/wrong_variance",     cheat_ln_wrong_var,       cheat_ln_wrong_var_kernel,         ref_layernorm, layernorm_inputs, l_spec),

        # Matmul
        ("matmul/partial_k_reduct",  cheat_mm_partial_k,    cheat_mm_partial_k_kernel,    ref_matmul, matmul_inputs, m_spec),
        ("matmul/skip_boundary",     cheat_mm_skip_boundary, cheat_mm_skip_boundary_kernel, ref_matmul, matmul_inputs, m_spec),
        ("matmul/swapped_strides",   cheat_mm_swapped,      cheat_mm_swapped_kernel,      ref_matmul, matmul_inputs, m_spec),
        ("matmul/wrong_dtype",       cheat_mm_wrong_dtype,  cheat_mm_wrong_dtype_kernel,  ref_matmul, matmul_inputs, m_spec),

        # Flash Attention
        ("flash_attn/approx_denom",  cheat_fa_approx, cheat_fa_approx_kernel, ref_flash_attention, fa_inputs, f_spec),
        ("flash_attn/drop_last_tile", cheat_fa_drop,  cheat_fa_drop_kernel,   ref_flash_attention, fa_inputs, f_spec),
        ("flash_attn/skip_rescaling", cheat_fa_skip,  cheat_fa_skip_kernel,   ref_flash_attention, fa_inputs, f_spec),
        ("flash_attn/wrong_mask",     cheat_fa_mask,  cheat_fa_mask_kernel,   ref_flash_attention, fa_inputs, f_spec),

        # RMSNorm
        ("rmsnorm/ignore_gamma",      rmsnorm_ignore_gamma,      rmsnorm_ignore_gamma_kernel,      ref_rmsnorm, rmsnorm_inputs, r_spec),
        ("rmsnorm/wrong_norm",        rmsnorm_wrong_norm,        rmsnorm_wrong_norm_kernel,        ref_rmsnorm, rmsnorm_inputs, r_spec),
        ("rmsnorm/partial_reduction", rmsnorm_partial_reduction, rmsnorm_partial_reduction_kernel, ref_rmsnorm, rmsnorm_inputs, r_spec),

        # LogSoftmax
        ("log_softmax/skip_max_subtraction", cheat_log_softmax, cheat_log_softmax_kernel, ref_log_softmax, make_log_softmax_inputs(), log_softmax_spec()),

        # Swish
        ("swish/linear_sigmoid_approx", cheat_swish, cheat_swish_kernel, ref_swish, make_swish_inputs(), swish_spec()),

        # GELU
        ("gelu/sigmoid_approx", cheat_gelu, cheat_gelu_kernel, ref_gelu, make_gelu_inputs(), gelu_spec()),

        # Reductions
        ("sum_reduction/partial_reduction", cheat_sum_reduction, cheat_sum_reduction_kernel, ref_sum_reduction, make_sum_mean_reduction_inputs(), sum_reduction_spec()),
        ("mean_reduction/partial_reduction", cheat_mean_reduction, cheat_mean_reduction_kernel, ref_mean_reduction, make_sum_mean_reduction_inputs(), mean_reduction_spec()),
        ("max_reduction/wrong_padding", cheat_max_reduction, cheat_max_reduction_kernel, ref_max_reduction, make_max_reduction_inputs(), max_reduction_spec()),
        ("min_reduction/wrong_padding", cheat_min_reduction, cheat_min_reduction_kernel, ref_min_reduction, make_min_reduction_inputs(), min_reduction_spec()),

        # Norms
        ("l1norm/partial_reduction", cheat_l1norm, cheat_l1norm_kernel, ref_l1norm, make_l1norm_inputs(), l1norm_spec()),
        ("l2norm/wrong_norm", cheat_l2norm, cheat_l2norm_kernel, ref_l2norm, make_l2norm_inputs(), l2norm_spec()),
        # frobenius_norm: two-kernel launch (sumsq + normalize), no single raw_kernel to pass for structural analysis
        ("frobenius_norm/wrong_norm", cheat_frobenius_norm, None, ref_frobenius_norm, make_frobenius_norm_inputs(), frobenius_norm_spec()),

        # Argmax/Argmin
        ("argmax/tiebreak", cheat_argmax, cheat_argmax_kernel, ref_argmax, make_argmax_inputs(), argmax_spec()),
        ("argmin/tiebreak", cheat_argmin, cheat_argmin_kernel, ref_argmin, make_argmin_inputs(), argmin_spec()),

        # InstanceNorm / GroupNorm / BatchNorm
        ("instancenorm/skip_eps", cheat_instancenorm, cheat_instancenorm_kernel, ref_instancenorm, make_instancenorm_inputs(), instancenorm_spec()),
        ("groupnorm/ignore_affine", cheat_groupnorm, cheat_groupnorm_kernel, ref_groupnorm, make_groupnorm_inputs(), groupnorm_spec()),
        ("batchnorm/wrong_running_stats_broadcast", cheat_batchnorm, cheat_batchnorm_kernel, ref_batchnorm, make_batchnorm_inputs(), batchnorm_spec()),

        # CrossEntropy
        ("cross_entropy/missing_max_subtraction", cheat_cross_entropy, cheat_cross_entropy_kernel, ref_cross_entropy, make_cross_entropy_inputs(), cross_entropy_spec()),

        # Pooling
        ("max_pool1d/wrong_padding", cheat_max_pool1d, cheat_max_pool1d_kernel, ref_max_pool1d, make_max_pool_inputs((2, 4, 17)), max_pool1d_spec()),
        ("max_pool2d/wrong_padding", cheat_max_pool2d, cheat_max_pool2d_kernel, ref_max_pool2d, make_max_pool_inputs((2, 4, 17, 17)), max_pool2d_spec()),
        ("max_pool3d/wrong_padding", cheat_max_pool3d, cheat_max_pool3d_kernel, ref_max_pool3d, make_max_pool_inputs((2, 4, 9, 9, 9)), max_pool3d_spec()),
        ("avg_pool1d/wrong_divisor", cheat_avg_pool1d, cheat_avg_pool1d_kernel, ref_avg_pool1d, make_avg_pool_inputs((2, 4, 17)), avg_pool1d_spec()),
        ("avg_pool2d/wrong_divisor", cheat_avg_pool2d, cheat_avg_pool2d_kernel, ref_avg_pool2d, make_avg_pool_inputs((2, 4, 17, 17)), avg_pool2d_spec()),
        ("avg_pool3d/wrong_divisor", cheat_avg_pool3d, cheat_avg_pool3d_kernel, ref_avg_pool3d, make_avg_pool_inputs((2, 4, 9, 9, 9)), avg_pool3d_spec()),

        # Attention
        ("scaled_dot_product_attention/wrong_mask", cheat_sdpa, cheat_sdpa_kernel, ref_sdpa, make_attention_inputs(), sdpa_spec()),
        ("causal_flash_attention/wrong_causal_mask", cheat_causal_fa, cheat_causal_fa_kernel, ref_causal_fa, make_attention_inputs(), causal_flash_attention_spec()),
    ]
    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    test_cases = build_test_cases()
    all_results = {}

    for name, candidate_fn, raw_kernel, reference_fn, inputs, spec in test_cases:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        checker = KernelChecker(spec)
        try:
            results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)
            print(checker.summary(results))
            all_results[name] = results
        except Exception as e:
            print(f"  ERRORED: {type(e).__name__}: {e}")
            all_results[name] = None

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print('='*60)
    print(f"  {'Kernel':<45} {'Verdict'}")
    print(f"  {'-'*45} {'-'*20}")
    n_caught, n_missed, n_errored = 0, 0, 0
    for name, results in all_results.items():
        if results is None:
            print(f"  {name:<45} ERROR")
            n_errored += 1
            continue
        checker = KernelChecker(None)
        verdict = checker.verdict(results)
        status = "PASS CAUGHT" if "FAIL" in verdict else "FAIL MISSED"
        print(f"  {name:<45} {status}")
        if status == "PASS CAUGHT":
            n_caught += 1
        else:
            n_missed += 1

    print(f"\n  {n_caught}/{len(all_results)} caught, {n_missed} missed, {n_errored} errored")


if __name__ == "__main__":
    run_all()