"""
run_checker.py  run the KernelChecker against every cheating kernel
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

# Imports
from verification.checker import KernelChecker
from verification.specs.softmax import get_spec as softmax_spec
from verification.specs.layernorm import get_spec as layernorm_spec
from verification.specs.matmul import get_spec as matmul_spec
from verification.specs.flash_attention import get_spec as flash_attention_spec

from TritonBench.reference.softmax import softmax as ref_softmax, softmax_kernel as ref_softmax_kernel
from TritonBench.reference.layernorm import layernorm as ref_layernorm, layernorm_kernel as ref_layernorm_kernel
from TritonBench.reference.mat_mult import matmul as ref_matmul, matmul_kernel as ref_matmul_kernel
from TritonBench.reference.flash_attention import flash_attention as ref_flash_attention, flash_attention_kernel as ref_flash_attention_kernel

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Test cases

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


# (kernel_name, candidate_fn, raw_kernel, reference_fn, inputs, spec)

def build_test_cases():
    softmax_inputs  = make_softmax_inputs()
    layernorm_inputs = make_layernorm_inputs()
    matmul_inputs   = make_matmul_inputs()
    fa_inputs       = make_flash_attention_inputs()

    s_spec = softmax_spec()
    l_spec = layernorm_spec()
    m_spec = matmul_spec()
    f_spec = flash_attention_spec()

    return [
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
    ]


# Runner

def run_all():
    test_cases = build_test_cases()
    all_results = {}

    for name, candidate_fn, raw_kernel, reference_fn, inputs, spec in test_cases:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        checker = KernelChecker(spec)
        results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)
        print(checker.summary(results))
        all_results[name] = results

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print('='*60)
    print(f"  {'Kernel':<35} {'Verdict'}")
    print(f"  {'-'*35} {'-'*20}")
    for name, results in all_results.items():
        checker = KernelChecker(None)
        verdict = checker.verdict(results)
        status = "PASS CAUGHT" if "FAIL" in verdict else "FAIL MISSED"
        print(f"  {name:<35} {status}")


if __name__ == "__main__":
    run_all()