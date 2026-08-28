"""
verification/layer3_properties/kernelbench_operator_registry.py

Maps a KernelBench problem filename stem (e.g. "26_GELU_" from
"26_GELU_.py") to the operator key used by verification/specs/<key>.py.
Lets kernel_adapter.py dispatch DIRECTLY to the correct spec's
algebraic_properties for a KNOWN operator, instead of shape-guessing.

WHY THIS EXISTS instead of extending the shape-based try_*_layer3
detectors (softmax_generic_properties.py etc.): most operators added
this conversation share IDENTICAL call shapes with each other and with
softmax. log_softmax, l1norm, l2norm, sum/mean/max/min reduction,
argmax, argmin, gelu, and swish are ALL "exactly one tensor argument,
>=2D" -- indistinguishable from try_softmax_layer3's own detection
heuristic and from each other. A shape-based try_l2norm_layer3 would
also fire on log_softmax/sum_reduction/argmax/etc. calls, silently
checking the WRONG operator's algebraic property against the RIGHT
operator's output (e.g. asserting rows-sum-to-1, true only for softmax,
against a correctly-implemented l2norm kernel) -- REJECTING CORRECT
KERNELS. Shape alone cannot disambiguate these. Operator identity can,
and it's available for free at the point kernel_adapter.py already
knows which KernelBench problem file it's processing -- this registry
just keeps that identity from being discarded before it's used.

VERIFIED 2026-08-27 against the real checkout: every stem below now
matches a file in KernelBench/KernelBench/level1/ (or level3/ for the
one attention entry). Three stems did NOT match before that check --
37/38/39 were rotated -- and the consequence was silent fallthrough to
shape-guessing on exactly the operators this file exists to protect.
See the fix note at those entries.

If a filename doesn't match, update the mapping here -- do not fall
back to shape-guessing for these operators, see the ambiguity argument
above. A cheap way to catch a repeat:
    python -c "import glob,os;from verification.layer3_properties.kernelbench_operator_registry import KERNELBENCH_FILENAME_TO_SPEC as R;\
kb={os.path.basename(f)[:-3] for d in ('level1','level3') for f in glob.glob(f'KernelBench/KernelBench/{d}/*.py')};\
print(sorted(k for k in R if k not in kb))"
"""

from typing import Optional

KERNELBENCH_FILENAME_TO_SPEC = {
    # Deliberately mapped to keys with NO corresponding
    # verification/specs/<key>.py file. There is no per-operator spec
    # for relu/sigmoid/tanh in this project, and the real "layernorm"
    # spec's candidate_fn(x, gamma, beta) convention doesn't match a
    # KernelBench-format candidate_fn(x) (gamma/beta live inside the
    # nn.Module, not as forward() args) -- registering under a
    # non-colliding key still buys the important thing (operator_key is
    # not None -> skip_shape_guessed_layer3=True, so the generic
    # try_softmax_layer3 heuristic never misidentifies these as
    # softmax-shaped, which is exactly what happened before this was
    # added: relu/sigmoid/tanh/layernorm's single-2D-tensor calls all
    # matched try_softmax_layer3's "exactly one tensor arg, >=2D"
    # detector and got asserted against softmax's own invariants).
    # _run_known_operator_properties treats a missing spec file as "no
    # properties to check" (empty dict), not a failure -- see its
    # docstring.
    "19_ReLU": "relu",
    "21_Sigmoid": "sigmoid",
    "22_Tanh": "tanh",
    "20_LeakyReLU": "leaky_relu",
    "31_ELU": "elu",
    "27_SELU_": "selu",
    "28_HardSigmoid": "hardsigmoid",
    "29_Softplus": "softplus",
    "30_Softsign": "softsign",
    "32_HardTanh": "hardtanh",
    "40_LayerNorm": "layernorm_1arg",

    "23_Softmax": "softmax",
    "24_LogSoftmax": "log_softmax",
    "25_Swish": "swish",
    "26_GELU_": "gelu",
    "1_Square_matrix_multiplication_": "matmul",
    # PREEMPTIVE FIX (same root cause as layernorm_1arg above, caught
    # before it caused a failure this time instead of after): KernelBench's
    # real 34_InstanceNorm.py / 33_BatchNorm.py / 35_GroupNorm_.py Model
    # classes all hold their affine/running-stat params as internal
    # nn.Module state (self.bn = nn.InstanceNorm2d(...) etc.) and their
    # forward() takes ONLY x -- but instancenorm/batchnorm/groupnorm's
    # real specs (verification/specs/*.py) expect multi-arg calls
    # (x, weight, bias) / (x, running_mean, running_var, weight, bias) /
    # (x, num_groups, weight, bias). Mapping to the real spec key would
    # crash known-operator properties on every KernelBench candidate for
    # these operators the same way "layernorm" did. Mapped to
    # non-colliding _1arg keys instead -- still gets skip_shape_guessed_
    # layer3=True (operator_key is not None), no per-op algebraic
    # properties (no spec file to load), same tradeoff as layernorm_1arg.
    "34_InstanceNorm": "instancenorm_1arg",
    "33_BatchNorm": "batchnorm_1arg",
    "35_GroupNorm_": "groupnorm_1arg",
    # FIXED 2026-08-27 -- these three stems were ROTATED and matched nothing.
    # The registry said 37=L1Norm / 38=L2Norm / 39=FrobeniusNorm; the actual
    # checkout is 37_FrobeniusNorm_ / 38_L1Norm_ / 39_L2Norm_. Confirmed by
    # listing KernelBench/KernelBench/level1/.
    #
    # This was NOT cosmetic. A stem that does not match makes
    # resolve_operator_key return None, which leaves
    # skip_shape_guessed_layer3 False -- so all three fell through to the
    # shape-based try_*_layer3 detectors. l1norm, l2norm and frobenius_norm
    # are all "exactly one tensor argument, >=2D", which is precisely
    # try_softmax_layer3's detection heuristic, so a CORRECT norm kernel would
    # have softmax's rows-sum-to-one invariant asserted against it and be
    # REJECTED. That is the exact failure this file's own docstring says it
    # exists to prevent, reached through a typo in the file itself.
    #
    # The docstring's "UNVERIFIED: built from the directory listing shared
    # earlier in this conversation, not confirmed against your actual
    # checkout" was correct to flag it. Now confirmed against the checkout.
    "37_FrobeniusNorm_": "frobenius_norm",
    "38_L1Norm_": "l1norm",
    "39_L2Norm_": "l2norm",
    "47_Sum_reduction_over_a_dimension": "sum_reduction",
    "48_Mean_reduction_over_a_dimension": "mean_reduction",
    "49_Max_reduction_over_a_dimension": "max_reduction",
    "53_Min_reduction_over_a_dimension": "min_reduction",
    "51_Argmax_over_a_dimension": "argmax",
    "52_Argmin_over_a_dimension": "argmin",
    "95_CrossEntropyLoss": "cross_entropy",
    "41_Max_Pooling_1D": "max_pool1d",
    "42_Max_Pooling_2D": "max_pool2d",
    "43_Max_Pooling_3D": "max_pool3d",
    "44_Average_Pooling_1D": "avg_pool1d",
    "45_Average_Pooling_2D": "avg_pool2d",
    "46_Average_Pooling_3D": "avg_pool3d",
    "97_ScaledDotProductAttention": "scaled_dot_product_attention",
    "43_MinGPTCausalAttention": "causal_flash_attention",

    # ---------------------------------------------------------------------
    # PHASE 1, added 2026-08-27. Every stem below was checked against its real
    # KernelBench forward() signature, and each maps to a spec whose calling
    # convention MATCHES it -- which is why these use the real spec key rather
    # than a non-colliding _1arg key the way layernorm/batchnorm/groupnorm had
    # to. Verified, not assumed:
    #   3, 4, 12, 15  forward(A, B)          -> 2-arg specs
    #   88, 89, 91, 92 forward(x)            -> SingleTensorSpec
    #   93            forward(x, mask)       -> MaskedScanKernelSpec
    #   94, 96, 98    forward(pred, targets) -> TargetLossKernelSpec
    #
    # NOTE the pre-existing activation entries above (19/20/21/22/27/28/29/31)
    # now resolve to spec files that DID NOT EXIST when their comment was
    # written. That comment's premise ("deliberately mapped to keys with NO
    # corresponding spec file") no longer holds for them. Checked rather than
    # left to chance: all eight load, run, and produce 0 false positives on a
    # correct candidate, and they now contribute 7 real algebraic checks where
    # they previously contributed none. 30_Softsign and 32_HardTanh keep the
    # old no-spec behaviour -- both were dropped from Phase 1 (absent from
    # TritonBench-G and KernelBenchX alike).
    "3_Batched_matrix_multiplication": "batched_matmul",
    "4_Matrix_vector_multiplication_": "matvec",
    "12_Matmul_with_diagonal_matrices_": "diagonal_matmul",
    "15_Matmul_for_lower_triangular_matrices": "triangular_matmul",
    "88_MinGPTNewGelu": "new_gelu",
    "89_cumsum": "cumsum",
    "91_cumsum_reverse": "cumsum_reverse",
    "92_cumsum_exclusive": "cumsum_exclusive",
    "93_masked_cumsum": "masked_cumsum",
    "94_MSELoss": "mse_loss",
    "96_HuberLoss": "huber_loss",
    "98_KLDivLoss": "kldiv_loss",
}


def resolve_operator_key(problem_filename_stem: str) -> Optional[str]:
    """problem_filename_stem: e.g. '26_GELU_' from Path('26_GELU_.py').stem."""
    return KERNELBENCH_FILENAME_TO_SPEC.get(problem_filename_stem)
