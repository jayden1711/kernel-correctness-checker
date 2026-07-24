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

UNVERIFIED: built from the KernelBench directory listing shared earlier
in this conversation, not confirmed against your actual checkout. If a
filename doesn't match, update the mapping here -- do not fall back to
shape-guessing for these operators, see the ambiguity argument above.
"""

from typing import Optional

KERNELBENCH_FILENAME_TO_SPEC = {
    "24_LogSoftmax": "log_softmax",
    "25_Swish": "swish",
    "26_GELU_": "gelu",
    "34_InstanceNorm": "instancenorm",
    "33_BatchNorm": "batchnorm",
    "35_GroupNorm_": "groupnorm",
    "37_L1Norm_": "l1norm",
    "38_L2Norm_": "l2norm",
    "39_FrobeniusNorm_": "frobenius_norm",
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
}


def resolve_operator_key(problem_filename_stem: str) -> Optional[str]:
    """problem_filename_stem: e.g. '26_GELU_' from Path('26_GELU_.py').stem."""
    return KERNELBENCH_FILENAME_TO_SPEC.get(problem_filename_stem)
