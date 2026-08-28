"""
The banked GPU evidence the scope detector's thresholds are derived from.

TRANSCRIBED, with provenance, from
`../../adaptive_tol_theory_2026-08-25/GPU_NATIVE.md`. Every row carries the
section it came from so a reader can check it against the source table rather
than trusting this file. Nothing here is invented, fitted, or smoothed.

  ATTENTION_VARIANTS  -- GPU_NATIVE.md Section 4 table, 11 rows. This is the
      ONLY place in the banked data where in-scope and out-of-scope
      invocations sit side by side with both signals measured, so it is what
      the separating thresholds are derived from. 5 independent seeds per
      variant; the `in_scope` column is the source table's own verdict.

  PRIMARY_DEFECTS     -- GPU_NATIVE.md Section 2 table, 27 rows, median
      linearisation defect per operator on the corpus's ordinary inputs. Used
      as the in-scope negative set for the defect threshold.

  Section 3 supplies the s/ulp facts that Section 2's table does not carry:
      every operator except `cross_entropy` has min s/ulp >= 3350 across all
      40 samples; `cross_entropy` has min 2.0 and median 360.
      That single operator is what forces the s/ulp statistic to be the
      MEDIAN rather than the MIN -- see derive_thresholds.py.
"""

# op, variant, peak_weight, s_over_ulp (median over 5 seeds), defect_pct,
# cv_median, in_scope, mechanism
ATTENTION_VARIANTS = [
    ("flash_attention", "primary",                 0.370,   8912.0,   0.2,  0.247, True,  None),
    ("flash_attention", "approx_denominator",      1.000,  91595.0,   0.3,  0.270, True,  None),
    ("flash_attention", "wrong_causal_mask",       1.000,   6408.0,   0.1,  0.228, True,  None),
    ("causal_flash_attention", "primary",          0.291,  15947.0,   0.1,  0.262, True,  None),
    ("scaled_dot_product_attention", "primary",    0.374,  15250.0,   0.1,  0.181, True,  None),
    ("flash_attention", "multi_tile_rescaling",    1.000,   2220.0,  99.3,  1.869, False, "saturation"),
    ("causal_flash_attention", "large_magnitude_qk", 1.000, 118355.0, 23.7, 1.033, False, "saturation"),
    ("scaled_dot_product_attention", "large_magnitude_qk", 1.000, 7699.0, 24.0, 1.074, False, "saturation"),
    ("flash_attention", "last_tile_dropped",       1.000,      2.00, 900.0, 0.315, False, "fp_floor"),
    ("flash_attention", "skip_rescaling",          1.000,      2.00, 900.0, 0.080, False, "fp_floor"),
    ("flash_attention", "equal_attention_weights", 0.016,      3.00, 900.0, 0.000, False, "fp_floor"),
]

# operator -> median linearisation defect (%), ordinary corpus inputs
PRIMARY_DEFECTS = {
    "avg_pool1d": 0.011, "avg_pool2d": 0.019, "avg_pool3d": 0.013,
    "batchnorm": 0.012, "causal_flash_attention": 0.071,
    "cross_entropy": 1.518, "flash_attention": 0.103,
    "frobenius_norm": 0.010, "gelu": 0.021, "groupnorm": 0.023,
    "instancenorm": 0.029, "l1norm": 0.008, "l2norm": 0.010,
    "layernorm": 0.014, "log_softmax": 0.053, "matmul": 0.013,
    "max_pool1d": 0.006, "max_pool2d": 0.009, "max_pool3d": 0.007,
    "max_reduction": 0.023, "mean_reduction": 0.024, "min_reduction": 0.020,
    "rmsnorm": 0.012, "scaled_dot_product_attention": 0.089,
    "softmax": 0.072, "sum_reduction": 0.024, "swish": 0.031,
}

# GPU_NATIVE.md Section 2 verdict line: the defect over all 228 in-scope
# invocations spans 0.0028% - 3.66%. The 3.66% worst case is cross_entropy and
# it is the binding in-scope observation for the defect threshold -- the
# per-operator medians above understate it.
WORST_IN_SCOPE_DEFECT_PCT = 3.66

# GPU_NATIVE.md Section 3.
CROSS_ENTROPY_SULP_MIN = 2.0
CROSS_ENTROPY_SULP_MEDIAN = 360.0
OTHER_OPS_SULP_MIN_FLOOR = 3350.0

# GPU_NATIVE.md Section 5: index-valued, J = 0 a.e., A3 fails. Excluded
# structurally, with no measurement involved.
STRUCTURALLY_EXCLUDED = ("argmax", "argmin")


# ---------------------------------------------------------------------------
# MEASURED ON THE CORPUS, 2026-08-26, T4, session `kccscope`.
# Arms in ../arms/. This is what the detector actually sees, as opposed to what
# the GPU_NATIVE.md probe measured, and the two disagree -- see FINDINGS.md.
#
# Both values are at the CONVERGED 40-delta defect probe (the 20- and 40-delta
# arms agree on 0 of 854 records).
# ---------------------------------------------------------------------------

# Worst defect on an invocation the detector left silent.
CORPUS_WORST_IN_SCOPE_DEFECT_PCT = 9.605      # cross_entropy/adversarial_large_magnitude_logits

# Smallest defect on one of the 6 variants GPU_NATIVE.md Section 4 labelled
# out of scope.
CORPUS_LEAST_OUT_OF_SCOPE_DEFECT_PCT = 6.6    # causal_flash_attention/adversarial_large_magnitude_qk

# Silent-set s/ulp margins, min over all 27 operators (cross_entropy) and max.
CORPUS_SULP_MARGIN_MIN = 9.0                  # 296 / 32
CORPUS_SULP_MARGIN_MAX = 381.0                # 12198 / 32
