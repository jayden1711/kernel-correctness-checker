# Checker Comparison -- Real Corpus

Corpus: 40 mutants across 29 operators (argmax, argmin, avg_pool1d, avg_pool2d, avg_pool3d, batchnorm, causal_flash_attention, cross_entropy, flash_attention, frobenius_norm, gelu, groupnorm, instancenorm, l1norm, l2norm, layernorm, log_softmax, matmul, max_pool1d, max_pool2d, max_pool3d, max_reduction, mean_reduction, min_reduction, rmsnorm, scaled_dot_product_attention, softmax, sum_reduction, swish).

## Headline comparison

| System | Catch rate | False positive rate | Mean latency (ms/check) |
|---|---|---|---|
| allclose | 57% | 0% | 0.1012 |
| autokernel_gate | 68% | 18% | 4.3911 |
| gpuemu (boundary_shape) | 65% | 0% | 4.0035 |
| gpuemu (adversarial_value) | 82% | 82% | 1.1549 |
| propilot | 10% | 0% | 0.4251 |
| your_checker (full) | 100% | 0% | 35.4444 |
| your_checker (numeric only) | 100% | 0% | 22.7380 |
| your_checker (algebraic only) | 45% | 0% | 3.5603 |
| your_checker (structural only) | 10% | 0% | 4.9073 |

## Per-operator catch rate

| System | argmax | argmin | avg_pool1d | avg_pool2d | avg_pool3d | batchnorm | causal_flash_attention | cross_entropy | flash_attention | frobenius_norm | gelu | groupnorm | instancenorm | l1norm | l2norm | layernorm | log_softmax | matmul | max_pool1d | max_pool2d | max_pool3d | max_reduction | mean_reduction | min_reduction | rmsnorm | scaled_dot_product_attention | softmax | sum_reduction | swish |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allclose | 0% | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 0% | 100% | 100% | 67% | 0% | 25% | 0% | 0% | 0% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 100% |
| autokernel_gate | 0% | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 0% | 100% | 100% | 100% | 0% | 100% | 0% | 0% | 0% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 100% |
| gpuemu (boundary_shape) | 0% | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 100% | 100% | 100% | 100% | 0% | 100% | 100% | 67% | 0% | 100% | 0% | 0% | 0% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 100% |
| gpuemu (adversarial_value) | 0% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 100% |
| propilot | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 100% | 0% |
| your_checker (full) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| your_checker (numeric only) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| your_checker (algebraic only) | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 50% | 100% | 0% | 0% | 0% | 100% | 100% | 67% | 0% | 50% | 0% | 0% | 0% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 0% |
| your_checker (structural only) | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 25% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 50% | 0% | 0% |

## Mutants each system missed

**allclose**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, cross_entropy/missing_max_subtraction, instancenorm/skip_eps, layernorm/wrong_variance_estimate, log_softmax/skip_max_subtraction, matmul/partial_k_reduct, matmul/skip_boundary_tiles, matmul/wrong_dtype, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding
**autokernel_gate**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, cross_entropy/missing_max_subtraction, instancenorm/skip_eps, log_softmax/skip_max_subtraction, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding
**gpuemu (boundary_shape)**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, cross_entropy/missing_max_subtraction, instancenorm/skip_eps, layernorm/wrong_variance_estimate, log_softmax/skip_max_subtraction, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding
**gpuemu (adversarial_value)**: argmax/tiebreak, argmin/tiebreak, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding
**propilot**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, batchnorm/wrong_running_stats_broadcast, causal_flash_attention/wrong_causal_mask, cross_entropy/missing_max_subtraction, flash_attention/approx_denom, flash_attention/drop_last_tile, flash_attention/skip_rescaling, flash_attention/wrong_mask, frobenius_norm/wrong_norm, gelu/sigmoid_approx, groupnorm/ignore_affine, instancenorm/skip_eps, l1norm/partial_reduction, l2norm/wrong_norm, layernorm/ignore_gamma_beta, layernorm/skip_mean_subtract, layernorm/wrong_variance_estimate, log_softmax/skip_max_subtraction, matmul/partial_k_reduct, matmul/wrong_dtype, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding, rmsnorm/ignore_gamma, rmsnorm/partial_reduction, rmsnorm/wrong_norm, scaled_dot_product_attention/wrong_mask, softmax/first_tile, softmax/wrong_reduction, swish/linear_sigmoid_approx
**your_checker (full)**: none
**your_checker (numeric only)**: none
**your_checker (algebraic only)**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, batchnorm/wrong_running_stats_broadcast, cross_entropy/missing_max_subtraction, flash_attention/drop_last_tile, flash_attention/skip_rescaling, gelu/sigmoid_approx, groupnorm/ignore_affine, instancenorm/skip_eps, layernorm/wrong_variance_estimate, log_softmax/skip_max_subtraction, matmul/partial_k_reduct, matmul/skip_boundary_tiles, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, min_reduction/wrong_padding, swish/linear_sigmoid_approx
**your_checker (structural only)**: argmax/tiebreak, argmin/tiebreak, avg_pool1d/wrong_divisor, avg_pool2d/wrong_divisor, avg_pool3d/wrong_divisor, batchnorm/wrong_running_stats_broadcast, cross_entropy/missing_max_subtraction, flash_attention/approx_denom, flash_attention/drop_last_tile, flash_attention/skip_rescaling, frobenius_norm/wrong_norm, gelu/sigmoid_approx, groupnorm/ignore_affine, instancenorm/skip_eps, l1norm/partial_reduction, l2norm/wrong_norm, layernorm/ignore_gamma_beta, layernorm/skip_mean_subtract, layernorm/wrong_variance_estimate, log_softmax/skip_max_subtraction, matmul/partial_k_reduct, matmul/skip_boundary_tiles, matmul/swapped_strides, matmul/wrong_dtype, max_pool1d/wrong_padding, max_pool2d/wrong_padding, max_pool3d/wrong_padding, max_reduction/wrong_padding, mean_reduction/partial_reduction, min_reduction/wrong_padding, rmsnorm/ignore_gamma, rmsnorm/partial_reduction, rmsnorm/wrong_norm, softmax/wrong_reduction, sum_reduction/partial_reduction, swish/linear_sigmoid_approx

## Per-operator false-positive rate (reference flagged as wrong; 0% cells omitted)

**autokernel_gate**: layernorm=100%, matmul=100%
**gpuemu (adversarial_value)**: avg_pool1d=100%, avg_pool2d=100%, avg_pool3d=100%, batchnorm=100%, causal_flash_attention=100%, cross_entropy=100%, flash_attention=100%, frobenius_norm=100%, gelu=100%, groupnorm=100%, instancenorm=60%, l1norm=100%, l2norm=100%, layernorm=100%, log_softmax=100%, matmul=100%, mean_reduction=100%, rmsnorm=100%, scaled_dot_product_attention=100%, softmax=100%, sum_reduction=100%, swish=100%

## False-positive example details (up to 3 per operator)

**autokernel_gate**:
  - layernorm: adversarial_stability
  - matmul: adversarial_stability
**gpuemu (adversarial_value)**:
  - avg_pool1d: adversarial_value(nan_inf): mismatch
  - avg_pool2d: adversarial_value(nan_inf): mismatch
  - avg_pool3d: adversarial_value(nan_inf): mismatch
  - batchnorm: adversarial_value(nan_inf): mismatch
  - causal_flash_attention: adversarial_value(nan_inf): mismatch
  - cross_entropy: adversarial_value(nan_inf): mismatch
  - flash_attention: adversarial_value(extreme): mismatch
  - frobenius_norm: adversarial_value(nan_inf): mismatch
  - gelu: adversarial_value(nan_inf): mismatch
  - groupnorm: adversarial_value(nan_inf): mismatch
  - instancenorm: adversarial_value(nan_inf): mismatch
  - l1norm: adversarial_value(nan_inf): mismatch
  - l2norm: adversarial_value(nan_inf): mismatch
  - layernorm: adversarial_value(nan_inf): mismatch
  - log_softmax: adversarial_value(nan_inf): mismatch
  - matmul: adversarial_value(nan_inf): mismatch
  - mean_reduction: adversarial_value(nan_inf): mismatch
  - rmsnorm: adversarial_value(nan_inf): mismatch
  - scaled_dot_product_attention: adversarial_value(extreme): mismatch
  - softmax: adversarial_value(nan_inf): mismatch
  - sum_reduction: adversarial_value(nan_inf): mismatch
  - swish: adversarial_value(nan_inf): mismatch
