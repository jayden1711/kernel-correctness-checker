# Per-operator tail — checker vs faithful AutoKernel

Pass-path (200 reference trials), Colab T4, warm Triton cache. Source:
`verification_runs/triton_cache_2026-08-25/run2/results_raw.json`.

`your_checker (full)` overall: mean 23.72 ms, p50 16.13, p90 48.91, p99 59.90
`autokernel_gate (faithful)` overall: mean 71.13 ms, p50 12.31, p90 341.62, p99 509.49

| operator | checker p50 | checker p90 | checker mean | p90/p50 | share of checker total | AutoKernel mean |
|---|---:|---:|---:|---:|---:|---:|
| `flash_attention` | 51.75 | 59.90 | 59.63 | 1.16 | 25.1% | 384.3 |
| `matmul` | 36.13 | 38.41 | 37.33 | 1.06 | 15.7% | 13.3 |
| `groupnorm` | 30.28 | 34.11 | 31.50 | 1.13 | 3.3% | 18.6 |
| `rmsnorm` | 28.73 | 29.91 | 28.84 | 1.04 | 9.1% | 12.2 |
| `layernorm` | 28.68 | 29.28 | 28.80 | 1.02 | 9.1% | 12.6 |
| `frobenius_norm` | 20.10 | 20.10 | 20.10 | 1.00 | 2.1% | 16.1 |
| `softmax` | 18.99 | 21.89 | 19.63 | 1.15 | 4.1% | 11.4 |
| `cross_entropy` | 15.65 | 18.31 | 18.06 | 1.17 | 1.9% | 12.4 |
| `min_reduction` | 17.08 | 17.92 | 17.38 | 1.05 | 1.8% | 11.0 |
| `instancenorm` | 16.06 | 16.13 | 16.18 | 1.00 | 1.7% | 12.6 |
| `sum_reduction` | 13.28 | 19.24 | 15.51 | 1.45 | 1.6% | 10.9 |
| `batchnorm` | 14.97 | 15.19 | 15.03 | 1.01 | 1.6% | 14.9 |
| `gelu` | 12.67 | 18.53 | 15.01 | 1.46 | 1.6% | 93.8 |
| `scaled_dot_product_attention` | 14.71 | 14.72 | 14.62 | 1.00 | 1.5% | 360.4 |
| `causal_flash_attention` | 14.34 | 14.34 | 14.39 | 1.00 | 1.5% | 365.2 |
| `log_softmax` | 13.85 | 13.96 | 13.95 | 1.01 | 1.5% | 11.4 |
| `max_reduction` | 13.28 | 13.36 | 13.28 | 1.01 | 1.4% | 11.0 |
| `mean_reduction` | 12.93 | 12.96 | 12.94 | 1.00 | 1.4% | 12.3 |
| `swish` | 12.59 | 12.78 | 12.76 | 1.02 | 1.3% | 90.4 |
| `l2norm` | 12.29 | 12.32 | 12.33 | 1.00 | 1.3% | 11.8 |
| `avg_pool3d` | 12.25 | 12.41 | 12.33 | 1.01 | 1.3% | 12.0 |
| `max_pool3d` | 12.29 | 12.33 | 12.21 | 1.00 | 1.3% | 12.0 |
| `l1norm` | 12.07 | 12.18 | 12.16 | 1.01 | 1.3% | 11.8 |
| `avg_pool2d` | 12.04 | 12.06 | 12.11 | 1.00 | 1.3% | 12.1 |
| `max_pool2d` | 12.08 | 12.12 | 12.08 | 1.00 | 1.3% | 11.6 |
| `avg_pool1d` | 11.69 | 12.07 | 11.79 | 1.03 | 1.2% | 11.7 |
| `max_pool1d` | 11.58 | 11.85 | 11.61 | 1.02 | 1.2% | 11.4 |
| `argmax` | 10.65 | 10.90 | 10.75 | 1.02 | 1.1% | 11.3 |
| `argmin` | 10.47 | 10.61 | 10.50 | 1.01 | 1.1% | 10.8 |

## Concentration

| | checker | AutoKernel |
|---|---:|---:|
| top 1 operators | 25.1% | 54.0% |
| top 3 operators | 50.0% | 79.5% |
| top 5 operators | 63.3% | 86.0% |
| top 8 operators | 70.6% | 90.5% |
| p90/p50 | 3.03 | 27.75 |
| mean/p50 | 1.47 | 5.78 |

## Reading

The checker's cost is **broadly distributed**: p90/p50 = 3.03 and no operator
has a within-operator p90/p50 above ~1.5. Its most expensive operator,
`flash_attention`, is 2.5x the checker's own mean but *uniformly* so
(p50 51.75 / p90 59.90), which is a cost profile rather than a blowup.

AutoKernel's is **concentrated**: three attention operators are 79.5% of its
total time, and its p90/p50 of 27.75 says the mean describes almost none of
its operators. Cite AutoKernel's p50, not its mean, unless the attention
family is specifically in scope.
