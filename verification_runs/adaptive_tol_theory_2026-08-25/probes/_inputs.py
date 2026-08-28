def _mk_single(rng):
    return (rng.normal(size=(64, 128)).astype(np.float32),)

def _mk_triple(shape_x, shape_w):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        w1 = rng.normal(size=shape_w).astype(np.float32)
        w2 = rng.normal(size=shape_w).astype(np.float32)
        return (x, w1, w2)
    return _mk

def _mk_rmsnorm(rng):
    x = rng.normal(size=(64, 128)).astype(np.float32)
    gamma = rng.normal(size=(128,)).astype(np.float32)
    return (x, gamma)

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

def _mk_groupnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    num_groups = 2
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, num_groups, weight, bias)

def _mk_batchnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    running_mean = rng.normal(size=(C,)).astype(np.float32)
    running_var = rng.uniform(0.5, 2.0, size=(C,)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, running_mean, running_var, weight, bias)

def _mk_cross_entropy(rng):
    n_rows, n_cols = 64, 32
    logits = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    targets = rng.integers(0, n_cols, size=(n_rows,)).astype(np.int64)
    return (logits, targets)

def _mk_pool(shape_x, kernel_size, stride, padding):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        return (x, kernel_size, stride, padding)
    return _mk

FAMILIES = {
    "single":        _mk_single,
    "layernorm":     _mk_triple((64, 128), (128,)),
    "instancenorm":  _mk_triple((2, 4, 4, 4), (4,)),
    "rmsnorm":       _mk_rmsnorm,
    "matmul":        _mk_matmul,
    "attention":     _mk_attention,
    "groupnorm":     _mk_groupnorm,
    "batchnorm":     _mk_batchnorm,
    "cross_entropy": _mk_cross_entropy,
    "pool1d":        _mk_pool((2, 3, 32), 4, 4, 0),
    "pool2d":        _mk_pool((2, 3, 16, 16), 4, 4, 0),
    "pool3d":        _mk_pool((2, 3, 8, 8, 8), 2, 2, 0),
}

OPS = [
    ("argmax", "single", ["tiebreak"]),
    ("argmin", "single", ["tiebreak"]),
    ("avg_pool1d", "pool1d", ["wrong_divisor"]),
    ("avg_pool2d", "pool2d", ["wrong_divisor"]),
    ("avg_pool3d", "pool3d", ["wrong_divisor"]),
    ("batchnorm", "batchnorm", ["wrong_running_stats_broadcast"]),
    ("causal_flash_attention", "attention", ["wrong_causal_mask"]),
    ("cross_entropy", "cross_entropy", ["missing_max_subtraction"]),
    ("flash_attention", "attention", ["approx_denom", "drop_last_tile",
                                      "skip_rescaling", "wrong_mask"]),
    ("frobenius_norm", "single", ["wrong_norm"]),
    ("gelu", "single", ["sigmoid_approx"]),
    ("groupnorm", "groupnorm", ["ignore_affine"]),
    ("instancenorm", "instancenorm", ["skip_eps"]),
    ("l1norm", "single", ["partial_reduction"]),
    ("l2norm", "single", ["wrong_norm"]),
    ("layernorm", "layernorm", ["ignore_gamma_beta", "skip_mean_subtract",
                                "wrong_variance_estimate"]),
    ("log_softmax", "single", ["skip_max_subtraction"]),
    ("matmul", "matmul", ["partial_k_reduct", "skip_boundary_tiles",
                          "swapped_strides", "wrong_dtype"]),
    ("max_pool1d", "pool1d", ["wrong_padding"]),
    ("max_pool2d", "pool2d", ["wrong_padding"]),
    ("max_pool3d", "pool3d", ["wrong_padding"]),
    ("max_reduction", "single", ["wrong_padding"]),
    ("mean_reduction", "single", ["partial_reduction"]),
    ("min_reduction", "single", ["wrong_padding"]),
    ("rmsnorm", "rmsnorm", ["ignore_gamma", "partial_reduction", "wrong_norm"]),
    ("scaled_dot_product_attention", "attention", ["wrong_mask"]),
    ("softmax", "single", ["first_tile", "wrong_reduction"]),
    ("sum_reduction", "single", ["partial_reduction"]),
    ("swish", "single", ["linear_sigmoid_approx"]),
]

