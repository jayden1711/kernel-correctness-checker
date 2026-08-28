"""Real @triton.jit reference kernels for the 27 Phase-1 operators.

WHY THIS FILE EXISTS. The GPU-native methodology (GPU_NATIVE.md §1) takes the
directional derivative OF THE KERNEL THAT SHIPS -- not of a torch stand-in --
so validating the Phase-1 derivations natively requires real Triton kernels,
and TritonBench/reference/ had only the original 29.

Each host wrapper matches its verification/specs/<op>.py calling convention
exactly, so the same spec drives both paths.

DETERMINISM. The scalar-output losses use a TWO-STAGE reduction (per-row
partials, then one program summing them) rather than tl.atomic_add. That is
deliberate: frobenius_norm's atomic_add reduction is measurably non-bitwise-
deterministic (GPU_NATIVE.md §3a, 2 of 6 invocations differ by 1 ulp) because
atomics do not fix summation order. Two-stage is deterministic by construction
and costs one extra tiny launch.
"""
import math
import torch
import triton
import triton.language as tl


# ===========================================================================
# ELEMENTWISE ACTIVATIONS -- 9 kernels, diagonal Jacobian
# ===========================================================================

@triton.jit
def _relu_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    tl.store(O + o, tl.where(x > 0, x, 0.0), mask=m)


@triton.jit
def _leaky_relu_kernel(X, O, N, SLOPE, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    tl.store(O + o, tl.where(x > 0, x, SLOPE * x), mask=m)


@triton.jit
def _sigmoid_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    # numerically stable both tails
    p = tl.where(x >= 0, 1.0 / (1.0 + tl.exp(-x)), tl.exp(x) / (1.0 + tl.exp(x)))
    tl.store(O + o, p, mask=m)


@triton.jit
def _tanh_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    tl.store(O + o, tl.where(x >= 0, t, -t), mask=m)


@triton.jit
def _selu_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    A = 1.6732632423543772
    S = 1.0507009873554805
    tl.store(O + o, S * tl.where(x > 0, x, A * (tl.exp(x) - 1.0)), mask=m)


@triton.jit
def _elu_kernel(X, O, N, ALPHA, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    tl.store(O + o, tl.where(x > 0, x, ALPHA * (tl.exp(x) - 1.0)), mask=m)


@triton.jit
def _softplus_kernel(X, O, N, BETA, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    bx = BETA * x
    # linear-regime fallback above 20 (torch's own threshold), so exp cannot
    # overflow -- this is the bug the spec's saturating_pos variant hunts.
    y = tl.where(bx > 20.0, bx, tl.log(1.0 + tl.exp(tl.minimum(bx, 20.0)))) / BETA
    tl.store(O + o, y, mask=m)


@triton.jit
def _hardsigmoid_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    y = (x + 3.0) / 6.0
    tl.store(O + o, tl.minimum(tl.maximum(y, 0.0), 1.0), mask=m)


@triton.jit
def _new_gelu_kernel(X, O, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    C = 0.7978845608028654          # sqrt(2/pi)
    u = C * (x + 0.044715 * x * x * x)
    e = tl.exp(-2.0 * tl.abs(u))
    t = (1.0 - e) / (1.0 + e)
    t = tl.where(u >= 0, t, -t)
    tl.store(O + o, 0.5 * x * (1.0 + t), mask=m)


def _ew(kern, x, extra=None):
    x = x.contiguous()
    o = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    if extra is None:
        kern[grid](x, o, n, BLOCK=BLOCK)
    else:
        kern[grid](x, o, n, extra, BLOCK=BLOCK)
    return o


def relu(x):                     return _ew(_relu_kernel, x)
def leaky_relu(x, s=0.01):       return _ew(_leaky_relu_kernel, x, float(s))
def sigmoid(x):                  return _ew(_sigmoid_kernel, x)
def tanh(x):                     return _ew(_tanh_kernel, x)
def selu(x):                     return _ew(_selu_kernel, x)
def elu(x, a=1.0):               return _ew(_elu_kernel, x, float(a))
def softplus(x, b=1.0):          return _ew(_softplus_kernel, x, float(b))
def hardsigmoid(x):              return _ew(_hardsigmoid_kernel, x)
def new_gelu(x):                 return _ew(_new_gelu_kernel, x)


# ===========================================================================
# SCANS -- row-wise, one tile per row
# ===========================================================================

@triton.jit
def _cumsum_kernel(X, O, NC, MODE: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < NC
    x = tl.load(X + r * NC + c, mask=m, other=0.0)
    if MODE == 1:                       # reverse / suffix
        idx = NC - 1 - c
        xr = tl.load(X + r * NC + idx, mask=m, other=0.0)
        s = tl.cumsum(xr, axis=0)
        tl.store(O + r * NC + idx, s, mask=m)
    else:
        s = tl.cumsum(x, axis=0)
        if MODE == 2:                   # exclusive
            s = s - x
        tl.store(O + r * NC + c, s, mask=m)


@triton.jit
def _masked_cumsum_kernel(X, M_, O, NC, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < NC
    x = tl.load(X + r * NC + c, mask=m, other=0.0)
    k = tl.load(M_ + r * NC + c, mask=m, other=0.0)
    tl.store(O + r * NC + c, tl.cumsum(x * k, axis=0), mask=m)


def _scan(x, mode):
    x = x.contiguous()
    nr, nc = x.reshape(-1, x.shape[-1]).shape
    o = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(nc)
    _cumsum_kernel[(nr,)](x, o, nc, MODE=mode, BLOCK=BLOCK)
    return o


def cumsum(x):            return _scan(x, 0)
def cumsum_reverse(x):    return _scan(x, 1)
def cumsum_exclusive(x):  return _scan(x, 2)


def masked_cumsum(x, mask):
    x, mask = x.contiguous(), mask.contiguous()
    nr, nc = x.reshape(-1, x.shape[-1]).shape
    o = torch.empty_like(x)
    _masked_cumsum_kernel[(nr,)](x, mask, o, nc, BLOCK=triton.next_power_of_2(nc))
    return o


# ===========================================================================
# MATMUL VARIANTS
# ===========================================================================

@triton.jit
def _matvec_kernel(A, V, O, M, K, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        c = k0 + tl.arange(0, BLOCK)
        m = c < K
        a = tl.load(A + r * K + c, mask=m, other=0.0)
        v = tl.load(V + c, mask=m, other=0.0)
        acc += tl.sum(a * v, axis=0)
    tl.store(O + r, acc)


def matvec(A, v):
    A, v = A.contiguous(), v.contiguous()
    M, K = A.shape
    o = torch.empty(M, device=A.device, dtype=A.dtype)
    _matvec_kernel[(M,)](A, v, o, M, K, BLOCK=min(1024, triton.next_power_of_2(K)))
    return o


@triton.jit
def _bmm_kernel(A, B, O, M, K, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + b * M * K + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        bb = tl.load(B + b * K * N + rk[:, None] * N + rn[None, :],
                     mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, bb)
    tl.store(O + b * M * N + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def batched_matmul(A, B):
    A, B = A.contiguous(), B.contiguous()
    Bt, M, K = A.shape
    N = B.shape[2]
    o = torch.empty(Bt, M, N, device=A.device, dtype=A.dtype)
    BM = BN = BK = 32
    _bmm_kernel[(Bt, triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, o, M, K, N, BM=BM, BN=BN, BK=BK)
    return o


@triton.jit
def _diagmm_kernel(D, B, O, NR, NC, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    d = tl.load(D + r)
    for c0 in range(0, NC, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        m = c < NC
        b = tl.load(B + r * NC + c, mask=m, other=0.0)
        tl.store(O + r * NC + c, d * b, mask=m)


def diagonal_matmul(d, B):
    d, B = d.contiguous(), B.contiguous()
    NR, NC = B.shape
    o = torch.empty_like(B)
    _diagmm_kernel[(NR,)](d, B, o, NR, NC, BLOCK=min(1024, triton.next_power_of_2(NC)))
    return o


@triton.jit
def _trilmm_kernel(A, B, O, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, N, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * N + rk[None, :],
                    mask=(rm[:, None] < N) & (rk[None, :] < N), other=0.0)
        b = tl.load(B + rk[:, None] * N + rn[None, :],
                    mask=(rk[:, None] < N) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    keep = rm[:, None] >= rn[None, :]
    acc = tl.where(keep, acc, 0.0)
    tl.store(O + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < N) & (rn[None, :] < N))


def triangular_matmul(A, B):
    A, B = A.contiguous(), B.contiguous()
    N = A.shape[0]
    o = torch.empty_like(A)
    BM = BN = BK = 32
    _trilmm_kernel[(triton.cdiv(N, BM), triton.cdiv(N, BN))](A, B, o, N, BM=BM, BN=BN, BK=BK)
    return o


# ===========================================================================
# LOSSES -- two-stage deterministic reduction (see module docstring)
# ===========================================================================

@triton.jit
def _loss_partial_kernel(X, T, P, NC, KIND: tl.constexpr, BETA, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, NC, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        m = c < NC
        x = tl.load(X + r * NC + c, mask=m, other=0.0)
        t = tl.load(T + r * NC + c, mask=m, other=0.0)
        if KIND == 0:                                   # mse
            v = (x - t) * (x - t)
        elif KIND == 1:                                 # huber / smooth-l1
            d = x - t
            ad = tl.abs(d)
            v = tl.where(ad < BETA, 0.5 * d * d / BETA, ad - 0.5 * BETA)
        elif KIND == 2:                                 # bce, x is a probability
            lp = tl.maximum(tl.log(x), -100.0)
            l1 = tl.maximum(tl.log(1.0 - x), -100.0)
            v = -(t * lp + (1.0 - t) * l1)
        else:                                           # kldiv, x is log q
            lt = tl.where(t > 0, tl.log(tl.maximum(t, 1e-12)), 0.0)
            v = t * (lt - x)
        acc += tl.sum(tl.where(m, v, 0.0), axis=0)
    tl.store(P + r, acc)


@triton.jit
def _nll_partial_kernel(X, T, P, NC, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    t = tl.load(T + r)
    tl.store(P + r, -tl.load(X + r * NC + t))


@triton.jit
def _sum_partials_kernel(P, O, NR, SCALE, BLOCK: tl.constexpr):
    c = tl.arange(0, BLOCK)
    acc = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, NR, BLOCK):
        i = c0 + c
        m = i < NR
        acc += tl.sum(tl.load(P + i, mask=m, other=0.0), axis=0)
    tl.store(O, acc * SCALE)


def _loss(x, t, kind, beta=1.0, denom=None):
    x, t = x.contiguous(), t.contiguous()
    NR, NC = x.shape
    p = torch.empty(NR, device=x.device, dtype=torch.float32)
    _loss_partial_kernel[(NR,)](x, t, p, NC, KIND=kind, BETA=float(beta),
                                BLOCK=min(1024, triton.next_power_of_2(NC)))
    o = torch.empty((), device=x.device, dtype=torch.float32)
    scale = 1.0 / (denom if denom is not None else (NR * NC))
    _sum_partials_kernel[(1,)](p, o, NR, scale, BLOCK=1024)
    return o


def mse_loss(x, t):    return _loss(x, t, 0)
def huber_loss(x, t):  return _loss(x, t, 1, beta=1.0)
def bce_loss(x, t):    return _loss(x, t, 2)
def kldiv_loss(x, t):  return _loss(x, t, 3, denom=x.shape[0])   # batchmean


def nll_loss(logp, tgt):
    logp = logp.contiguous()
    tgt = tgt.contiguous().to(torch.int32)
    NR, NC = logp.shape
    p = torch.empty(NR, device=logp.device, dtype=torch.float32)
    _nll_partial_kernel[(NR,)](logp, tgt, p, NC, BLOCK=1)
    o = torch.empty((), device=logp.device, dtype=torch.float32)
    _sum_partials_kernel[(1,)](p, o, NR, 1.0 / NR, BLOCK=1024)
    return o


# ===========================================================================
# ROPE / SWIGLU / LOGSUMEXP / STD / VAR
# ===========================================================================

@triton.jit
def _rope_kernel(X, C, S, O, W, H, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < H
    x1 = tl.load(X + r * W + c, mask=m, other=0.0)
    x2 = tl.load(X + r * W + H + c, mask=m, other=0.0)
    co = tl.load(C + r * H + c, mask=m, other=0.0)
    si = tl.load(S + r * H + c, mask=m, other=0.0)
    tl.store(O + r * W + c,      x1 * co - x2 * si, mask=m)
    tl.store(O + r * W + H + c,  x1 * si + x2 * co, mask=m)


def rope(x, cos, sin):
    x, cos, sin = x.contiguous(), cos.contiguous(), sin.contiguous()
    R, W = x.shape
    H = W // 2
    o = torch.empty_like(x)
    _rope_kernel[(R,)](x, cos, sin, o, W, H, BLOCK=triton.next_power_of_2(H))
    return o


@triton.jit
def _swiglu_kernel(X, O, W, H, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < H
    a = tl.load(X + r * W + c, mask=m, other=0.0)
    b = tl.load(X + r * W + H + c, mask=m, other=0.0)
    s = tl.where(a >= 0, 1.0 / (1.0 + tl.exp(-a)), tl.exp(a) / (1.0 + tl.exp(a)))
    tl.store(O + r * H + c, (a * s) * b, mask=m)


def swiglu(x):
    x = x.contiguous()
    R, W = x.shape
    H = W // 2
    o = torch.empty(R, H, device=x.device, dtype=x.dtype)
    _swiglu_kernel[(R,)](x, o, W, H, BLOCK=triton.next_power_of_2(H))
    return o


@triton.jit
def _logsumexp_kernel(X, O, NC, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < NC
    x = tl.load(X + r * NC + c, mask=m, other=-float("inf"))
    mx = tl.max(x, axis=0)
    s = tl.sum(tl.where(m, tl.exp(x - mx), 0.0), axis=0)
    tl.store(O + r, mx + tl.log(s))


def logsumexp(x):
    x = x.contiguous()
    NR, NC = x.shape
    o = torch.empty(NR, device=x.device, dtype=x.dtype)
    _logsumexp_kernel[(NR,)](x, o, NC, BLOCK=triton.next_power_of_2(NC))
    return o


@triton.jit
def _stdvar_kernel(X, O, NC, IS_STD: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < NC
    x = tl.load(X + r * NC + c, mask=m, other=0.0)
    mu = tl.sum(tl.where(m, x, 0.0), axis=0) / NC
    d = tl.where(m, x - mu, 0.0)
    v = tl.sum(d * d, axis=0) / (NC - 1)
    tl.store(O + r, tl.sqrt(v) if IS_STD else v)


def _sv(x, is_std):
    x = x.contiguous()
    NR, NC = x.shape
    o = torch.empty(NR, device=x.device, dtype=x.dtype)
    _stdvar_kernel[(NR,)](x, o, NC, IS_STD=is_std, BLOCK=triton.next_power_of_2(NC))
    return o


def std_reduction(x):  return _sv(x, True)
def var_reduction(x):  return _sv(x, False)


KERNELS = {
    "relu": relu, "leaky_relu": leaky_relu, "sigmoid": sigmoid, "tanh": tanh,
    "selu": selu, "elu": elu, "softplus": softplus, "hardsigmoid": hardsigmoid,
    "new_gelu": new_gelu,
    "cumsum": cumsum, "cumsum_reverse": cumsum_reverse,
    "cumsum_exclusive": cumsum_exclusive, "masked_cumsum": masked_cumsum,
    "matvec": matvec, "batched_matmul": batched_matmul,
    "diagonal_matmul": diagonal_matmul, "triangular_matmul": triangular_matmul,
    "mse_loss": mse_loss, "huber_loss": huber_loss, "bce_loss": bce_loss,
    "kldiv_loss": kldiv_loss, "nll_loss": nll_loss,
    "rope": rope, "swiglu": swiglu, "logsumexp": logsumexp,
    "std_reduction": std_reduction, "var_reduction": var_reduction,
}
