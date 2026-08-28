"""Real @triton.jit convolution kernels for Phase 2 -- 8 forms.

DIRECT convolution, deliberately. These are REFERENCE kernels whose job is to
be obviously correct and to be the thing the directional-derivative probe
differentiates; they are not tuned for speed. An im2col+dot formulation would
be faster and would also put a second, separately-fallible transformation
between the operator and its measured Jacobian.

Layout: one program per (batch * out_channel, block of flattened output
spatial positions). Taps are looped at runtime, so one kernel body covers every
(stride, padding, dilation, groups, asymmetric-kernel) combination rather than
specialising -- which is what makes 35 KernelBench problems reduce to 8 kernels.

TRANSPOSED forms are implemented as GATHERS, not scatters. For output position
o and tap k, the contributing input index is i = (o + p - k*d) / s, which
contributes only when that division is exact and i is in range. Written this
way there are no atomics and no write conflicts, so the kernels are bitwise
deterministic -- the property frobenius_norm's atomic_add reduction lacks
(GPU_NATIVE.md 3a).
"""
import torch
import triton
import triton.language as tl


# ===========================================================================
# 1-D
# ===========================================================================

@triton.jit
def _conv1d_kernel(X, W, O, N, CIN, COUT, LIN, LOUT, K,
                   STRIDE, PAD, DIL, GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    ol = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = ol < LOUT
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for k in range(K):
            il = ol * STRIDE - PAD + k * DIL
            ok = m & (il >= 0) & (il < LIN)
            xv = tl.load(X + (n * CIN + cin) * LIN + il, mask=ok, other=0.0)
            wv = tl.load(W + (co * GCIN + ci) * K + k)
            acc += xv * wv
    tl.store(O + (n * COUT + co) * LOUT + ol, acc, mask=m)


@triton.jit
def _convT1d_kernel(X, W, O, N, CIN, COUT, LIN, LOUT, K,
                    STRIDE, PAD, DIL, GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    ol = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = ol < LOUT
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for k in range(K):
            num = ol + PAD - k * DIL
            il = num // STRIDE
            ok = m & (num >= 0) & (num % STRIDE == 0) & (il >= 0) & (il < LIN)
            xv = tl.load(X + (n * CIN + cin) * LIN + il, mask=ok, other=0.0)
            wv = tl.load(W + (cin * GCOUT + (co % GCOUT)) * K + k)
            acc += xv * wv
    tl.store(O + (n * COUT + co) * LOUT + ol, acc, mask=m)


# ===========================================================================
# 2-D
# ===========================================================================

@triton.jit
def _conv2d_kernel(X, W, O, N, CIN, COUT, IH, IW, OH, OW, KH, KW,
                   SH, SW, PH, PW, DH, DW, GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = idx < OH * OW
    oh = idx // OW
    ow = idx % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for kh in range(KH):
            ih = oh * SH - PH + kh * DH
            for kw in range(KW):
                iw = ow * SW - PW + kw * DW
                ok = m & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                xv = tl.load(X + ((n * CIN + cin) * IH + ih) * IW + iw,
                             mask=ok, other=0.0)
                wv = tl.load(W + ((co * GCIN + ci) * KH + kh) * KW + kw)
                acc += xv * wv
    tl.store(O + ((n * COUT + co) * OH) * OW + idx, acc, mask=m)


@triton.jit
def _convT2d_kernel(X, W, O, N, CIN, COUT, IH, IW, OH, OW, KH, KW,
                    SH, SW, PH, PW, DH, DW, GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = idx < OH * OW
    oh = idx // OW
    ow = idx % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for kh in range(KH):
            nh = oh + PH - kh * DH
            ih = nh // SH
            okh = (nh >= 0) & (nh % SH == 0) & (ih >= 0) & (ih < IH)
            for kw in range(KW):
                nw = ow + PW - kw * DW
                iw = nw // SW
                ok = m & okh & (nw >= 0) & (nw % SW == 0) & (iw >= 0) & (iw < IW)
                xv = tl.load(X + ((n * CIN + cin) * IH + ih) * IW + iw,
                             mask=ok, other=0.0)
                wv = tl.load(W + ((cin * GCOUT + (co % GCOUT)) * KH + kh) * KW + kw)
                acc += xv * wv
    tl.store(O + ((n * COUT + co) * OH) * OW + idx, acc, mask=m)


# ===========================================================================
# 3-D
# ===========================================================================

@triton.jit
def _conv3d_kernel(X, W, O, N, CIN, COUT, ID, IH, IW, OD, OH, OW,
                   KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW,
                   GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = idx < OD * OH * OW
    od = idx // (OH * OW)
    rem = idx % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for kd in range(KD):
            id_ = od * SD - PD + kd * DD
            okd = (id_ >= 0) & (id_ < ID)
            for kh in range(KH):
                ih = oh * SH - PH + kh * DH
                okh = okd & (ih >= 0) & (ih < IH)
                for kw in range(KW):
                    iw = ow * SW - PW + kw * DW
                    ok = m & okh & (iw >= 0) & (iw < IW)
                    xv = tl.load(X + (((n * CIN + cin) * ID + id_) * IH + ih) * IW + iw,
                                 mask=ok, other=0.0)
                    wv = tl.load(W + ((((co * GCIN + ci) * KD + kd) * KH + kh) * KW + kw))
                    acc += xv * wv
    tl.store(O + (n * COUT + co) * (OD * OH * OW) + idx, acc, mask=m)


@triton.jit
def _convT3d_kernel(X, W, O, N, CIN, COUT, ID, IH, IW, OD, OH, OW,
                    KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW,
                    GCIN, GCOUT, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    n = pid_nc // COUT
    co = pid_nc % COUT
    g = co // GCOUT
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = idx < OD * OH * OW
    od = idx // (OH * OW)
    rem = idx % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(GCIN):
        cin = g * GCIN + ci
        for kd in range(KD):
            nd_ = od + PD - kd * DD
            id_ = nd_ // SD
            okd = (nd_ >= 0) & (nd_ % SD == 0) & (id_ >= 0) & (id_ < ID)
            for kh in range(KH):
                nh = oh + PH - kh * DH
                ih = nh // SH
                okh = okd & (nh >= 0) & (nh % SH == 0) & (ih >= 0) & (ih < IH)
                for kw in range(KW):
                    nw = ow + PW - kw * DW
                    iw = nw // SW
                    ok = m & okh & (nw >= 0) & (nw % SW == 0) & (iw >= 0) & (iw < IW)
                    xv = tl.load(X + (((n * CIN + cin) * ID + id_) * IH + ih) * IW + iw,
                                 mask=ok, other=0.0)
                    wv = tl.load(W + ((((cin * GCOUT + (co % GCOUT)) * KD + kd) * KH + kh) * KW + kw))
                    acc += xv * wv
    tl.store(O + (n * COUT + co) * (OD * OH * OW) + idx, acc, mask=m)


# ===========================================================================
# Host wrappers -- signatures match verification/specs/<op>.py
# ===========================================================================

def _p(v, n):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n


def _oshape(I, k, s, p, d):
    return (I + 2 * p - d * (k - 1) - 1) // s + 1


def _oshapeT(I, k, s, p, d, op=0):
    return (I - 1) * s - 2 * p + d * (k - 1) + 1 + op


def conv1d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, LIN = x.shape
    COUT, GCIN, K = W.shape
    LOUT = _oshape(LIN, K, stride, padding, dilation)
    o = torch.empty(N, COUT, LOUT, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(LOUT))
    _conv1d_kernel[(N * COUT, triton.cdiv(LOUT, BLOCK))](
        x, W, o, N, CIN, COUT, LIN, LOUT, K, stride, padding, dilation,
        GCIN, COUT // groups, BLOCK=BLOCK)
    return o


def conv_transpose1d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, LIN = x.shape
    _, GCOUT, K = W.shape
    COUT = GCOUT * groups
    LOUT = _oshapeT(LIN, K, stride, padding, dilation)
    o = torch.empty(N, COUT, LOUT, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(LOUT))
    _convT1d_kernel[(N * COUT, triton.cdiv(LOUT, BLOCK))](
        x, W, o, N, CIN, COUT, LIN, LOUT, K, stride, padding, dilation,
        CIN // groups, GCOUT, BLOCK=BLOCK)
    return o


def conv2d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, IH, IW = x.shape
    COUT, GCIN, KH, KW = W.shape
    SH, SW = _p(stride, 2); PH, PW = _p(padding, 2); DH, DW = _p(dilation, 2)
    OH = _oshape(IH, KH, SH, PH, DH); OW = _oshape(IW, KW, SW, PW, DW)
    o = torch.empty(N, COUT, OH, OW, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(OH * OW))
    _conv2d_kernel[(N * COUT, triton.cdiv(OH * OW, BLOCK))](
        x, W, o, N, CIN, COUT, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW,
        GCIN, COUT // groups, BLOCK=BLOCK)
    return o


def conv_transpose2d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, IH, IW = x.shape
    _, GCOUT, KH, KW = W.shape
    COUT = GCOUT * groups
    SH, SW = _p(stride, 2); PH, PW = _p(padding, 2); DH, DW = _p(dilation, 2)
    OH = _oshapeT(IH, KH, SH, PH, DH); OW = _oshapeT(IW, KW, SW, PW, DW)
    o = torch.empty(N, COUT, OH, OW, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(OH * OW))
    _convT2d_kernel[(N * COUT, triton.cdiv(OH * OW, BLOCK))](
        x, W, o, N, CIN, COUT, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW,
        CIN // groups, GCOUT, BLOCK=BLOCK)
    return o


def conv3d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, ID, IH, IW = x.shape
    COUT, GCIN, KD, KH, KW = W.shape
    SD, SH, SW = _p(stride, 3); PD, PH, PW = _p(padding, 3); DD, DH, DW = _p(dilation, 3)
    OD = _oshape(ID, KD, SD, PD, DD); OH = _oshape(IH, KH, SH, PH, DH); OW = _oshape(IW, KW, SW, PW, DW)
    o = torch.empty(N, COUT, OD, OH, OW, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(OD * OH * OW))
    _conv3d_kernel[(N * COUT, triton.cdiv(OD * OH * OW, BLOCK))](
        x, W, o, N, CIN, COUT, ID, IH, IW, OD, OH, OW, KD, KH, KW,
        SD, SH, SW, PD, PH, PW, DD, DH, DW, GCIN, COUT // groups, BLOCK=BLOCK)
    return o


def conv_transpose3d(x, W, stride=1, padding=0, dilation=1, groups=1):
    x, W = x.contiguous(), W.contiguous()
    N, CIN, ID, IH, IW = x.shape
    _, GCOUT, KD, KH, KW = W.shape
    COUT = GCOUT * groups
    SD, SH, SW = _p(stride, 3); PD, PH, PW = _p(padding, 3); DD, DH, DW = _p(dilation, 3)
    OD = _oshapeT(ID, KD, SD, PD, DD); OH = _oshapeT(IH, KH, SH, PH, DH); OW = _oshapeT(IW, KW, SW, PW, DW)
    o = torch.empty(N, COUT, OD, OH, OW, device=x.device, dtype=x.dtype)
    BLOCK = min(256, triton.next_power_of_2(OD * OH * OW))
    _convT3d_kernel[(N * COUT, triton.cdiv(OD * OH * OW, BLOCK))](
        x, W, o, N, CIN, COUT, ID, IH, IW, OD, OH, OW, KD, KH, KW,
        SD, SH, SW, PD, PH, PW, DD, DH, DW, CIN // groups, GCOUT, BLOCK=BLOCK)
    return o


def depthwise_conv2d(x, W, stride=1, padding=1, dilation=1):
    """groups == C_in; W is (C, 1, KH, KW)."""
    return conv2d(x, W, stride, padding, dilation, groups=x.shape[1])


def pointwise_conv2d(x, W):
    """1x1 conv; W is (C_out, C_in, 1, 1)."""
    return conv2d(x, W, 1, 0, 1, 1)


KERNELS = {
    "conv1d": conv1d, "conv2d": conv2d, "conv3d": conv3d,
    "conv_transpose1d": conv_transpose1d,
    "conv_transpose2d": conv_transpose2d,
    "conv_transpose3d": conv_transpose3d,
    "depthwise_conv2d": depthwise_conv2d,
    "pointwise_conv2d": pointwise_conv2d,
}
