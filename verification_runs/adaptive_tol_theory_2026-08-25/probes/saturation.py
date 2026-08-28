"""
Direct confirmation that the attention-family ceiling violations are softmax
saturation -- i.e. the argmax mechanism, not a new failure mode.

For each attention input, report:
  * peak attention weight  max_ij softmax(QK^T/sqrt(D))_ij   (1.0 = hard select)
  * the C1 relative error   | s - ||J d||_inf | / s          (jvp)
  * the sensitivity CV
"""
import math, statistics as st
import torch
from torch.func import jvp

DT = torch.float32
torch.manual_seed(0)


def attn(Q, K, V, causal=False):
    S = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
    if causal:
        N = Q.shape[0]
        i = torch.arange(N).unsqueeze(1); j = torch.arange(N).unsqueeze(0)
        S = S.masked_fill(j > i, float("-inf"))
    return torch.softmax(S, -1) @ V


def peak_weight(Q, K, causal=False):
    S = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
    if causal:
        N = Q.shape[0]
        i = torch.arange(N).unsqueeze(1); j = torch.arange(N).unsqueeze(0)
        S = S.masked_fill(j > i, float("-inf"))
    return torch.softmax(S, -1).max().item()


def _make_qkv(N, D):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(N, D, generator=g), torch.randn(N, D, generator=g),
            torch.randn(N, D, generator=g))


def multi_tile_rescaling(N=192, D=64):
    B = 32
    Q, K, V = _make_qkv(N, D)
    K = K.clone()
    K[:B, :] *= 1e-6
    K[B:2 * B, :] *= 1.0
    K[2 * B:, :] *= 1e4
    return Q, K, V


CASES = [
    ("flash_attention / primary",              _make_qkv(64, 32),          False),
    ("flash_attention / multi_tile_rescaling", multi_tile_rescaling(),     False),
    ("causal / primary",                       _make_qkv(64, 32),          True),
    ("causal / large_magnitude_qk (x20)",
     tuple(t * 20 if i < 2 else t for i, t in enumerate(_make_qkv(64, 32))), True),
    ("sdpa / large_magnitude_qk (x20)",
     tuple(t * 20 if i < 2 else t for i, t in enumerate(_make_qkv(64, 32))), False),
]

print("%-40s %10s %11s %9s" % ("input", "peak wgt", "C1 relerr", "CV"))
print("-" * 74)
for name, (Q, K, V), causal in CASES:
    f = lambda q: attn(q, K, V, causal)
    base = f(Q)
    sig = 1e-3 * Q.float().std().item()
    g = torch.Generator().manual_seed(1)
    sens, rels = [], []
    for k in range(40):
        d = torch.randn(Q.shape, generator=g, dtype=DT) * sig
        sa = (f(Q + d) - base).abs().max().item()
        sens.append(sa)
        if k < 12:
            _, jd = jvp(f, (Q,), (d,))
            sl = jd.abs().max().item()
            if sa > 0:
                rels.append(abs(sa - sl) / sa)
    cv = st.stdev(sens) / st.fmean(sens) if st.fmean(sens) > 0 else float('nan')
    print("%-40s %10.6f %10.2f%% %9.4f"
          % (name, peak_weight(Q, K, causal),
             100 * (st.median(rels) if rels else float('nan')), cv))

print()
print("Reference: half-normal ceiling CV = %.4f;  a C1 relative error of a few")
print("hundredths of a percent is what the 27 in-scope operators show.")
print("Peak weight -> 1.0 means softmax has collapsed to a hard select, i.e.")
print("the operator has become argmax-like and its Jacobian is ~0 a.e.")
