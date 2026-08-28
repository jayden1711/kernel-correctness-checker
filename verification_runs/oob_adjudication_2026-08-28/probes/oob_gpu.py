"""
OOB adjudication, GPU stage. Runs on the Colab VM (/content).

Question (NORM_ADJUDICATION standard): when the layernorm/rmsnorm
`non_power_of_two` adversarial variant feeds a width-333 x with the captured
length-128 gamma/beta, is the resulting record a kernel bug or an invalid
input the kernel was never defined on?

Stages:
  A  POISON RECOVERY. Allocate gamma(128), beta(128), then poison blocks, in
     tight sequence so the CUDA caching allocator packs them into one
     segment (each 128-float tensor occupies exactly one 512-byte block).
     Run the reference kernel at x(64,333) and RECOVER the effective
     per-column (gamma_j, beta_j) the kernel actually used, by least squares
     against the kernel-faithful normalized values. Prediction if the loads
     are out of bounds and the segment is packed: recovered gamma[128:256]
     == beta's storage, recovered gamma[256:333] == poison A's values,
     recovered beta[128:256] == poison A, beta[256:333] == poison B.
     Columns < 128 must recover the true gamma/beta (method validation).
  B  IN-BOUNDS EMULATION. Same x with length-512 and length-333 companions:
     kernel output must equal the kernel-faithful float64 emulation
     (layernorm: unmasked pad-lane variance, the documented bug; rmsnorm:
     exact) using gamma[:333] -- i.e., with valid-length companions the
     variant is a well-defined test and the kernel computes exactly its own
     arithmetic. Also reports kernel vs IDEAL (masked-variance) math to
     restate the known, separately-tracked variance-bug delta.
  C  STABILITY / NEIGHBOR DEPENDENCE. Re-run with tensors held: bitwise
     identical. Free the poison, allocate a different-valued block of the
     same size, re-run: the leaked columns must CHANGE while columns < 128
     stay bit-identical -- the output depends on unrelated allocations,
     which no function of the op's inputs can.
  D  EXPOSURE TABLE. For base widths {64, 128, 256, 333, 512}: whether the
     variant reads out of bounds and how many columns leak.

Writes /content/oob_gpu.json with every recovered array and comparison.
"""

import json
import math

import torch

torch.manual_seed(0)
assert torch.cuda.is_available()
dev = "cuda"

import sys
sys.path.insert(0, "/content")
from TritonBench.reference.layernorm import layernorm
from TritonBench.reference.rmsnorm import rmsnorm

EPS = 1e-5
OUT = {}


def kernel_faithful_z_layernorm(x64, width_block):
    """Normalized values under the kernel's actual arithmetic: mean over
    n_cols (pads contribute 0 to the sum), variance with the UNMASKED
    pad-lane term (pads contribute mean^2 each)."""
    n = x64.shape[1]
    pad = width_block - n
    mean = x64.sum(dim=1, keepdim=True) / n
    var = (((x64 - mean) ** 2).sum(dim=1, keepdim=True)
           + pad * mean ** 2) / n
    return (x64 - mean) / torch.sqrt(var + EPS)


def recover_affine(y64, z64):
    """Per-column least squares y = z*g + b over rows."""
    zbar = z64.mean(dim=0)
    ybar = y64.mean(dim=0)
    cov = ((z64 - zbar) * (y64 - ybar)).mean(dim=0)
    var = ((z64 - zbar) ** 2).mean(dim=0)
    g = cov / var
    b = ybar - g * zbar
    return g, b


def block512(n_floats, fill=None):
    """A tensor occupying exactly ceil(n_floats/128) 512-byte blocks."""
    t = (torch.full((n_floats,), fill, device=dev)
         if fill is not None else torch.randn(n_floats, device=dev))
    return t


def match(a, b, tol=1e-3):
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).abs().max()), bool(torch.allclose(a, b, atol=tol, rtol=tol))


# ---------------------------------------------------------------- stage A --
x = torch.randn(64, 333, device=dev)
x64 = x.double()

# Warm the allocator with throwaway blocks so subsequent small allocations
# land in a fresh packed region, then allocate in tight sequence.
_warm = [torch.randn(4096, device=dev) for _ in range(4)]
del _warm
gamma = block512(128)
beta = block512(128)
poisonA = block512(128, 111111.0)
poisonB = block512(128, 222222.0)
poisonC = block512(128, 333333.0)

NAMED = {"gamma": gamma, "beta": beta,
         "poisonA": poisonA, "poisonB": poisonB, "poisonC": poisonC}


def predict_from_addresses(base_ptr, n_cols, tensors):
    """For each column j, the float that lives at base_ptr + 4j, looked up
    legally through whichever named tensor's storage contains that address.
    No packing assumption: pure pointer arithmetic on actual data_ptr()s."""
    pred = torch.full((n_cols,), float("nan"))
    src = ["?"] * n_cols
    for name, t in tensors.items():
        lo = t.data_ptr()
        hi = lo + t.numel() * 4
        for j in range(n_cols):
            a = base_ptr + 4 * j
            if lo <= a < hi:
                pred[j] = float(t.view(-1)[(a - lo) // 4])
                src[j] = name
    return pred, src


y_ln = layernorm(x, gamma, beta)
z = kernel_faithful_z_layernorm(x64, 512)
g_rec, b_rec = recover_affine(y_ln.double(), z)

pred_g, src_g = predict_from_addresses(gamma.data_ptr(), 333, NAMED)
pred_b, src_b = predict_from_addresses(beta.data_ptr(), 333, NAMED)


def score_leak(rec, pred, src):
    """Compare recovered values against address-predicted values on the OOB
    columns whose addresses fall inside a named tensor."""
    idx = [j for j in range(128, 333) if src[j] != "?"]
    if not idx:
        return dict(n_mapped=0)
    r = rec.detach().double().cpu()[idx]
    p = pred.double()[idx]
    rel = ((r - p).abs() / p.abs().clamp_min(1e-12))
    return dict(n_mapped=len(idx), n_unmapped=205 - len(idx),
                max_rel_err=float(rel.max()),
                agree=bool((rel < 1e-3).all()),
                sources=sorted(set(src[j] for j in idx)))


in_g = match(g_rec[:128], gamma)
in_b = match(b_rec[:128], beta)
leak_g = score_leak(g_rec, pred_g, src_g)
leak_b = score_leak(b_rec, pred_b, src_b)
OUT["A_layernorm"] = dict(
    recovered_inbounds_gamma_maxerr=in_g, recovered_inbounds_beta_maxerr=in_b,
    gamma_leak=leak_g, beta_leak=leak_b,
    addr_gaps=dict(beta_minus_gamma=beta.data_ptr() - gamma.data_ptr(),
                   poisA_minus_beta=poisonA.data_ptr() - beta.data_ptr()),
    g_rec_sample=[float(v) for v in g_rec[126:134]],
)
print("A layernorm: in-bounds recovery err g/b:",
      f"{in_g[0]:.2e}/{in_b[0]:.2e}",
      "| gamma leak:", leak_g, "| beta leak:", leak_b, flush=True)

# rmsnorm: fresh tight sequence
gamma_r = block512(128)
poisonR = block512(128, 444444.0)
poisonR2 = block512(128, 555555.0)
y_rm = rmsnorm(x, gamma_r)
rms = torch.sqrt((x64 ** 2).sum(dim=1, keepdim=True) / 333 + EPS)
g_eff = (y_rm.double() * rms / x64).median(dim=0).values
pred_r, src_r = predict_from_addresses(
    gamma_r.data_ptr(), 333,
    {"gamma_r": gamma_r, "poisonR": poisonR, "poisonR2": poisonR2})
r_in = match(g_eff[:128], gamma_r)
leak_r = score_leak(g_eff, pred_r, src_r)
OUT["A_rmsnorm"] = dict(inbounds_maxerr=r_in, leak=leak_r,
                        g_eff_sample=[float(v) for v in g_eff[126:134]])
print("A rmsnorm: in-bounds recovery err:", f"{r_in[0]:.2e}",
      "| leak:", leak_r, flush=True)

# ---------------------------------------------------------------- stage B --
res_b = {}
for width in (512, 333):
    g_full = torch.randn(width, device=dev)
    b_full = torch.randn(width, device=dev)
    y = layernorm(x, g_full, b_full).double()
    z = kernel_faithful_z_layernorm(x64, 512)
    want_kf = z * g_full.double()[:333] + b_full.double()[:333]
    # ideal (masked-variance) math
    mean = x64.mean(dim=1, keepdim=True)
    var = ((x64 - mean) ** 2).mean(dim=1, keepdim=True)
    want_ideal = ((x64 - mean) / torch.sqrt(var + EPS)) * g_full.double()[:333] \
        + b_full.double()[:333]
    d_kf = float((y - want_kf).abs().max())
    d_ideal = float((y - want_ideal).abs().max())
    y_r = rmsnorm(x, g_full).double()
    want_r = x64 / rms * g_full.double()[:333]
    res_b[width] = dict(layernorm_vs_kernel_faithful=d_kf,
                        layernorm_vs_ideal=d_ideal,
                        rmsnorm_vs_math=float((y_r - want_r).abs().max()))
    print(f"B width={width}: ln vs kernel-faithful {d_kf:.2e}, "
          f"ln vs ideal {d_ideal:.2e}, rms vs math "
          f"{res_b[width]['rmsnorm_vs_math']:.2e}", flush=True)
OUT["B_inbounds"] = res_b

# ---------------------------------------------------------------- stage C --
y1 = layernorm(x, gamma, beta)
y2 = layernorm(x, gamma, beta)
identical = bool(torch.equal(y1, y2))
# replace poisonA with different contents at the same allocation.
# NAMED holds a live reference; drop it or the block is never freed.
addrA = poisonA.data_ptr()
NAMED.clear()
del poisonA
newA = block512(128, 999999.0)
same_slot = (newA.data_ptr() == addrA)
y3 = layernorm(x, gamma, beta)
inb_same = bool(torch.equal(y1[:, :128], y3[:, :128]))
leak_changed = bool(not torch.equal(y1[:, 128:], y3[:, 128:]))
OUT["C_stability"] = dict(rerun_bitwise_identical=identical,
                          poison_realloc_same_slot=same_slot,
                          inbounds_cols_unchanged=inb_same,
                          leaked_cols_changed=leak_changed)
print("C: rerun identical:", identical, "| realloc same slot:", same_slot,
      "| cols<128 unchanged:", inb_same, "| cols>=128 changed:",
      leak_changed, flush=True)

# ---------------------------------------------------------------- stage D --
OUT["D_exposure"] = {
    w: dict(oob=(w < 333), leaked_cols=max(0, 333 - w))
    for w in (64, 128, 256, 333, 512)
}
print("D:", OUT["D_exposure"], flush=True)

json.dump(OUT, open("/content/oob_gpu.json", "w"), indent=1)
print("wrote /content/oob_gpu.json")
