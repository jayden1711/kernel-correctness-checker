"""
A GPU-side test of assumption (A3) that needs no GPU.

Under linearisation the sensitivity is s = ||J d||_inf = max_i |Z_i| with Z
jointly Gaussian, centred.  CLAIM: for ANY centred joint Gaussian in R^m,

        CV( max_i |Z_i| )  <=  CV(|N(0,1)|)  =  sqrt(pi/2 - 1) = 0.7555

i.e. the half-normal (m_eff = 1) case is the worst case.  Verified below by
Monte Carlo over random covariance structures.  Any operator whose measured
sensitivity CV materially exceeds that ceiling therefore CANNOT be in the
linear regime -- a falsification test applicable directly to the banked
Triton-on-T4 vectors, with no jvp and no GPU.

Sampling error: with n = 40 draws, SE(CV) ~ CV*sqrt((1+2 CV^2)/(2n)) ~ 0.12 at
CV = 0.76, so the 2-sigma decision threshold is about 1.0.
"""
import gzip, json, math, statistics as st
import torch

CEIL = math.sqrt(math.pi / 2 - 1)
print("half-normal ceiling CV = %.4f" % CEIL)

# ---- verify the ceiling numerically ---------------------------------------
torch.manual_seed(0)
worst = 0.0; worst_cfg = None
for trial in range(400):
    m = int(torch.randint(1, 60, (1,)).item())
    r = int(torch.randint(1, 8, (1,)).item())          # rank of the structure
    A = torch.randn(m, r)
    if trial % 4 == 0:                                  # wildly unequal scales
        A *= torch.exp(3 * torch.randn(m, 1))
    Z = torch.randn(200000, r) @ A.T
    s = Z.abs().max(dim=1).values
    cv = (s.std() / s.mean()).item()
    if cv > worst:
        worst, worst_cfg = cv, (m, r, trial % 4 == 0)
print("max CV over 400 random centred-Gaussian structures: %.4f  (m=%d, rank=%d, scaled=%s)"
      % (worst, *worst_cfg))
print("ceiling respected: %s\n" % (worst <= CEIL + 0.01))

# ---- apply to the banked Triton vectors -----------------------------------
d = json.load(gzip.open(
    'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))


def walk(o):
    if isinstance(o, dict):
        if o.get('kind') == 'perturbation_sensitivities':
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


per_op = {}
for e in d['entries']:
    op = e['op']
    recs = list(walk(e['mutant'])) + [x for r in (e.get('refs') or []) for x in walk(r)]
    for r in recs:
        s = r['sensitivities']
        per_op.setdefault(op, []).append(s)

print("=" * 96)
print("BANKED TRITON-ON-T4 SENSITIVITY VECTORS -- all invocations incl. adversarial inputs")
print("=" * 96)
print("%-30s %5s %6s %8s %8s %8s %7s" %
      ("op", "n_inv", "n_zero", "CV med", "CV max", "ceiling", "verdict"))
out = {}
for op in sorted(per_op):
    vs = per_op[op]
    nz = sum(1 for s in vs if max(s) <= 0)
    cvs = [st.stdev(s) / st.fmean(s) for s in vs if st.fmean(s) > 0]
    if not cvs:
        print("%-30s %5d %6d %8s %8s %8.4f %7s"
              % (op, len(vs), nz, "-", "-", CEIL, "ZERO-J"))
        out[op] = ("ZERO-J", None, None, len(vs), nz)
        continue
    med, mx = st.median(cvs), max(cvs)
    over = sum(1 for c in cvs if c > 1.0)
    verdict = "OK" if over == 0 else "VIOL:%d" % over
    print("%-30s %5d %6d %8.4f %8.4f %8.4f %7s"
          % (op, len(vs), nz, med, mx, CEIL, verdict))
    out[op] = (verdict, med, mx, len(vs), nz)

json.dump(out, open('/tmp/gputest.json', 'w'))
print()
viol = {k: v for k, v in out.items() if v[0].startswith("VIOL")}
print("operators whose banked GPU sensitivity CV exceeds the linear-regime")
print("ceiling (=> assumption A3 falsified ON REAL TRITON DATA): %s"
      % (", ".join(sorted(viol)) if viol else "NONE"))
for k in sorted(viol):
    print("   %-28s CV med %.4f  max %.4f  (%d invocations)"
          % (k, out[k][1], out[k][2], out[k][3]))
