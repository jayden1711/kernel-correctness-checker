"""Analyse the GPU-native run and compare against the CPU-derived round."""
import json, math, statistics as st
from collections import defaultdict

P = ('/private/tmp/claude-501/-Users-jaydenvasquez-Library-CloudStorage-'
     'GoogleDrive-jaydenvasquez1711-gmail-com-My-Drive-kernel-correctness-checker/'
     '2b7d1407-7ba4-4941-afbe-5634d0489329/scratchpad/gpu_native.jsonl')
rows = [json.loads(l) for l in open(P)]
prim = [r for r in rows if r.get('kind') == 'primary' and 'error' not in r]
errs = [r for r in rows if 'error' in r]
attn = [r for r in rows if r.get('kind') == 'attn' and 'error' not in r]

print("primary invocations: %d   attention variants: %d   errors: %d"
      % (len(prim), len(attn), len(errs)))
for e in errs[:10]:
    print("   ERROR %s/%s: %s" % (e['op'], e.get('inv'), e['error']))

# CPU-derived numbers from the prior round (probes/coverage2.py output)
CPU = {
 "avg_pool1d":(5.078e-03,0.001),"avg_pool2d":(2.549e-03,0.002),"avg_pool3d":(4.062e-03,0.001),
 "batchnorm":(1.632e-02,0.002),"causal_flash_attention":(6.676e-03,0.031),
 "cross_entropy":(6.544e-04,0.387),"flash_attention":(4.210e-03,0.037),
 "frobenius_norm":(1.500e-04,0.008),"gelu":(1.363e-02,0.015),"groupnorm":(1.348e-02,0.016),
 "instancenorm":(1.689e-02,0.028),"l1norm":(1.368e-04,0.007),"l2norm":(1.231e-03,0.014),
 "layernorm":(2.866e-02,0.010),"log_softmax":(1.343e-02,0.012),"matmul":(5.460e-02,0.002),
 "max_pool1d":(9.582e-03,0.001),"max_pool2d":(1.015e-02,0.001),"max_pool3d":(1.168e-02,0.001),
 "max_reduction":(9.994e-03,0.003),"mean_reduction":(8.463e-04,0.003),
 "min_reduction":(9.847e-03,0.002),"rmsnorm":(2.725e-02,0.012),
 "scaled_dot_product_attention":(3.731e-03,0.037),"softmax":(9.669e-04,0.058),
 "sum_reduction":(1.175e-01,0.003),"swish":(1.242e-02,0.022)}
# banked-Triton tolerance medians from the prior round
GPUTOL = {
 "avg_pool1d":4.714e-03,"avg_pool2d":2.408e-03,"avg_pool3d":4.322e-03,"batchnorm":1.614e-02,
 "causal_flash_attention":6.783e-03,"cross_entropy":7.501e-04,"flash_attention":4.160e-03,
 "frobenius_norm":1.529e-04,"gelu":1.326e-02,"groupnorm":1.220e-02,"instancenorm":1.665e-02,
 "l1norm":1.383e-04,"l2norm":1.253e-03,"layernorm":2.848e-02,"log_softmax":1.362e-02,
 "matmul":5.459e-02,"max_pool1d":1.013e-02,"max_pool2d":1.009e-02,"max_pool3d":1.121e-02,
 "max_reduction":9.962e-03,"mean_reduction":8.502e-04,"min_reduction":9.999e-03,
 "rmsnorm":2.748e-02,"scaled_dot_product_attention":4.272e-03,"softmax":9.952e-04,
 "sum_reduction":1.090e-01,"swish":1.280e-02}

by = defaultdict(list)
for r in prim:
    by[r['op']].append(r)

print()
print("=" * 122)
print("GPU-NATIVE SANDWICH  --  L, m, adaptive_tol and the linearisation test all")
print("computed by executing the Triton kernels")
print("=" * 122)
print("%-30s %3s %7s %10s %10s %8s %9s %8s %7s %8s" %
      ("operator", "n", "m", "tol(nat)", "L(nat)", "tol/3sL", "defect", "slope",
       "sand", "vs bank"))
tot = ok = 0
worst = []
for op in sorted(by):
    v = by[op]
    n = len(v)
    lo_ok = sum(1 for r in v if r['ok_lo'])
    hi_ok = sum(1 for r in v if r['ok_hi'])
    both = sum(1 for r in v if r['ok_lo'] and r['ok_hi'])
    tot += n; ok += both
    tol = st.median([r['tol'] for r in v])
    L = st.median([r['L'] for r in v])
    ratio = st.median([r['ratio'] for r in v])
    dfs = [r['defect_t01'] for r in v if r.get('defect_t01') is not None]
    sls = [r['slope'] for r in v if r.get('slope') is not None]
    bank = GPUTOL.get(op)
    rel = tol / bank if bank else float('nan')
    print("%-30s %3d %7d %10.3e %10.3e %8.3f %8.3f%% %8.4f %7s %8.3f"
          % (op, n, v[0]['m'], tol, L, ratio,
             100 * st.median(dfs) if dfs else float('nan'),
             st.median(sls) if sls else float('nan'),
             "%d/%d" % (both, n), rel))
    worst.append((abs(math.log10(rel)) if rel > 0 else 9, op, rel,
                  100 * st.median(dfs) if dfs else float('nan')))

print("-" * 122)
print("TOTAL: %d/%d invocations satisfy BOTH sides, across %d operators"
      % (ok, tot, len(by)))

print()
print("Divergence from the banked/CPU round (native tol / banked Triton tol):")
worst.sort(reverse=True)
for w, op, rel, df in worst[:6]:
    print("   %-30s %.3f   (native linearisation defect %.3f%%)" % (op, rel, df))
allrel = [x[2] for x in worst]
print("   range %.3f - %.3f, median %.3f" % (min(allrel), max(allrel), st.median(allrel)))

print()
print("=" * 122)
print("ATTENTION, INPUT-CONDITIONAL, NATIVE")
print("=" * 122)
CEIL = math.sqrt(math.pi / 2 - 1)
print("%-30s %-26s %9s %8s %10s %8s %6s" %
      ("op", "input", "peak wgt", "CV", "defect", "slope", "sand"))
for r in attn:
    print("%-30s %-26s %9.6f %8.4f %9.2f%% %8.4f %6s"
          % (r['op'], r['variant'],
             r['peak_weight'] if r.get('peak_weight') is not None else float('nan'),
             r['cv'] if r['cv'] is not None else float('nan'),
             100 * r['defect_t01'] if r.get('defect_t01') is not None else float('nan'),
             r['slope'] if r.get('slope') is not None else float('nan'),
             "ok" if (r['ok_lo'] and r['ok_hi']) else "FAIL"))
print("\nhalf-normal ceiling CV = %.4f" % CEIL)
