import json, math, statistics as st
NEW = [json.loads(l) for l in open("/private/tmp/claude-501/-Users-jaydenvasquez-Library-CloudStorage-GoogleDrive-jaydenvasquez1711-gmail-com-My-Drive-kernel-correctness-checker/7f64e343-8c25-4367-9961-8cbc1945494d/scratchpad/gpu/native_run/phase1_native.jsonl")]
GEN = [json.loads(l) for l in open("/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker/verification_runs/adaptive_tol_theory_2026-08-25/generalization/data/gen_native.jsonl")]
GPU = [json.loads(l) for l in open("/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker/verification_runs/adaptive_tol_theory_2026-08-25/generalization/data/gpu_native.jsonl")]

def r2(pred, act):
    mu = sum(act)/len(act); ss = sum((a-mu)**2 for a in act)
    rs = sum((p-a)**2 for p, a in zip(pred, act)); return 1 - rs/ss

# ---- original 27: banked matched invocations --------------------------------
P = [r for r in GPU if r.get('kind')=='primary' and 'error' not in r]
old = []
for g in GEN:
    c = [r for r in P if r['op']==g['op'] and abs(r['sigma']-g['sigma'])/g['sigma'] < 1e-6]
    if c:
        old.append(dict(op=g['op'], m=g['m'], yM3=g['y_M3'],
                        ymeas=c[0]['tol']/(3*c[0]['sigma']*g['L_struct']), grp='original'))

# ---- new 27: one matched pair per invocation --------------------------------
new = [dict(op=r['op'], m=r['m'], yM3=r['y_M3'],
            ymeas=r['tol']/(3*r['sigma']*r['L_closed']), grp='phase1')
       for r in NEW if r.get('kind')=='primary' and r.get('y_M3') and r.get('L_closed')]

print("="*78)
print("M3 RE-FIT -- zero fitted constants, so this is prediction vs measurement")
print("="*78)
for label, rows in (("original 27 (reproduced)", old), ("Phase-1 27 (new)", new),
                    ("FULL 54-operator corpus", old+new)):
    pr=[r['yM3'] for r in rows]; ac=[r['ymeas'] for r in rows]
    rat=[p/a for p,a in zip(pr,ac)]
    print(f"\n{label}:  n={len(rows)}  ops={len(set(r['op'] for r in rows))}")
    print(f"  R^2 = {r2(pr,ac):.4f}")
    print(f"  pred/meas  min {min(rat):.3f}  median {st.median(rat):.3f}  max {max(rat):.3f}"
          f"  spread {max(rat)/min(rat):.2f}x")
    print(f"  within +-10%: {sum(1 for x in rat if 0.9<=x<=1.1)}/{len(rat)}"
          f"   +-25%: {sum(1 for x in rat if 0.75<=x<=1.25)}/{len(rat)}")

# ---- m=1 specifically -------------------------------------------------------
print("\n" + "="*78)
print("THE m=1 QUESTION -- corpus went from 1 to 6 m=1 operators")
print("="*78)
allr = old + new
m1 = [r for r in allr if r['m']==1]
mn = [r for r in allr if r['m']>1]
for lbl, rows in (("m = 1", m1), ("m > 1", mn)):
    if not rows: continue
    rat=[r['yM3']/r['ymeas'] for r in rows]
    print(f"  {lbl:8s} n={len(rows):3d}  ops={len(set(r['op'] for r in rows)):2d}  "
          f"median pred/meas {st.median(rat):.3f}  min {min(rat):.3f}  max {max(rat):.3f}")
print("\n  per m=1 operator (pred/meas, >1 = M3 over-predicts):")
for op in sorted(set(r['op'] for r in m1)):
    rr=[r['yM3']/r['ymeas'] for r in m1 if r['op']==op]
    print(f"    {op:16s} n={len(rr)}  median {st.median(rr):.3f}   "
          f"over-prediction {100*(st.median(rr)-1):+.1f}%")
r2_no_m1 = r2([r['yM3'] for r in mn], [r['ymeas'] for r in mn])
print(f"\n  R^2 over the FULL corpus EXCLUDING m=1: {r2_no_m1:.4f}")

# ---- closed-form L vs native L ---------------------------------------------
print("\n" + "="*78)
print("CLOSED-FORM L vs NATIVE PROBED L (K=400), Phase-1 operators")
print("="*78)
rr = [(r['op'], r['L']/r['L_closed']) for r in NEW
      if r.get('kind')=='primary' and r.get('L_closed')]
vals=[v for _,v in rr]
print(f"  L_native/L_closed  min {min(vals):.3f}  median {st.median(vals):.3f}  max {max(vals):.3f}")
print(f"  within +-15% of 1.0: {sum(1 for v in vals if 0.85<=v<=1.15)}/{len(vals)}")
print("  NOTE the original round measured K=400 as biased HIGH by ~8% (median 1.081)")
print("  and only converging onto the closed form at K=20000.")
print("\n  worst 6 operators:")
byop={}
for op,v in rr: byop.setdefault(op,[]).append(v)
for op,v in sorted(byop.items(), key=lambda kv:-abs(st.median(kv[1])-1))[:6]:
    print(f"    {op:20s} median {st.median(v):.3f}")

# ---- spread -----------------------------------------------------------------
print("\n" + "="*78)
print("SPREAD -- native, from live kernel execution")
print("="*78)
sp={}
for r in NEW:
    if r.get('kind')=='primary' and r.get('prof_spread') and math.isfinite(r['prof_spread']):
        sp.setdefault(r['op'],[]).append(r['prof_spread'])
print(f"  {'operator':20s} {'native spread':>16s} {'zero_frac':>10s} {'sandwich':>10s} {'y=tol/3sL':>10s}")
for op in sorted(sp, key=lambda o:-st.median(sp[o]))[:10]:
    rs=[r for r in NEW if r['op']==op and r.get('kind')=='primary']
    zf=st.median([r['prof_zero_frac'] for r in rs])
    sw=f"{sum(r['ok_lo'] and r['ok_hi'] for r in rs)}/{len(rs)}"
    yy=st.median([r['ratio'] for r in rs])
    print(f"  {op:20s} {st.median(sp[op]):16.3e} {zf:10.3f} {sw:>10s} {yy:10.3f}")
