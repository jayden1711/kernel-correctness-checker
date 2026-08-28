import json, math, statistics as st
NEW=[json.loads(l) for l in open("/private/tmp/claude-501/-Users-jaydenvasquez-Library-CloudStorage-GoogleDrive-jaydenvasquez1711-gmail-com-My-Drive-kernel-correctness-checker/7f64e343-8c25-4367-9961-8cbc1945494d/scratchpad/gpu/native_run/phase1_native.jsonl")]
DT =[json.loads(l) for l in open("/private/tmp/claude-501/-Users-jaydenvasquez-Library-CloudStorage-GoogleDrive-jaydenvasquez1711-gmail-com-My-Drive-kernel-correctness-checker/7f64e343-8c25-4367-9961-8cbc1945494d/scratchpad/gpu/native_run/diagtri.jsonl")]
GEN=[json.loads(l) for l in open("/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker/verification_runs/adaptive_tol_theory_2026-08-25/generalization/data/gen_native.jsonl")]
GPU=[json.loads(l) for l in open("/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker/verification_runs/adaptive_tol_theory_2026-08-25/generalization/data/gpu_native.jsonl")]
def r2(p,a):
    mu=sum(a)/len(a); ss=sum((v-mu)**2 for v in a); rs=sum((x-y)**2 for x,y in zip(p,a))
    return 1-rs/ss
P=[r for r in GPU if r.get('kind')=='primary' and 'error' not in r]
old=[]
for g in GEN:
    c=[r for r in P if r['op']==g['op'] and abs(r['sigma']-g['sigma'])/g['sigma']<1e-6]
    if c: old.append(dict(op=g['op'],m=g['m'],yM3=g['y_M3'],
                          ymeas=c[0]['tol']/(3*c[0]['sigma']*g['L_struct'])))
new=[dict(op=r['op'],m=r['m'],yM3=r['y_M3'],ymeas=r['tol']/(3*r['sigma']*r['L_closed']))
     for r in NEW if r.get('kind')=='primary' and r.get('y_M3') and r.get('L_closed')]
new+= [dict(op=r['op'],m=None,yM3=r['y_M3'],ymeas=r['y_meas']) for r in DT]

print("="*80); print("FINAL M3 RE-FIT"); print("="*80)
for lbl,rows in (("original 27",old),("Phase-1 27",new),("FULL 54-op corpus",old+new)):
    pr=[r['yM3'] for r in rows]; ac=[r['ymeas'] for r in rows]
    rat=[p/a for p,a in zip(pr,ac)]
    print(f"{lbl:22s} n={len(rows):4d} ops={len(set(r['op'] for r in rows)):3d}  "
          f"R2={r2(pr,ac):7.4f}  med {st.median(rat):.3f}  spread {max(rat)/min(rat):.2f}x  "
          f"+-10%: {sum(1 for x in rat if 0.9<=x<=1.1)}/{len(rat)}")

allr=old+new
print()
print("=== driver of the drop: per-family median pred/meas ===")
FAM={'scan':['cumsum','cumsum_reverse','cumsum_exclusive','masked_cumsum'],
     'loss(m=1)':['mse_loss','huber_loss','bce_loss','kldiv_loss','nll_loss','cross_entropy'],
     'activation':['relu','leaky_relu','sigmoid','tanh','selu','elu','softplus','hardsigmoid','new_gelu','gelu','swish'],
     'matmul-var':['matvec','batched_matmul','diagonal_matmul','triangular_matmul','matmul'],
     'other-new':['rope','swiglu','logsumexp','std_reduction','var_reduction']}
for fam,ops in FAM.items():
    rs=[r for r in allr if r['op'] in ops]
    if not rs: continue
    rat=[r['yM3']/r['ymeas'] for r in rs]
    print(f"  {fam:14s} n={len(rs):3d}  median {st.median(rat):.3f}  ({100*(st.median(rat)-1):+.1f}%)")
sc=[r for r in allr if r['op'] in FAM['scan']]
rest=[r for r in allr if r['op'] not in FAM['scan']]
print(f"\n  R2 with scans EXCLUDED: {r2([r['yM3'] for r in rest],[r['ymeas'] for r in rest]):.4f}"
      f"   (scans alone n={len(sc)})")
