"""Do the Phase-2 conv specs separate correct kernels from buggy ones?

Same protocol and same exclusions as Phase 1's run_phase1.py: Layer-1 AST
checks are NOT run (they test for Triton launches and would fail a torch
candidate including the correct one). Reports catch rate AND false-positive
rate on the correct implementation, which is the number that matters.

Sweeps EVERY config in valid_shapes, not just the first -- `ignores_dilation`
is by construction invisible on the d==1 configs, so a single-config run would
score it as an escape when it is really a config-coverage question.
"""
import importlib, json, os, sys
import torch
ROOT="/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutants.conv_defs import OPS
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance

torch.manual_seed(0)
ATOL=RTOL=1e-4

def judge(spec, cand, ref, inputs):
    out={}
    for pn, pf in spec.algebraic_properties:
        try:
            r=pf(cand, inputs); out[f"L2:{pn}"]=(bool(r[0]) if isinstance(r,(tuple,list)) else bool(r),"")
        except Exception as e: out[f"L2:{pn}"]=(False,str(e))
    for an, ai in spec.get_adversarial_inputs(inputs):
        try:
            c=spec.run_candidate(cand, ai); r=spec.run_reference(ref, ai)
            if c.shape!=r.shape: out[f"L3:adv_{an}"]=(False,"shape"); continue
            ok=torch.allclose(c.float(), r.float(), atol=ATOL, rtol=RTOL)
            out[f"L3:adv_{an}"]=(ok, f"max_err={(c-r).abs().max().item():.3e}")
        except Exception as e: out[f"L3:adv_{an}"]=(False,str(e)[:60])
    try:
        prim=spec.primary_input(inputs)
        _c=lambda t: spec.run_candidate(cand,(t,)+tuple(inputs[1:]))
        _r=lambda t: spec.run_reference(ref,(t,)+tuple(inputs[1:]))
        res=check_perturbation_tolerance(_c,_r,prim,batch_samples=spec.batch_samples,
                                         op_name=spec.name, companions=tuple(inputs[1:]))
        out["L3:perturbation"]=(bool(res[0]), str(res[1])[:80])
    except Exception as e: out["L3:perturbation"]=(False,str(e)[:80])
    return out

rows=[]; tot=caught=fp_ops=0
print(f"{'operator':20s} {'FP':>3s} {'caught':>8s}   per-mutant (config that caught it)")
print("-"*100)
for op,d in OPS.items():
    spec=importlib.import_module(f"verification.specs.{op}").get_spec()
    fp_any=set(); detail=[]
    ncfg=len(spec.valid_shapes)
    for mname,mfn in d["mutants"].items():
        hit=None
        for ci in range(ncfg):
            inputs=spec.make_inputs(spec.valid_shapes[ci],"cpu",torch.float32)
            cj=judge(spec,d["correct"],d["ref"],inputs)
            fp=[k for k,(p,_) in cj.items() if not p]
            fp_any |= set(fp)
            mj=judge(spec,mfn,d["ref"],inputs)
            fl=[k for k,(p,_) in mj.items() if not p and k not in fp]
            if fl and hit is None: hit=(ci,fl[0])
        rows.append(dict(op=op,mutant=mname,caught=hit is not None,
                         config=hit[0] if hit else None, by=hit[1] if hit else None))
        detail.append(f"{mname}->cfg{hit[0]}:{hit[1].split(':')[1]}" if hit else f"{mname}->ESCAPED")
        tot+=1; caught+= 1 if hit else 0
    if fp_any: fp_ops+=1
    print(f"{op:20s} {len(fp_any):3d} {'':8s}   " + "; ".join(detail)[:78])
    if fp_any: print(f"{'':20s}   FP on CORRECT: {sorted(fp_any)}")
print("-"*100)
print(f"operators: {len(OPS)}  mutants: {tot}  caught: {caught} ({100*caught/tot:.1f}%)")
print(f"operators with a false positive on the CORRECT implementation: {fp_ops}")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"conv_catch.jsonl"),"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
