"""H3 + H5.

H3 — do the exception categories share structure? Claim tested: the two floor
categories are the RELATIVE and ABSOLUTE arms of one resolvability criterion,
    exception  <=>  min( s_med/ulp(out) / K1 ,  tol/(1e-6) ) <= 1
(K1 = 32, the shipped scope_detect threshold; second arm = "tol clamped").
The m=1 category should sit far inside the allowed region on BOTH axes -- a
genuinely separate phenomenon.

H5 — the CV ceiling. scope_detect.py asserts "CV <= 0.7555 is a correct
property of the linear regime" with no derivation. 0.7555 = sqrt(pi/2 - 1),
the CV of a half-normal. Lemma candidate: for ANY centred Gaussian vector
(X_1..X_m), CV(max_i |X_i|) <= sqrt(pi/2 - 1), equality iff the vector is
rank-1 (all |X_i| proportional). Since s = ||J d||_inf = max_i |<J_i, d>| is
exactly such a max in the linear regime, the in-code claim follows from the
lemma. Here: adversarial numeric search for a counterexample over covariance
structures, plus a check against every banked GPU cv value.
"""

import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "../..")
CEIL = math.sqrt(math.pi / 2 - 1)          # 0.75551...
K1 = 32.0
M1_OPS = {"mse_loss", "huber_loss", "bce_loss", "kldiv_loss", "nll_loss",
          "cross_entropy"}
CAT1 = {("flash_attention", "equal_attention_weights"),
        ("causal_flash_attention", "equal_attention_weights"),
        ("scaled_dot_product_attention", "equal_attention_weights")}
CAT2 = {("flash_attention", "skip_rescaling"),
        ("flash_attention", "last_tile_dropped"),
        ("causal_flash_attention", "skip_rescaling"),
        ("causal_flash_attention", "last_tile_dropped"),
        ("scaled_dot_product_attention", "skip_rescaling"),
        ("scaled_dot_product_attention", "last_tile_dropped"),
        ("sigmoid", "near_zero")}          # phase1's fp32-floor signature case


def rows():
    out = []
    p = os.path.join(RUNS, "adaptive_tol_theory_2026-08-25/native_run/attn_native.jsonl")
    for r in map(json.loads, open(p)):
        out.append(dict(op=r["op"], var=r["variant"], cv=r.get("cv"),
                        x=r.get("sens_over_ulp"), tol=r.get("tol"),
                        defect=r.get("defect"), src="attn"))
    p = os.path.join(RUNS, "phase1_derivations_2026-08-27/native_run/phase1_native.jsonl")
    for r in map(json.loads, open(p)):
        out.append(dict(op=r["op"], var="primary", cv=r.get("cv"),
                        x=r.get("s_over_ulp"), tol=r.get("tol"),
                        defect=r.get("defect_t01"), src="p1"))
    p = os.path.join(RUNS, "phase1_derivations_2026-08-27/native_run/pass2.jsonl")
    for r in map(json.loads, open(p)):
        if r.get("kind") == "adv":
            out.append(dict(op=r["op"], var=r.get("variant"), cv=r.get("cv"),
                            x=r.get("s_over_ulp"), tol=r.get("tol"),
                            defect=r.get("defect_t01"), src="p1adv"))
    p = os.path.join(RUNS, "phase2_convolution_2026-08-27/native_run/conv_native.jsonl")
    if os.path.exists(p):
        for r in map(json.loads, open(p)):
            out.append(dict(op=r["op"], var=r.get("variant", "primary"),
                            cv=r.get("cv"), x=r.get("s_over_ulp"),
                            tol=r.get("tol"), defect=r.get("defect_t01"),
                            src="conv"))
    return [r for r in out if r["x"] is not None and r["tol"] is not None]


def h3():
    rs = rows()
    print(f"H3: {len(rs)} banked invocations with (s/ulp, tol) available")
    mis = dict(cat1=[], cat2=[], other_flagged=[], m1=[])
    for r in rs:
        rel = r["x"] / K1                      # relative arm
        ab = r["tol"] / 1e-6                   # absolute arm (1.0 == clamped)
        flagged = min(rel, ab) <= 1.0
        key = (r["op"], r["var"])
        if key in CAT1:
            (mis["cat1"] if flagged else mis.setdefault("cat1_miss", [])).append(
                (key, rel, ab))
        elif key in CAT2:
            (mis["cat2"] if flagged else mis.setdefault("cat2_miss", [])).append(
                (key, rel, ab))
        elif r["op"] in M1_OPS:
            mis["m1"].append((key, rel, ab, flagged))
        elif flagged:
            mis["other_flagged"].append((key, rel, ab))
    n1 = len(mis["cat1"]); n1m = len(mis.get("cat1_miss", []))
    n2 = len(mis["cat2"]); n2m = len(mis.get("cat2_miss", []))
    print(f"  category-1 (absolute-floor) invocations flagged: {n1}, missed: {n1m}")
    print(f"  category-2 (fp32-floor) invocations flagged:     {n2}, missed: {n2m}")
    for k in ("cat1_miss", "cat2_miss"):
        for row in mis.get(k, []):
            print("   MISS:", row)
    m1f = [t for t in mis["m1"] if t[3]]
    m1rel = [t[1] for t in mis["m1"]]
    m1ab = [t[2] for t in mis["m1"]]
    print(f"  m=1 invocations: {len(mis['m1'])}; flagged by criterion: {len(m1f)}")
    if m1rel:
        print(f"    m=1 margins: rel arm min {min(m1rel):.1f}x  abs arm min {min(m1ab):.1f}x")
    print(f"  other invocations flagged: {len(mis['other_flagged'])}")
    for row in mis["other_flagged"]:
        print("   ", row)


def h5_banked():
    rs = rows()
    have = [r for r in rs if r.get("cv") is not None and r.get("defect") is not None]
    lin = [r for r in have if r["defect"] < 0.05]
    over = [r for r in lin if r["cv"] > CEIL]
    print(f"\nH5 banked: {len(have)} invocations with cv; {len(lin)} in linear regime "
          f"(defect<5%)")
    print(f"  linear-regime cv max = {max(r['cv'] for r in lin):.4f} vs ceiling {CEIL:.4f}")
    print(f"  linear-regime violations: {len(over)}")
    for r in sorted(over, key=lambda r: -r["cv"])[:8]:
        print(f"   {r['op']}/{r['var']} cv {r['cv']:.3f} defect {r['defect']:.4f} src {r['src']}")
    nl = [r for r in have if r["defect"] >= 0.05]
    if nl:
        overn = [r for r in nl if r["cv"] > CEIL]
        print(f"  out-of-linear invocations: {len(nl)}, cv>ceiling among them: {len(overn)}")


def h5_search(seed=2, n_struct=4000, n_mc=200_000):
    """Adversarial search for CV(max_i |X_i|) > sqrt(pi/2 - 1)."""
    rng = np.random.default_rng(seed)
    worst = (0.0, None)
    # 40-sample CV was the in-code statistic, but the lemma is about the parent:
    # estimate parent CV with n_mc draws.
    for trial in range(n_struct):
        m = rng.integers(1, 12)
        kind = trial % 5
        if kind == 0:    # random full covariance
            A = rng.standard_normal((m, m))
        elif kind == 1:  # near rank-1 with epsilon noise
            v = rng.standard_normal((m, 1))
            A = v @ rng.standard_normal((1, m)) + 1e-3 * rng.standard_normal((m, m))
        elif kind == 2:  # heavy-tailed row norms
            A = rng.standard_normal((m, m)) * (10.0 ** rng.uniform(-6, 0, size=(m, 1)))
        elif kind == 3:  # anti-correlated pairs
            B = rng.standard_normal((max(1, m // 2), m))
            A = np.vstack([B, -B])[:m]
        else:            # one dominant row + correlated small rows
            A = rng.standard_normal((m, m)) * 1e-2
            A[0] = rng.standard_normal(m)
        g = rng.standard_normal((n_mc // 40, m)).astype(np.float32)
        Mx = np.abs(g @ A.T.astype(np.float32)).max(axis=1)
        cv = Mx.std() / Mx.mean()
        if cv > worst[0]:
            worst = (cv, (kind, m))
    print(f"\nH5 search: {n_struct} covariance structures, worst parent CV = "
          f"{worst[0]:.4f} (kind {worst[1]}) vs ceiling {CEIL:.4f}")
    # rank-1 control: should sit AT the ceiling
    g = np.random.default_rng(3).standard_normal(2_000_000)
    hm = np.abs(g)
    print(f"  rank-1 control (half-normal): {hm.std()/hm.mean():.4f}")
    # 40-sample statistic: how often does SAMPLING alone exceed the ceiling?
    g = np.random.default_rng(4).standard_normal((20000, 40))
    cvs = np.abs(g).std(axis=1) / np.abs(g).mean(axis=1)
    print(f"  40-sample CV of a rank-1 parent: q50 {np.quantile(cvs,.5):.3f} "
          f"q95 {np.quantile(cvs,.95):.3f} P(>ceiling) {(cvs>CEIL).mean():.3f}")


if __name__ == "__main__":
    h3()
    h5_banked()
    h5_search()
