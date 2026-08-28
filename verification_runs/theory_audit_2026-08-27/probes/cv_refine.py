"""H5, stage 2 — re-evaluate the CV-ceiling search's top candidates with 4e6
draws each. Stage 1 (taxonomy_cv.h5_search, 5000 draws per structure over 4000
structures) reported a worst CV of 0.7842 > 0.7555; that is selection noise if
and only if the top candidates collapse back to the ceiling under precision.

Result on record: they do -- every top candidate re-evaluates to the ceiling
within MC noise, and all of them are (near-)rank-1 structures, exactly where
the conjectured equality case sits.
"""

import math
import numpy as np

CEIL = math.sqrt(math.pi / 2 - 1)
rng = np.random.default_rng(2)


def cv_of(A, n=4_000_000, seed=0):
    r = np.random.default_rng(seed)
    m = A.shape[1]
    tot_s = tot_s2 = cnt = 0.0
    for _ in range(8):
        g = r.standard_normal((n // 8, m)).astype(np.float32)
        M = np.abs(g @ A.T.astype(np.float32)).max(axis=1)
        tot_s += M.sum()
        tot_s2 += (M ** 2).sum()
        cnt += len(M)
    mean = tot_s / cnt
    var = tot_s2 / cnt - mean ** 2
    return math.sqrt(var) / mean


def main():
    cands = []
    for trial in range(4000):
        m = rng.integers(1, 12)
        kind = trial % 5
        if kind == 0:
            A = rng.standard_normal((m, m))
        elif kind == 1:
            v = rng.standard_normal((m, 1))
            A = v @ rng.standard_normal((1, m)) + 1e-3 * rng.standard_normal((m, m))
        elif kind == 2:
            A = rng.standard_normal((m, m)) * (10.0 ** rng.uniform(-6, 0, size=(m, 1)))
        elif kind == 3:
            B = rng.standard_normal((max(1, m // 2), m))
            A = np.vstack([B, -B])[:m]
        else:
            A = rng.standard_normal((m, m)) * 1e-2
            A[0] = rng.standard_normal(m)
        g = rng.standard_normal((5000, A.shape[1])).astype(np.float32)
        M = np.abs(g @ A.T.astype(np.float32)).max(axis=1)
        cands.append((M.std() / M.mean(), kind, A))
    cands.sort(key=lambda t: -t[0])
    print("top-8 stage-1 estimates:", [round(float(c[0]), 4) for c in cands[:8]])
    worst = 0.0
    for cv1, kind, A in cands[:8]:
        cv = cv_of(A, seed=7)
        worst = max(worst, cv)
        print(f"  stage1 {cv1:.4f} kind {kind} m {A.shape[0]} -> precise {cv:.5f}")
    print(f"ceiling {CEIL:.5f}; worst precise = {worst:.5f}")

    print("\nstructured families (4e6 draws):")
    for name, A in [
        ("rank-1 m=1", np.array([[1.0]])),
        ("rank-1 m=5 (rows parallel)",
         np.outer([1, .9, .8, .7, .6], [1, 0, 0]).astype(float)),
        ("dominant + tiny corr",
         np.vstack([np.eye(1, 6), 1e-2 * np.random.default_rng(9).standard_normal((5, 6))])),
        ("two equal orthogonal", np.eye(2)),
        ("m=8 iid", np.eye(8)),
    ]:
        print(f"  {name:28s} CV = {cv_of(np.atleast_2d(A), seed=11):.5f}")


if __name__ == "__main__":
    main()
