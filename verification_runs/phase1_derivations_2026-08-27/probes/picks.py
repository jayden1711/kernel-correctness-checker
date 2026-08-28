"""Test each planned Phase 1 pick for real presence in TritonBench-G and
KernelBenchX. `hits` counts problem files whose stem matches the operator's
regex -- a crude but honest popularity proxy, and the raw stems are printed so
the count can be argued with."""
import re, os

SP = os.path.dirname(os.path.abspath(__file__))
TBG = os.path.join(SP, "corpora/TritonBench/data/TritonBench_G_v1")
KBX = os.path.join(SP, "corpora/KernelBenchX/data/kernelbenchx")

tbg = sorted(f[:-3] for f in os.listdir(TBG) if f.endswith(".py"))
kbx = []
for root, _, files in os.walk(KBX):
    for f in files:
        if f.endswith(".py"):
            kbx.append((os.path.basename(root), f[:-3]))
kbx.sort()

# planned Phase 1 picks from CORPUS_EXPANSION_PLAN.md
PICKS = [
    ("A1 activation", "relu",          r"(?<!leaky_)(?<!_)relu"),
    ("A1 activation", "leaky_relu",    r"leaky_relu|leakyrelu"),
    ("A1 activation", "sigmoid",       r"(?<!hard)sigmoid"),
    ("A1 activation", "tanh",          r"(?<!hard)tanh"),
    ("A1 activation", "selu",          r"(?<!\w)selu"),
    ("A1 activation", "elu",           r"(?<!s)(?<!r)elu"),
    ("A1 activation", "softplus",      r"softplus"),
    ("A1 activation", "softsign",      r"softsign"),
    ("A1 activation", "hardtanh",      r"hardtanh"),
    ("A1 activation", "hardsigmoid",   r"hardsigmoid"),
    ("A1 activation", "new_gelu",      r"gelu"),
    ("A3 scan",       "cumsum",        r"cumsum"),
    ("A3 scan",       "cumsum_reverse",r"reversed_cumsum|cumsum_rev"),
    ("A3 scan",       "cumsum_excl",   r"exclusive"),
    ("A3 scan",       "masked_cumsum", r"masked_cumsum|decay_cumsum"),
    ("A2 matmul",     "matvec",        r"matrix_vector|vecmat|_mv_|^mv_|matrix_vector_dot"),
    ("A2 matmul",     "mat_scalar",    r"matrix_scalar|scalar_mult|int_scaled"),
    ("A2 matmul",     "diagonal_mm",   r"diag"),
    ("A2 matmul",     "triangular_mm", r"tril|triu|triangular"),
    ("A2 matmul",     "batched_mm",    r"bmm|batched|batch_mat"),
    ("A4 loss",       "mse_loss",      r"mse"),
    ("A4 loss",       "huber/smoothl1",r"huber|smooth_l1"),
    ("A4 loss",       "hinge_loss",    r"hinge"),
    ("A4 loss",       "kldiv",         r"kldiv|kl_div"),
    ("A4 loss",       "triplet_margin",r"triplet"),
]

# candidates NOT in the plan, surfaced by the real lists
NEW = [
    ("NEW", "rope/rotary",     r"rope|rotary|rbe"),
    ("NEW", "swiglu/geglu",    r"swiglu|geglu|glu"),
    ("NEW", "logsumexp",       r"logsumexp"),
    ("NEW", "std/var",         r"(?<!\w)std|(?<!\w)var(?!_len)"),
    ("NEW", "addmm",           r"addmm"),
    ("NEW", "binary_cross_ent",r"binary_cross"),
    ("NEW", "nll_loss",        r"nll"),
    ("NEW", "rsqrt",           r"rsqrt"),
    ("NEW", "reciprocal",      r"reciprocal"),
    ("NEW", "exp",             r"(?<!\w)exp(?!and)"),
    ("NEW", "log/log1p",       r"(?<!\w)log(?!s|i|_soft)"),
    ("NEW", "sqrt",            r"(?<!r)(?<!\w_)sqrt"),
    ("NEW", "abs",             r"(?<!\w)abs"),
    ("NEW", "erf",             r"erf"),
    ("NEW", "cosine_sim",      r"cosine_simil"),
]

def hits(rx, names):
    p = re.compile(rx)
    return [n for n in names if p.search(n.lower())]

tbg_names = tbg
kbx_names = [s for _, s in kbx]

print(f"{'bucket':16s} {'operator':17s} {'TB-G':>5s} {'KBX':>5s}   evidence")
print("-" * 108)
for bucket, op, rx in PICKS + [("", "", "")] + NEW:
    if not op:
        print("-" * 108); continue
    a, b = hits(rx, tbg_names), hits(rx, kbx_names)
    ev = ", ".join((a + b)[:4]) + ("..." if len(a) + len(b) > 4 else "")
    flag = "  <-- ABSENT FROM BOTH" if not a and not b else ""
    print(f"{bucket:16s} {op:17s} {len(a):5d} {len(b):5d}   {ev[:60]}{flag}")
