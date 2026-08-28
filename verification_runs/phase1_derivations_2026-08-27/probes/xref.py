"""Cross-reference TritonBench-G (184) and KernelBenchX (184) against the
checker's existing 29 operators and CORPUS_EXPANSION_PLAN.md's Phase 1 picks.

Families are matched by regex on the problem filename stem. Every stem lands in
exactly one family (first match wins), and the UNMATCHED bucket is printed in
full so nothing is silently dropped.
"""
import re, os, collections

SP = os.path.dirname(os.path.abspath(__file__))
TBG = os.path.join(SP, "corpora/TritonBench/data/TritonBench_G_v1")
KBX = os.path.join(SP, "corpora/KernelBenchX/data/kernelbenchx")

# (family, regex). ORDER MATTERS -- first match wins, so specific before generic.
FAMILIES = [
    ("attention",      r"attn|attention|flash_decode|token_attn|swa_|block_sparse"),
    ("rope/rotary",    r"rope|rotary|rbe"),
    ("kvcache/copy",   r"kcache|kv_cache|destindex|cache_transform|quantize_kv|var_len_copy|quantize_copy"),
    ("quantization",   r"quant|int4|int8|fp4|fp8|w8a8|w4a16|dequant|f8_conv"),
    ("linear-attn/ssm",r"chunk_|retention|gla_|rwkv|hgrn|recurrent|delta_fwd|lightning|ssm|gate_recurrence"),
    ("scan/cumsum",    r"cumsum|cumprod|scan"),
    ("matmul",         r"matmul|mm_|_mm|gemm|bmm|matrix_mult|vecmat|matrix_vector|mv_|addmm|tensordot|streamk|square_matrix|linear_activation|lora|bgmv|sgmv|dot"),
    ("softmax",        r"softmax"),
    ("layernorm",      r"layer_?norm|layernorm"),
    ("rmsnorm",        r"rms_?norm|rms_matmul|rms_rbe"),
    ("norm-other",     r"l2_norm|batch_norm|instance_norm|group_norm|spectral_norm|normalize|cosine_simil|pairwise_dist"),
    ("loss",           r"cross_entropy|ce_loss|kldiv|kl_div|mse_loss|nll_loss|smooth_l1|binary_cross|hinge|triplet|cosine_embedding"),
    ("gated-act",      r"swiglu|geglu|glu"),
    ("activation",     r"relu|gelu|sigmoid|tanh|selu|elu|softplus|softsign|hardtanh|hardsigmoid|hardshrink|silu|swish|activation|logit"),
    ("reduction",      r"reduction|reduce|logsumexp|^sum|^mean|^min|^max|^std|^var|argmax|argmin|_sum|_mean|_std"),
    ("pooling",        r"pool"),
    ("conv",           r"conv|pixel_shuffle"),
    ("linalg",         r"cholesky|svd|qr|lu$|_lu|det|eig|solve|invert|pseudoinv|ldl|matrix_power|low_rank"),
    ("index/gather",   r"index_|gather|scatter|masked_select|embedding|expand_where|permute_copy|select|repeat_interleave|hstack|tile"),
    ("random/dropout", r"dropout|rand|multinomial|uniform_sampl|logspace|seeded"),
    ("optimizer",      r"adam|sgd|lion|rmsprop|adamw|apply_penalty|update"),
    ("elementwise-math",r"^abs|^add|^sub|^mul|^div|^exp|^log|^sqrt|^rsqrt|^pow|^cos|^sin|^asin|^erf|^floor|^trunc|^reciprocal|^i0|^signbit|^bitwise|^rad2deg|^zeta|^digamma|^gammaln|^polygamma|^airy|^bessel|^chebyshev|^fftn|^ifftshift|^isfinite|vector_addition|add_value|sin_|masked_add|mul_exponent|pow_scalar|sqrt_|exp_|log_|_sqrt|_exp"),
    ("interp/spatial", r"grid_sample|interpolate|sph_harmonics"),
    ("transpose/layout",r"transpose|strided_buffer|nested_loops|spinning_lock|cosine_compute|diag_ssm"),
]

def classify(stem):
    s = stem.lower()
    for fam, rx in FAMILIES:
        if re.search(rx, s):
            return fam
    return "UNMATCHED"

def load(path, recurse=False):
    out = []
    if recurse:
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".py"):
                    out.append((os.path.basename(root), f[:-3]))
    else:
        for f in sorted(os.listdir(path)):
            if f.endswith(".py"):
                out.append(("", f[:-3]))
    return out

tbg = load(TBG)
kbx = load(KBX, recurse=True)

print(f"TritonBench-G  : {len(tbg)} problem files")
print(f"KernelBenchX   : {len(kbx)} problem files, "
      f"{len(set(c for c,_ in kbx))} categories")
print()

for name, items in (("TritonBench-G", tbg), ("KernelBenchX", kbx)):
    cnt = collections.Counter(classify(s) for _, s in items)
    print(f"=== {name} by family ===")
    for fam, n in cnt.most_common():
        print(f"  {fam:20s} {n:4d}")
    print(f"  {'TOTAL':20s} {sum(cnt.values()):4d}")
    un = [s for _, s in items if classify(s) == "UNMATCHED"]
    if un:
        print(f"  UNMATCHED ({len(un)}): {', '.join(un)}")
    print()
