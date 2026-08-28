"""
Offline validation of the scope detector -- Gram-screen edition (2026-08-27).

Three groups:

  1. classify() is pure arithmetic over scalars; its rule is pinned here,
     including the RETIREMENT of the falsified defect screen (the 2026-08-26
     corpus run measured the in/out classes overlapping at 0.68x; the code was
     removed, and these tests keep a "fix by nudging the constant back in"
     from ever passing review silently).
  2. measure_gram() is exercised end-to-end on tiny CPU tensors: for a linear
     operator the exact directional derivative reproduces a synthetically
     "measured" sensitivity to float64 precision, so the ratio is 1 and the
     screen is silent; scaling the measured side by 2x fires it.
  3. math_refs' hand-written float64 definitions are checked against the
     corresponding torch built-ins on random inputs -- a transcription bug in
     an eps placement or a pooling convention would surface here, not on the
     GPU corpus run.

What this file cannot settle -- deliberately not faked -- is whether the
measured Triton sensitivities land inside the factor-2 band on the real
corpus. That needs the GPU run; see
verification_runs/gram_screen_2026-08-27/FINDINGS.md.
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verification.layer2_numeric_oracle import scope_detect as S
from verification.layer2_numeric_oracle import math_refs as M


def flagged(gram_log10=None, sulp=None, discrete=False, tol_floor=False):
    rs = S.classify(gram_log10, sulp, discrete, tol_floor)
    return [r["reason"] for r in rs if r["severity"] != "advisory"]


# --------------------------------------------------------------------------
# 1. The rule.
# --------------------------------------------------------------------------

def test_gram_threshold_is_factor_two_pre_registered():
    assert S.GRAM_MAX_ABS_LOG10 == pytest.approx(math.log10(2.0))
    assert S.SULP_MIN_MEDIAN == 32.0


def test_gram_fires_symmetrically_beyond_factor_two():
    assert flagged(gram_log10=0.35, sulp=1000.0) == [S.REASON_GRAM]
    assert flagged(gram_log10=-0.35, sulp=1000.0) == [S.REASON_GRAM]
    assert flagged(gram_log10=3.0, sulp=1000.0) == [S.REASON_GRAM]


def test_gram_silent_inside_the_curvature_band():
    # Worst measured in-scope curvature is ~10-15%; give it the whole band
    # up to 2x and require silence.
    for r in (1.0, 1.1, 1.5, 1.99, 1 / 1.99):
        assert flagged(gram_log10=math.log10(r), sulp=1000.0) == [], r


def test_floor_screen_unchanged():
    assert flagged(sulp=3.0) == [S.REASON_FP_FLOOR]
    assert flagged(sulp=31.9) == [S.REASON_FP_FLOOR]
    assert flagged(sulp=360.0) == []


def test_screens_are_or_not_and():
    """Gram divergence shows at ordinary s/ulp; the fp floor shows a tiny
    s/ulp. AND-ing them would miss each."""
    assert flagged(gram_log10=1.0, sulp=2220.0) == [S.REASON_GRAM]
    assert S.REASON_FP_FLOOR in flagged(gram_log10=None, sulp=3.0)


def test_missing_signals_do_not_flag():
    """A probe that could not be taken must not read as a divergence."""
    assert flagged(None, None) == []


def test_index_valued_ops_flagged_without_measurement():
    rs = S.classify(None, None, output_is_discrete=True, tol_at_floor=False)
    assert [r["reason"] for r in rs] == [S.REASON_STRUCTURAL]
    assert rs[0]["severity"] == "excluded"


def test_tolerance_floor_is_advisory_only():
    rs = S.classify(0.01, 5000.0, False, tol_at_floor=True)
    assert [r["reason"] for r in rs] == [S.REASON_TOL_FLOOR]
    assert all(r["severity"] == "advisory" for r in rs)
    assert flagged(0.01, 5000.0, tol_floor=True) == []


def test_rejected_signals_are_not_consulted():
    """The defect ladder, CV and peak attention weight were falsified;
    classify() must not take them as arguments at all."""
    import inspect
    params = set(inspect.signature(S.classify).parameters)
    assert params == {"gram_log10_med", "sulp_med", "output_is_discrete",
                      "tol_at_floor"}, params


def test_defect_screen_is_retired_not_retuned():
    """The 2026-08-26 corpus run falsified the linearisation-defect screen
    (classes overlap 0.68x, scope_detect_2026-08-26/FINDINGS.md 2b). The fix
    was REPLACEMENT by the Gram screen, not a constant nudge; this pins the
    removal so a threshold cannot quietly come back."""
    for name in ("DEFECT_MAX_PCT", "SUB_SCALE", "measure_defect",
                 "_DEFECT_SAMPLES", "REASON_SATURATION"):
        assert not hasattr(S, name), f"{name} resurrected"


def test_gram_sample_default_is_the_converged_count():
    """20 deltas: the count at which a median-over-deltas statistic converged
    on this corpus (20 vs 40 arms: 0/854 disagreements)."""
    assert S._GRAM_SAMPLES == 20 or os.environ.get("KCC_SCOPE_GRAM_SAMPLES")


# --------------------------------------------------------------------------
# 2. measure_gram end-to-end on tiny tensors.
# --------------------------------------------------------------------------

def _linear_measured(x, deltas):
    """Synthetic 'measured' sensitivities for sum_reduction: what a perfect
    kernel would produce, s_k = ||f(x+d)-f(x)||_inf = |sum d|_max, computed
    in float64 so the test isolates the screen from fp32 cancellation."""
    x64 = x.double()
    return [float(((x64 + d.double()).sum(dim=-1)
                   - x64.sum(dim=-1)).abs().max()) for d in deltas]


def test_measure_gram_ratio_one_for_linear_op():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(8)]
    s = _linear_measured(x, deltas)
    g = S.measure_gram("sum_reduction", x, (), deltas, s)
    assert g is not None and g["n_valid"] == 8
    assert abs(g["log10_median"]) < 1e-6
    assert flagged(gram_log10=g["log10_median"], sulp=1000.0) == []


def test_measure_gram_fires_on_doubled_response():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(8)]
    s = [2.0 * v for v in _linear_measured(x, deltas)]
    g = S.measure_gram("sum_reduction", x, (), deltas, s)
    assert g["log10_median"] == pytest.approx(math.log10(2.0), abs=1e-9)
    assert flagged(gram_log10=g["log10_median"], sulp=1000.0) == [S.REASON_GRAM]


def test_measure_gram_declines_on_unknown_op():
    x = torch.randn(4, 4)
    assert S.measure_gram("no_such_op", x, (), [torch.randn_like(x)], [0.1]) is None


def test_measure_gram_withholds_signal_below_min_valid():
    x = torch.randn(4, 4)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(3)]
    s = _linear_measured(x, deltas)  # 3 < GRAM_MIN_VALID
    g = S.measure_gram("sum_reduction", x, (), deltas, s)
    assert g["log10_median"] is None
    assert flagged(gram_log10=g["log10_median"], sulp=1000.0) == []


def test_measure_gram_skips_zero_and_nonfinite_samples():
    x = torch.randn(6, 6)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(6)]
    s = _linear_measured(x, deltas)
    s[0] = 0.0
    s[1] = float("nan")
    g = S.measure_gram("sum_reduction", x, (), deltas, s)
    assert g["n_valid"] == 4 and g["n_skipped"] == 2


def test_measure_gram_uses_companions():
    torch.manual_seed(1)
    a = torch.randn(6, 5)
    b = torch.randn(5, 4)
    deltas = [torch.randn_like(a) * 1e-3 for _ in range(6)]
    s = [float((d @ b.double()).abs().max()) for d in
         (dd.double() for dd in deltas)]
    g = S.measure_gram("matmul", a, (b,), deltas, s)
    assert abs(g["log10_median"]) < 1e-9


def test_build_record_floor_gate_withholds_gram(monkeypatch):
    """Below the s/ulp floor the measured side is quantisation; the Gram
    screen must not run there -- the floor screen owns the record."""
    monkeypatch.setattr(S, "_ENABLED", True)
    x = torch.randn(4, 8)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(6)]
    # ref_base magnitude 1e4 -> ulp ~1e0 at float32; sensitivities ~1e-3 are
    # far below 32 ulp.
    ref_base = torch.full((4,), 1.0e4)
    sens = torch.tensor([1e-3] * 6)
    rec = S.build_record(sens, ref_base, None, x, deltas, 1e-2,
                         op_name="sum_reduction")
    assert rec is not None
    assert rec["gram_log10_median"] is None
    assert any(r["reason"] == S.REASON_FP_FLOOR for r in rec["reasons"])


def test_build_record_gram_fires_end_to_end(monkeypatch):
    monkeypatch.setattr(S, "_ENABLED", True)
    torch.manual_seed(2)
    x = torch.randn(8, 16)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(8)]
    ref_base = x.sum(dim=-1)
    sens = torch.tensor([10.0 * v for v in _linear_measured(x, deltas)])
    rec = S.build_record(sens, ref_base, None, x, deltas, 1e-2,
                         op_name="sum_reduction")
    assert any(r["reason"] == S.REASON_GRAM for r in rec["reasons"])
    assert rec["in_scope"] is False
    assert rec["gram_n_valid"] == 8
    assert len(rec["gram_log10_ratios"]) == 8


def test_build_record_draws_no_rng(monkeypatch):
    """Arms A and B must consume the torch generators identically; the screen
    reuses the loop's deltas and adds only deterministic float64 arithmetic."""
    monkeypatch.setattr(S, "_ENABLED", True)
    torch.manual_seed(3)
    x = torch.randn(8, 16)
    deltas = [torch.randn_like(x) * 1e-3 for _ in range(8)]
    sens = torch.tensor(_linear_measured(x, deltas))
    before = torch.get_rng_state()
    S.build_record(sens, x.sum(dim=-1), None, x, deltas, 1e-2,
                   op_name="sum_reduction")
    assert torch.equal(before, torch.get_rng_state())


# --------------------------------------------------------------------------
# 3. math_refs vs torch built-ins (float64, random CPU inputs).
# --------------------------------------------------------------------------

def _close(a, b, tol=1e-12):
    assert torch.allclose(a, b, rtol=tol, atol=tol), (a - b).abs().max()


def test_mathref_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(7, 33, dtype=torch.float64)
    g = torch.randn(33, dtype=torch.float64)
    b = torch.randn(33, dtype=torch.float64)
    _close(M.get("layernorm")(x, g, b),
           F.layer_norm(x, (33,), g, b, eps=1e-5), 1e-10)


def test_mathref_layernorm_slices_oversized_companions():
    """The non_power_of_two variant feeds width-333 x with the original
    width-512 gamma/beta; the kernel reads the first 333 entries."""
    torch.manual_seed(0)
    x = torch.randn(5, 333, dtype=torch.float64)
    g = torch.randn(512, dtype=torch.float64)
    b = torch.randn(512, dtype=torch.float64)
    _close(M.get("layernorm")(x, g, b),
           F.layer_norm(x, (333,), g[:333], b[:333], eps=1e-5), 1e-10)


def test_mathref_rmsnorm_matches_formula():
    torch.manual_seed(0)
    x = torch.randn(6, 40, dtype=torch.float64)
    g = torch.randn(40, dtype=torch.float64)
    want = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g
    _close(M.get("rmsnorm")(x, g), want)


def test_mathref_groupnorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 5, 7, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64)
    _close(M.get("groupnorm")(x, 4, w, b),
           F.group_norm(x, 4, w, b, eps=1e-5), 1e-10)


def test_mathref_instancenorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 6, 6, dtype=torch.float64)
    w = torch.randn(4, dtype=torch.float64)
    b = torch.randn(4, dtype=torch.float64)
    _close(M.get("instancenorm")(x, w, b),
           F.instance_norm(x, weight=w, bias=b, eps=1e-5), 1e-10)


def test_mathref_batchnorm_matches_torch_inference():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5, 5, dtype=torch.float64)
    rm = torch.randn(6, dtype=torch.float64)
    rv = torch.rand(6, dtype=torch.float64) + 0.5
    w = torch.randn(6, dtype=torch.float64)
    b = torch.randn(6, dtype=torch.float64)
    _close(M.get("batchnorm")(x, rm, rv, w, b),
           F.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5), 1e-10)


def test_mathref_cross_entropy_matches_torch():
    torch.manual_seed(0)
    lo = torch.randn(9, 13, dtype=torch.float64)
    t = torch.randint(0, 13, (9,))
    _close(M.get("cross_entropy")(lo, t),
           F.cross_entropy(lo, t, reduction="mean"), 1e-12)


def test_mathref_gelu_swish_match_torch():
    torch.manual_seed(0)
    x = torch.randn(100, dtype=torch.float64)
    _close(M.get("gelu")(x), F.gelu(x, approximate="none"), 1e-12)
    _close(M.get("swish")(x), F.silu(x), 1e-12)


@pytest.mark.parametrize("name,dims,k,s,p", [
    ("avg_pool1d", (2, 4, 17), 3, 2, 1),
    ("avg_pool2d", (2, 4, 17, 17), 3, 2, 1),
    ("avg_pool3d", (2, 4, 9, 9, 9), 3, 2, 1),
    ("avg_pool1d", (2, 4, 33), 2, 2, 0),
    ("max_pool1d", (2, 4, 17), 3, 2, 1),
    ("max_pool2d", (2, 4, 17, 17), 3, 2, 1),
    ("max_pool3d", (2, 4, 9, 9, 9), 3, 2, 1),
    ("max_pool1d", (1, 8, 17), 3, 1, 1),
])
def test_mathref_pools_match_torch(name, dims, k, s, p):
    torch.manual_seed(0)
    x = torch.randn(*dims, dtype=torch.float64)
    tf = {"avg_pool1d": lambda: F.avg_pool1d(x, k, s, p, count_include_pad=True),
          "avg_pool2d": lambda: F.avg_pool2d(x, k, s, p, count_include_pad=True),
          "avg_pool3d": lambda: F.avg_pool3d(x, k, s, p, count_include_pad=True),
          "max_pool1d": lambda: F.max_pool1d(x, k, s, p),
          "max_pool2d": lambda: F.max_pool2d(x, k, s, p),
          "max_pool3d": lambda: F.max_pool3d(x, k, s, p)}[name]
    _close(M.get(name)(x, k, s, p), tf(), 1e-12)


def test_mathref_attention_softmax_reductions():
    torch.manual_seed(0)
    q, k, v = (torch.randn(10, 8, dtype=torch.float64) for _ in range(3))
    want = torch.softmax(q @ k.T / math.sqrt(8), dim=-1) @ v
    _close(M.get("flash_attention")(q, k, v), want)
    _close(M.get("scaled_dot_product_attention")(q, k, v), want)
    causal = M.get("causal_flash_attention")(q, k, v)
    want_c = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True)[0]
    _close(causal, want_c, 1e-10)
    x = torch.randn(5, 12, dtype=torch.float64)
    _close(M.get("softmax")(x), torch.softmax(x, -1))
    _close(M.get("log_softmax")(x), torch.log_softmax(x, -1))
    _close(M.get("frobenius_norm")(x), x / (x.norm() + 1e-12))


def test_mathref_registry_covers_the_perturbation_routed_corpus():
    """Every float-output corpus operator has a math definition; the two
    index-valued ones are structurally excluded instead."""
    need = {"softmax", "log_softmax", "gelu", "swish", "l1norm", "l2norm",
            "frobenius_norm", "layernorm", "rmsnorm", "groupnorm",
            "instancenorm", "batchnorm", "sum_reduction", "mean_reduction",
            "max_reduction", "min_reduction", "cross_entropy", "matmul",
            "flash_attention", "scaled_dot_product_attention",
            "causal_flash_attention", "avg_pool1d", "avg_pool2d",
            "avg_pool3d", "max_pool1d", "max_pool2d", "max_pool3d"}
    assert need <= set(M.registered_ops())
    assert "argmax" not in M.registered_ops()
    assert "argmin" not in M.registered_ops()
