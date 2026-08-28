"""
Spec shape-contract test -- the systemic half of the OOB fix
(verification_runs/oob_adjudication_2026-08-28/ FINDINGS §5.3).

Every spec's get_adversarial_inputs is executed at EVERY entry of its own
valid_shapes AND at the autokernel corpus's shapes (tritonbench_registry),
and every returned tuple is checked against the shape contract:

  R1  a 1-D floating companion whose base length equals the base primary's
      last dimension (a per-feature companion: gamma, beta, weight, bias)
      must, in every adversarial tuple, have length equal to THAT tuple's
      primary's last dimension;
  R2  a companion with exactly the base primary's shape (elementwise
      companion: masks) must have the adversarial primary's shape;
  R3  where a float64 math definition exists (math_refs), it must execute
      on the adversarial tuple without raising -- an executable
      shape-consistency check covering companion structures R1/R2 do not
      (matmul operands, attention K/V).

This is the test that would have caught the layernorm/rmsnorm
non_power_of_two out-of-bounds construction before it ran (205 floats read
past the captured gamma/beta at the corpus's width-128 shapes, three rounds
undetected). It failed on the pre-fix specs at (2048, 128) and at the
registry shapes; it must pass forever after.

Coverage is honest, not total: specs whose companions are scalars, 2-D
operands, or absent are exercised through R3 or vacuously; the file reports
how many (spec, shape) pairs and rule applications actually ran, and pins a
floor on both so silent erosion of coverage fails the test.

Also here: unit pins for the fixed variant itself -- the width adaptation
table and the DRAW-THEN-SLICE stream-preservation property (the fix must
not shift the torch draws of variants generated after non_power_of_two).
The reference-wrapper ValueError cannot be tested on this machine (the
wrappers import triton); the GPU round exercises it.
"""

import glob
import importlib
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "autokernel", "files"))

from verification.layer2_numeric_oracle import math_refs

SPEC_DIR = os.path.join(ROOT, "verification", "specs")
SKIP_MODULES = {"base_spec", "__init__"}


def spec_modules():
    out = []
    for p in sorted(glob.glob(os.path.join(SPEC_DIR, "*.py"))):
        name = os.path.basename(p)[:-3]
        if name not in SKIP_MODULES:
            out.append(name)
    return out


def load_spec(name):
    mod = importlib.import_module(f"verification.specs.{name}")
    return mod.get_spec()


def to64(t):
    if torch.is_tensor(t):
        return t.detach().double() if t.is_floating_point() else t.detach()
    return t


def check_contract(spec, inputs, where):
    """Apply R1/R2/R3 to every adversarial tuple. Returns rule-application
    counts; raises AssertionError with a precise message on violation."""
    applied = {"R1": 0, "R2": 0, "R3": 0}
    base_primary = spec.primary_input(inputs)
    base_comps = tuple(inputs[1:]) if isinstance(inputs, tuple) else ()

    feature_idx = [i for i, c in enumerate(base_comps)
                   if torch.is_tensor(c) and c.is_floating_point()
                   and c.dim() == 1
                   and c.numel() == base_primary.shape[-1]]
    # R2 is the structural fallback for specs WITHOUT an executable math
    # definition: for those, a companion shaped exactly like the primary is
    # mask-like and must track the primary's shape. Where math_refs covers
    # the op, R3 validates companion consistency executably and R2 must not
    # apply -- a square matmul's B coincidentally matches A's shape, and its
    # shape-changing variants legitimately regenerate B at a different,
    # consistent shape.
    samewise_idx = [] if math_refs.get(spec.name) is not None else [
        i for i, c in enumerate(base_comps)
        if torch.is_tensor(c)
        and tuple(c.shape) == tuple(base_primary.shape)
        and i not in feature_idx]

    for name, adv in spec.get_adversarial_inputs(inputs):
        adv_primary = spec.primary_input(adv)
        adv_comps = tuple(adv[1:]) if isinstance(adv, tuple) else ()
        loc = f"{where}/{name}"
        for i in feature_idx:
            applied["R1"] += 1
            c = adv_comps[i]
            assert torch.is_tensor(c) and c.numel() == adv_primary.shape[-1], (
                f"{loc}: companion #{i} length {c.numel() if torch.is_tensor(c) else c} "
                f"!= adversarial primary width {adv_primary.shape[-1]} "
                f"(the OOB construction of oob_adjudication_2026-08-28)")
        for i in samewise_idx:
            applied["R2"] += 1
            c = adv_comps[i]
            assert torch.is_tensor(c) and tuple(c.shape) == tuple(adv_primary.shape), (
                f"{loc}: elementwise companion #{i} shape "
                f"{tuple(c.shape) if torch.is_tensor(c) else c} != primary "
                f"{tuple(adv_primary.shape)}")
        fn = math_refs.get(spec.name)
        if fn is not None:
            applied["R3"] += 1
            try:
                fn(to64(adv_primary), *[to64(c) for c in adv_comps])
            except Exception as e:
                raise AssertionError(
                    f"{loc}: math definition raised on the adversarial "
                    f"tuple -- shape-inconsistent construction: {e!r}")
    return applied


def _collect_cases():
    """(spec_name, shape) pairs from every spec's own valid_shapes."""
    cases = []
    for name in spec_modules():
        try:
            spec = load_spec(name)
            shapes = spec.valid_shapes
        except Exception:
            continue
        for sh in shapes:
            cases.append((name, sh))
    return cases


CASES = _collect_cases()


@pytest.mark.parametrize("name,shape", CASES,
                         ids=[f"{n}-{s}" for n, s in CASES])
def test_adversarial_tuples_satisfy_shape_contract(name, shape):
    torch.manual_seed(0)
    spec = load_spec(name)
    try:
        inputs = spec.make_inputs(shape, "cpu", torch.float32)
    except NotImplementedError:
        pytest.skip("spec has no make_inputs")
    try:
        check_contract(spec, inputs, f"{name}@{shape}")
    except NotImplementedError:
        pytest.skip("spec has no adversarial battery")


def test_contract_at_registry_corpus_shapes():
    """The corpus where the OOB construction actually ran: the registry's
    own input builders (numpy, converted to CPU torch), including the
    width-128 layernorm/rmsnorm shapes."""
    from tritonbench_registry import OPS, FAMILIES
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_checked = 0
    for spec_key, _ref, _cheat, family, _muts in OPS:
        try:
            spec = load_spec(spec_key)
        except Exception:
            continue
        np_args = FAMILIES[family][0](rng)
        ts = tuple(torch.from_numpy(a) if isinstance(a, np.ndarray) else a
                   for a in np_args)
        inputs = ts if len(ts) > 1 else ts[0]
        try:
            check_contract(spec, inputs, f"{spec_key}@registry")
            n_checked += 1
        except NotImplementedError:
            continue
    assert n_checked >= 25, f"registry contract coverage eroded: {n_checked}"


def test_coverage_floor():
    """The contract test is only as good as what it exercises; pin it."""
    assert len(CASES) >= 100, f"only {len(CASES)} (spec, shape) cases collected"
    torch.manual_seed(0)
    r1 = r3 = 0
    for name, shape in CASES:
        spec = load_spec(name)
        try:
            inputs = spec.make_inputs(shape, "cpu", torch.float32)
            a = check_contract(spec, inputs, f"{name}@{shape}")
        except NotImplementedError:
            continue
        r1 += a["R1"]
        r3 += a["R3"]
    assert r1 >= 40, f"R1 (feature-companion) applications collapsed: {r1}"
    assert r3 >= 200, f"R3 (executable math) applications collapsed: {r3}"


# ---------------------------------------------------------------------------
# Unit pins for the fixed variant.
# ---------------------------------------------------------------------------

def test_non_pow2_width_table():
    from verification.specs.layernorm import _non_pow2_width as w_ln
    from verification.specs.rmsnorm import _non_pow2_width as w_rm
    for f in (w_ln, w_rm):
        assert f(512) == 333          # >= 333: unchanged pre-fix behaviour
        assert f(1024) == 333
        assert f(333) == 333          # non-pow2 base kept as-is
        assert f(128) == 127          # the corpus configuration
        assert f(256) == 255
        assert f(129) == 129
        assert f(4) == 3
        w = f(128)
        assert w & (w - 1) != 0       # non-power-of-two, always


def test_variant_width_never_exceeds_companions():
    """The exact defect, pinned: at every base width the variant's primary
    width equals its sliced companions' length."""
    from verification.specs.layernorm import get_spec as ln
    from verification.specs.rmsnorm import get_spec as rm
    for base_cols in (64, 128, 333, 512, 2048):
        torch.manual_seed(0)
        for spec in (ln(), rm()):
            inputs = spec.make_inputs((8, base_cols), "cpu", torch.float32)
            for name, adv in spec.get_adversarial_inputs(inputs):
                x = adv[0]
                for c in adv[1:]:
                    assert c.numel() == x.shape[-1], (spec.name, name,
                                                      base_cols)


def test_draw_then_slice_preserves_the_rng_stream():
    """Variants generated AFTER non_power_of_two must receive exactly the
    draws they received pre-fix: the fix draws the old (rows, 333) shape and
    slices, so the reseeded stream position is unchanged. Emulate the
    pre-fix generator sequence for rmsnorm and compare the post-fix battery's
    constant_rows / large_variance tensors draw for draw."""
    from verification.specs.rmsnorm import get_spec
    spec = get_spec()
    rows, cols = 8, 128
    torch.manual_seed(1234)
    x = torch.randn(rows, cols)
    gamma = torch.randn(cols).abs() + 0.1

    torch.manual_seed(42)
    got = dict(spec.get_adversarial_inputs((x, gamma)))

    torch.manual_seed(42)                      # pre-fix sequence, verbatim
    _ = torch.randn_like(x) * 1e4              # large_magnitude
    _ = torch.randn_like(x) * 1e-8             # near_zero
    _ = torch.randn(rows, 333)                 # non_power_of_two (old draw)
    vals = torch.randn(rows, 1) * 10.0         # constant_rows
    want_const = vals.expand(rows, cols).contiguous()
    want_lv = torch.zeros(rows, cols)
    mid = cols // 2
    want_lv[:, mid:] = torch.randn(rows, cols - mid) * 1e4  # large_variance

    assert torch.equal(got["constant_rows"][0], want_const)
    assert torch.equal(got["large_variance"][0], want_lv)
    # and the fixed variant itself is the old draw's left slice
    torch.manual_seed(42)
    _ = torch.randn_like(x); _ = torch.randn_like(x)
    old_npw = torch.randn(rows, 333)
    assert torch.equal(got["non_power_of_two"][0], old_npw[:, :127])


def test_contract_rejects_the_prefix_construction():
    """Negative control: the exact pre-fix tuple -- width-333 primary over
    the captured width-128 companions -- must FAIL the contract check. If
    this ever passes, the contract test has stopped guarding the defect
    class it was written for."""
    from verification.specs.rmsnorm import get_spec
    spec = get_spec()
    torch.manual_seed(0)
    inputs = spec.make_inputs((8, 128), "cpu", torch.float32)
    x, gamma = inputs
    prefix_tuple = [("non_power_of_two",
                     (torch.randn(8, 333), gamma))]      # pre-fix construction
    spec.get_adversarial_inputs = lambda _inputs: prefix_tuple
    with pytest.raises(AssertionError, match="OOB construction"):
        check_contract(spec, inputs, "negative-control")
