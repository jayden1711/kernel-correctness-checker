"""
tests/verification/adversarial_search/test_materializer.py

Tests for materializer.py: TensorDescriptor → torch.Tensor.

Covers shape, dtype, fill strategies, scale/shift, patches,
edge cases (non-power-of-two shapes, extreme values), and the
tensors_to_inputs calling-convention mapping.

No GPU required.
"""

import math
import uuid
import pytest
import torch

from verification.adversarial_search.schemas import TensorDescriptor, InputProposal
from verification.adversarial_search.materializer import (
    materialize_proposal,
    tensors_to_inputs,
    _safe_eval_index,
    _parse_part,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _desc(**kwargs) -> TensorDescriptor:
    defaults = dict(shape=[4, 16], dtype="float32", fill="randn", scale=1.0, shift=0.0)
    defaults.update(kwargs)
    return TensorDescriptor(**defaults)


def _proposal(operator="softmax", tensors=None) -> InputProposal:
    if tensors is None:
        tensors = {"x": _desc()}
    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id="t",
        iteration=0,
        operator=operator,
        tensors=tensors,
        rationale="",
        predicted_failure_mode="",
    )


def materialize(proposal, device="cpu"):
    return materialize_proposal(proposal, device=device)


# ── Fill strategies ───────────────────────────────────────────────────────────

class TestFillStrategies:
    def test_randn_shape(self):
        p = _proposal(tensors={"x": _desc(shape=[8, 33], fill="randn")})
        t = materialize(p)["x"]
        assert t.shape == (8, 33)

    def test_ones_values(self):
        p = _proposal(tensors={"x": _desc(fill="ones", scale=1.0, shift=0.0)})
        t = materialize(p)["x"]
        assert (t == 1.0).all()

    def test_zeros_values(self):
        p = _proposal(tensors={"x": _desc(fill="zeros")})
        t = materialize(p)["x"]
        assert (t == 0.0).all()

    def test_arange_shape(self):
        p = _proposal(tensors={"x": _desc(fill="arange", shape=[2, 8])})
        t = materialize(p)["x"]
        assert t.shape == (2, 8)
        assert t.flatten()[0].item() == pytest.approx(0.0)
        assert t.flatten()[-1].item() == pytest.approx(15.0)

    def test_literal_values(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        p = _proposal(tensors={"x": _desc(
            fill="literal", literal_values=vals, shape=[2, 2]
        )})
        t = materialize(p)["x"]
        assert t.flatten().tolist() == pytest.approx(vals)

    def test_invalid_fill_raises(self):
        p = _proposal(tensors={"x": _desc(fill="gaussian")})
        with pytest.raises(ValueError, match="Unknown fill"):
            materialize(p)

    def test_literal_without_values_raises(self):
        p = _proposal(tensors={"x": _desc(fill="literal", literal_values=None)})
        with pytest.raises(ValueError, match="literal_values"):
            materialize(p)


# ── Scale and shift ───────────────────────────────────────────────────────────

class TestScaleShift:
    def test_scale(self):
        p = _proposal(tensors={"x": _desc(fill="ones", scale=3.0, shift=0.0)})
        t = materialize(p)["x"]
        assert t.unique().item() == pytest.approx(3.0)

    def test_shift(self):
        p = _proposal(tensors={"x": _desc(fill="zeros", scale=1.0, shift=5.0)})
        t = materialize(p)["x"]
        assert t.unique().item() == pytest.approx(5.0)

    def test_scale_and_shift(self):
        p = _proposal(tensors={"x": _desc(fill="ones", scale=2.0, shift=1.0)})
        t = materialize(p)["x"]
        assert t.unique().item() == pytest.approx(3.0)   # 1.0 * 2.0 + 1.0

    def test_negative_scale(self):
        p = _proposal(tensors={"x": _desc(fill="ones", scale=-1.0, shift=0.0)})
        t = materialize(p)["x"]
        assert t.unique().item() == pytest.approx(-1.0)

    def test_large_scale(self):
        p = _proposal(tensors={"x": _desc(fill="randn", scale=1e4)})
        t = materialize(p)["x"]
        assert t.abs().max().item() > 100.0  # should be much larger than unit normal


# ── Patches ───────────────────────────────────────────────────────────────────

class TestPatches:
    def test_last_column_spike(self):
        p = _proposal(tensors={"x": _desc(
            fill="zeros", patches=[{"indices": "[:, -1]", "value": 1e9}]
        )})
        t = materialize(p)["x"]
        assert (t[:, -1] == 1e9).all()
        assert (t[:, :-1] == 0.0).all()

    def test_last_n_columns(self):
        p = _proposal(tensors={"x": _desc(
            shape=[4, 32], fill="zeros",
            patches=[{"indices": "[:, -8:]", "value": 1e4}]
        )})
        t = materialize(p)["x"]
        assert (t[:, -8:] == 1e4).all()
        assert (t[:, :-8] == 0.0).all()

    def test_first_row(self):
        p = _proposal(tensors={"x": _desc(
            fill="zeros", patches=[{"indices": "[0, :]", "value": 99.0}]
        )})
        t = materialize(p)["x"]
        assert (t[0, :] == 99.0).all()
        assert (t[1:, :] == 0.0).all()

    def test_multiple_patches_applied_in_order(self):
        p = _proposal(tensors={"x": _desc(
            fill="zeros",
            patches=[
                {"indices": "[:, :]", "value": 1.0},
                {"indices": "[:, -1]", "value": 999.0},
            ]
        )})
        t = materialize(p)["x"]
        assert (t[:, -1] == 999.0).all()
        assert (t[:, :-1] == 1.0).all()

    def test_alternating_rows(self):
        p = _proposal(tensors={"x": _desc(
            shape=[6, 4], fill="ones",
            patches=[{"indices": "[1::2, :]", "value": -1.0}]
        )})
        t = materialize(p)["x"]
        assert (t[0::2, :] == 1.0).all()
        assert (t[1::2, :] == -1.0).all()

    def test_bad_patch_index_raises(self):
        p = _proposal(tensors={"x": _desc(
            fill="zeros",
            patches=[{"indices": "[invalid_expr]", "value": 1.0}]
        )})
        with pytest.raises((ValueError, Exception)):
            materialize(p)


# ── Dtypes ────────────────────────────────────────────────────────────────────

class TestDtypes:
    @pytest.mark.parametrize("dtype_str,expected", [
        ("float32",  torch.float32),
        ("float16",  torch.float16),
        ("bfloat16", torch.bfloat16),
    ])
    def test_dtype_cast(self, dtype_str, expected):
        p = _proposal(tensors={"x": _desc(dtype=dtype_str)})
        t = materialize(p)["x"]
        assert t.dtype == expected

    def test_unknown_dtype_defaults_to_float32(self):
        p = _proposal(tensors={"x": _desc(dtype="unknown_dtype")})
        t = materialize(p)["x"]
        assert t.dtype == torch.float32


# ── Edge-case shapes ──────────────────────────────────────────────────────────

class TestEdgeCaseShapes:
    @pytest.mark.parametrize("shape", [
        [1, 1], [1, 512], [512, 1], [333, 333],
        [1000, 777], [2048, 128],
    ])
    def test_various_shapes(self, shape):
        p = _proposal(tensors={"x": _desc(shape=shape)})
        t = materialize(p)["x"]
        assert list(t.shape) == shape

    def test_non_power_of_two_cols(self):
        """Non-power-of-two shapes are critical adversarial inputs — must materialise cleanly."""
        p = _proposal(tensors={"x": _desc(shape=[512, 333])})
        t = materialize(p)["x"]
        assert t.shape == (512, 333)
        assert not torch.isnan(t).any()

    def test_single_row(self):
        p = _proposal(tensors={"x": _desc(shape=[1, 512])})
        t = materialize(p)["x"]
        assert t.shape == (1, 512)


# ── Multi-tensor operators ────────────────────────────────────────────────────

class TestMultiTensor:
    def test_layernorm_all_tensors_materialised(self):
        p = _proposal(
            operator="layernorm",
            tensors={
                "x":     _desc(shape=[8, 16]),
                "gamma": _desc(shape=[16], fill="ones"),
                "beta":  _desc(shape=[16], fill="zeros"),
            }
        )
        tensors = materialize(p)
        assert set(tensors.keys()) == {"x", "gamma", "beta"}
        assert (tensors["gamma"] == 1.0).all()
        assert (tensors["beta"]  == 0.0).all()

    def test_missing_tensor_key_raises(self):
        """materialize_proposal must propagate if a descriptor fails."""
        p = _proposal(tensors={"x": _desc(fill="literal", literal_values=None)})
        with pytest.raises(ValueError):
            materialize(p)


# ── tensors_to_inputs ─────────────────────────────────────────────────────────

class TestTensorsToInputs:
    def test_softmax_returns_tensor(self):
        tensors = {"x": torch.randn(4, 16)}
        inp = tensors_to_inputs("softmax", tensors)
        assert isinstance(inp, torch.Tensor)

    def test_layernorm_returns_tuple(self):
        tensors = {
            "x":     torch.randn(4, 16),
            "gamma": torch.ones(16),
            "beta":  torch.zeros(16),
        }
        inp = tensors_to_inputs("layernorm", tensors)
        assert isinstance(inp, tuple) and len(inp) == 3

    def test_matmul_returns_tuple(self):
        tensors = {"A": torch.randn(8, 16), "B": torch.randn(16, 8)}
        inp = tensors_to_inputs("matmul", tensors)
        assert isinstance(inp, tuple) and len(inp) == 2

    def test_rmsnorm_returns_tuple(self):
        tensors = {"x": torch.randn(4, 16), "gamma": torch.ones(16)}
        inp = tensors_to_inputs("rmsnorm", tensors)
        assert isinstance(inp, tuple) and len(inp) == 2

    def test_flash_attention_returns_tuple(self):
        tensors = {
            "Q": torch.randn(32, 64),
            "K": torch.randn(32, 64),
            "V": torch.randn(32, 64),
        }
        inp = tensors_to_inputs("flash_attention", tensors)
        assert isinstance(inp, tuple) and len(inp) == 3

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            tensors_to_inputs("not_an_op", {})


# ── Index parser ──────────────────────────────────────────────────────────────

class TestSafeEvalIndex:
    @pytest.mark.parametrize("expr,shape,expected", [
        ("[:, -1]",    (4, 16), (slice(None), -1)),
        ("[:, -32:]",  (4, 64), (slice(None), slice(-32, None))),
        ("[0, :]",     (4, 16), (0, slice(None))),
        ("[1::2, :]",  (6, 4),  (slice(1, None, 2), slice(None))),
        ("[:, :]",     (4, 4),  (slice(None), slice(None))),
    ])
    def test_valid_expressions(self, expr, shape, expected):
        result = _safe_eval_index(expr, shape)
        assert result == expected

    def test_single_int_index(self):
        result = _safe_eval_index("[0]", (8,))
        assert result == 0

    def test_negative_index(self):
        result = _safe_eval_index("[-1]", (8,))
        assert result == -1