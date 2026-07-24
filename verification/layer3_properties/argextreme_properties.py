"""
verification/layer3_properties/argextreme_properties.py

Shared properties for argmax/argmin. Both TRUE exactly:
  - shift invariance: argmax(x+c) == argmax(x) for any c (adding a
    constant to every element cannot change which index is extremal).
  - positive_scale_invariance: argmax(c*x) == argmax(x) for c > 0
    (order preserved under positive scaling). Deliberately NOT tested
    for c < 0, where argmax/argmin swap roles.
"""

import torch


def check_shift_invariance(kernel_fn, x: torch.Tensor, shift: float = 37.0):
    out1 = kernel_fn(x)
    out2 = kernel_fn(x + shift)
    ok = torch.equal(out1, out2)
    return ok, "index unchanged under shift" if ok else "index CHANGED under shift"


def check_positive_scale_invariance(kernel_fn, x: torch.Tensor, scale: float = 3.3):
    out1 = kernel_fn(x)
    out2 = kernel_fn(x * scale)
    ok = torch.equal(out1, out2)
    return ok, "index unchanged under positive rescale" if ok else "index CHANGED under positive rescale"
