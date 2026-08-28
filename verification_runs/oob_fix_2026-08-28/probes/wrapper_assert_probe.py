"""The wrapper shape-asserts, exercised on the GPU (the wrappers import
triton, so this cannot run on the dev machine). Matched companions must
work; short companions must raise ValueError BEFORE any kernel launch."""
import sys
import torch

sys.path.insert(0, "/content")
from TritonBench.reference.layernorm import layernorm
from TritonBench.reference.rmsnorm import rmsnorm

torch.manual_seed(0)
x = torch.randn(8, 333, device="cuda")
ok = layernorm(x, torch.randn(333, device="cuda"),
               torch.randn(333, device="cuda"))
assert ok.shape == x.shape
ok = rmsnorm(x, torch.randn(333, device="cuda"))
assert ok.shape == x.shape
print("matched companions: both wrappers ran", flush=True)

for fn, args, tag in [
        (layernorm, (x, torch.randn(128, device="cuda"),
                     torch.randn(333, device="cuda")), "layernorm short gamma"),
        (layernorm, (x, torch.randn(333, device="cuda"),
                     torch.randn(128, device="cuda")), "layernorm short beta"),
        (rmsnorm, (x, torch.randn(128, device="cuda")), "rmsnorm short gamma")]:
    try:
        fn(*args)
    except ValueError as e:
        print(f"{tag}: ValueError as required -- {e}", flush=True)
    else:
        print(f"{tag}: *** NO ERROR -- ASSERT MISSING ***", flush=True)
        sys.exit(1)
print("WRAPPER-ASSERTS-OK", flush=True)
