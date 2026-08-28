"""Minimal repro for compute-sanitizer: one layernorm launch at the exact
corpus configuration (x width 333, gamma/beta length 128). An in-bounds
control launch (length-333 companions) runs first so any report can be
attributed to the mismatched call, not to the kernel per se."""
import sys
import torch

sys.path.insert(0, "/content")
from TritonBench.reference.layernorm import layernorm

torch.manual_seed(0)
x = torch.randn(8, 333, device="cuda")
g333 = torch.randn(333, device="cuda")
b333 = torch.randn(333, device="cuda")
layernorm(x, g333, b333)          # control: valid-length companions
torch.cuda.synchronize()
print("CONTROL-DONE", flush=True)
g128 = torch.randn(128, device="cuda")
b128 = torch.randn(128, device="cuda")
layernorm(x, g128, b128)          # the corpus configuration
torch.cuda.synchronize()
print("MISMATCH-DONE", flush=True)
