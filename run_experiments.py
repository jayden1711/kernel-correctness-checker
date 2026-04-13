import torch
from TritonBench.reference.softmax_reference import softmax as ref_softmax
from TritonBench.kernels.softmax_cheating_first_tile import softmax as cheat_first_tile
from TritonBench.kernels.softmax_cheating_wrong_reduction import softmax as cheat_wrong_reduction

x = torch.randn(512, 2048, device='cuda')  # n_cols > BLOCK_SIZE so first-tile cheat activates

print("Experiment 1: First Tile Cheat")
ref = ref_softmax(x)
cheat = cheat_first_tile(x)
print(f"Passes allclose: {torch.allclose(ref, cheat, atol=1e-4, rtol=1e-2)}")
print(f"Max absolute error: {torch.max(torch.abs(ref - cheat)):.6f}")

print("Experiment 2: Wrong Reduction Cheat")
cheat = cheat_wrong_reduction(x, PARTIAL_SIZE=2040) 
print(f"Passes allclose: {torch.allclose(ref, cheat, atol=1e-4, rtol=1e-2)}")
print(f"Max absolute error: {torch.max(torch.abs(ref - cheat)):.6f}")