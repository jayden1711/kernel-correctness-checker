import torch
from TritonBench.reference.softmax_reference import softmax as ref_softmax
from TritonBench.kernels.softmax_cheating_first_tile import softmax as cheat_first_tile, softmax_kernel_cheat_first_tile
from TritonBench.kernels.softmax_cheating_wrong_reduction import softmax as cheat_wrong_reduction
from TritonBench.reference.softmax_reference import softmax_kernel as ref_kernel
from checks.properties import check_rows_sum_to_one, check_shift_invariance
from checks.structural import check_all_tiles_visited
from checks.oracle import check_adversarial

#shared inputs
x      = torch.randn(512, 2048, device='cuda')
x_small = torch.randn(4,   2048, device='cuda')
x_adv  = torch.randn(512, 2048, device='cuda')
x_adv[:, -1] = 5.0

ref        = ref_softmax(x)
wrong_fn   = lambda x: cheat_wrong_reduction(x, PARTIAL_SIZE=2040)

#Experiment 1: First Tile Cheat
print("Experiment 1: First Tile Cheat")
cheat = cheat_first_tile(x)
print(f"  Passes allclose : {torch.allclose(ref, cheat, atol=1e-4, rtol=1e-2)}")
print(f"  Max abs error   : {torch.max(torch.abs(ref - cheat)):.6f}")

#Experiment 2: Wrong Reduction Cheat
print("\nExperiment 2: Wrong Reduction Cheat")
cheat = wrong_fn(x)
print(f"  Passes allclose : {torch.allclose(ref, cheat, atol=1e-4, rtol=1e-2)}")
print(f"  Max abs error   : {torch.max(torch.abs(ref - cheat)):.6f}")

#Experiment 3: Algebraic Property Tests
print("\nExperiment 3: Algebraic Property Tests")
for name, fn in [("first_tile", cheat_first_tile), ("wrong_reduction", wrong_fn)]:
    output   = fn(x)
    rows_sum = check_rows_sum_to_one(output)
    shift_inv = check_shift_invariance(fn, x)
    print(f"  {name}: rows_sum_to_one={rows_sum}, shift_invariant={shift_inv}")

#Experiment 4: Structural Access Pattern
print("\nExperiment 4: Structural Access Pattern Check")
passed, fail_row, cols = check_all_tiles_visited(ref_softmax, ref_kernel, x_small)
print(f"  Reference   : all_tiles_visited={passed}")
passed, fail_row, cols = check_all_tiles_visited(cheat_first_tile, softmax_kernel_cheat_first_tile, x_small)
print(f"  First tile  : all_tiles_visited={passed}, fail_row={fail_row}, cols={cols}/2048")

#xperiment 5: Adversarial Input Oracle
print("\nExperiment 5: Adversarial Input Oracle")
ref_adv = ref_softmax(x_adv)
cheat   = wrong_fn(x_adv)
passes  = torch.allclose(ref_adv, cheat, atol=1e-4, rtol=1e-2)
err     = torch.max(torch.abs(ref_adv - cheat))
print(f"  Wrong reduction on adversarial input: passes={passes}, max_err={err:.6f}")