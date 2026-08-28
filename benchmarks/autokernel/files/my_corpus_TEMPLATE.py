"""
FILL THIS IN with your real 29-operator corpus. Copy this file to
my_corpus.py and replace the example entries.

Each CORPUS entry needs:
  op:          str
  mutant_name: str  (unique per op, e.g. "no_max_subtract")
  ref_fn:      callable(*args) -> array-like (numpy array, or torch tensor
               moved to CPU numpy via .detach().cpu().numpy() inside the fn)
  mutant_fn:   callable(*args) -> array-like, same shape as ref_fn's output
  input_fn:    callable(rng) -> tuple of positional args

If your kernels are Triton/CUDA, wrap the launch + device-to-host copy
inside ref_fn/mutant_fn so the harness only ever sees plain arrays:

    def my_softmax_ref(x_np):
        x = torch.from_numpy(x_np).cuda()
        out = torch.softmax(x, dim=-1)
        return out.detach().cpu().numpy()

    def my_softmax_mutant(x_np):
        x = torch.from_numpy(x_np).cuda()
        out = my_triton_softmax_kernel(x)   # your real kernel launch
        return out.detach().cpu().numpy()

Run `python corpus_contract.py my_corpus.py` after filling this in --
it will smoke-test every entry and tell you exactly which one is broken
before you run the full (slower) benchmark.
"""
import numpy as np

# TODO: import your real reference + mutant kernel callables here, e.g.:
#   from kernelchecker.operators.softmax import softmax_ref, MUTANTS as softmax_mutants
#   import torch


def _example_input_fn(rng, shape=(8, 128)):
    return (rng.normal(size=shape),)


# TODO: replace this entire list by either:
#   (a) hand-writing one dict per (operator, mutant) pair, matching the
#       shape below, or
#   (b) programmatically building it from your existing operator registry,
#       e.g.:
#         CORPUS = []
#         for op_name, spec in your_operator_registry.items():
#             for mutant_name, mutant_fn in spec.mutants.items():
#                 CORPUS.append(dict(
#                     op=op_name, mutant_name=mutant_name,
#                     ref_fn=spec.reference, mutant_fn=mutant_fn,
#                     input_fn=spec.input_fn,
#                 ))
CORPUS = [
    dict(
        op="EXAMPLE_op_name",
        mutant_name="EXAMPLE_mutant_name",
        ref_fn=lambda x: x,       # TODO replace
        mutant_fn=lambda x: x,    # TODO replace
        input_fn=_example_input_fn,
    ),
]
