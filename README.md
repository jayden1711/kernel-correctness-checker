# Kernel Correctness Checker
 
This project checks whether AI-generated GPU kernels (written in Triton) are actually *correct*, not just "looks right" or "ran without crashing."
 
## The problem this solves
 
If you ask an LLM to write a GPU kernel and you only check "does the output roughly match a reference on random input," you'll miss a lot of real bugs. A kernel can pass that kind of check and still be wrong, as it might only process half the input, skip a normalization step, use the wrong tolerance, or break the moment the input isn't a nice round shape like 512x512.
 
This checker is more thorough and runs each kernel through three layers of verification:
 
1. **Structural checks** — did the kernel actually run, is the output the right shape and type, is it deterministic, did every part of the output actually get written to (not just left over from memory)?
2. **Numeric checks** — does the output match a trusted reference, not just on one easy input, but across different shapes and a set of intentionally tricky inputs (extreme values, odd shapes, near-zero variance, etc.)?
3. **Algebraic checks** — does the output obey the math it's supposed to obey? (Softmax rows should sum to 1. Layernorm should have zero mean. That kind of thing.)
A kernel is only verified if it survives all three.
 
## Two ways this gets used
 
**1. Testing kernels you (or an LLM) write yourself**, against four operators: softmax, layernorm, matmul, and flash attention. This is the original, hand-built part of the project. The checker was validated against 13 deliberately-broken kernels before ever being trusted on real output.
 
**2. Testing kernels against TritonBench**, a public benchmark of ~180 real-world Triton kernels scraped from GitHub, plus a big set of LLM-generated attempts at those same kernels from models like DeepSeek, Qwen, GPT-4o, and Claude. This part lets you check how good a given model actually is at writing correct Triton code, and not just whether it compiles.
 
## Setup
 
```bash
pip install torch triton triton-viz hypothesis litellm
```
 
You'll also need a Triton-capable GPU
 
## Quick start

### First experimental checks showing bugs that this checker catches where naive allclose passes
 
```bash
python run_experiments.py
```
 
This runs through each operator and shows the exact progression described above: a broken kernel that passes a naive check, then gets caught once algebraic checks or adversarial inputs are added. No setup needed — good first thing to run if you want to see the point of this project before digging into anything else.
 
### Check one kernel file you already have
 
```bash
python run_checker_manual.py path/to/your_kernel.py --operator softmax
```
 
Operators supported: `softmax`, `layernorm`, `matmul`, `flash_attention`. It'll try to find the right function in your file automatically, or you can point it directly with `--func your_function_name`.
 
### Generate kernels with an LLM and check them automatically
 
```bash
python run_generation.py --softmax --n 3 --model claude-sonnet-5 --api-key sk-ant-...
```
 
This asks the model to write 3 softmax kernels, saves them, and immediately runs each one through the full checker so you see a verdict right away — no separate step needed.
 
### Run the whole hand-built test suite
 
```bash
python run_checker.py
```
 
This runs the checker against a set of 13 kernels that were built with specific, known bugs in them (dropped normalization, wrong axis, missing masking, etc.) — good for confirming the checker itself still works correctly.
 
### Check real-world TritonBench kernels, or LLM attempts at them
 
```bash
python run_tritonbench_llm.py \
  --jsonl path/to/some_model_output.jsonl \
  --repo path/to/TritonBench_G_v1 \
  --stats path/to/TritonBench_G_v1.json \
  --n 30
```
 
This matches each LLM-generated kernel to its real ground-truth counterpart and reports whether it's actually correct, split into categories: passed, genuinely wrong, or never even ran (with the reason why).
 
## What works
 
- All four operators (softmax, layernorm, matmul, flash attention) have at least one confirmed real bug they catch that a naive comparison would miss.
- The TritonBench adapter has been tested against real kernels from the benchmark and real LLM output, not just self-comparisons.


## What's still open
 
- Only a small sample of the full TritonBench dataset has been run through the LLM-candidate checker so far (a few hundred kernels out of several thousand available).
- A handful of TritonBench's own reference kernels fail for reasons unrelated to correctness - GPU memory limits, older Triton syntax, etc. These are logged and skipped, not silently ignored.
- The tile-coverage check (making sure every part of the output actually got computed) works well for softmax specifically, and has a separate, more general version for the other operators, but the two aren't quite the same check under the hood, which is worth knowing.

## Project layout
 
```
verification/           the core three-layer checker
  checker.py             runs all the checks in order
  specs/                 defines input/output shape per operator
  layer1_structural/      structural checks
  layer2_numeric_oracle/  numeric comparison + adversarial inputs
  layer3_properties/      algebraic invariant checks
  tritonbench_adapter.py  the TritonBench integration
 
TritonBench/             hand-built reference kernels and known-broken test kernels
run_experiments.py       demo showing why naive checks aren't enough, per operator
run_generation.py        ask an LLM to write kernels, check them right away
run_checker.py           run the built-in test suite of known-broken kernels
run_checker_manual.py    check a single kernel file
run_tritonbench_llm.py   check LLM attempts at real TritonBench kernels
```

## Related Work

- [AccelOpt](https://arxiv.org/pdf/2511.12638) — motivating paper
- [robust-kbench](https://arxiv.org/abs/2509.14279) — empirical evidence cheating is widespread
- [TTrace](https://arxiv.org/pdf/2506.09280) — perturbation-based tolerance estimation
- [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) — runtime instrumentation backend
- [Volta](https://arxiv.org/pdf/2311.02737) — PTX-level formal equivalence checking, long-term direction
- [AutoKernel](https://arxiv.org/abs/2603.21331) — adversarial input harness design