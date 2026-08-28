"""
NEAR-MISS corpus: 50 deliberately-marginal mutants (5 ops x design margins
{0.5, 0.8, 1.0, 1.25, 2.0}x, two series), built 2026-08-28 so tolerance
experiments have a non-flat response surface:

  m-series  margins target the adaptive PERTURBATION tolerance (still
            verdict-caught by tighter checks -- check-level instrument);
  v-series  margins target the op's BINDING check (the smallest flip-delta
            in the whole pipeline: layernorm affine_correctness, softmax
            adversarial_max_in_last_tile, gelu adversarial_near_global_min,
            l2norm/sum_reduction cross_shape) -- VERDICT-level instrument,
            GPU-validated straddle 0/0/42/100/100% across the ladder
            (verification_runs/near_miss_verdict_2026-08-28/).

NOT part of the published corpus (my_corpus.CORPUS) and never to be merged
into it -- the published 40/40, 0/200 numbers are defined on the original
mutant set. Design, GPU validation (response curve 0/6/42/90/100% caught
across the margin ladder) and the scale-identifiability demonstration:
verification_runs/near_miss_2026-08-28/FINDINGS.md.

Entries have the same shape as my_corpus.CORPUS entries and run through
checker_adapter unchanged:

    from near_miss_corpus import NEAR_MISS_CORPUS
"""
import importlib

from tritonbench_registry import FAMILIES, _find_raw_kernel

NEAR_MISS_OPS = [
    ("layernorm", "layernorm", "layernorm"),
    ("softmax", "softmax", "single"),
    ("gelu", "gelu", "single"),
    ("l2norm", "l2norm", "single"),
    ("sum_reduction", "sum_reduction", "single"),
]
MARGINS = ["m050", "m080", "m100", "m125", "m200",
           "v050", "v080", "v100", "v125", "v200"]


def build_near_miss_corpus():
    corpus = []
    for spec_key, ref_file, family in NEAR_MISS_OPS:
        mk_fn, to_torch = FAMILIES[family]
        ref_module = importlib.import_module(
            f"TritonBench.reference.{ref_file}")
        ref_torch_fn = getattr(ref_module, spec_key)
        raw_kernel_ref = _find_raw_kernel(ref_module, spec_key)
        spec = importlib.import_module(
            f"verification.specs.{spec_key}").get_spec()
        for mname in MARGINS:
            mod = importlib.import_module(
                f"TritonBench.near_miss.{spec_key}.{mname}")
            mutant_torch_fn = getattr(mod, spec_key)
            raw_kernel_mutant = _find_raw_kernel(mod, spec_key)

            def _make_np_fn(torch_fn, to_torch=to_torch):
                def _fn(*np_args):
                    torch_inputs = to_torch(np_args)
                    if isinstance(torch_inputs, tuple):
                        out = torch_fn(*torch_inputs)
                    else:
                        out = torch_fn(torch_inputs)
                    return out.detach().cpu().numpy()
                return _fn

            corpus.append(dict(
                op=spec_key,
                mutant_name=f"near_miss_{mname}",
                ref_fn=_make_np_fn(ref_torch_fn),
                mutant_fn=_make_np_fn(mutant_torch_fn),
                input_fn=mk_fn,
                family=family,
                to_torch=to_torch,
                spec=spec,
                torch_ref_fn=ref_torch_fn,
                torch_mutant_fn=mutant_torch_fn,
                raw_kernel_ref=raw_kernel_ref,
                raw_kernel_mutant=raw_kernel_mutant,
            ))
    return corpus


NEAR_MISS_CORPUS = None  # built lazily: requires a GPU + triton to import


def get_corpus():
    global NEAR_MISS_CORPUS
    if NEAR_MISS_CORPUS is None:
        NEAR_MISS_CORPUS = build_near_miss_corpus()
    return NEAR_MISS_CORPUS
