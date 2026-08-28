"""
Run this against your real corpus before wiring it into the harness:

    python corpus_contract.py your_corpus.py

It checks the required shape and does one smoke-test call per entry so
you find out about a broken input_fn/ref_fn/mutant_fn signature in
seconds, not after a confusing harness result.
"""
import sys
import importlib
import numpy as np

REQUIRED_KEYS = {"op", "mutant_name", "ref_fn", "mutant_fn", "input_fn"}


def validate_corpus(corpus, rng_seed=0):
    if not isinstance(corpus, list) or len(corpus) == 0:
        raise ValueError("corpus must be a non-empty list of dicts")

    rng = np.random.default_rng(rng_seed)
    errors = []

    for i, entry in enumerate(corpus):
        tag = f"entry {i} (op={entry.get('op', '?')}, mutant={entry.get('mutant_name', '?')})"

        missing = REQUIRED_KEYS - set(entry.keys())
        if missing:
            errors.append(f"{tag}: missing keys {missing}")
            continue

        for key in ("ref_fn", "mutant_fn", "input_fn"):
            if not callable(entry[key]):
                errors.append(f"{tag}: '{key}' is not callable")
        if errors and errors[-1].startswith(tag):
            continue

        # Smoke test: input_fn -> ref_fn / mutant_fn must actually run and
        # return same-shape, finite output. This is where real integration
        # bugs (wrong arg order, GPU tensor vs numpy mismatch, etc.) surface.
        try:
            args = entry["input_fn"](rng)
            if not isinstance(args, tuple):
                errors.append(f"{tag}: input_fn(rng) must return a tuple of args, got {type(args)}")
                continue
        except Exception as e:
            errors.append(f"{tag}: input_fn(rng) raised {type(e).__name__}: {e}")
            continue

        try:
            ref_out = entry["ref_fn"](*args)
        except Exception as e:
            errors.append(f"{tag}: ref_fn(*args) raised {type(e).__name__}: {e}")
            continue

        try:
            mutant_out = entry["mutant_fn"](*args)
        except Exception as e:
            errors.append(f"{tag}: mutant_fn(*args) raised {type(e).__name__}: {e}")
            continue

        ref_arr = np.asarray(ref_out)
        mut_arr = np.asarray(mutant_out)
        if ref_arr.shape != mut_arr.shape:
            errors.append(f"{tag}: ref_fn output shape {ref_arr.shape} != "
                           f"mutant_fn output shape {mut_arr.shape} -- harness "
                           f"can't compare these")
        if not np.all(np.isfinite(ref_arr)):
            errors.append(f"{tag}: WARNING ref_fn output contains non-finite "
                           f"values on a random input -- is this really the "
                           f"correct reference?")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python corpus_contract.py your_corpus_module.py")
        sys.exit(1)

    module_path = sys.argv[1].replace(".py", "").replace("/", ".")
    mod = importlib.import_module(module_path)

    if not hasattr(mod, "CORPUS"):
        print(f"ERROR: {module_path} has no top-level CORPUS list")
        sys.exit(1)

    errors = validate_corpus(mod.CORPUS)
    if errors:
        print(f"{len(errors)} problem(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK -- {len(mod.CORPUS)} corpus entries validated, all smoke-tested clean.")
