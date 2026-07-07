"""
run_generation.py  Generate Triton kernels for softmax, layernorm, matmul,
and flash attention using any LLM via LiteLLM, then save them to generated/.

Usage:
    python run_generation.py --all                                  # all operators, default model
    python run_generation.py --softmax --matmul                     # specific operators
    python run_generation.py --all --n 10                           # override count
    python run_generation.py --all --model gpt-4o                   # use GPT-4o
    python run_generation.py --all --model deepseek/deepseek-coder  # use DeepSeek

Supported models (set the corresponding env var):
    claude-sonnet-4-20250514    export ANTHROPIC_API_KEY=...
    gpt-4o                      export OPENAI_API_KEY=...
    deepseek/deepseek-coder     export DEEPSEEK_API_KEY=...
    gemini/gemini-2.5-flash     export GEMINI_API_KEY=...
    Any model supported by LiteLLM: https://docs.litellm.ai/docs/providers

Requirements:
    pip install litellm -q

Each generated kernel is saved to:
    generated/<model_short>/<operator>/kernel_<i>.py
"""

import os
import re
import argparse
import time
import litellm

# Suppress litellm verbose logging
litellm.set_verbose = False

# Problem specs

PROBLEM_SPECS = {
    "softmax": {
        "description": "Row-wise softmax over a 2D tensor.",
        "pytorch_ref": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 2048

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return []
""",
    },
    "layernorm": {
        "description": "Layer normalization over the last dimension of a 2D tensor, with learnable gamma and beta.",
        "pytorch_ref": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

batch_size = 512
dim = 512

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return [(dim,)]
""",
    },
    "matmul": {
        "description": "Matrix multiplication C = A @ B for 2D tensors.",
        "pytorch_ref": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

M, K, N = 512, 512, 512

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]

def get_init_inputs():
    return []
""",
    },
    "flash_attention": {
        "description": (
            "Flash Attention: compute softmax(Q @ K^T / sqrt(d)) @ V "
            "in a memory-efficient tiled manner for 2D inputs "
            "(seq_len x head_dim)."
        ),
        "pytorch_ref": """
import torch

def flash_attention_reference(Q, K, V):
    \"\"\"Reference: scaled dot-product attention.\"\"\"
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

seq_len = 128
head_dim = 64

def get_inputs():
    Q = torch.rand(seq_len, head_dim)
    K = torch.rand(seq_len, head_dim)
    V = torch.rand(seq_len, head_dim)
    return [Q, K, V]

def get_init_inputs():
    return []
""",
    },
}

FUNC_NAMES = {
    "softmax":         "softmax",
    "layernorm":       "layernorm",
    "matmul":          "matmul",
    "flash_attention": "flash_attention",
}

# Prompts

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton.
When asked to implement an operator in Triton, you write correct, complete,
runnable Python code using the triton and triton.language (tl) libraries.
Your implementation must:
1. Include a @triton.jit decorated kernel function.
2. Include a Python wrapper function that launches the kernel.
3. Be numerically correct  match the reference PyTorch implementation
   on all valid inputs, not just typical ones.
4. Handle edge cases: non-power-of-two sizes, non-square matrices,
   boundary tiles, arbitrary batch sizes.
5. NOT take shortcuts that only work on specific input sizes.
Output ONLY the Python code, no explanation, no markdown fences.\
"""

USER_TEMPLATE = """\
Implement the following operator as a Triton kernel.

Operator: {description}

Reference PyTorch implementation:
{pytorch_ref}

Requirements:
- Your file must define a callable Python function with the same
  interface as the Model.forward() above (or the reference function
  for flash attention).
- The function must be named exactly: {func_name}
- It must accept the same arguments as the reference and return a
  torch.Tensor of the same shape and dtype.
- Use @triton.jit for the GPU kernel and launch it from the wrapper.
- Do not use torch.softmax, torch.nn.LayerNorm, torch.matmul, or any
  other PyTorch high-level op as a shortcut inside your implementation.

Output only valid Python code.\
"""

# Helpers

def extract_code(text: str) -> str:
    """Strip markdown fences if the model wraps code in them."""
    text = text.strip()
    fenced = re.match(r"^```(?:python)?\n(.*?)```$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def model_short_name(model: str) -> str:
    """Turn 'claude-sonnet-4-20250514' into 'claude-sonnet' for folder names."""
    base = model.split("/")[-1]          # strip provider prefix e.g. deepseek/
    parts = base.split("-")
    # Keep first two meaningful parts, drop date suffixes
    meaningful = [p for p in parts if not p.isdigit() and len(p) > 2]
    return "-".join(meaningful[:2]) if len(meaningful) >= 2 else base[:20]


def generate_kernel(model: str, operator: str) -> str:
    spec = PROBLEM_SPECS[operator]
    user_msg = USER_TEMPLATE.format(
        description=spec["description"],
        pytorch_ref=spec["pytorch_ref"].strip(),
        func_name=FUNC_NAMES[operator],
    )
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=4096,
    )
    raw = response.choices[0].message.content
    return extract_code(raw)


def save_kernel(model: str, operator: str, index: int,
                code: str, out_dir: str) -> str:
    op_dir = os.path.join(out_dir, model_short_name(model), operator)
    os.makedirs(op_dir, exist_ok=True)
    path = os.path.join(op_dir, f"kernel_{index}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def check_api_key(model: str):
    """Warn early if the likely required env var is missing."""
    key_map = {
        "claude":   "ANTHROPIC_API_KEY",
        "gpt":      "OPENAI_API_KEY",
        "openai":   "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini":   "GEMINI_API_KEY",
        "mistral":  "MISTRAL_API_KEY",
    }
    model_lower = model.lower()
    for prefix, env_var in key_map.items():
        if prefix in model_lower:
            if not os.environ.get(env_var):
                print(f"WARNING: {env_var} not set  {model} calls will likely fail.")
            return
        
def _verify_reference_consistency():
    """
    Sanity check: the plain-PyTorch pytorch_ref text shown to the LLM must
    numerically match TritonBench's actual reference implementation.
    If this ever fails, PROBLEM_SPECS has drifted from ground truth and
    every generated kernel is being scored against the wrong target.
    """
    import torch
    from TritonBench.reference.softmax import softmax as ref_softmax
    from TritonBench.reference.layernorm import layernorm as ref_layernorm
    from TritonBench.reference.mat_mult import matmul as ref_matmul
    from TritonBench.reference.flash_attention import flash_attention as ref_flash_attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    x = torch.rand(4096, 2048, device=device)
    assert torch.allclose(torch.softmax(x, dim=1), ref_softmax(x), atol=1e-4), \
        "softmax pytorch_ref no longer matches TritonBench.reference.softmax"

    x = torch.rand(512, 512, device=device)
    gamma = torch.ones(512, device=device)
    beta = torch.zeros(512, device=device)
    assert torch.allclose(
        torch.nn.functional.layer_norm(x, [512]),
        ref_layernorm(x, gamma, beta), atol=1e-4
    ), "layernorm pytorch_ref no longer matches TritonBench.reference.layernorm"

    A = torch.rand(512, 512, device=device)
    B = torch.rand(512, 512, device=device)
    assert torch.allclose(torch.matmul(A, B), ref_matmul(A, B), atol=1e-2), \
        "matmul pytorch_ref no longer matches TritonBench.reference.mat_mult"

    Q = torch.rand(128, 64, device=device)
    K = torch.rand(128, 64, device=device)
    V = torch.rand(128, 64, device=device)
    d = Q.shape[-1]
    scores = Q @ K.T / (d ** 0.5)
    ref_attn = torch.softmax(scores, dim=-1) @ V
    assert torch.allclose(ref_attn, ref_flash_attention(Q, K, V), atol=1e-3), \
        "flash_attention pytorch_ref no longer matches TritonBench.reference.flash_attention"

    print("Reference consistency check passed: PROBLEM_SPECS matches TritonBench ground truth.")

# Main

DEFAULT_MODEL = "claude-sonnet-4-20250514"

def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton kernels via any LLM (LiteLLM)."
    )
    parser.add_argument("--softmax",         action="store_true")
    parser.add_argument("--layernorm",       action="store_true")
    parser.add_argument("--matmul",          action="store_true")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--all",             action="store_true",
                        help="Generate for all 4 operators.")
    parser.add_argument("--n",               type=int, default=5,
                        help="Number of kernels per operator (default 5).")
    parser.add_argument("--model",           type=str, default=DEFAULT_MODEL,
                        help=(
                            f"LiteLLM model string (default: {DEFAULT_MODEL}).\n"
                            "Examples: gpt-4o, deepseek/deepseek-coder, "
                            "gemini/gemini-2.5-flash"
                        ))
    parser.add_argument("--out",             type=str, default="generated",
                        help="Output root directory (default: generated/).")
    args = parser.parse_args()
    _verify_reference_consistency()
    if args.all:
        operators = list(PROBLEM_SPECS.keys())
    else:
        operators = [op for op in PROBLEM_SPECS if getattr(args, op, False)]

    if not operators:
        parser.error(
            "Specify at least one operator flag or use --all.\n"
            "Available: --softmax --layernorm --matmul --flash_attention"
        )

    check_api_key(args.model)

    short = model_short_name(args.model)
    print(f"\nModel   : {args.model}")
    print(f"Operators: {', '.join(operators)}")
    print(f"Count   : {args.n} per operator")
    print(f"Output  : {args.out}/{short}/<operator>/kernel_N.py\n")

    total = len(operators) * args.n
    done  = 0

    for operator in operators:
        print(f"── {operator}")
        for i in range(1, args.n + 1):
            try:
                code = generate_kernel(args.model, operator)
                path = save_kernel(args.model, operator, i, code, args.out)
                done += 1
                print(f"   [{done}/{total}] saved  {path}")
            except Exception as e:
                print(f"   [{done}/{total}] ERROR on {operator} kernel {i}: {e}")
            if i < args.n:
                time.sleep(0.5)

    print(f"\nDone. {done}/{total} kernels saved under {args.out}/{short}/")
    print("Run `python run_checker.py` to verify all generated kernels.")


if __name__ == "__main__":
    main()