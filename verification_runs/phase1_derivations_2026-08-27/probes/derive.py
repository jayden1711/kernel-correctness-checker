"""Closed-form ||J_i||_2 for the Phase-1 operators, each checked against an
autograd-computed exact Jacobian.

WHAT THIS VALIDATES AND WHAT IT DOES NOT.
  Validates: the CALCULUS. Did we differentiate the operator correctly, and does
  the transcribed formula equal the true Jacobian row norm of the torch
  reference? That is a mathematical identity and is device-independent, so CPU
  is the right place to check it.
  Does NOT validate: anything about a Triton kernel, the probed L, or
  adaptive_tol. Those need a GPU and are out of scope here by the standing rule
  in SESSION_HANDOFF.md 0 ("do not substitute CPU approximations ... for a
  number that requires a GPU").

Method: build the full Jacobian with torch.autograd.functional.jacobian, take
each output row's 2-norm, compare against the closed form. Tolerance 1e-4
relative -- these are float64 exact identities, not statistical agreements.
"""
import math, torch
import torch.nn.functional as F

torch.manual_seed(0)
DT = torch.float64


# ---------------------------------------------------------------------------
# CLOSED FORMS  --  candidates for structural_l.row_norms
# ---------------------------------------------------------------------------

def cf_elementwise(op, x, rest):
    """Diagonal Jacobian: ||J_i|| = |phi'(x_i)|. Same template as the existing
    gelu/swish entries."""
    if op == "relu":        d = (x > 0).to(x.dtype)
    elif op == "leaky_relu":
        s = rest[0]
        d = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, s))
    elif op == "sigmoid":   s = torch.sigmoid(x); d = s * (1 - s)
    elif op == "tanh":      t = torch.tanh(x);    d = 1 - t * t
    elif op == "selu":
        a, sc = 1.6732632423543772, 1.0507009873554805
        d = sc * torch.where(x > 0, torch.ones_like(x), a * torch.exp(x))
    elif op == "elu":
        a = rest[0]
        d = torch.where(x > 0, torch.ones_like(x), a * torch.exp(x))
    elif op == "softplus":
        b = rest[0]; d = torch.sigmoid(b * x)
    elif op == "hardsigmoid":
        d = ((x > -3) & (x < 3)).to(x.dtype) / 6.0
    elif op == "new_gelu":
        c = math.sqrt(2.0 / math.pi); k = 0.044715
        u = c * (x + k * x ** 3)
        t = torch.tanh(u)
        d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * k * x * x)
    else: raise KeyError(op)
    return d.abs().flatten()


def cf_scan(op, x, rest):
    """Scan over the last dim, length n. J is a (masked) triangular block of
    ones, so ||J_i|| = sqrt(#contributing inputs) -- SHAPE-ONLY for the three
    unmasked variants."""
    n = x.shape[-1]
    lead = x.shape[:-1].numel()
    dev = x.device
    i = torch.arange(n, device=dev, dtype=x.dtype)
    if op == "cumsum":            r = torch.sqrt(i + 1)
    elif op == "cumsum_reverse":  r = torch.sqrt(n - i)
    elif op == "cumsum_exclusive":r = torch.sqrt(i)
    elif op == "masked_cumsum":
        # mask rides as a companion; ||J_i|| = sqrt(# unmasked j <= i)
        mask = rest[0].to(x.dtype)
        return torch.sqrt(torch.cumsum(mask * mask, dim=-1)).flatten()
    else: raise KeyError(op)
    return r.repeat(lead)


def cf_matmul_variant(op, x, rest):
    if op == "matvec":
        # y = A @ v, primary A (M,K). dy_i/dA_pq = v_q delta_ip -> ||J_i|| = ||v||
        v = rest[0]
        return v.norm().repeat(x.shape[0])
    if op == "batched_matmul":
        # (B,M,K) @ (B,K,N), primary A. Per batch b: ||J_(b,i,j)|| = ||B_b[:,j]||
        Bm = rest[0]
        cn = Bm.norm(dim=1)                      # (B, N)
        return cn.unsqueeze(1).expand(Bm.shape[0], x.shape[1], Bm.shape[2]).flatten()
    if op == "diagonal_matmul":
        # C = diag(d) @ B, primary d (N,). dC_ij/dd_p = delta_ip B_ij
        Bm = rest[0]
        return Bm.abs().flatten()
    if op == "triangular_matmul":
        # C = tril(A @ B), primary A. Rows above the diagonal are identically
        # zero -- a genuine structural zero, not a small number.
        Bm = rest[0]
        M, N = x.shape[0], Bm.shape[1]
        cn = Bm.norm(dim=0)                       # (N,)
        keep = torch.tril(torch.ones(M, N, device=x.device, dtype=x.dtype))
        return (keep * cn.unsqueeze(0)).flatten()
    raise KeyError(op)


def cf_loss(op, x, rest):
    """Scalar output: m = 1, ||J|| = ||grad_x loss||."""
    if op == "mse_loss":
        t = rest[0]; N = x.numel()
        return (2.0 * (x - t) / N).norm().reshape(1)
    if op == "huber_loss":
        t, beta = rest[0], rest[1]; N = x.numel()
        d = x - t
        g = torch.where(d.abs() < beta, d / beta, torch.sign(d)) / N
        return g.norm().reshape(1)
    if op == "kldiv_loss":
        # F.kl_div(input=log q, target=p), reduction='batchmean'
        t = rest[0]; N = x.shape[0]
        return (-t / N).norm().reshape(1)
    if op == "bce_loss":
        # binary_cross_entropy(p, t), p already in (0,1)
        t = rest[0]; N = x.numel()
        g = (-(t / x) + (1 - t) / (1 - x)) / N
        return g.norm().reshape(1)
    if op == "nll_loss":
        # F.nll_loss(log_probs, targets), reduction='mean'. grad is -1/N at
        # each (row, target) cell and 0 elsewhere -> ||J|| = sqrt(N)/N
        N = x.shape[0]
        return torch.tensor([math.sqrt(N) / N], dtype=x.dtype, device=x.device)
    raise KeyError(op)


def cf_new(op, x, rest):
    if op == "rope":
        # Pairwise rotation. Rows of J are (cos_k, -sin_k) and (sin_k, cos_k),
        # so ||J_i|| = sqrt(cos^2 + sin^2) -- EXACTLY 1 for a genuine rotation
        # table, independent of theta AND of x.
        #
        # The general form is kept rather than hardcoding 1.0. A table that is
        # not a unit rotation (a scaled or mis-built cos/sin cache -- a real
        # kernel bug) then reports its true row norm instead of being silently
        # assumed orthogonal. Costs one cheap elementwise pass; buys the
        # difference between a derivation and an assumption.
        cos, sin = rest[0], rest[1]
        r = torch.sqrt(cos * cos + sin * sin)
        h = x.shape[-1] // 2
        r = r.expand(x.shape[:-1] + (h,)) if r.dim() < x.dim() else r
        return torch.cat([r, r], dim=-1).flatten()
    if op == "swiglu":
        # x is (..., 2h) split into (a, b); y = silu(a) * b
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = torch.sigmoid(a)
        silu = a * s
        dsilu = s * (1 + a * (1 - s))
        return torch.sqrt((dsilu * b) ** 2 + silu ** 2).flatten()
    if op == "logsumexp":
        # y_r = log sum_j exp(x_rj); dy_r/dx_rj = softmax -> ||J_r|| = ||p_r||
        p = torch.softmax(x, -1)
        return p.norm(dim=-1).flatten()
    if op == "std_reduction":
        # y = sqrt(sum (x-m)^2 / (n-1)).  dy/dx_j = (x_j-m)/((n-1) y)
        # ||J|| = ||x-m||/((n-1)y) = sqrt((n-1)y^2)/((n-1)y) = 1/sqrt(n-1)
        # SHAPE-ONLY.  NOTE: this is std, NOT var -- var's row norm is
        # 2||x-m||/(n-1), which IS input-dependent.
        n = x.shape[-1]
        return torch.full((x.shape[:-1].numel(),), 1.0 / math.sqrt(n - 1),
                          device=x.device, dtype=x.dtype)
    if op == "var_reduction":
        n = x.shape[-1]
        m = x.mean(-1, keepdim=True)
        return (2.0 * (x - m).norm(dim=-1) / (n - 1)).flatten()
    raise KeyError(op)


CLOSED = {}
for _o in ["relu","leaky_relu","sigmoid","tanh","selu","elu","softplus",
           "hardsigmoid","new_gelu"]:                      CLOSED[_o] = cf_elementwise
for _o in ["cumsum","cumsum_reverse","cumsum_exclusive","masked_cumsum"]:
                                                            CLOSED[_o] = cf_scan
for _o in ["matvec","batched_matmul","diagonal_matmul","triangular_matmul"]:
                                                            CLOSED[_o] = cf_matmul_variant
for _o in ["mse_loss","huber_loss","kldiv_loss","bce_loss","nll_loss"]:
                                                            CLOSED[_o] = cf_loss
for _o in ["rope","swiglu","logsumexp","std_reduction","var_reduction"]:
                                                            CLOSED[_o] = cf_new


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATIONS (torch) + input builders
# ---------------------------------------------------------------------------

def rope_ref(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

REFS = {
 "relu":        (lambda x: F.relu(x),                       lambda: (torch.randn(6,7,dtype=DT), ())),
 "leaky_relu":  (lambda x,s: F.leaky_relu(x,s),             lambda: (torch.randn(6,7,dtype=DT), (0.01,))),
 "sigmoid":     (lambda x: torch.sigmoid(x),                lambda: (torch.randn(6,7,dtype=DT), ())),
 "tanh":        (lambda x: torch.tanh(x),                   lambda: (torch.randn(6,7,dtype=DT), ())),
 "selu":        (lambda x: F.selu(x),                       lambda: (torch.randn(6,7,dtype=DT), ())),
 "elu":         (lambda x,a: F.elu(x,a),                    lambda: (torch.randn(6,7,dtype=DT), (1.0,))),
 "softplus":    (lambda x,b: F.softplus(x,beta=b),          lambda: (torch.randn(6,7,dtype=DT), (1.0,))),
 "hardsigmoid": (lambda x: F.hardsigmoid(x),                lambda: (torch.randn(6,7,dtype=DT)*2, ())),
 "new_gelu":    (lambda x: F.gelu(x,approximate="tanh"),    lambda: (torch.randn(6,7,dtype=DT), ())),

 "cumsum":           (lambda x: torch.cumsum(x,-1),                    lambda: (torch.randn(4,9,dtype=DT), ())),
 "cumsum_reverse":   (lambda x: torch.cumsum(x.flip(-1),-1).flip(-1),  lambda: (torch.randn(4,9,dtype=DT), ())),
 "cumsum_exclusive": (lambda x: torch.cumsum(x,-1)-x,                  lambda: (torch.randn(4,9,dtype=DT), ())),
 "masked_cumsum":    (lambda x,m: torch.cumsum(x*m,-1),                lambda: (torch.randn(4,9,dtype=DT), (torch.randint(0,2,(4,9)).to(DT),))),

 "matvec":            (lambda A,v: A@v,                     lambda: (torch.randn(5,6,dtype=DT), (torch.randn(6,dtype=DT),))),
 "batched_matmul":    (lambda A,B: torch.bmm(A,B),          lambda: (torch.randn(3,4,5,dtype=DT), (torch.randn(3,5,6,dtype=DT),))),
 "diagonal_matmul":   (lambda d,B: torch.diag(d)@B,         lambda: (torch.randn(5,dtype=DT), (torch.randn(5,6,dtype=DT),))),
 "triangular_matmul": (lambda A,B: torch.tril(A@B),         lambda: (torch.randn(6,6,dtype=DT), (torch.randn(6,6,dtype=DT),))),

 "mse_loss":   (lambda x,t: F.mse_loss(x,t),                          lambda: (torch.randn(5,4,dtype=DT), (torch.randn(5,4,dtype=DT),))),
 "huber_loss": (lambda x,t,b: F.smooth_l1_loss(x,t,beta=b),           lambda: (torch.randn(5,4,dtype=DT), (torch.randn(5,4,dtype=DT), 1.0))),
 "kldiv_loss": (lambda x,t: F.kl_div(x,t,reduction="batchmean"),      lambda: (torch.log_softmax(torch.randn(5,4,dtype=DT),-1), (torch.softmax(torch.randn(5,4,dtype=DT),-1),))),
 "bce_loss":   (lambda x,t: F.binary_cross_entropy(x,t),              lambda: (torch.rand(5,4,dtype=DT)*0.8+0.1, (torch.rand(5,4,dtype=DT).round(),))),
 "nll_loss":   (lambda x,t: F.nll_loss(x,t),                          lambda: (torch.log_softmax(torch.randn(5,4,dtype=DT),-1), (torch.randint(0,4,(5,)),))),

 "rope":          (rope_ref,                        lambda: (torch.randn(4,8,dtype=DT), (torch.cos(torch.randn(4,dtype=DT)), torch.sin(torch.randn(4,dtype=DT))))),
 "swiglu":        (lambda x: F.silu(x[...,:x.shape[-1]//2])*x[...,x.shape[-1]//2:], lambda: (torch.randn(4,8,dtype=DT), ())),
 "logsumexp":     (lambda x: torch.logsumexp(x,-1),         lambda: (torch.randn(5,7,dtype=DT), ())),
 "std_reduction": (lambda x: x.std(-1),                     lambda: (torch.randn(5,7,dtype=DT), ())),
 "var_reduction": (lambda x: x.var(-1),                     lambda: (torch.randn(5,7,dtype=DT), ())),
}

# rope's cos/sin must match the half-width, fix that builder
def _rope_inputs():
    theta = torch.randn(4, 4, dtype=DT)          # (rows, half-width)
    return torch.randn(4, 8, dtype=DT), (torch.cos(theta), torch.sin(theta))
REFS["rope"] = (rope_ref, _rope_inputs)

def _rope_inputs_nonunit():
    """Deliberately NOT a rotation -- a scaled cos/sin cache. The general form
    must still match autograd here; a hardcoded 1.0 would not."""
    theta = torch.randn(4, 4, dtype=DT)
    s_ = 1.7
    return torch.randn(4, 8, dtype=DT), (s_ * torch.cos(theta), s_ * torch.sin(theta))
REFS["rope_nonunit"] = (rope_ref, _rope_inputs_nonunit)
CLOSED["rope_nonunit"] = lambda op, x, rest: cf_new("rope", x, rest)


def autograd_row_norms(fn, x, rest):
    """Exact ||J_i||_2 for every output element, via the full Jacobian."""
    f = lambda t: fn(t, *rest).reshape(-1)
    J = torch.autograd.functional.jacobian(f, x)
    J = J.reshape(J.shape[0], -1)          # (n_out, n_in)
    return J.norm(dim=1)


def main():
    print(f"{'operator':20s} {'m':>6s} {'max rel err':>12s}  verdict")
    print("-" * 60)
    ok = bad = 0
    for op, (fn, mk) in REFS.items():
        x, rest = mk()
        try:
            ag = autograd_row_norms(fn, x, rest)
            cf = CLOSED[op](op, x, list(rest)).to(ag.dtype)
            if cf.shape != ag.shape:
                print(f"{op:20s} {'--':>6s} {'SHAPE':>12s}  FAIL cf{tuple(cf.shape)} vs ag{tuple(ag.shape)}")
                bad += 1; continue
            denom = ag.abs().clamp_min(1e-12)
            rel = ((cf - ag).abs() / denom).max().item()
            v = "OK" if rel < 1e-6 else ("FAIL" if rel > 1e-4 else "WARN")
            ok, bad = (ok + 1, bad) if v == "OK" else (ok, bad + 1)
            print(f"{op:20s} {ag.numel():6d} {rel:12.3e}  {v}")
        except Exception as e:
            print(f"{op:20s} {'--':>6s} {'ERR':>12s}  {type(e).__name__}: {e}")
            bad += 1
    print("-" * 60)
    print(f"{ok} verified, {bad} not")

main()
