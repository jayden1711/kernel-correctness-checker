"""H4 by-product — the diversity penalty is provably inert at shipped defaults.

Claim: with diversity_weight=3.0 and beam_width<=4 (the defaults: beam_width
defaults to --workers=4), DiverseBeamStrategy.select returns exactly
BeamSearchStrategy.select's top-B whenever every candidate's beam_score > 9.
Since any reference-passing proposal with fewer than 2 errored mutants scores
>= 10, the penalty can only ever discriminate among already-broken proposals
(failed reference, or >= 2 crashed mutants).

Proof from the constants: penalty = overlap * 3 <= (B-1)*3 = 9 within a beam of
4, and the accept test is `effective_score > 0 or len(selected) < B//2` in rank
order with a rank-order tail fill -- so any score > 9 is accepted at every
overlap, and rank order is never re-sorted.

This probe verifies the claim by exhaustive randomized enumeration against the
real classes, and then finds the boundary where the two strategies first
diverge (candidates with score <= 9, i.e. broken proposals).
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../.."))

from verification.adversarial_search.strategy.beam import BeamSearchStrategy
from verification.adversarial_search.strategy.diverse import DiverseBeamStrategy


class P:  # stand-in for InputProposal
    def __init__(self, pid, mode):
        self.proposal_id = pid
        self.predicted_failure_mode = mode

    def __repr__(self):
        return f"P({self.proposal_id},{self.predicted_failure_mode})"


class V:  # stand-in for ProposalVerdict
    def __init__(self, score):
        self.beam_score = score


def run(pool, bw=4):
    beam = BeamSearchStrategy().select(list(pool), bw)
    div = DiverseBeamStrategy(3.0).select(list(pool), bw)
    return [p.proposal_id for p, _ in beam], [p.proposal_id for p, _ in div]


rng = random.Random(0)
MODES = ["partial_tile", "wrong_reduction", "missing_max_shift"]

# 1: all-valid pools (score >= 10): identity must hold on every trial
diverge = 0
for trial in range(20000):
    n = rng.randint(1, 10)
    pool = [(P(i, rng.choice(MODES)), V(rng.choice([10, 12, 18, 26, 34])))
            for i in range(n)]
    b, d = run(pool)
    if b != d:
        diverge += 1
        if diverge <= 3:
            print("DIVERGENCE on valid pool:", [(p.proposal_id, p.predicted_failure_mode,
                                                 v.beam_score) for p, v in pool])
print(f"valid-only pools (score>=10): {20000-diverge}/20000 identical "
      f"({diverge} divergences)")

# 2: pools including broken proposals (score <= 9): find first divergence
diverge2 = []
for trial in range(20000):
    n = rng.randint(1, 10)
    pool = [(P(i, rng.choice(MODES)),
             V(rng.choice([-5, 6, 8, 10, 12, 18])))
            for i in range(n)]
    b, d = run(pool)
    if b != d:
        diverge2.append(pool)
print(f"pools with broken proposals: {len(diverge2)}/20000 diverge")
if diverge2:
    pool = diverge2[0]
    print("example:", [(p.proposal_id, p.predicted_failure_mode, v.beam_score)
                       for p, v in pool])
    print("  beam:", run(pool)[0], " diverse:", run(pool)[1])

# 3: boundary scan -- lowest uniform score at which identity is guaranteed
for s in [8, 9, 9.001, 10]:
    bad = 0
    for trial in range(5000):
        n = rng.randint(4, 10)
        pool = [(P(i, MODES[0]), V(s)) for i in range(n)]  # all same mode: max overlap
        b, d = run(pool)
        if b != d:
            bad += 1
    print(f"uniform score {s}: {bad}/5000 divergences (same-mode worst case)")
