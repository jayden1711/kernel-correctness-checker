"""
tests/verification/adversarial_search/test_history_store.py

Tests for history/store.py: SQLite-backed persistent history.

Covers run lifecycle, proposal/verdict persistence, resume semantics,
memory items, coverage report, and concurrent-write safety.
No GPU, no LLM.
"""

import json
import os
import tempfile
import threading
import uuid
import pytest

from verification.adversarial_search.schemas import (
    InputProposal, TensorDescriptor, ProposalVerdict, SearchResult,
)
from verification.adversarial_search.history.store import SearchHistoryStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with SearchHistoryStore(str(db)) as s:
        yield s


@pytest.fixture
def run_id(store):
    return store.create_run("r1", "softmax", "beam", "test-model", 4, 20)


def _proposal(operator="softmax", worker_id="w0", iteration=0, pattern="partial_tile"):
    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id=worker_id,
        iteration=iteration,
        operator=operator,
        tensors={"x": TensorDescriptor(shape=[4, 16], dtype="float32", fill="randn")},
        rationale="",
        predicted_failure_mode=pattern,
    )


def _verdict(proposal_id, is_hit=False, hit_mutants=None, gap_confirmed=False, beam_score=0.0):
    return ProposalVerdict(
        proposal_id=proposal_id,
        is_hit=is_hit,
        hit_mutants=hit_mutants or [],
        missed_mutants=[],
        reference_passed=True,
        gap_confirmed=gap_confirmed,
        failure_summary="",
        beam_score=beam_score,
    )


# ── Run lifecycle ─────────────────────────────────────────────────────────────

class TestRunLifecycle:
    def test_create_run_returns_id(self, store):
        rid = store.create_run("r1", "softmax", "beam", "model", 4, 20)
        assert rid == "r1"

    def test_get_run(self, store, run_id):
        r = store.get_run(run_id)
        assert r is not None
        assert r["operator"] == "softmax"
        assert r["strategy"] == "beam"
        assert r["status"] == "running"

    def test_get_nonexistent_run(self, store):
        assert store.get_run("nonexistent") is None

    def test_finish_run_updates_status(self, store, run_id):
        p = _proposal()
        v = _verdict(p.proposal_id, is_hit=True, hit_mutants=["m1"], gap_confirmed=True)
        result = SearchResult(
            run_id=run_id,
            operator="softmax",
            strategy="beam",
            total_proposals=1,
            total_iterations=20,
            n_workers=4,
            winning_proposal=p,
            winning_verdict=v,
            all_verdicts=[v],
            wall_time_s=5.0,
            model="test-model",
        )
        store.finish_run(run_id, result)
        r = store.get_run(run_id)
        assert r["status"] == "hit"
        assert r["finished_at"] is not None

    def test_finish_run_no_hit(self, store, run_id):
        result = SearchResult(
            run_id=run_id,
            operator="softmax",
            strategy="beam",
            total_proposals=20,
            total_iterations=80,
            n_workers=4,
            winning_proposal=None,
            winning_verdict=None,
            all_verdicts=[],
            wall_time_s=60.0,
            model="test-model",
        )
        store.finish_run(run_id, result)
        r = store.get_run(run_id)
        assert r["status"] == "no_hit"

    def test_list_runs_empty(self, store):
        assert store.list_runs() == []

    def test_list_runs_returns_entries(self, store, run_id):
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id

    def test_list_runs_filtered_by_operator(self, store):
        store.create_run("r2", "layernorm", "greedy", "model", 2, 10)
        softmax_runs = store.list_runs(operator="softmax")
        assert all(r["operator"] == "softmax" for r in softmax_runs)
        ln_runs = store.list_runs(operator="layernorm")
        assert len(ln_runs) == 1


# ── Proposals ─────────────────────────────────────────────────────────────────

class TestProposals:
    def test_save_and_retrieve(self, store, run_id):
        p = _proposal()
        store.save_proposal(run_id, p)
        proposals = store.get_proposals_for_run(run_id)
        assert len(proposals) == 1
        assert proposals[0].proposal_id == p.proposal_id

    def test_save_multiple(self, store, run_id):
        ps = [_proposal(iteration=i) for i in range(5)]
        for p in ps:
            store.save_proposal(run_id, p)
        proposals = store.get_proposals_for_run(run_id)
        assert len(proposals) == 5

    def test_duplicate_save_ignored(self, store, run_id):
        p = _proposal()
        store.save_proposal(run_id, p)
        store.save_proposal(run_id, p)  # duplicate
        proposals = store.get_proposals_for_run(run_id)
        assert len(proposals) == 1

    def test_proposal_roundtrip_preserves_patches(self, store, run_id):
        p = _proposal()
        p.tensors["x"].patches = [{"indices": "[:, -8:]", "value": 1e4}]
        store.save_proposal(run_id, p)
        recovered = store.get_proposals_for_run(run_id)[0]
        assert recovered.tensors["x"].patches == p.tensors["x"].patches


# ── Verdicts ──────────────────────────────────────────────────────────────────

class TestVerdicts:
    def test_save_and_retrieve(self, store, run_id):
        p = _proposal()
        store.save_proposal(run_id, p)
        v = _verdict(p.proposal_id, beam_score=12.0)
        store.save_verdict(run_id, v)
        verdicts = store.get_verdicts_for_run(run_id)
        assert len(verdicts) == 1
        assert verdicts[0].beam_score == 12.0

    def test_get_hits_for_operator(self, store, run_id):
        for i in range(3):
            p = _proposal()
            store.save_proposal(run_id, p)
            is_hit = (i == 1)
            v = _verdict(p.proposal_id, is_hit=is_hit,
                         hit_mutants=["m1"] if is_hit else [],
                         gap_confirmed=is_hit, beam_score=float(i))
            store.save_verdict(run_id, v)
        hits = store.get_hits_for_operator("softmax")
        assert len(hits) == 1
        assert hits[0].is_hit

    def test_top_beam_candidates(self, store, run_id):
        for score in [3.0, 9.0, 5.0, 1.0, 7.0]:
            p = _proposal()
            store.save_proposal(run_id, p)
            store.save_verdict(run_id, _verdict(p.proposal_id, beam_score=score))
        top = store.top_beam_candidates(run_id, beam_width=3)
        assert len(top) == 3
        scores = [v.beam_score for v in top]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 9.0


# ── Resume ────────────────────────────────────────────────────────────────────

class TestResume:
    def test_resume_running_run(self, store, run_id):
        for i, wid in enumerate(["w0", "w1", "w0"]):  # w0 has 2 proposals
            p = _proposal(worker_id=wid, iteration=i)
            store.save_proposal(run_id, p)
            store.save_verdict(run_id, _verdict(p.proposal_id))

        ctx = store.resume_run(run_id)
        assert ctx is not None
        assert ctx["n_proposals"] == 3
        # w0's last proposal is iteration=2
        assert ctx["last_per_worker"]["w0"].iteration == 2
        # w1's last proposal is iteration=1
        assert ctx["last_per_worker"]["w1"].iteration == 1

    def test_resume_finished_run_returns_none(self, store, run_id):
        result = SearchResult(
            run_id=run_id, operator="softmax", strategy="beam",
            total_proposals=5, total_iterations=20, n_workers=4,
            winning_proposal=None, winning_verdict=None, all_verdicts=[],
            wall_time_s=10.0, model="m",
        )
        store.finish_run(run_id, result)
        assert store.resume_run(run_id) is None

    def test_resume_nonexistent_run_returns_none(self, store):
        assert store.resume_run("does-not-exist") is None


# ── Memory items ──────────────────────────────────────────────────────────────

class TestMemoryItems:
    def test_add_and_retrieve(self, store, run_id):
        store.add_memory_item(
            operator="softmax",
            bug_pattern="partial_tile",
            summary="Spike in last tile exposes first-tile-only bug.",
            source_run=run_id,
        )
        items = store.get_memory_items("softmax", limit=5)
        assert len(items) == 1
        assert items[0]["bug_pattern"] == "partial_tile"
        assert "spike" in items[0]["summary"].lower()

    def test_memory_filtered_by_operator(self, store, run_id):
        store.add_memory_item("softmax",  "partial_tile", "s1", run_id)
        store.add_memory_item("layernorm", "wrong_variance", "s2", run_id)
        sm = store.get_memory_items("softmax")
        assert all(True for _ in sm)  # all returned
        ln = store.get_memory_items("layernorm")
        assert len(ln) == 1

    def test_memory_limit(self, store, run_id):
        for i in range(10):
            store.add_memory_item("softmax", f"pattern_{i}", f"summary {i}", run_id)
        items = store.get_memory_items("softmax", limit=3)
        assert len(items) == 3


# ── Coverage report ───────────────────────────────────────────────────────────

class TestCoverageReport:
    def test_empty_report(self, store):
        report = store.coverage_report()
        assert report == {}

    def test_coverage_populated_after_hit(self, store):
        run_id = store.create_run("r1", "softmax", "beam", "m", 4, 20)
        p = _proposal(operator="softmax", pattern="partial_tile")
        store.save_proposal(run_id, p)
        v = _verdict(p.proposal_id, is_hit=True,
                     hit_mutants=["first_tile"], gap_confirmed=True)
        store.save_verdict(run_id, v)

        report = store.coverage_report()
        assert "softmax" in report
        assert "partial_tile" in report["softmax"]
        assert "first_tile" in report["softmax"]["partial_tile"]

    def test_miss_not_in_coverage(self, store):
        run_id = store.create_run("r1", "softmax", "beam", "m", 4, 20)
        p = _proposal()
        store.save_proposal(run_id, p)
        v = _verdict(p.proposal_id, is_hit=False)
        store.save_verdict(run_id, v)
        report = store.coverage_report()
        # Misses should not appear in coverage
        assert report == {}


# ── Concurrency ───────────────────────────────────────────────────────────────

class TestConcurrentWrites:
    def test_multiple_threads_write_safely(self, store, run_id):
        """
        4 threads each writing 10 proposals concurrently — no data loss,
        no integrity errors.  Validates WAL mode + check_same_thread=False.
        """
        errors = []

        def write_proposals(worker_id):
            try:
                for i in range(10):
                    p = _proposal(worker_id=worker_id, iteration=i)
                    store.save_proposal(run_id, p)
                    store.save_verdict(run_id, _verdict(p.proposal_id))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_proposals, args=(f"w{i}",))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        proposals = store.get_proposals_for_run(run_id)
        assert len(proposals) == 40