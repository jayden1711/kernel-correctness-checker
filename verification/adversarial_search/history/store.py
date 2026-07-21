"""
verification/adversarial_search/history/store.py

SQLite-backed persistent search history.

Responsibilities:
  - Persist every proposal and verdict to disk after every iteration
  - Allow a crashed search to resume from where it stopped
  - Provide a distilled "memory" of what worked (for beam search
    and for LLM prompt injection)
  - Support cross-run queries: "which bug patterns have been confirmed
    for this operator?" — important for the paper's coverage tables

Design decisions:
  - SQLite, not JSON files: atomic writes, concurrent-reader safe,
    queryable without loading everything into memory
  - One DB file per output_dir (shared across operators and runs)
  - Schema is append-only: no updates, no deletes
    (immutable audit trail, safe to resume)
  - Memory items are stored separately and survive across runs
"""

from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from verification.adversarial_search.schemas import (
    InputProposal,
    ProposalVerdict,
    SearchResult,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    operator    TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    model       TEXT NOT NULL,
    n_workers   INTEGER NOT NULL,
    max_iter    INTEGER NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | done | hit | no_hit
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    operator    TEXT NOT NULL,
    worker_id   TEXT NOT NULL,
    iteration   INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    proposal_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS verdicts (
    proposal_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    operator        TEXT NOT NULL,
    is_hit          INTEGER NOT NULL,
    gap_confirmed   INTEGER NOT NULL,
    hit_mutants     TEXT NOT NULL,   -- JSON list
    missed_mutants  TEXT NOT NULL,   -- JSON list
    beam_score      REAL NOT NULL DEFAULT 0.0,
    verdict_json    TEXT NOT NULL,
    created_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS memory_items (
    item_id     TEXT PRIMARY KEY,
    operator    TEXT NOT NULL,
    bug_pattern TEXT NOT NULL,
    summary     TEXT NOT NULL,       -- distilled 1–2 sentence insight
    source_run  TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proposals_run    ON proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run     ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_operator ON verdicts(operator);
CREATE INDEX IF NOT EXISTS idx_memory_operator  ON memory_items(operator);
"""


class SearchHistoryStore:
    """
    Thread-safe SQLite-backed history store.
    One instance per search session; safe to share across worker threads
    because SQLite handles concurrent writes with WAL mode.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        operator: str,
        strategy: str,
        model: str,
        n_workers: int,
        max_iter: int,
    ) -> str:
        self._conn.execute(
            "INSERT INTO runs (run_id, operator, strategy, model, n_workers, "
            "max_iter, started_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, operator, strategy, model, n_workers, max_iter, time.time()),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, result: SearchResult):
        status = "hit" if result.winning_proposal else "no_hit"
        self._conn.execute(
            "UPDATE runs SET finished_at=?, status=?, result_json=? WHERE run_id=?",
            (time.time(), status, result.to_json(), run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM runs LIMIT 0").description]
        return dict(zip(cols, row))

    def list_runs(self, operator: Optional[str] = None) -> List[dict]:
        if operator:
            rows = self._conn.execute(
                "SELECT run_id, operator, strategy, model, status, started_at, finished_at "
                "FROM runs WHERE operator=? ORDER BY started_at DESC",
                (operator,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT run_id, operator, strategy, model, status, started_at, finished_at "
                "FROM runs ORDER BY started_at DESC"
            ).fetchall()
        cols = ["run_id", "operator", "strategy", "model", "status", "started_at", "finished_at"]
        return [dict(zip(cols, r)) for r in rows]

    # ── Proposals ─────────────────────────────────────────────────────────────

    def save_proposal(self, run_id: str, proposal: InputProposal):
        self._conn.execute(
            "INSERT OR IGNORE INTO proposals "
            "(proposal_id, run_id, operator, worker_id, iteration, created_at, proposal_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                proposal.proposal_id,
                run_id,
                proposal.operator,
                proposal.worker_id,
                proposal.iteration,
                time.time(),
                proposal.to_json(),
            ),
        )
        self._conn.commit()

    def get_proposals_for_run(self, run_id: str) -> List[InputProposal]:
        rows = self._conn.execute(
            "SELECT proposal_json FROM proposals WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [InputProposal.from_dict(json.loads(r[0])) for r in rows]

    # ── Verdicts ──────────────────────────────────────────────────────────────

    def save_verdict(self, run_id: str, verdict: ProposalVerdict):
        self._conn.execute(
            "INSERT OR IGNORE INTO verdicts "
            "(proposal_id, run_id, operator, is_hit, gap_confirmed, "
            "hit_mutants, missed_mutants, beam_score, verdict_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                verdict.proposal_id,
                run_id,
                # operator stored in proposal; look it up or pass it here
                self._operator_for_run(run_id),
                int(verdict.is_hit),
                int(verdict.gap_confirmed),
                json.dumps(verdict.hit_mutants),
                json.dumps(verdict.missed_mutants),
                verdict.beam_score,
                json.dumps(verdict.to_dict()),
                time.time(),
            ),
        )
        self._conn.commit()

    def get_verdicts_for_run(self, run_id: str) -> List[ProposalVerdict]:
        rows = self._conn.execute(
            "SELECT verdict_json FROM verdicts WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [ProposalVerdict.from_dict(json.loads(r[0])) for r in rows]

    def get_hits_for_operator(self, operator: str) -> List[ProposalVerdict]:
        rows = self._conn.execute(
            "SELECT verdict_json FROM verdicts "
            "WHERE operator=? AND is_hit=1 ORDER BY beam_score DESC",
            (operator,),
        ).fetchall()
        return [ProposalVerdict.from_dict(json.loads(r[0])) for r in rows]

    def top_beam_candidates(
        self, run_id: str, beam_width: int
    ) -> List[ProposalVerdict]:
        """
        Return the top-B verdicts by beam_score for beam search selection.
        """
        rows = self._conn.execute(
            "SELECT verdict_json FROM verdicts WHERE run_id=? "
            "ORDER BY beam_score DESC LIMIT ?",
            (run_id, beam_width),
        ).fetchall()
        return [ProposalVerdict.from_dict(json.loads(r[0])) for r in rows]

    # ── Memory items ──────────────────────────────────────────────────────────

    def add_memory_item(
        self,
        operator: str,
        bug_pattern: str,
        summary: str,
        source_run: str,
    ) -> str:
        item_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO memory_items (item_id, operator, bug_pattern, summary, "
            "source_run, created_at) VALUES (?,?,?,?,?,?)",
            (item_id, operator, bug_pattern, summary, source_run, time.time()),
        )
        self._conn.commit()
        return item_id

    def get_memory_items(
        self, operator: str, limit: int = 5
    ) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT bug_pattern, summary FROM memory_items "
            "WHERE operator=? ORDER BY created_at DESC LIMIT ?",
            (operator, limit),
        ).fetchall()
        return [{"bug_pattern": r[0], "summary": r[1]} for r in rows]

    # ── Resume support ────────────────────────────────────────────────────────

    def resume_run(self, run_id: str) -> Optional[Dict]:
        """
        Returns resumption context for a previously started run:
          - which workers were active
          - last proposal per worker
          - all verdicts so far
        Returns None if run_id is unknown or already finished.
        """
        run = self.get_run(run_id)
        if run is None or run["status"] in ("done", "hit", "no_hit"):
            return None

        proposals = self.get_proposals_for_run(run_id)
        verdicts = self.get_verdicts_for_run(run_id)

        # Last proposal per worker
        last_per_worker: Dict[str, InputProposal] = {}
        for p in proposals:
            if p.worker_id not in last_per_worker or \
               p.iteration > last_per_worker[p.worker_id].iteration:
                last_per_worker[p.worker_id] = p

        return {
            "run": run,
            "last_per_worker": last_per_worker,
            "verdicts": verdicts,
            "n_proposals": len(proposals),
        }

    # ── Coverage report ───────────────────────────────────────────────────────

    def coverage_report(self) -> Dict:
        """
        Returns a summary of which operators and bug patterns have been
        confirmed with adversarial inputs.  For the paper's Table 1.
        """
        rows = self._conn.execute(
            "SELECT v.operator, p.proposal_json, v.hit_mutants "
            "FROM verdicts v JOIN proposals p ON v.proposal_id=p.proposal_id "
            "WHERE v.is_hit=1 AND v.gap_confirmed=1"
        ).fetchall()

        report: Dict[str, Dict] = {}
        for operator, prop_json, hit_json in rows:
            prop = InputProposal.from_dict(json.loads(prop_json))
            hits = json.loads(hit_json)
            if operator not in report:
                report[operator] = {}
            pattern = prop.predicted_failure_mode
            if pattern not in report[operator]:
                report[operator][pattern] = []
            report[operator][pattern].extend(hits)

        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _operator_for_run(self, run_id: str) -> str:
        row = self._conn.execute(
            "SELECT operator FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else "unknown"

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()