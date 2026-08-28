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
  - threading.Lock() serialises ALL operations (reads and writes).
    The Python sqlite3 module is not safe for concurrent access on a
    shared connection object even with check_same_thread=False.
    WAL mode is kept for future multi-process access but within this
    process all access goes through the lock.
"""

from __future__ import annotations
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from verification.adversarial_search.schemas import (
    InputProposal,
    KernelExecutionResult,
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
    status      TEXT NOT NULL DEFAULT 'running',
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id   TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    operator      TEXT NOT NULL,
    worker_id     TEXT NOT NULL,
    iteration     INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    proposal_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS verdicts (
    proposal_id    TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    operator       TEXT NOT NULL,
    is_hit         INTEGER NOT NULL,
    gap_confirmed  INTEGER NOT NULL,
    hit_mutants    TEXT NOT NULL,
    missed_mutants TEXT NOT NULL,
    beam_score     REAL NOT NULL DEFAULT 0.0,
    verdict_json   TEXT NOT NULL,
    created_at     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS memory_items (
    item_id     TEXT PRIMARY KEY,
    operator    TEXT NOT NULL,
    bug_pattern TEXT NOT NULL,
    summary     TEXT NOT NULL,
    source_run  TEXT NOT NULL,
    created_at  REAL NOT NULL
);

-- One row per (proposal, kernel) execution. Added because the executor
-- computed full per-check pass/fail/details for the reference AND every mutant
-- on every proposal, used it transiently for passed_checker / failure_summary /
-- feedback hints, then discarded it -- so "did the checker catch this mutant,
-- and which check caught it?" was unanswerable from stored data. The
-- causal_flash_attention post-mortem had to reconstruct that from proposal
-- shapes instead (adversarial_results/CFA_NONHIT_ROOTCAUSE.md).
--
-- passed_checker / passed_naive / error_type / n_checks / n_failed are
-- deliberately DENORMALISED out of check_results_json: the common queries are
-- exactly those, and forcing every one of them through a JSON parse is what
-- made the previous investigation slow and error-prone.
CREATE TABLE IF NOT EXISTS executions (
    proposal_id        TEXT NOT NULL,
    run_id             TEXT NOT NULL,
    operator           TEXT NOT NULL,
    kernel_id          TEXT NOT NULL,
    passed_checker     INTEGER NOT NULL,
    passed_naive       INTEGER NOT NULL,
    wall_time_ms       REAL NOT NULL,
    total_wall_time_ms REAL,
    exec_mode          TEXT,
    batch_spawn_ms     REAL,
    kernel_wall_time_ms REAL,
    startup_phases_json TEXT,
    error_type         TEXT,
    n_checks           INTEGER NOT NULL,
    n_failed           INTEGER NOT NULL,
    check_results_json TEXT NOT NULL,
    error_json         TEXT,
    created_at         REAL NOT NULL,
    PRIMARY KEY (proposal_id, kernel_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_proposals_run      ON proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run       ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_operator  ON verdicts(operator);
CREATE INDEX IF NOT EXISTS idx_memory_operator    ON memory_items(operator);
CREATE INDEX IF NOT EXISTS idx_executions_run      ON executions(run_id);
CREATE INDEX IF NOT EXISTS idx_executions_operator ON executions(operator);
CREATE INDEX IF NOT EXISTS idx_executions_kernel   ON executions(kernel_id);
"""


# Columns added to `executions` after the table's first release, in the order
# they were introduced. Every entry is NULLABLE by design: a row written before
# the column existed genuinely has no measurement, and NULL says exactly that
# where 0.0 would claim, say, a free subprocess spawn.
#
# Appending here is the ONLY supported way to extend the table. Editing the
# schema block above alone reaches fresh databases only -- `CREATE TABLE IF NOT
# EXISTS` is a no-op on every DB that already exists, which is the defect
# `_migrate_unlocked` exists to prevent.
_LATE_EXECUTION_COLUMNS = [
    ("total_wall_time_ms",  "REAL"),   # parent-stamped spawn-to-result, single path only
    ("exec_mode",           "TEXT"),   # "single" | "batched" -- read this FIRST
    ("batch_spawn_ms",      "REAL"),   # shared startup, IDENTICAL across a batch
    ("kernel_wall_time_ms", "REAL"),   # per-kernel interval, populated on both paths
    ("startup_phases_json", "TEXT"),   # decomposition of batch_spawn_ms
    ("start_method",        "TEXT"),   # "spawn" | "forkserver" -- as USED, not requested
]


class SearchHistoryStore:
    """
    Thread-safe SQLite-backed history store.
    One instance per search session; safe to share across worker threads.
    All database operations (reads and writes) are serialised via a
    single threading.Lock() to prevent sqlite3 InterfaceError on
    concurrent access to a shared connection object.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_unlocked()
            self._conn.commit()

    def _migrate_unlocked(self):
        """Add columns introduced after a DB was first created.

        `CREATE TABLE IF NOT EXISTS` is a no-op on a table that already exists,
        so it cannot add a column to an existing `search_history.db` -- the
        schema block above only covers fresh databases. Every real DB in this
        repo predates `total_wall_time_ms`, so without this they would keep
        silently writing the old column set.

        Additive and idempotent: checks `PRAGMA table_info` first and only
        issues ALTER TABLE ADD COLUMN, which never rewrites or reorders
        existing rows. The new column is nullable on purpose -- rows written
        before the migration genuinely have no spawn measurement, and NULL says
        that, where 0.0 would claim a free spawn.

        Verified against a copy of the real DB in
        tests/instrumentation/check_execution_persistence.py.

        `_LATE_EXECUTION_COLUMNS` is the full list of post-creation additions,
        applied in order. Adding to that list is the ONLY supported way to
        extend the table: editing the schema block above alone silently skips
        every DB that already exists, which is the exact defect this method was
        written for.
        """
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(executions)")}
        if not existing:
            return
        for column, decl in _LATE_EXECUTION_COLUMNS:
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE executions ADD COLUMN {column} {decl}")

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
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, operator, strategy, model, n_workers, "
                "max_iter, started_at) VALUES (?,?,?,?,?,?,?)",
                (run_id, operator, strategy, model, n_workers, max_iter, time.time()),
            )
            self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, result: SearchResult):
        status = "hit" if result.winning_proposal else "no_hit"
        result_json = result.to_json()
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at=?, status=?, result_json=? WHERE run_id=?",
                (time.time(), status, result_json, run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        cols = ["run_id", "operator", "strategy", "model", "n_workers", "max_iter",
                "started_at", "finished_at", "status", "result_json"]
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))

    def list_runs(self, operator: Optional[str] = None) -> List[dict]:
        cols = ["run_id", "operator", "strategy", "model", "status",
                "started_at", "finished_at"]
        col_str = ", ".join(cols)
        with self._lock:
            if operator:
                rows = self._conn.execute(
                    f"SELECT {col_str} FROM runs WHERE operator=? "
                    f"ORDER BY started_at DESC",
                    (operator,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT {col_str} FROM runs ORDER BY started_at DESC"
                ).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ── Proposals ─────────────────────────────────────────────────────────────

    def save_proposal(self, run_id: str, proposal: InputProposal):
        proposal_json = proposal.to_json()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO proposals "
                "(proposal_id, run_id, operator, worker_id, iteration, "
                "created_at, proposal_json) VALUES (?,?,?,?,?,?,?)",
                (
                    proposal.proposal_id,
                    run_id,
                    proposal.operator,
                    proposal.worker_id,
                    proposal.iteration,
                    time.time(),
                    proposal_json,
                ),
            )
            self._conn.commit()

    def get_proposals_for_run(self, run_id: str) -> List[InputProposal]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT proposal_json FROM proposals "
                "WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [InputProposal.from_dict(json.loads(r[0])) for r in rows]

    # ── Verdicts ──────────────────────────────────────────────────────────────

    def save_verdict(self, run_id: str, verdict: ProposalVerdict):
        verdict_json   = json.dumps(verdict.to_dict())
        hit_json       = json.dumps(verdict.hit_mutants)
        missed_json    = json.dumps(verdict.missed_mutants)
        operator       = self._operator_for_run_unlocked(run_id)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO verdicts "
                "(proposal_id, run_id, operator, is_hit, gap_confirmed, "
                "hit_mutants, missed_mutants, beam_score, verdict_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    verdict.proposal_id,
                    run_id,
                    operator,
                    int(verdict.is_hit),
                    int(verdict.gap_confirmed),
                    hit_json,
                    missed_json,
                    verdict.beam_score,
                    verdict_json,
                    time.time(),
                ),
            )
            self._conn.commit()

    # ── Executions ────────────────────────────────────────────────────────────

    def save_execution(self, run_id: str, result: KernelExecutionResult):
        """
        Persist one (proposal, kernel) execution, including the full per-check
        results and any ExecutionError.

        Called once per execute_proposal() return -- reference first, then each
        mutant -- rather than batched at the end of the proposal. A timeout or
        crash partway through the mutant loop is precisely the case whose detail
        is most worth keeping, and a batched write would lose all of it. This
        mirrors save_proposal(), which is likewise written before the work it
        describes completes.

        INSERT OR IGNORE, keyed on (proposal_id, kernel_id): a resumed run that
        re-executes a proposal will not duplicate or clobber the first record.
        """
        checks = result.check_results or []
        n_failed = sum(1 for c in checks if not c.get("passed", True))
        error_dict = result.error.to_dict() if result.error else None
        operator = self._operator_for_run_unlocked(run_id)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO executions "
                "(proposal_id, run_id, operator, kernel_id, passed_checker, "
                "passed_naive, wall_time_ms, total_wall_time_ms, "
                "exec_mode, batch_spawn_ms, kernel_wall_time_ms, "
                "startup_phases_json, start_method, error_type, "
                "n_checks, n_failed, "
                "check_results_json, error_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.proposal_id,
                    run_id,
                    operator,
                    result.kernel_id,
                    int(result.passed_checker),
                    int(result.passed_naive),
                    float(result.wall_time_ms),
                    (None if result.total_wall_time_ms is None
                     else float(result.total_wall_time_ms)),
                    getattr(result, "exec_mode", "single"),
                    (None if getattr(result, "batch_spawn_ms", None) is None
                     else float(result.batch_spawn_ms)),
                    (None if getattr(result, "kernel_wall_time_ms", None) is None
                     else float(result.kernel_wall_time_ms)),
                    (None if getattr(result, "startup_phases", None) is None
                     else json.dumps(result.startup_phases)),
                    getattr(result, "start_method", None),
                    result.error.error_type if result.error else None,
                    len(checks),
                    n_failed,
                    json.dumps(checks),
                    json.dumps(error_dict) if error_dict is not None else None,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_executions(
        self,
        run_id: Optional[str] = None,
        proposal_id: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> List[Dict]:
        """
        Read execution records back, with check_results/error already parsed.

        Deliberately a thin accessor and not a report generator -- there is no
        pending question this needs to answer in a fixed shape, and the
        investigation this table exists to support was done with ad-hoc SQL.
        """
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id=?"); params.append(run_id)
        if proposal_id is not None:
            clauses.append("proposal_id=?"); params.append(proposal_id)
        if operator is not None:
            clauses.append("operator=?"); params.append(operator)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        # The timing columns are selected here, not left to ad-hoc SQL: they
        # were added to answer "where does the 71% go", and an accessor that
        # silently drops them makes the instrumentation invisible to every
        # caller that does not already know it exists.
        #
        # THE LATE COLUMNS ARE READ OFF `_LATE_EXECUTION_COLUMNS` RATHER THAN
        # RETYPED. They were retyped once, and adding `start_method` to the
        # migration, the INSERT and the dataclass while missing this list
        # produced a column that wrote correctly and read back NULL every time
        # -- caught only because a test asserted the round-tripped VALUE rather
        # than the column's existence. Two hand-maintained copies of one list is
        # the same silent-divergence shape as the seven operator tables (§2.3
        # B2); here it costs one line to remove entirely.
        base = ["proposal_id", "run_id", "operator", "kernel_id",
                "passed_checker", "passed_naive", "wall_time_ms", "error_type",
                "n_checks", "n_failed", "check_results_json", "error_json",
                "created_at"]
        cols = base + [name for name, _ in _LATE_EXECUTION_COLUMNS]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM executions{where} "
                "ORDER BY created_at",
                tuple(params),
            ).fetchall()

        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["passed_checker"] = bool(d["passed_checker"])
            d["passed_naive"] = bool(d["passed_naive"])
            d["check_results"] = json.loads(d.pop("check_results_json"))
            sp = d.pop("startup_phases_json")
            d["startup_phases"] = json.loads(sp) if sp else None
            ej = d.pop("error_json")
            d["error"] = json.loads(ej) if ej else None
            out.append(d)
        return out

    def get_verdicts_for_run(self, run_id: str) -> List[ProposalVerdict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT verdict_json FROM verdicts "
                "WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [ProposalVerdict.from_dict(json.loads(r[0])) for r in rows]

    def get_hits_for_operator(self, operator: str) -> List[ProposalVerdict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT verdict_json FROM verdicts "
                "WHERE operator=? AND is_hit=1 ORDER BY beam_score DESC",
                (operator,),
            ).fetchall()
        return [ProposalVerdict.from_dict(json.loads(r[0])) for r in rows]

    def top_beam_candidates(
        self, run_id: str, beam_width: int
    ) -> List[ProposalVerdict]:
        with self._lock:
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
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_items "
                "(item_id, operator, bug_pattern, summary, source_run, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (item_id, operator, bug_pattern, summary, source_run, time.time()),
            )
            self._conn.commit()
        return item_id

    def get_memory_items(
        self, operator: str, limit: int = 5
    ) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT bug_pattern, summary FROM memory_items "
                "WHERE operator=? ORDER BY created_at DESC LIMIT ?",
                (operator, limit),
            ).fetchall()
        return [{"bug_pattern": r[0], "summary": r[1]} for r in rows]

    # ── Resume support ────────────────────────────────────────────────────────

    def resume_run(self, run_id: str) -> Optional[Dict]:
        run = self.get_run(run_id)
        if run is None or run["status"] in ("done", "hit", "no_hit"):
            return None

        proposals = self.get_proposals_for_run(run_id)
        verdicts  = self.get_verdicts_for_run(run_id)

        last_per_worker: Dict[str, InputProposal] = {}
        for p in proposals:
            if p.worker_id not in last_per_worker or \
               p.iteration > last_per_worker[p.worker_id].iteration:
                last_per_worker[p.worker_id] = p

        return {
            "run":             run,
            "last_per_worker": last_per_worker,
            "verdicts":        verdicts,
            "n_proposals":     len(proposals),
        }

    # ── Coverage report ───────────────────────────────────────────────────────

    def coverage_report(self) -> Dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.operator, p.proposal_json, v.hit_mutants "
                "FROM verdicts v "
                "JOIN proposals p ON v.proposal_id = p.proposal_id "
                "WHERE v.is_hit=1 AND v.gap_confirmed=1"
            ).fetchall()

        report: Dict[str, Dict] = {}
        for operator, prop_json, hit_json in rows:
            prop    = InputProposal.from_dict(json.loads(prop_json))
            hits    = json.loads(hit_json)
            pattern = prop.predicted_failure_mode
            report.setdefault(operator, {}).setdefault(pattern, []).extend(hits)

        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _operator_for_run_unlocked(self, run_id: str) -> str:
        """Read operator without acquiring lock — caller must hold lock or
        call before any concurrent access begins."""
        row = self._conn.execute(
            "SELECT operator FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else "unknown"

    def _operator_for_run(self, run_id: str) -> str:
        with self._lock:
            return self._operator_for_run_unlocked(run_id)

    def close(self):
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()