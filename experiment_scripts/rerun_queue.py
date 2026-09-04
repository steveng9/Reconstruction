#!/usr/bin/env python
"""
rerun_queue.py — durable work queue for the post-encoding-fix re-run.

Every unit of work (generate one synth.csv, or score one attack) is a row in
`rerun_queue.db`. Workers claim rows atomically, so any number of them can run
concurrently, a worker can die without losing work, and the whole thing can be
stopped and relaunched at will. Nothing lives in memory, so logging out is safe.

    JOB KINDS
      generate : write {sample}/{SDG}_eps{e}/synth.csv for one (sample, gen, eps)
      attack   : score one (dataset, sample, sdg, qi, attack) and insert into
                 results.db

    TIERS (workers always drain the lowest non-empty tier first)
      0  generation — adult 10k, the primary dataset for the epsilon figure
      1  generation — cdc 1k and adult 1k
      2  attacks    — the epsilon-sweep trio (RandomForest/NaiveBayes/CoBP-RA)
                      on every DP generator; this is what the new figure needs
      3  attacks    — the rest of the Table 1 / appendix attack set
      4  attacks    — exploratory attack variants (appendix only)

CLI
    python experiment_scripts/rerun_queue.py build          # populate (idempotent)
    python experiment_scripts/rerun_queue.py status         # progress by tier/kind
    python experiment_scripts/rerun_queue.py reset --failed # retry failed jobs
    python experiment_scripts/rerun_queue.py reap           # free jobs from dead workers
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT


import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_ROOT = Path(str(DATA_ROOT))
QUEUE_DB = Path(__file__).with_name("rerun_queue.db")
RESULTS_DB = Path(__file__).with_name("results.db")

EPSILONS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
DP_GENERATORS = ["MST", "AIM", "PrivBayes", "PrivSyn", "MWEMPGM", "PrivateGSD"]
N_SAMPLES = 5

# dataset key -> (data dir, db dataset name, db size, meta.json, dataset type)
DATASETS = {
    "adult10k": (DATA_ROOT / "adult/size_10000", "adult", 10000,
                 DATA_ROOT / "adult/meta.json", "categorical"),
    "adult1k":  (DATA_ROOT / "adult/size_1000", "adult", 1000,
                 DATA_ROOT / "adult/meta.json", "categorical"),
    "cdc1k":    (DATA_ROOT / "cdc_diabetes/size_1000", "cdc_diabetes", 1000,
                 DATA_ROOT / "cdc_diabetes/meta.json", "categorical"),
}
GEN_TIER = {"adult10k": 0, "adult1k": 1, "cdc1k": 1}

# The epsilon-sweep trio — the attacks the new multi-generator figure is built on.
SWEEP_ATTACKS = {
    "RandomForest": "RandomForest",
    "NaiveBayes": "NaiveBayes",
    "CoBP-RA": "CoBP-RA",
}

# archived attack_label -> registry attack name. Labels are preserved on re-insert
# so existing LaTeX/table scripts keep resolving; only the *method* is renamed
# (see the 2026-06 attack rename). Anything not listed here is queued at tier 4
# only if its label is also a valid registry name.
LABEL_TO_METHOD = {
    "Random": "Random", "Mode": "Mode", "KNN": "KNN", "NaiveBayes": "NaiveBayes",
    "LogisticRegression": "LogisticRegression", "RandomForest": "RandomForest",
    "MLP": "MLP", "SVM": "SVM", "TabPFN": "TabPFN",
    "MeasureDeid": "MeasureDeid", "LinearReconstruction": "LinearReconstruction",
    "LinearReconstructionCategorical": "LinearReconstructionCategorical",
    "LinearReconstructionJoint": "LinearReconstructionJoint",
    "CoBP-RA": "CoBP-RA",
    "MarginalRF": "CoBP-RA",
    "MarginalRF_graphQI_entropyBP": "CoBP-RA_graphQI_entropyBP",
    "TabDDPM": "CondDDPM", "TabDDPMWithMLP": "CondDDPMWithMLP",
    "PartialMST": "CondMST", "PartialMSTBounded": "CondMSTBounded",
    "PartialMSTIndependent": "CondMSTIndependent",
    "ConditionedRePaint": "CondRePaint", "RePaint": "RePaint",
    "Attention": "ARFFormer", "JointMLP": "MultiHeadMLP",
}
# Attacks in the manuscript's main + appendix tables (tier 3); everything else
# that maps cleanly goes to tier 4.
CORE_LABELS = {
    "Random", "Mode", "KNN", "NaiveBayes", "LogisticRegression", "RandomForest",
    "MLP", "SVM", "TabPFN", "MeasureDeid", "LinearReconstruction",
    "CoBP-RA", "MarginalRF_graphQI_entropyBP", "TabDDPM", "PartialMST",
    "ConditionedRePaint", "Attention", "JointMLP",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    tier         INTEGER NOT NULL,
    kind         TEXT    NOT NULL,          -- 'generate' | 'attack'
    ds_key       TEXT    NOT NULL,          -- adult10k | adult1k | cdc1k
    dataset      TEXT    NOT NULL,
    dataset_size INTEGER NOT NULL,
    sample       INTEGER NOT NULL,
    sdg_method   TEXT    NOT NULL,          -- dir name, e.g. 'MST_eps1'
    sdg_base     TEXT    NOT NULL,          -- 'MST'
    epsilon      REAL,
    qi           TEXT,                      -- attack jobs only
    attack_label TEXT,                      -- attack jobs only
    attack_method TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    worker       TEXT,
    heartbeat    TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    elapsed_s    REAL,
    error        TEXT,
    UNIQUE(kind, dataset, dataset_size, sample, sdg_method, qi, attack_label)
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, tier, job_id);
"""


def connect(path=QUEUE_DB, timeout=60.0):
    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")      # concurrent readers + one writer
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def init(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def sdg_dirname(base, eps):
    return f"{base}_eps{eps:g}"


# ── queue construction ────────────────────────────────────────────────────────

def build(conn, verbose=True):
    """Populate the queue. Idempotent — existing rows are left untouched."""
    init(conn)
    added = {"generate": 0, "attack": 0}

    # ---- generation: fill the full 6 generators x 9 epsilons x 5 samples grid
    for ds_key, (root, dataset, size, _meta, _typ) in DATASETS.items():
        for base in DP_GENERATORS:
            for eps in EPSILONS:
                for s in range(N_SAMPLES):
                    d = sdg_dirname(base, eps)
                    if (root / f"sample_{s:02d}" / d / "synth.csv").exists():
                        continue
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO jobs (tier, kind, ds_key, dataset, "
                        "dataset_size, sample, sdg_method, sdg_base, epsilon) "
                        "VALUES (?,'generate',?,?,?,?,?,?,?)",
                        (GEN_TIER[ds_key], ds_key, dataset, size, s, d, base, eps))
                    added["generate"] += cur.rowcount

    # ---- attacks: restore everything the archive lost
    res = sqlite3.connect(RESULTS_DB)
    archived = res.execute("""
        SELECT DISTINCT dataset, dataset_size, sample, qi, sdg_method, attack_label
        FROM runs_superseded""").fetchall()
    res.close()

    ds_by_db = {(v[1], v[2]): k for k, v in DATASETS.items()}

    for dataset, size, sample, qi, sdg_method, label in archived:
        ds_key = ds_by_db.get((dataset, size))
        if ds_key is None:
            continue                      # arizona / sbo / adult20k: see `build --all`
        method = LABEL_TO_METHOD.get(label)
        if method is None:
            continue
        tier = 2 if label in SWEEP_ATTACKS else (3 if label in CORE_LABELS else 4)
        base = sdg_method.split("_eps")[0]
        eps = None
        if "_eps" in sdg_method:
            try:
                eps = float(sdg_method.split("_eps")[1])
            except ValueError:
                pass
        cur = conn.execute(
            "INSERT OR IGNORE INTO jobs (tier, kind, ds_key, dataset, dataset_size, "
            "sample, sdg_method, sdg_base, epsilon, qi, attack_label, attack_method) "
            "VALUES (?,'attack',?,?,?,?,?,?,?,?,?,?)",
            (tier, ds_key, dataset, size, sample, sdg_method, base, eps,
             qi, label, method))
        added["attack"] += cur.rowcount

    # ---- attacks: the sweep trio on every DP generator/epsilon/sample/QI,
    #      including the cells that never existed (new PrivateGSD/AIM coverage)
    res = sqlite3.connect(RESULTS_DB)
    for ds_key, (root, dataset, size, _meta, _typ) in DATASETS.items():
        qis = [r[0] for r in res.execute(
            "SELECT DISTINCT qi FROM runs WHERE dataset=? AND dataset_size=? "
            "UNION SELECT DISTINCT qi FROM runs_superseded WHERE dataset=? AND dataset_size=?",
            (dataset, size, dataset, size))]
        qis = [q for q in qis if q in ("QI1", "QI_large", "QI_behavioral")]
        for base in DP_GENERATORS:
            for eps in EPSILONS:
                for s in range(N_SAMPLES):
                    for qi in qis:
                        for label, method in SWEEP_ATTACKS.items():
                            cur = conn.execute(
                                "INSERT OR IGNORE INTO jobs (tier, kind, ds_key, dataset, "
                                "dataset_size, sample, sdg_method, sdg_base, epsilon, qi, "
                                "attack_label, attack_method) VALUES (?,'attack',?,?,?,?,?,?,?,?,?,?)",
                                (2, ds_key, dataset, size, s, sdg_dirname(base, eps),
                                 base, eps, qi, label, method))
                            added["attack"] += cur.rowcount
    res.close()
    conn.commit()

    if verbose:
        print(f"added {added['generate']} generate jobs, {added['attack']} attack jobs")
        status(conn)


# ── worker-facing helpers ─────────────────────────────────────────────────────

def claim(conn, worker_id, skip_ids=(), tiers=None, kinds=None,
          sdg_bases=None, exclude_sdg_bases=None):
    """Atomically take the cheapest pending job in the lowest tier.

    `tiers` / `kinds` / `sdg_bases` / `exclude_sdg_bases` let a worker be pinned
    to part of the queue. Without them every worker drains the lowest non-empty
    tier, which starves cheap work behind slow generation; see launch_rerun.sh,
    which splits the pool into generators, attackers, and an AIM-only lane.

    Ordering within a tier is cost-aware, not by job_id. AIM's runtime scales
    with epsilon (adult 10k: ~2 min at eps=0.1, ~3.5 h at eps=10, >6 h and
    timing out at eps>=300 even on the 1k datasets) while every other generator
    finishes in seconds to minutes. Because AIM happens to hold the lowest
    job_ids in each tier, plain `ORDER BY job_id` handed every free worker an
    AIM job and left ~175 cheap jobs untouched for 8+ hours. Sorting AIM last
    keeps it progressing without letting it monopolise the pool.
    """
    now = datetime.now(timezone.utc).isoformat()
    for _ in range(50):
        conn.execute("BEGIN IMMEDIATE")
        try:
            filters, params = ["status='pending'"], []
            if skip_ids:
                filters.append(f"job_id NOT IN ({','.join('?' * len(skip_ids))})")
                params.extend(skip_ids)
            if tiers:
                filters.append(f"tier IN ({','.join('?' * len(tiers))})")
                params.extend(tiers)
            if kinds:
                filters.append(f"kind IN ({','.join('?' * len(kinds))})")
                params.extend(kinds)
            if sdg_bases:
                filters.append(f"sdg_base IN ({','.join('?' * len(sdg_bases))})")
                params.extend(sdg_bases)
            if exclude_sdg_bases:
                filters.append(
                    f"sdg_base NOT IN ({','.join('?' * len(exclude_sdg_bases))})")
                params.extend(exclude_sdg_bases)
            # Non-AIM work keeps tier order. AIM ignores tiers and walks its own
            # cost gradient instead: cheapest epsilon first, and within an
            # epsilon the 1k datasets before the 10k one. That finishes the most
            # cells per CPU-hour and defers the multi-hour runs to the very end.
            q = (f"SELECT job_id FROM jobs WHERE {' AND '.join(filters)} "
                 f"ORDER BY (sdg_base='AIM'), "
                 f"CASE WHEN sdg_base='AIM' THEN 0 ELSE tier END, "
                 f"CASE WHEN sdg_base='AIM' THEN epsilon ELSE 0 END, "
                 f"CASE WHEN sdg_base='AIM' THEN dataset_size ELSE 0 END, "
                 f"job_id LIMIT 1")
            row = conn.execute(q, tuple(params)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            job_id = row[0]
            conn.execute(
                "UPDATE jobs SET status='running', worker=?, started_at=?, "
                "heartbeat=?, attempts=attempts+1 WHERE job_id=? AND status='pending'",
                (worker_id, now, now, job_id))
            conn.execute("COMMIT")
        except sqlite3.OperationalError:
            conn.execute("ROLLBACK")
            continue
        cur = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, cur.fetchone()))
    return None


def finish(conn, job_id, ok, error=None, elapsed=None, max_attempts=3):
    now = datetime.now(timezone.utc).isoformat()
    if ok:
        conn.execute("UPDATE jobs SET status='done', finished_at=?, elapsed_s=?, "
                     "error=NULL WHERE job_id=?", (now, elapsed, job_id))
    else:
        attempts = conn.execute("SELECT attempts FROM jobs WHERE job_id=?",
                                (job_id,)).fetchone()[0]
        new = "pending" if attempts < max_attempts else "failed"
        conn.execute("UPDATE jobs SET status=?, finished_at=?, error=? WHERE job_id=?",
                     (new, now, (error or "")[:4000], job_id))
    conn.commit()


def release(conn, job_id, status="pending", error=None):
    conn.execute("UPDATE jobs SET status=?, attempts=MAX(attempts-1,0), error=? "
                 "WHERE job_id=?", (status, (error or "")[:2000], job_id))
    conn.commit()


def beat(conn, job_id):
    conn.execute("UPDATE jobs SET heartbeat=? WHERE job_id=?",
                 (datetime.now(timezone.utc).isoformat(), job_id))
    conn.commit()


def reap(conn, stale_minutes=180, verbose=True):
    """Return jobs whose worker vanished (no heartbeat) to the pending pool."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()
    n = conn.execute(
        "UPDATE jobs SET status='pending', worker=NULL, "
        "error='reaped: worker heartbeat stale' "
        "WHERE status='running' AND (heartbeat IS NULL OR heartbeat < ?)",
        (cutoff,)).rowcount
    conn.commit()
    if verbose:
        print(f"reaped {n} stale running job(s)")
    return n


def status(conn):
    init(conn)
    print(f"\n{'tier':>4} {'kind':10} {'pending':>8} {'running':>8} {'done':>8} "
          f"{'failed':>8} {'blocked':>8} {'cancel':>8}")
    rows = conn.execute("""
        SELECT tier, kind,
          SUM(status='pending'), SUM(status='running'), SUM(status='done'),
          SUM(status='failed'),  SUM(status='blocked'), SUM(status='cancelled')
        FROM jobs GROUP BY tier, kind ORDER BY tier, kind""").fetchall()
    for t, k, p, r, d, f, b, c in rows:
        print(f"{t:>4} {k:10} {p or 0:>8} {r or 0:>8} {d or 0:>8} {f or 0:>8} "
              f"{b or 0:>8} {c or 0:>8}")
    # 'cancelled' is a manual terminal state (never set by a worker): it takes a
    # job permanently out of the pool without pretending it succeeded. Excluded
    # from the denominator so the percentage tracks work we still intend to do.
    tot = conn.execute(
        "SELECT COUNT(*), SUM(status='done'), SUM(status='failed'), "
        "SUM(status='cancelled') FROM jobs").fetchone()
    live = tot[0] - (tot[3] or 0)
    if live:
        print(f"\ntotal {tot[0]}  ({tot[3] or 0} cancelled)  done {tot[1] or 0}"
              f"/{live} ({100 * (tot[1] or 0) / live:.1f}%)  failed {tot[2] or 0}")
    errs = conn.execute(
        "SELECT error, COUNT(*) c FROM jobs WHERE status='failed' AND error IS NOT NULL "
        "GROUP BY substr(error,1,80) ORDER BY c DESC LIMIT 5").fetchall()
    if errs:
        print("\ntop failure modes:")
        for e, c in errs:
            print(f"  {c:>5}  {e.splitlines()[0][:100] if e else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("status")
    sub.add_parser("reap").add_argument("--stale-minutes", type=int, default=180)
    r = sub.add_parser("reset")
    r.add_argument("--failed", action="store_true")
    r.add_argument("--running", action="store_true")
    r.add_argument("--tier", type=int)
    args = ap.parse_args()

    conn = connect()
    if args.cmd == "build":
        build(conn)
    elif args.cmd == "status":
        status(conn)
    elif args.cmd == "reap":
        reap(conn, args.stale_minutes)
    elif args.cmd == "reset":
        where, params = [], []
        if args.failed:
            where.append("status='failed'")
        if args.running:
            where.append("status='running'")
        if args.tier is not None:
            where.append("tier=?"); params.append(args.tier)
        if not where:
            ap.error("reset needs --failed and/or --running and/or --tier")
        n = conn.execute(
            f"UPDATE jobs SET status='pending', attempts=0, error=NULL, worker=NULL "
            f"WHERE {' AND '.join(where) if args.tier is not None and len(where) > 1 else ' OR '.join(where)}",
            params).rowcount
        conn.commit()
        print(f"reset {n} job(s) to pending")
    conn.close()


if __name__ == "__main__":
    main()
