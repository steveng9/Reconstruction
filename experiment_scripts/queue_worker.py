#!/usr/bin/env python
"""
queue_worker.py — one worker process for the rerun_queue.

Claims jobs from rerun_queue.db and executes them until the queue is empty or it
is asked to stop. Run as many as the box will take (see launch_rerun.sh).

ISOLATION
    Every job runs in a *child* process with a timeout. A segfaulting attack, an
    OOM kill, or a generator that wedges therefore costs one job, not the worker
    and not the queue. The parent heartbeats while it waits, so a genuinely dead
    worker can be told apart from a slow one and its job returned to the pool by
    `rerun_queue.py reap`.

RETRIES
    A failed job goes back to 'pending' until it has burned `--max-attempts`
    tries, then it is parked as 'failed' and skipped. Fix the cause and use
    `rerun_queue.py reset --failed` to put them all back.

SIGNALS
    SIGTERM/SIGINT finish the job in flight, release nothing, and exit cleanly.

USAGE
    python experiment_scripts/queue_worker.py --worker-id w01
    python experiment_scripts/queue_worker.py --worker-id w01 --tiers 0 1
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import MIA_ON_DIFFUSION, RECON_SYNTH, REPO_ROOT


import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_queue as q

REPO = str(REPO_ROOT)
_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"[worker] signal {signum} received — finishing current job then exiting",
          flush=True)


def _setup_paths():
    for p in [str(MIA_ON_DIFFUSION),
              str(MIA_ON_DIFFUSION / 'midst_models' / 'single_table_TabDDPM'),
              str(RECON_SYNTH),
              str(RECON_SYNTH / 'attacks'),
              str(RECON_SYNTH / 'attacks' / 'solvers')]:
        if p not in sys.path:
            sys.path.append(p)
    if REPO in sys.path:
        sys.path.remove(REPO)
    sys.path.insert(0, REPO)


# ── job bodies (run inside the child process) ─────────────────────────────────

def _do_generate(job):
    """Generate one synth.csv."""
    import pandas as pd
    from sdg import get_sdg

    root, dataset, size, meta_path, _typ = q.DATASETS[job["ds_key"]]
    sample_dir = root / f"sample_{job['sample']:02d}"
    out_dir = sample_dir / job["sdg_method"]
    out_csv = out_dir / "synth.csv"
    if out_csv.exists():
        return {"status": "already_present"}

    meta = json.loads(Path(meta_path).read_text())
    train_df = pd.read_csv(sample_dir / "train.csv")

    kwargs = {"epsilon": job["epsilon"]}
    if job["sdg_base"] in ("MST", "AIM"):
        # Pre-binning keeps SmartNoise's private bound estimation from collapsing
        # at small epsilon, and is what every existing MST/AIM release used.
        kwargs.update(bin_continuous_as_ordinal=True, n_bins=20)

    synth = get_sdg(job["sdg_base"])(train_df, meta, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "synth.csv.partial"
    synth.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)               # atomic: never leave a half-written file

    ints = [c for c in train_df.columns
            if train_df[c].dtype.kind in "iu" and c in synth.columns]
    bad = [c for c in ints if synth[c].dtype.kind == "f"]
    return {"status": "generated", "rows": len(synth),
            "float_encoded_int_cols": bad}       # must be empty post-fix


def _attack_params_for(method):
    """Attack params for `method`, with re-run-specific overrides.

    The CondMST family caches its fitted graphical model at
    <sample>/<sdg>/partial_mst_artifacts_<QI>/mst_model.pkl and reuses it
    whenever the file exists. Every checkpoint on disk was written between
    March and May 2026, which makes them stale in two independent ways:

      1. They were pickled when the class was named _PartialMSTSynthesizer.
         It is now _CondMSTSynthesizer (the 2026-06 attack rename), so
         pickle.load raises AttributeError -- this is the 45 failed jobs.
      2. They were fit on synth.csv files that still carried float
         bin-midpoint values, i.e. before the August encoding repair. Even
         if they unpickled cleanly they would model the wrong data.

    Forcing retrain makes each job refit on the repaired synth and dump a
    fresh checkpoint over the stale one, so the bad files are corrected in
    place rather than deleted.
    """
    # Imported here, not at module scope: _do_attack does its heavy imports
    # lazily inside the function, so ATTACK_PARAM_DEFAULTS is not a module
    # global and a top-level reference to it raises NameError.
    from attack_defaults import ATTACK_PARAM_DEFAULTS

    params = dict(ATTACK_PARAM_DEFAULTS.get(method, {}))
    if method.startswith("CondMST"):
        params["retrain"] = True
    return params


def _do_attack(job):
    """Score one attack and write results to results.db."""
    import numpy as np
    from attack_defaults import ATTACK_PARAM_DEFAULTS
    from get_data import load_data
    from master_experiment_script import (_prepare_config, _run_attack,
                                          _score_reconstruction)
    from results_db import ResultsDB

    root, dataset, size, _meta, dtype = q.DATASETS[job["ds_key"]]
    sample_dir = root / f"sample_{job['sample']:02d}"
    method = job["attack_method"]

    # Some samples were drawn non-disjointly and carry a NO_HOLDOUT marker; they
    # may neither serve as a holdout nor be paired with one (load_data refuses
    # both). Walk forward to the next usable sample, and fall back to no
    # memorization test rather than losing the ordinary score.
    holdout_dir = None
    if not (sample_dir / "NO_HOLDOUT").exists():
        for step in range(1, q.N_SAMPLES):
            cand = root / f"sample_{(job['sample'] + step) % q.N_SAMPLES:02d}"
            if cand != sample_dir and cand.exists() and not (cand / "NO_HOLDOUT").exists():
                holdout_dir = cand
                break

    cfg = _prepare_config({
        "dataset": {"name": dataset, "dir": str(sample_dir), "size": size,
                    "type": dtype},
        "QI": job["qi"],
        "data_type": dtype,
        "sdg_method": job["sdg_base"],
        "sdg_params": {"epsilon": job["epsilon"]} if job["epsilon"] is not None else None,
        "attack_method": method,
        "memorization_test": ({"enabled": True, "holdout_dir": str(holdout_dir)}
                              if holdout_dir else {"enabled": False}),
        "attack_params": {method: _attack_params_for(method)},
    })

    train_df, synth_df, qi_feats, hidden, holdout_df = load_data(cfg)

    recon = _run_attack(cfg, synth_df, train_df, qi_feats, hidden)
    tr_scores = _score_reconstruction(train_df, recon, hidden, dtype)
    ra_train = float(np.mean(tr_scores))
    tr_feats = {f: float(s) for f, s in zip(hidden, tr_scores)}

    ra_nt, nt_feats = None, None
    if holdout_df is not None:
        recon_nt = _run_attack(cfg, synth_df, holdout_df, qi_feats, hidden)
        nt_scores = _score_reconstruction(holdout_df, recon_nt, hidden, dtype)
        ra_nt = float(np.mean(nt_scores))
        nt_feats = {f: float(s) for f, s in zip(hidden, nt_scores)}

    common = dict(dataset=dataset, dataset_size=size, sample=job["sample"],
                  qi=job["qi"], sdg_method=job["sdg_method"],
                  attack_label=job["attack_label"],
                  sdg_params={"epsilon": job["epsilon"]} if job["epsilon"] is not None else None,
                  attack_params=dict(ATTACK_PARAM_DEFAULTS.get(method, {})),
                  source_file="queue_worker.py",
                  confidence="certain",
                  confidence_notes="post-encoding-fix re-run (integer-decoded synth)")
    with ResultsDB() as db:
        # 'standard' and 'train' are the same quantity in the historical schema
        # (RA against training-set targets); both are written so table scripts
        # keyed on either split resolve.
        db.insert_run(split="standard", ra_mean=ra_train, feature_scores=tr_feats, **common)
        db.insert_run(split="train", ra_mean=ra_train, feature_scores=tr_feats, **common)
        if ra_nt is not None:
            db.insert_run(split="nontraining", ra_mean=ra_nt,
                          feature_scores=nt_feats, **common)

    return {"ra_mean_train": ra_train, "ra_mean_nontraining": ra_nt,
            "delta": None if ra_nt is None else ra_train - ra_nt}


def _child(job, out_path):
    _setup_paths()
    sys.argv = sys.argv[:1]
    try:
        res = _do_generate(job) if job["kind"] == "generate" else _do_attack(job)
        Path(out_path).write_text(json.dumps({"ok": True, "result": res}))
    except Exception as e:
        Path(out_path).write_text(json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}))


# ── worker loop ───────────────────────────────────────────────────────────────

def run_job_isolated(job, timeout, conn):
    """Run one job in a child process, heartbeating until it finishes."""
    ctx = mp.get_context("spawn")
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    proc = ctx.Process(target=_child, args=(job, out_path))
    t0 = time.time()
    proc.start()
    try:
        while proc.is_alive():
            proc.join(timeout=30)
            q.beat(conn, job["job_id"])
            if time.time() - t0 > timeout:
                proc.terminate()
                proc.join(30)
                if proc.is_alive():
                    proc.kill()
                return False, f"timeout after {timeout}s", time.time() - t0
        elapsed = time.time() - t0
        raw = Path(out_path).read_text() if Path(out_path).exists() else ""
        if not raw:
            return False, f"child died with exit code {proc.exitcode}", elapsed
        payload = json.loads(raw)
        if payload.get("ok"):
            return True, payload.get("result"), elapsed
        return False, payload.get("traceback") or payload.get("error"), elapsed
    finally:
        Path(out_path).unlink(missing_ok=True)


def synth_ready(job):
    root = q.DATASETS[job["ds_key"]][0]
    return (root / f"sample_{job['sample']:02d}" / job["sdg_method"] / "synth.csv").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", default=f"w{os.getpid()}")
    ap.add_argument("--generate-timeout", type=int, default=6 * 3600)
    ap.add_argument("--attack-timeout", type=int, default=3 * 3600)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--idle-exit", action="store_true",
                    help="exit when the queue is empty instead of polling")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--tiers", type=int, nargs="+",
                    help="only claim jobs in these tiers")
    ap.add_argument("--kinds", nargs="+", choices=["generate", "attack"],
                    help="only claim jobs of these kinds")
    ap.add_argument("--sdg-bases", nargs="+",
                    help="only claim jobs for these base generators")
    ap.add_argument("--exclude-sdg-bases", nargs="+",
                    help="never claim jobs for these base generators")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    conn = q.connect()
    q.init(conn)
    wid = args.worker_id
    print(f"[{wid}] started (pid {os.getpid()})", flush=True)

    done = failed = 0
    deferred: list[int] = []       # attack jobs whose synth is not generated yet

    excluded = args.exclude_sdg_bases
    borrowed = False        # currently ignoring our own exclusion

    while not _STOP:
        job = q.claim(conn, wid, skip_ids=tuple(deferred[-200:]),
                      tiers=args.tiers, kinds=args.kinds,
                      sdg_bases=args.sdg_bases,
                      exclude_sdg_bases=None if borrowed else excluded)
        if job is None:
            if deferred:
                deferred.clear()   # give blocked jobs another chance
                continue
            # Nothing left in our lane. Rather than idle, pick up the work we
            # were excluded from — this is what repurposes a fast-lane worker
            # onto AIM once every cheap generator is done. Reset afterwards so
            # the worker returns to its own lane if new work shows up there.
            if excluded and not borrowed:
                borrowed = True
                print(f"[{wid}] fast lane empty — borrowing {excluded} work",
                      flush=True)
                continue
            borrowed = False
            if args.idle_exit:
                print(f"[{wid}] queue empty — exiting", flush=True)
                break
            time.sleep(args.poll_seconds)
            continue
        borrowed = False

        # An attack cannot run before its release exists; put it back and move on
        # rather than burning an attempt on it.
        if job["kind"] == "attack" and not synth_ready(job):
            q.release(conn, job["job_id"], "pending", "waiting on synth generation")
            deferred.append(job["job_id"])
            continue

        tag = (f"{job['ds_key']}/s{job['sample']}/{job['sdg_method']}"
               + (f"/{job['qi']}/{job['attack_label']}" if job["kind"] == "attack" else ""))
        print(f"[{wid}] {job['kind']} tier{job['tier']} #{job['job_id']} {tag}", flush=True)

        timeout = args.generate_timeout if job["kind"] == "generate" else args.attack_timeout
        ok, payload, elapsed = run_job_isolated(job, timeout, conn)
        q.finish(conn, job["job_id"], ok,
                 error=None if ok else str(payload),
                 elapsed=elapsed, max_attempts=args.max_attempts)
        if ok:
            done += 1
            print(f"[{wid}]   ok {elapsed:7.1f}s  {payload}", flush=True)
        else:
            failed += 1
            first = str(payload).strip().splitlines()[-1][:200] if payload else "?"
            print(f"[{wid}]   FAIL {elapsed:7.1f}s  {first}", flush=True)

    print(f"[{wid}] exiting — {done} done, {failed} failed", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
