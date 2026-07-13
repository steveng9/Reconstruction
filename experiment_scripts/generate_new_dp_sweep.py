#!/usr/bin/env python
"""
generate_new_dp_sweep.py

Generate synth.csv for the new DP generators (PrivBayes, MWEMPGM, and — if it
lands in time — PrivateGSD) across the same 9-point epsilon sweep used for MST
in the manuscript's `mst_eps_sweep` table (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000),
for Adult 10k.

Writes to the standard layout so downstream attack scripts / evaluate_synth.py
work unmodified:
    {DATA_ROOT}/sample_{i:02d}/{method}_eps{eps:g}/synth.csv

Trial 1 = sample_00 only (default). Samples 1-4 (trials 2-5) can be generated
later by re-running with --samples 1 2 3 4.

Usage:
    conda activate recon_
    python experiment_scripts/generate_new_dp_sweep.py --dry-run
    python experiment_scripts/generate_new_dp_sweep.py --samples 0
    python experiment_scripts/generate_new_dp_sweep.py --samples 0 --methods PrivBayes
    python experiment_scripts/generate_new_dp_sweep.py --samples 1 2 3 4 --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, "/home/golobs/Reconstruction")

DATA_ROOT = Path("/home/golobs/data/reconstruction_data/adult/size_10000")
META_PATH = Path("/home/golobs/data/reconstruction_data/adult/meta.json")

EPSILONS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
METHODS = ["PrivBayes", "MWEMPGM"]


def _sdg_dirname(method, eps):
    return f"{method}_eps{eps:g}"


def _job_list(samples, methods, epsilons, data_root, meta_path):
    jobs = []
    for s in samples:
        for m in methods:
            for e in epsilons:
                jobs.append((s, m, e, str(data_root), str(meta_path)))
    return jobs


def run_one(job):
    sample_idx, method, eps, data_root, meta_path = job
    data_root = Path(data_root)
    import pandas as pd
    from sdg import get_sdg
    from sdg.evaluate_synth import evaluate_one

    sample_dir = data_root / f"sample_{sample_idx:02d}"
    train_csv = sample_dir / "train.csv"
    out_dir = sample_dir / _sdg_dirname(method, eps)
    out_csv = out_dir / "synth.csv"

    meta = json.loads(Path(meta_path).read_text())
    train_df = pd.read_csv(train_csv)

    if out_csv.exists():
        synth_df = pd.read_csv(out_csv)
        status = "skip_existing"
        elapsed = 0.0
    else:
        gen = get_sdg(method)
        t0 = time.time()
        synth_df = gen(train_df, meta, epsilon=eps)
        elapsed = time.time() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(out_csv, index=False)
        status = "generated"

    try:
        results, _ = evaluate_one(train_df, synth_df, meta)
    except Exception as e:
        results = {"error": str(e)}

    return {
        "sample": sample_idx, "method": method, "epsilon": eps,
        "status": status, "elapsed": elapsed,
        "out_csv": str(out_csv), **results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS)
    parser.add_argument("--epsilons", type=float, nargs="+", default=EPSILONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--meta-path", type=str, default=str(META_PATH))
    args = parser.parse_args()

    jobs = _job_list(args.samples, args.methods, args.epsilons, args.data_root, args.meta_path)
    print(f"Total jobs: {len(jobs)}  (samples={args.samples}, methods={args.methods}, epsilons={args.epsilons})")

    if args.dry_run:
        for j in jobs:
            s, m, e = j[0], j[1], j[2]
            print(f"  sample_{s:02d}  {_sdg_dirname(m, e)}")
        return

    results = []
    if args.serial:
        for job in jobs:
            r = run_one(job)
            results.append(r)
            print(_fmt(r), flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"sample": j[0], "method": j[1], "epsilon": j[2],
                         "status": "ERROR", "error": str(e) + "\n" + traceback.format_exc()}
                results.append(r)
                print(_fmt(r), flush=True)

    ok = [r for r in results if r.get("status") in ("generated", "skip_existing")]
    print(f"\n{len(ok)}/{len(jobs)} succeeded.")
    for r in results:
        if r.get("status") not in ("generated", "skip_existing"):
            print(f"  FAILED: sample_{r['sample']:02d} {r['method']} eps={r['epsilon']} -> {r.get('error')}")


def _fmt(r):
    if r.get("status") == "ERROR":
        return f"  [ERROR] sample_{r['sample']:02d} {r['method']} eps={r['epsilon']}: {r.get('error', '')[:200]}"
    tvd = r.get("mean_tvd")
    jsd = r.get("mean_jsd")
    ptvd = r.get("pairwise_tvd")
    merr = r.get("mean_mean_err%")
    tvd_s = f"{tvd:.3f}" if tvd is not None else "n/a"
    jsd_s = f"{jsd:.3f}" if jsd is not None else "n/a"
    ptvd_s = f"{ptvd:.3f}" if ptvd is not None else "n/a"
    merr_s = f"{merr:.1f}%" if merr is not None else "n/a"
    return (f"  [{r['status']:14s}] sample_{r['sample']:02d} {r['method']:10s} eps={r['epsilon']:<6g} "
            f"({r.get('elapsed', 0):.1f}s)  tvd={tvd_s} jsd={jsd_s} pair_tvd={ptvd_s} mean_err={merr_s}")


if __name__ == "__main__":
    main()
