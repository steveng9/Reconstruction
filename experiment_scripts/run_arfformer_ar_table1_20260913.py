#!/usr/bin/env python
"""Autoregressive ARFFormer on Table 1's 13 generator columns (adult 10k, QI1, 5 samples).

Table 1's printed ARFFormer row comes from the per-feature transformer (DB label
`Attention`); the paper describes ARFFormer as autoregressive. This fills the
causal model (registry `ARFFormerAutoregressive`, DB label
`AttentionAutoregressive`) on exactly the printed cells so the row can be swapped
without touching the prose. See CAMERA_READY_PLAN.md §0.17.

Run one process per GPU (``--gpu 0`` and ``--gpu 1``). Each takes every other job
and runs ``NW`` single-threaded workers on its card. Jobs already in results.db
(split='standard') are skipped, so a relaunch resumes where it stopped.
"""
import argparse, os, sys, pathlib, sqlite3, traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

LABEL, METHOD = "AttentionAutoregressive", "ARFFormerAutoregressive"
SDGS = [("RankSwap", "RankSwap", None), ("CellSuppression", "CellSuppression", None),
        ("Synthpop", "Synthpop", None), ("TVAE", "TVAE", None), ("CTGAN", "CTGAN", None),
        ("ARF", "ARF", None), ("TabDDPM", "TabDDPM", None),
        ("MST_eps0.1", "MST", 0.1), ("MST_eps1", "MST", 1.0), ("MST_eps10", "MST", 10.0),
        ("MST_eps100", "MST", 100.0), ("MST_eps1000", "MST", 1000.0), ("AIM_eps1", "AIM", 1.0)]


def all_jobs():
    return [dict(ds_key="adult10k", sample=s, qi="QI1", sdg_method=d, sdg_base=b, epsilon=e,
                 attack_label=LABEL, attack_method=METHOD)
            for s in range(5) for d, b, e in SDGS]


def done_keys():
    con = sqlite3.connect(HERE / "results.db", timeout=60)
    rows = con.execute("SELECT sample, sdg_method FROM runs WHERE dataset='adult' AND dataset_size=10000 "
                       "AND qi='QI1' AND split='standard' AND attack_label=?", (LABEL,)).fetchall()
    con.close()
    return set(rows)


def init_worker():
    # Spawned children inherit this script's argv, and master_experiment_script
    # parses sys.argv at import time, so --gpu would abort every job.
    sys.argv = sys.argv[:1]
    import torch
    torch.set_num_threads(1)


def run(job):
    sys.argv = sys.argv[:1]
    import queue_worker as w
    try:
        w._setup_paths()
        return job, w._do_attack(job), None
    except BaseException:
        return job, None, traceback.format_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="shard index; set CUDA_VISIBLE_DEVICES to match")
    ap.add_argument("--n-gpus", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import rerun_queue as q
    root = q.DATASETS["adult10k"][0]
    js = [j for i, j in enumerate(all_jobs()) if i % a.n_gpus == a.gpu]
    missing = [(j["sample"], j["sdg_method"]) for j in js
               if not (root / f"sample_{j['sample']:02d}" / j["sdg_method"] / "synth.csv").exists()]
    if missing:
        sys.exit(f"synth.csv missing: {missing}")
    done = done_keys()
    js = [j for j in js if (j["sample"], j["sdg_method"]) not in done]
    print(f"shard {a.gpu}: {len(js)} jobs to run ({len(done)} already in DB overall)", flush=True)
    if a.dry_run or not js:
        sys.exit(0)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NW", "8")),
                             mp_context=mp.get_context("spawn"), initializer=init_worker) as ex:
        for f in as_completed([ex.submit(run, j) for j in js]):
            job, res, err = f.result()
            tag = f"s{job['sample']} {job['sdg_method']}"
            if err:
                bad += 1; print(f"FAIL {tag}\n{err}", flush=True)
            else:
                ok += 1
                print(f"ok {ok + bad}/{len(js)} {tag} ra_train={res['ra_mean_train']:.2f} "
                      f"ra_nt={res['ra_mean_nontraining']}", flush=True)
    print(f"=== DONE shard {a.gpu} ok={ok} fail={bad}", flush=True)
