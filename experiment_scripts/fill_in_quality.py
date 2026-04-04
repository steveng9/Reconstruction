#!/usr/bin/env python
"""
fill_in_quality.py — fill missing synthetic data quality metrics.

Three phases:
  Phase 1 — Re-evaluate: find synth.csv files on disk missing from any existing
             results CSV; evaluate them and save their metrics.
  Phase 2 — Generate: create new synth.csv files for targeted missing combos
             (new epsilon variants, AIM_eps3 on other datasets, etc.)
             Generation skips any synth.csv that already exists on disk.
  Phase 3 — Evaluate: compute quality metrics for all Phase 2 outputs.

Resume safety:
  - A checkpoint JSON (fill_in_checkpoint.json) records completed generation
    and evaluation job keys.  Relaunching skips anything in the checkpoint.
  - Phase 1 also cross-checks the most-recent existing results CSV so it does
    not re-evaluate jobs already recorded there.
  - Results are appended to a single output CSV one row at a time (crash-safe).

Memory throttle:
  - Before submitting each new job to the worker pool, the script checks
    psutil RAM usage and waits until it drops below --mem-watermark (%).
  - This prevents OOM cascades on long-running AIM jobs at large sizes.

Usage:
    conda activate recon_
    python experiment_scripts/fill_in_quality.py
    python experiment_scripts/fill_in_quality.py --dry-run
    python experiment_scripts/fill_in_quality.py --phase 1
    python experiment_scripts/fill_in_quality.py --phase 2
    python experiment_scripts/fill_in_quality.py --sdg-workers 4 --eval-workers 8
    python experiment_scripts/fill_in_quality.py --mem-watermark 80
    python experiment_scripts/fill_in_quality.py --dataset adult --method AIM_eps3
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_ROOT   = Path("/home/golobs/data/reconstruction_data")
GENERATE_PY = REPO_ROOT / "sdg" / "generate_synth.py"

CHECKPOINT_PATH = SCRIPTS_DIR / "fill_in_checkpoint.json"
PROGRESS_LOG    = REPO_ROOT / "outfiles" / "fill_in_quality_progress.log"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


# ── Dataset eval config (matches evaluate_synth_quality.py DATASET_CONFIGS) ──

DATASET_EVAL_CONFIG: dict[tuple[str, str], dict] = {
    ("adult",            "size_1000"):           {"target": "income",         "task": "classification"},
    ("adult",            "size_10000"):           {"target": "income",         "task": "classification"},
    ("adult",            "size_20000"):           {"target": "income",         "task": "classification"},
    ("nist_arizona_data","size_10000_25feat"):    {"target": "EMPSTAT",        "task": "classification"},
    ("nist_sbo",         "size_1000"):            {"target": "SEX1",           "task": "classification"},
    ("cdc_diabetes",     "size_1000"):            {"target": "Diabetes_binary","task": "classification"},
    ("cdc_diabetes",     "size_100000"):          {"target": "Diabetes_binary","task": "classification"},
    ("california",       "size_1000"):            {"target": "MedHouseVal",    "task": "regression"},
}

N_SAMPLES = 5  # expected samples per (dataset, size_dir)


# ── Generation targets ────────────────────────────────────────────────────────
# Each entry: (dataset, size_dir, samples, method, config)
# The script skips any synth.csv that already exists on disk.
# bin_continuous_as_ordinal=True avoids BinTransformer failures on skewed
# continuous columns at small epsilon (see generate_synth.py comments).

GENERATION_TARGETS: list[tuple[str, str, list[int], str, dict]] = [

    # ── adult / size_1000 ─────────────────────────────────────────────────────
    # MST_eps300: new epsilon, none generated anywhere yet
    ("adult", "size_1000", [0,1,2,3,4], "MST", {"epsilon": 300.0, "bin_continuous_as_ordinal": True}),
    # AIM_eps100: sample_00 exists, fill in the remaining 4
    ("adult", "size_1000", [1,2,3,4],   "AIM", {"epsilon": 100.0, "bin_continuous_as_ordinal": True}),

    # ── adult / size_10000 ────────────────────────────────────────────────────
    # MST intermediate epsilons (0.3, 3, 30 only generated for size_1000)
    ("adult", "size_10000", [0,1,2,3,4], "MST", {"epsilon": 0.3,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_10000", [0,1,2,3,4], "MST", {"epsilon": 3.0,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_10000", [0,1,2,3,4], "MST", {"epsilon": 30.0, "bin_continuous_as_ordinal": True}),
    ("adult", "size_10000", [0,1,2,3,4], "MST", {"epsilon": 300.0,"bin_continuous_as_ordinal": True}),
    # AIM low-to-mid epsilons (0.3 and 3 only generated for size_1000)
    ("adult", "size_10000", [0,1,2,3,4], "AIM", {"epsilon": 0.3,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_10000", [0,1,2,3,4], "AIM", {"epsilon": 3.0,  "bin_continuous_as_ordinal": True}),
    # AIM_eps1 for size_10000 already exists — skipped by existence check

    # ── adult / size_20000 ────────────────────────────────────────────────────
    ("adult", "size_20000", [0,1,2,3,4], "MST", {"epsilon": 0.3,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_20000", [0,1,2,3,4], "MST", {"epsilon": 3.0,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_20000", [0,1,2,3,4], "MST", {"epsilon": 30.0, "bin_continuous_as_ordinal": True}),
    ("adult", "size_20000", [0,1,2,3,4], "MST", {"epsilon": 300.0,"bin_continuous_as_ordinal": True}),
    ("adult", "size_20000", [0,1,2,3,4], "AIM", {"epsilon": 0.3,  "bin_continuous_as_ordinal": True}),
    ("adult", "size_20000", [0,1,2,3,4], "AIM", {"epsilon": 3.0,  "bin_continuous_as_ordinal": True}),
    # AIM_eps10: sample_04 exists, fill remaining 4
    ("adult", "size_20000", [0,1,2,3],   "AIM", {"epsilon": 10.0, "bin_continuous_as_ordinal": True}),

    # ── other datasets: AIM_eps3 only ─────────────────────────────────────────
    # california: all cols continuous, bin_continuous needed
    ("california",      "size_1000",         [0,1,2,3,4], "AIM", {"epsilon": 3.0, "bin_continuous_as_ordinal": True}),
    # cdc_diabetes: BMI/MentHlth/PhysHlth are skewed continuous
    ("cdc_diabetes",    "size_1000",         [0,1,2,3,4], "AIM", {"epsilon": 3.0, "bin_continuous_as_ordinal": True}),
    # nist_arizona 25-feat: INCWAGE/WKSWORK1/etc have large ranges
    ("nist_arizona_data","size_10000_25feat",[0,1,2,3,4], "AIM", {"epsilon": 3.0, "bin_continuous_as_ordinal": True}),
    # nist_sbo: AIM is computationally infeasible for 130 columns — intentionally omitted

]


# ── Checkpoint ────────────────────────────────────────────────────────────────

class Checkpoint:
    """Persistent JSON checkpoint with atomic saves."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {"gen_done": [], "eval_done": [], "gen_failed": [], "eval_failed": []}
        if path.exists():
            try:
                with open(path) as f:
                    loaded = json.load(f)
                self._data.update(loaded)
            except Exception as e:
                print(f"[WARN] Could not load checkpoint: {e}. Starting fresh.")
        self._gen_done  = set(self._data["gen_done"])
        self._eval_done = set(self._data["eval_done"])

    def gen_done(self, key: str) -> bool:
        return key in self._gen_done

    def eval_done(self, key: str) -> bool:
        return key in self._eval_done

    def mark_gen_done(self, key: str):
        if key not in self._gen_done:
            self._gen_done.add(key)
            self._data["gen_done"].append(key)
            self._save()

    def mark_eval_done(self, key: str):
        if key not in self._eval_done:
            self._eval_done.add(key)
            self._data["eval_done"].append(key)
            self._save()

    def mark_gen_failed(self, key: str, error: str):
        self._data["gen_failed"].append({"key": key, "error": error[:300],
                                          "ts": datetime.now().isoformat()})
        self._save()

    def mark_eval_failed(self, key: str, error: str):
        self._data["eval_failed"].append({"key": key, "error": error[:300],
                                           "ts": datetime.now().isoformat()})
        self._save()

    def _save(self):
        """Atomic save via temp file → rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self.path)

    @property
    def stats(self) -> str:
        return (f"gen_done={len(self._gen_done)}  eval_done={len(self._eval_done)}  "
                f"gen_failed={len(self._data['gen_failed'])}  "
                f"eval_failed={len(self._data['eval_failed'])}")


# ── Memory throttle ───────────────────────────────────────────────────────────

class MemoryThrottle:
    """Pause before submitting a job if RAM usage exceeds watermark."""

    def __init__(self, watermark: float = 75.0, poll: float = 20.0):
        self.watermark = watermark
        self.poll = poll
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            print("[WARN] psutil not installed — memory throttle disabled.")
            self._psutil = None

    def wait(self, label: str = ""):
        if self._psutil is None:
            return
        while True:
            pct = self._psutil.virtual_memory().percent
            if pct < self.watermark:
                return
            _log(f"[MEM] {pct:.0f}% ≥ {self.watermark:.0f}% — waiting {self.poll:.0f}s "
                 f"before next submission{(' (' + label + ')') if label else ''}")
            time.sleep(self.poll)


# ── SDG generation (subprocess) ───────────────────────────────────────────────

def _sdg_dirname(method: str, config: dict) -> str:
    """Derive canonical SDG directory name — mirrors sdg.sdg_dirname."""
    config = config or {}
    eps = config.get("epsilon") or config.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method


def _gen_job_key(dataset: str, size_dir: str, sample: int, method_dir: str) -> str:
    return f"{dataset}|{size_dir}|{sample:02d}|{method_dir}"


def _eval_job_key(dataset: str, size_dir: str, sample: str, method_dir: str) -> str:
    return f"{dataset}|{size_dir}|{sample}|{method_dir}"


def run_sdg_generation(dataset: str, size_dir: str, sample_idx: int,
                        method: str, config: dict) -> str:
    """Generate one synth.csv by calling generate_synth.py --job.

    Returns the output_csv path on success.
    Raises RuntimeError on failure.
    """
    import json as _json

    ds_dir     = DATA_ROOT / dataset / size_dir
    sample_dir = ds_dir / f"sample_{sample_idx:02d}"
    train_csv  = sample_dir / "train.csv"
    dirname    = _sdg_dirname(method, config)
    output_csv = sample_dir / dirname / "synth.csv"

    if output_csv.exists():
        return str(output_csv)   # already done

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    # Build filtered meta (handles 25-feat subsets automatically)
    import pandas as pd
    meta_path = DATA_ROOT / dataset / "meta.json"
    with open(meta_path) as f:
        meta_raw = _json.load(f)
    train_df = pd.read_csv(train_csv, nrows=0)  # header only for column list
    cols = set(train_df.columns)
    meta_filtered = {
        "categorical": [c for c in meta_raw.get("categorical", []) if c in cols],
        "continuous":  [c for c in meta_raw.get("continuous",  []) if c in cols],
        "ordinal":     [c for c in meta_raw.get("ordinal",     []) if c in cols],
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_csv.parent / "sdg.log"

    cmd = [
        sys.executable, str(GENERATE_PY), "--job",
        method,
        str(train_csv),
        str(output_csv),
        _json.dumps(meta_filtered),
        _json.dumps(config),
    ]

    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONWARNINGS": "ignore",
                 "PYKEOPS_VERBOSE": "0", "TOKENIZERS_PARALLELISM": "false"},
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"SDG subprocess failed (exit {proc.returncode}). "
            f"See log: {log_path}"
        )
    if not output_csv.exists():
        raise RuntimeError(f"SDG succeeded but synth.csv not written: {output_csv}")

    return str(output_csv)


# ── Evaluation worker ─────────────────────────────────────────────────────────

def _eval_worker(job: dict) -> tuple[str, dict | None, str | None]:
    """Worker: runs in subprocess via ProcessPoolExecutor."""
    import sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    key = _eval_job_key(job["dataset"], job["size_dir"], job["sample"], job["method"])
    try:
        from evaluate_synth_quality import evaluate_job
        result = evaluate_job(job)
        return key, result, None
    except Exception:
        return key, None, traceback.format_exc()


# ── Results CSV writer ────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "dataset", "size_dir", "sample", "method",
    "n_rows_train", "n_rows_synth",
    "mean_tvd", "mean_jsd", "cat_coverage", "pairwise_tvd",
    "mean_mean_err_pct", "mean_std_err_pct", "mean_wasserstein", "corr_diff",
    "sdv_col_shapes", "sdv_col_pairs",
    "tstr_score", "trtr_score", "tstr_ratio", "prop_score",
]


class ResultsWriter:
    """Append results to a CSV one row at a time (crash-safe)."""

    def __init__(self, path: Path):
        self.path = path
        self._wrote_header = path.exists()

    def write(self, row: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


# ── Progress log ──────────────────────────────────────────────────────────────

def _log(msg: str):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


# ── Load existing results (for Phase 1 dedup) ─────────────────────────────────

def _load_existing_eval_keys() -> set[str]:
    """Read all results CSVs and return set of already-evaluated keys."""
    done = set()
    for csv_path in sorted(SCRIPTS_DIR.glob("synth_quality_results_*.csv")) + \
                    sorted(SCRIPTS_DIR.glob("fill_in_quality_results_*.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, usecols=["dataset", "size_dir", "sample", "method"])
            for _, row in df.iterrows():
                done.add(_eval_job_key(row["dataset"], row["size_dir"],
                                       str(row["sample"]), row["method"]))
        except Exception as e:
            _log(f"[WARN] Could not read {csv_path.name}: {e}")
    return done


# ── Phase 1: re-evaluate existing uneval'd synth data ─────────────────────────

def _discover_unevaluated_jobs(existing_keys: set[str], checkpoint: Checkpoint,
                                ds_filter: list[str] | None,
                                method_filter: list[str] | None) -> list[dict]:
    """Find synth.csv files on disk not yet evaluated."""
    jobs = []
    for (dataset, size_dir), eval_cfg in DATASET_EVAL_CONFIG.items():
        if ds_filter and dataset not in ds_filter:
            continue
        ds_dir    = DATA_ROOT / dataset / size_dir
        meta_path = DATA_ROOT / dataset / "meta.json"
        if not ds_dir.exists() or not meta_path.exists():
            continue
        for sample_idx in range(N_SAMPLES):
            sample_name = f"sample_{sample_idx:02d}"
            sample_dir  = ds_dir / sample_name
            train_path  = sample_dir / "train.csv"
            if not train_path.exists():
                continue
            for method_dir in sorted(sample_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                synth_path = method_dir / "synth.csv"
                if not synth_path.exists():
                    continue
                if method_filter and method_dir.name not in method_filter:
                    continue
                key = _eval_job_key(dataset, size_dir, sample_name, method_dir.name)
                if key in existing_keys or checkpoint.eval_done(key):
                    continue
                jobs.append({
                    "dataset":     dataset,
                    "size_dir":    size_dir,
                    "sample":      sample_name,
                    "method":      method_dir.name,
                    "train_path":  str(train_path),
                    "synth_path":  str(synth_path),
                    "meta_path":   str(meta_path),
                    "target":      eval_cfg["target"],
                    "task":        eval_cfg["task"],
                    "is_baseline": False,
                })
    return jobs


# ── Phase 2: generate new SDG data ────────────────────────────────────────────

def _build_generation_jobs(checkpoint: Checkpoint,
                            ds_filter: list[str] | None,
                            method_filter: list[str] | None) -> list[tuple]:
    """Return list of (dataset, size_dir, sample_idx, method, config, key) to generate."""
    pending = []
    for (dataset, size_dir, samples, method, config) in GENERATION_TARGETS:
        if ds_filter and dataset not in ds_filter:
            continue
        dirname = _sdg_dirname(method, config)
        if method_filter and dirname not in method_filter:
            continue
        for sample_idx in samples:
            key = _gen_job_key(dataset, size_dir, sample_idx, dirname)
            synth_path = DATA_ROOT / dataset / size_dir / f"sample_{sample_idx:02d}" / dirname / "synth.csv"
            if synth_path.exists():
                continue   # already on disk
            if checkpoint.gen_done(key):
                continue   # completed in a prior run (shouldn't happen if file exists)
            pending.append((dataset, size_dir, sample_idx, method, config, key))
    return pending


# ── Run a pool of eval jobs ────────────────────────────────────────────────────

def _run_eval_pool(jobs: list[dict], checkpoint: Checkpoint, writer: ResultsWriter,
                   n_workers: int, throttle: MemoryThrottle, phase_label: str):
    if not jobs:
        _log(f"[{phase_label}] No evaluation jobs to run.")
        return

    _log(f"[{phase_label}] Evaluating {len(jobs)} jobs with {n_workers} workers.")
    ctx = mp.get_context("spawn")
    done = skipped = failed = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        future_to_job = {}
        for job in jobs:
            throttle.wait(f"{job['dataset']}/{job['size_dir']}/{job['method']}")
            fut = pool.submit(_eval_worker, job)
            future_to_job[fut] = job

        for fut in as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                key, result, err = fut.result()
            except Exception as exc:
                err = traceback.format_exc()
                key = _eval_job_key(job["dataset"], job["size_dir"],
                                    job["sample"], job["method"])
                result = None

            if err:
                _log(f"[{phase_label}] FAIL  {key}: {err.splitlines()[-1]}")
                checkpoint.mark_eval_failed(key, err)
                failed += 1
            else:
                writer.write(result)
                checkpoint.mark_eval_done(key)
                _log(f"[{phase_label}] OK    {key}")
                done += 1

    _log(f"[{phase_label}] Eval complete — ok={done}  failed={failed}  "
         f"checkpoint: {checkpoint.stats}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fill in missing synth quality metrics (3 phases)."
    )
    parser.add_argument("--phase",         type=int, default=None, choices=[1, 2],
                        help="Run only phase 1 (re-evaluate) or phase 2 (generate+evaluate). "
                             "Default: run both.")
    parser.add_argument("--sdg-workers",   type=int, default=2,
                        help="Parallel SDG generation workers (default 2).")
    parser.add_argument("--eval-workers",  type=int, default=8,
                        help="Parallel evaluation workers (default 8).")
    parser.add_argument("--mem-watermark", type=float, default=75.0,
                        help="Pause submissions when RAM %% exceeds this (default 75).")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print planned jobs and exit without running.")
    parser.add_argument("--dataset",       nargs="+", default=None,
                        help="Restrict to these dataset names.")
    parser.add_argument("--method",        nargs="+", default=None,
                        help="Restrict to these method directory names (e.g. AIM_eps3).")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Delete the checkpoint file before starting (full re-run).")
    args = parser.parse_args()

    if args.reset_checkpoint and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        _log("Checkpoint deleted — full re-run.")

    checkpoint = Checkpoint(CHECKPOINT_PATH)
    throttle   = MemoryThrottle(watermark=args.mem_watermark)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = SCRIPTS_DIR / f"fill_in_quality_results_{ts}.csv"
    writer      = ResultsWriter(results_csv)

    _log("=" * 70)
    _log(f"fill_in_quality.py  —  {ts}")
    _log(f"  checkpoint : {CHECKPOINT_PATH}")
    _log(f"  results    : {results_csv}")
    _log(f"  sdg-workers: {args.sdg_workers}   eval-workers: {args.eval_workers}")
    _log(f"  mem-wmark  : {args.mem_watermark}%")
    _log(f"  checkpoint : {checkpoint.stats}")
    _log("=" * 70)

    run_gen  = args.phase in (None, 2)
    run_eval = args.phase in (None, 1, 2)

    # ── Phase 1: Generate new SDG data ────────────────────────────────────────
    if run_gen:
        _log("\n── Phase 1: Generate new SDG data ──")
        gen_jobs = _build_generation_jobs(checkpoint, args.dataset, args.method)
        _log(f"  {len(gen_jobs)} generation jobs pending.")

        if args.dry_run:
            for (dataset, size_dir, sample_idx, method, config, key) in gen_jobs:
                dirname = _sdg_dirname(method, config)
                print(f"  [GEN] {dataset}/{size_dir}/sample_{sample_idx:02d}/{dirname}")
        else:
            ctx = mp.get_context("spawn")
            _log(f"  Generating with {args.sdg_workers} workers, "
                 f"mem watermark {args.mem_watermark}%.")
            with ProcessPoolExecutor(max_workers=args.sdg_workers, mp_context=ctx) as pool:
                future_to_meta = {}
                for (dataset, size_dir, sample_idx, method, config, key) in gen_jobs:
                    throttle.wait(f"{dataset}/{size_dir}/sample_{sample_idx:02d}/"
                                  f"{_sdg_dirname(method, config)}")
                    fut = pool.submit(run_sdg_generation, dataset, size_dir,
                                      sample_idx, method, config)
                    future_to_meta[fut] = (dataset, size_dir, sample_idx,
                                           _sdg_dirname(method, config), key)

                gen_ok = gen_fail = 0
                for fut in as_completed(future_to_meta):
                    dataset, size_dir, sample_idx, dirname, key = future_to_meta[fut]
                    label = f"{dataset}/{size_dir}/sample_{sample_idx:02d}/{dirname}"
                    try:
                        fut.result()
                        checkpoint.mark_gen_done(key)
                        _log(f"[P1-gen] OK    {label}")
                        gen_ok += 1
                    except Exception as exc:
                        tb = traceback.format_exc()
                        checkpoint.mark_gen_failed(key, tb)
                        _log(f"[P1-gen] FAIL  {label}: {str(exc)[:120]}")
                        gen_fail += 1

            _log(f"  Generation complete — ok={gen_ok}  failed={gen_fail}")

    # ── Phase 2: Evaluate everything on disk not yet evaluated ────────────────
    # Runs after generation so it catches both newly generated files AND any
    # survivors from previous crashed runs (the old "Phase 1 before gen" missed
    # those because they didn't exist at script start time).
    if run_eval:
        _log("\n── Phase 2: Evaluate all uneval'd synth data on disk ──")
        existing_keys = _load_existing_eval_keys()
        _log(f"  Loaded {len(existing_keys)} already-evaluated keys from existing CSVs.")

        eval_jobs = _discover_unevaluated_jobs(
            existing_keys, checkpoint, args.dataset, args.method
        )
        _log(f"  Found {len(eval_jobs)} unevaluated synth.csv files on disk.")

        if args.dry_run:
            for j in eval_jobs:
                print(f"  [EVAL] {j['dataset']}/{j['size_dir']}/{j['sample']}/{j['method']}")
        else:
            _run_eval_pool(eval_jobs, checkpoint, writer, args.eval_workers,
                           throttle, "P2-eval")

    _log("\n" + "=" * 70)
    _log(f"Done.  Results written to: {results_csv}")
    _log(f"Final checkpoint: {checkpoint.stats}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
