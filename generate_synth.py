#!/usr/bin/env python
"""
Generate synthetic datasets for reconstruction attack experiments.

Two-step usage:
    Step 1 — Sample training data:
        python generate_synth.py sample

    Step 2 — Generate synthetic data (after verifying samples):
        python generate_synth.py sdg

    Runs in background by default, logging to sdg_log.txt in the dataset's size dir.
    Tail progress with:  tail -f /home/golobs/data/reconstruction_data/adult/size_1000/sdg_log.txt

Single-job mode (called internally by sdg step):
    python generate_synth.py --job <method> <train_csv> <output_csv> <meta_json> <config_json>
"""

import os
import sys
import json
import time
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
#  CONFIGURATION — edit this section to control what's generated
# ============================================================

DATA_ROOT = Path("/home/golobs/data/reconstruction_data")
DATASET = "adult"

SAMPLE_SIZE = 1000
NUM_SAMPLES = 10       # total disjoint training samples to create
RANDOM_SEED = 42

# Which samples to generate SDG for (0-indexed).
# First batch: range(5). Later: range(5, 10).
#SAMPLES_TO_GENERATE = range(1,5)
SAMPLES_TO_GENERATE = range(5)

# Max parallel SDG jobs per sample.
# GPU methods (TVAE, CTGAN, ARF, TabDDPM) share GPU memory — tune accordingly.
MAX_WORKERS = 1

# Column metadata — loaded from meta.json next to full_data.csv
META_PATH = DATA_ROOT / DATASET / "meta.json"
with open(META_PATH) as f:
    META = json.load(f)

# SDG jobs: (method_name, config_kwargs)
# Directory names are derived automatically via sdg_dirname(method, config).
# To add more epsilon settings, just add another tuple.
SDG_JOBS = [
    # Differentially private — vary epsilon
    ("MST",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 10.0}),
    ("MST",  {"epsilon": 100.0}),
    ("MST",  {"epsilon": 1000.0}),
    ("AIM",  {"epsilon": 1.0}),

    # Deep generative models
    ("TVAE",    {}),
    ("CTGAN",   {}),
    ("ARF",     {}),
    ("TabDDPM", {}),

    # R-based / de-identification
    ("Synthpop",        {}),
    ("RankSwap",        {
        "swap_features": ["age", "fnlwgt", "education-num",
                          "capital-gain", "capital-loss", "hours-per-week"],
    }),
    ("CellSuppression", {
        #"key_vars": ["age", "workclass", "education", "sex", "race", "native-country"],
        "key_vars": ["age", "sex", "race"],
        "k": 5,
    }),
]

SDG_JOBS = [
    ("TabDDPM", {}),
]

# Environment variables to suppress noisy warnings in subprocesses
_QUIET_ENV = {
    **os.environ,
    "PYTHONWARNINGS": "ignore",
    "PYKEOPS_VERBOSE": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}


# ============================================================
#  SINGLE-JOB WORKER (called as subprocess)
# ============================================================

def run_single_job(argv):
    """Run one SDG method on one training set. Called via --job flag."""
    # Suppress warnings inside worker
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    _, _, method, train_csv, output_csv, meta_json, config_json = argv
    meta = json.loads(meta_json)
    config = json.loads(config_json)

    sys.path.insert(0, "/home/golobs/Reconstruction")
    from sdg import get_sdg

    train_df = pd.read_csv(train_csv)
    generate = get_sdg(method)

    print(f"[{method}] Generating → {output_csv}", flush=True)
    t0 = time.time()
    synth_df = generate(train_df, meta, **config)
    elapsed = time.time() - t0
    print(f"[{method}] Done in {elapsed:.1f}s — {len(synth_df)} rows", flush=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    synth_df.to_csv(output_csv, index=False)


# ============================================================
#  STEP 1: SAMPLE
# ============================================================

def do_sample():
    """Load full dataset, clean it, create and save disjoint samples."""
    full_path = DATA_ROOT / DATASET / "full_data.csv"
    df = pd.read_csv(full_path)

    # Clean Adult-specific issues
    if DATASET == "adult":
        df["income"] = df["income"].str.strip().str.rstrip(".")
        df = df.dropna().reset_index(drop=True)

    print(f"Full dataset: {len(df)} rows after cleaning")

    # Shuffle deterministically
    rng = np.random.RandomState(RANDOM_SEED)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    needed = NUM_SAMPLES * SAMPLE_SIZE
    if needed > len(df):
        raise ValueError(f"Need {needed} rows but only have {len(df)}")

    base = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"
    for i in range(NUM_SAMPLES):
        start = i * SAMPLE_SIZE
        end = start + SAMPLE_SIZE
        sample_df = df.iloc[start:end]

        sample_dir = base / f"sample_{i:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_dir / "train.csv", index=False)
        print(f"  sample_{i:02d}/train.csv — {len(sample_df)} rows")

    print(f"\nSaved {NUM_SAMPLES} samples to {base}/")
    print(f"Meta: {META_PATH}")
    print(f"Verify, then run:  python generate_synth.py sdg")


# ============================================================
#  STEP 2: SDG
# ============================================================

def do_sdg():
    """Generate synthetic data for each sample using parallel subprocesses."""
    base = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"

    # Verify samples exist
    for i in SAMPLES_TO_GENERATE:
        train_path = base / f"sample_{i:02d}" / "train.csv"
        if not train_path.exists():
            log(f"ERROR: {train_path} not found. Run 'python generate_synth.py sample' first.")
            sys.exit(1)

    log(f"Dataset: {DATASET}")
    log(f"Generating SDG for samples: {list(SAMPLES_TO_GENERATE)}")
    log(f"SDG jobs per sample: {len(SDG_JOBS)}")
    log(f"Max parallel workers: {MAX_WORKERS}")
    log("")

    for i in SAMPLES_TO_GENERATE:
        log(f"=== Sample {i:02d} ===")
        launch_sdg_jobs(i)
        log("")

    log("All done!")


def launch_sdg_jobs(sample_idx):
    """Launch all SDG jobs for one sample in parallel, wait for completion."""
    sys.path.insert(0, "/home/golobs/Reconstruction")
    from sdg import sdg_dirname

    base = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"
    sample_dir = base / f"sample_{sample_idx:02d}"
    train_csv = str(sample_dir / "train.csv")

    # Build list of jobs that haven't run yet
    pending = []
    for method, config in SDG_JOBS:
        dirname = sdg_dirname(method, config)
        output_csv = str(sample_dir / dirname / "synth.csv")
        if os.path.exists(output_csv):
            log(f"  [SKIP] {dirname} — synth.csv already exists")
            continue
        pending.append((method, dirname, output_csv, config))

    if not pending:
        log(f"  All SDG jobs already complete for sample_{sample_idx:02d}")
        return

    # Launch subprocesses (up to MAX_WORKERS at a time)
    script = os.path.abspath(__file__)
    running = {}  # proc -> (dirname, log_path)

    def wait_for_slot():
        """Block until a running process finishes."""
        while len(running) >= MAX_WORKERS:
            for proc, (name, job_log) in list(running.items()):
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        log(f"  [FAIL] {name} (exit code {ret}) — see {job_log}")
                    else:
                        log(f"  [DONE] {name}")
                    del running[proc]
                    return
            time.sleep(1)

    for method, dirname, output_csv, config in pending:
        wait_for_slot()
        # Each job gets its own log file in its output directory
        job_log_dir = Path(output_csv).parent
        job_log_dir.mkdir(parents=True, exist_ok=True)
        job_log = job_log_dir / "sdg.log"
        job_log_fh = open(job_log, "w")

        cmd = [
            sys.executable, script, "--job",
            method, train_csv, output_csv,
            json.dumps(META), json.dumps(config),
        ]
        proc = subprocess.Popen(
            cmd, stdout=job_log_fh, stderr=subprocess.STDOUT, env=_QUIET_ENV,
        )
        running[proc] = (dirname, str(job_log))
        log(f"  [START] {dirname} (pid {proc.pid})")

    # Wait for remaining processes
    for proc, (name, job_log) in running.items():
        proc.wait()
        if proc.returncode != 0:
            log(f"  [FAIL] {name} (exit code {proc.returncode}) — see {job_log}")
        else:
            log(f"  [DONE] {name}")


# ============================================================
#  LOGGING
# ============================================================

_log_fh = None

def log(msg):
    """Print to both stdout and the log file with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if "--job" in sys.argv:
        run_single_job(sys.argv)
    elif len(sys.argv) < 2 or sys.argv[1] not in ("sample", "sdg"):
        print("Usage:")
        print("  python generate_synth.py sample   # Step 1: create disjoint training samples")
        print("  python generate_synth.py sdg      # Step 2: generate synthetic data")
        print()
        print("To run in background:")
        print("  nohup python generate_synth.py sdg &")
        base = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"
        print(f"  tail -f {base}/sdg_log.txt")
        sys.exit(1)
    elif sys.argv[1] == "sample":
        do_sample()
    elif sys.argv[1] == "sdg":
        # Open log file
        base = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"
        base.mkdir(parents=True, exist_ok=True)
        _log_fh = open(base / "sdg_log.txt", "a")
        log(f"--- Started: {datetime.now().isoformat()} ---")
        try:
            do_sdg()
        finally:
            _log_fh.close()
