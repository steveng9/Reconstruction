#!/usr/bin/env python
"""
Generate synthetic datasets for reconstruction attack experiments.

Two-step usage:
    Step 1 — Sample training data:
        python sdg/generate_synth.py sample

    Step 2 — Generate synthetic data (after verifying samples):
        python sdg/generate_synth.py sdg

    Runs in background by default, logging to sdg_log.txt in the dataset's size dir.
    Tail progress with:  tail -f /home/golobs/data/reconstruction_data/adult/size_1000/sdg_log.txt

Single-job mode (called internally by sdg step):
    python sdg/generate_synth.py --job <method> <train_csv> <output_csv> <meta_json> <config_json>
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
#DATASET = "adult"
DATASET = "nist_arizona_data"
#DATASET = "nist_sbo"
#DATASET = "cdc_diabetes"
#DATASET = "california"
#DATASET = "match2-2017"

SAMPLE_SIZE = 10_000
NUM_SAMPLES = 10       # total training samples to create

# Whether samples must be disjoint (non-overlapping).
# Set False when XxN exceeds total available data, or when samples won't be used as holdout sets.
# Non-disjoint sampling writes a NO_HOLDOUT marker in each sample dir.
DISJOINT = True

RANDOM_SEED = 42

# Feature subset — set to a list of column names to restrict BOTH sample() and sdg() to
# those columns. None = use all columns. When set, the size directory gets a "_Nfeat"
# suffix so 50-feat and 98-feat experiments live in separate directories
# (e.g. size_10000_50feat/ vs size_10000/).
#FEATURE_SUBSET = None
# nist_arizona_data 25-col subset (mirrors NIST CRC competition, F-code encoding):
FEATURE_SUBSET = [
    'AGE','AGEMARR','BPL','CITIZEN','DURUNEMP',
    'EDUC','EMPSTAT','FAMSIZE','FARM','GQ','GQTYPE',
    'HISPAN','INCWAGE','IND','LABFORCE','MARST','MIGRATE5',
    'MTONGUE','NATIVITY','OWNERSHP','RACE','SEX','URBAN',
    'VETSTAT','WKSWORK1',
]
# nist_arizona_data 50-col subset:
# FEATURE_SUBSET = [
#     'AGE','AGEMARR','BPL','CHBORN','CITIZEN',
#     'CITY','CLASSWKR','COUNTY','DURUNEMP',
#     'EDUC','EMPSTAT','FAMSIZE','FARM','FBPL',
#     'GQ','GQFUNDS','GQTYPE','HISPAN','HRSWORK1',
#     'INCNONWG','INCWAGE','IND','LABFORCE',
#     'MARRNO','MARST','MBPL','METAREA','METRO',
#     'MIGCITY5','MIGRATE5','MIGTYPE5','MTONGUE',
#     'NATIVITY','NCHLT5','OCC','OWNERSHP','RACE',
#     'RENT','SAMEPLAC','SCHOOL','SEX','SSENROLL',
#     'URBAN','VALUEH','VETCHILD','VETPER',
#     'VETSTAT','VETWWI','WARD','WKSWORK1',
# ]

# Which samples to generate SDG for (0-indexed).
SAMPLES_TO_GENERATE = range(5)
#SAMPLES_TO_GENERATE = range(5, 10)
#SAMPLES_TO_GENERATE = [4]

# Max parallel SDG jobs per sample.
# GPU methods (TVAE, CTGAN, ARF, TabDDPM) share GPU memory — tune accordingly.
MAX_WORKERS = 8

# Column metadata — loaded from meta.json next to full_data.csv
META_PATH = DATA_ROOT / DATASET / "meta.json"
with open(META_PATH) as f:
    META = json.load(f)


# ============================================================
#  SDG JOBS — per-dataset configs
#
#  Each dataset entry is a list of (method, config_kwargs) tuples.
#  Directory names are derived automatically via sdg_dirname(method, config).
#  Changing DATASET above auto-selects the right job list.
# ============================================================

_SDG_JOBS_BY_DATASET = {

    "adult": [
        # Deep generative models
        ("TabDDPM", {}),
        ("TVAE",    {}),
        ("CTGAN",   {}),
        ("ARF",     {}),
        # Differentially private — vary epsilon
        ("MST",  {"epsilon": 0.1}),
        ("MST",  {"epsilon": 1.0}),
        ("MST",  {"epsilon": 10.0}),
        ("MST",  {"epsilon": 100.0}),
        ("MST",  {"epsilon": 1000.0}),
        ("AIM",  {"epsilon": 1.0}),
        ("AIM",  {"epsilon": 10.0}),
        # R-based / de-identification
        ("Synthpop",        {}),
        ("RankSwap",        {"swap_features": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]}),
        ("CellSuppression", {"key_vars": ["age", "workclass", "education", "sex", "race", "native-country"]}),
    ],

    "cdc_diabetes": [
        ("MST",  {"epsilon": 0.1}),
        ("MST",  {"epsilon": 1.0}),
        ("MST",  {"epsilon": 10.0}),
        ("MST",  {"epsilon": 100.0}),
        ("MST",  {"epsilon": 1000.0}),
        ("AIM",  {"epsilon": 1.0}),
        ("AIM",  {"epsilon": 10.0}),
        ("TVAE",    {}),
        ("CTGAN",   {}),
        ("ARF",     {}),
        ("TabDDPM", {}),
        ("Synthpop",        {}),
        ("RankSwap",        {"swap_features": ["BMI", "MentHlth", "PhysHlth"]}),
        ("CellSuppression", {"key_vars": ["Sex", "Age", "Income", "Education"]}),
    ],

    "california": [
        ("MST",  {"epsilon": 1.0}),
        ("MST",  {"epsilon": 10.0}),
        ("MST",  {"epsilon": 100.0}),
        ("MST",  {"epsilon": 1000.0}),
        ("AIM",  {"epsilon": 1.0}),
        ("AIM",  {"epsilon": 10.0}),
        ("TVAE",    {}),
        ("CTGAN",   {}),
        ("ARF",     {}),
        ("TabDDPM", {}),
        ("Synthpop",        {}),
        # All 9 columns are continuous — swap all
        ("RankSwap", {"swap_features": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]}),
        # CellSuppression requires categorical QI columns — omit for purely continuous datasets
    ],

    "nist_sbo": [
        # Deep generative models
        # TabDDPM: wider network for 130-column dataset (one-hot expands to 500+ features).
        # Bump middle layers to 2048 and train longer.
        ("TabDDPM", {
            "d_layers": [1024, 2048, 2048, 2048, 1024],
            "iterations": 300000,
        }),
        ("TVAE",    {}),
        ("CTGAN",   {}),
        ("ARF",     {}),
        # Differentially private
        # NOTE: AIM is computationally infeasible for 130 columns — omitted.
        # bin_continuous_as_ordinal=True: pre-bins the 5 continuous columns before fitting,
        # bypassing SmartNoise's private BinTransformer bound estimation. With small epsilon
        # (e.g. 0.1), preprocessor_eps=0.03 is too small for approx_bounds to work on
        # skewed financial data (range 0–140k). MST bins continuous data internally anyway —
        # pre-binning with actual data bounds is equivalent and gives the full epsilon to synthesis.
        ("MST",  {"epsilon": 0.1,    "bin_continuous_as_ordinal": True}),
        ("MST",  {"epsilon": 1.0,    "bin_continuous_as_ordinal": True}),
        ("MST",  {"epsilon": 10.0,   "bin_continuous_as_ordinal": True}),
        ("MST",  {"epsilon": 100.0,  "bin_continuous_as_ordinal": True}),
        ("MST",  {"epsilon": 1000.0, "bin_continuous_as_ordinal": True}),
        # R-based / de-identification
        ("Synthpop",        {}),
        # 5 continuous columns
        ("RankSwap",        {"swap_features": ["TABWGT", "EMPLOYMENT_NOISY", "PAYROLL_NOISY", "RECEIPTS_NOISY", "PCT1"]}),
        # Geographic + industry + owner demographics as QI
        ("CellSuppression", {"key_vars": ["FIPST", "SECTOR", "ETH1", "SEX1", "AGE1"]}),
    ],

    "nist_arizona_data": [
        # Deep generative models
        #("TabDDPM", {}),
        #("TVAE",    {}),
        #("CTGAN",   {}),
        #("ARF",     {}),
        # Differentially private — pre-bin continuous cols to avoid BinTransformer failures
        # at small epsilon (INCWAGE range 0–999998 and VALUEH range 1–9999999 cause
        # approx_bounds to return None with preprocessor_eps < ~0.1)
        #("MST", {"epsilon": 0.1,   "bin_continuous_as_ordinal": True}),
        #("MST", {"epsilon": 1.0,   "bin_continuous_as_ordinal": True}),
        #("MST", {"epsilon": 10.0,  "bin_continuous_as_ordinal": True}),
        #("MST", {"epsilon": 100.0, "bin_continuous_as_ordinal": True}),
        #("MST", {"epsilon": 1000.0, "bin_continuous_as_ordinal": True}),
        #("AIM", {"epsilon": 1.0}),
        #("AIM", {"epsilon": 10.0}),
        #("AIM", {"epsilon": 100.0}),
        # R-based / de-identification
        ("Synthpop", {}),
        #("RankSwap",        {"swap_features": ["AGE", "AGEMARR", "FAMSIZE", "INCWAGE"]}),
        #("CellSuppression", {"key_vars": ["RACE", "SEX", "AGE"], "k": 6}),
    ],

    "match2-2017": [
        ("MST",  {"epsilon": 1.0}),
        ("MST",  {"epsilon": 10.0}),
        ("TVAE",    {}),
        ("CTGAN",   {}),
        ("ARF",     {}),
        ("TabDDPM", {}),
        ("Synthpop",        {}),
        ("RankSwap",        {"swap_features": ["Location - Lng", "Location - Lat", "Number of Alarms", "Unit sequence in call dispatch"]}),
        ("CellSuppression", {"key_vars": ["Zipcode of Incident", "Station Area", "Battalion", "Supervisor District"]}),
    ],

}

if DATASET not in _SDG_JOBS_BY_DATASET:
    raise ValueError(
        f"No SDG job config found for dataset '{DATASET}'. "
        f"Add an entry to _SDG_JOBS_BY_DATASET in generate_synth.py."
    )

SDG_JOBS = _SDG_JOBS_BY_DATASET[DATASET]


def _base_dir() -> Path:
    """Size directory, with a '_Nfeat' suffix when FEATURE_SUBSET is active."""
    if FEATURE_SUBSET is not None:
        return DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}_{len(FEATURE_SUBSET)}feat"
    return DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}"


def _effective_meta() -> dict:
    """Return META filtered to FEATURE_SUBSET columns (or the full META if unset)."""
    if FEATURE_SUBSET is None:
        return META
    subset = set(FEATURE_SUBSET)
    return {
        "categorical": [c for c in META.get("categorical", []) if c in subset],
        "continuous":  [c for c in META.get("continuous",  []) if c in subset],
        "ordinal":     [c for c in META.get("ordinal",     []) if c in subset],
    }


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
    """Load full dataset, clean it, create and save training samples.

    When DISJOINT=True (default), samples are non-overlapping slices.
    When DISJOINT=False, each sample is drawn independently (may overlap);
    a NO_HOLDOUT marker is written to each sample dir.
    """
    full_path = DATA_ROOT / DATASET / "full_data.csv"
    df = pd.read_csv(full_path)

    # Dataset-specific cleaning
    if DATASET == "adult":
        df["income"] = df["income"].str.strip().str.rstrip(".")
        df = df.dropna().reset_index(drop=True)

    if DATASET == "nist_sbo":
        # 23% of rows are partial respondents with 116/130 columns blank (structural skip pattern).
        # They're a fundamentally different kind of record — drop them for clean SDG input.
        # Leaves ~123,892 fully-complete rows out of 161,079.
        df = df.dropna().reset_index(drop=True)

    if FEATURE_SUBSET is not None:
        missing = set(FEATURE_SUBSET) - set(df.columns)
        if missing:
            raise ValueError(f"FEATURE_SUBSET columns not in data: {sorted(missing)}")
        n_total = len(df.columns)
        df = df[list(FEATURE_SUBSET)]
        print(f"Feature subset active: using {len(FEATURE_SUBSET)} of {n_total} columns")

    print(f"Full dataset: {len(df)} rows, {len(df.columns)} columns after cleaning")

    base = _base_dir()

    if DISJOINT:
        # Non-overlapping sequential slices from a shuffled dataset
        needed = NUM_SAMPLES * SAMPLE_SIZE
        if needed > len(df):
            raise ValueError(
                f"DISJOINT=True requires {needed} rows ({NUM_SAMPLES} x {SAMPLE_SIZE}) "
                f"but only {len(df)} available. Use DISJOINT=False for overlapping samples."
            )
        rng = np.random.RandomState(RANDOM_SEED)
        df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

        for i in range(NUM_SAMPLES):
            start = i * SAMPLE_SIZE
            end = start + SAMPLE_SIZE
            sample_df = df.iloc[start:end]

            sample_dir = base / f"sample_{i:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(sample_dir / "train.csv", index=False)
            print(f"  sample_{i:02d}/train.csv — {len(sample_df)} rows  [disjoint]")

    else:
        # Independent random draws — samples may overlap each other
        if SAMPLE_SIZE > len(df):
            raise ValueError(
                f"SAMPLE_SIZE={SAMPLE_SIZE} exceeds available rows ({len(df)})."
            )

        for i in range(NUM_SAMPLES):
            sample_df = df.sample(n=SAMPLE_SIZE, replace=False, random_state=RANDOM_SEED + i)
            sample_df = sample_df.reset_index(drop=True)

            sample_dir = base / f"sample_{i:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(sample_dir / "train.csv", index=False)

            # Mark as non-holdout-eligible — these samples may overlap each other
            # and therefore cannot be used as membership inference holdout sets.
            (sample_dir / "NO_HOLDOUT").touch()
            print(f"  sample_{i:02d}/train.csv — {len(sample_df)} rows  [non-disjoint, NO_HOLDOUT marked]")

    print(f"\nSaved {NUM_SAMPLES} samples to {base}/")
    print(f"Meta: {META_PATH}" + (f"  (filtered to {len(FEATURE_SUBSET)} cols)" if FEATURE_SUBSET else ""))
    print(f"Verify, then run:  python generate_synth.py sdg")


# ============================================================
#  STEP 2: SDG
# ============================================================

def do_sdg():
    """Generate synthetic data for each sample using parallel subprocesses."""
    base = _base_dir()

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

    base = _base_dir()
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
            json.dumps(_effective_meta()), json.dumps(config),
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
#  COUNT
# ============================================================

def do_count(args):
    """Count generated synth.csv files, optionally filtered by dataset name."""
    datasets = args if args else sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())

    total = 0
    for ds in datasets:
        ds_dir = DATA_ROOT / ds
        if not ds_dir.is_dir():
            print(f"[SKIP] {ds}: not found")
            continue
        meta_path = ds_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                ds_meta = json.load(f)
            n_cat  = len(ds_meta.get("categorical", []))
            n_cont = len(ds_meta.get("continuous", []))
            n_ord  = len(ds_meta.get("ordinal", []))
            total_cols = n_cat + n_cont + n_ord
            parts = [f"{n_cat} categorical", f"{n_cont} continuous"]
            if n_ord:
                parts.append(f"{n_ord} ordinal")
            col_info = f"  [{total_cols} cols: {', '.join(parts)}]"
        else:
            col_info = "  [no meta.json]"
        full_data_path = ds_dir / "full_data.csv"
        if full_data_path.exists():
            n_rows = sum(1 for _ in open(full_data_path)) - 1  # subtract header
            row_info = f"  [{n_rows:,} rows in full_data.csv]"
        else:
            row_info = "  [no full_data.csv]"
        print(f"\n{ds}/{col_info}{row_info}")
        ds_total = 0
        for size_dir in sorted(ds_dir.glob("size_*")):
            samples = sorted(size_dir.glob("sample_*"))
            methods = {}
            for sample_dir in samples:
                for synth in sample_dir.glob("*/synth.csv"):
                    method = synth.parent.name
                    methods[method] = methods.get(method, 0) + 1
            if not methods:
                continue
            n = sum(methods.values())
            ds_total += n
            print(f"  {size_dir.name}: {n} files across {len(samples)} samples")
            for m in sorted(methods):
                print(f"    {m}: {methods[m]}")
        print(f"  total: {ds_total}")
        total += ds_total

    print(f"\nGrand total: {total} synth.csv files")


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
    elif len(sys.argv) >= 2 and sys.argv[1] == "count":
        do_count(sys.argv[2:])
    elif len(sys.argv) < 2 or sys.argv[1] not in ("sample", "sdg"):
        print("Usage:")
        print("  python sdg/generate_synth.py sample          # Step 1: create training samples")
        print("  python sdg/generate_synth.py sdg             # Step 2: generate synthetic data")
        print("  python sdg/generate_synth.py count [dataset]  # Count generated synth.csv files")
        print()
        print("To run in background:")
        print("  nohup python sdg/generate_synth.py sdg &")
        base = _base_dir()
        print(f"  tail -f {base}/sdg_log.txt")
        sys.exit(1)
    elif sys.argv[1] == "sample":
        do_sample()
    elif sys.argv[1] == "sdg":
        # Open log file
        base = _base_dir()
        base.mkdir(parents=True, exist_ok=True)
        _log_fh = open(base / "sdg_log.txt", "a")
        log(f"--- Started: {datetime.now().isoformat()} ---")
        try:
            do_sdg()
        finally:
            _log_fh.close()
