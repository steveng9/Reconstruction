#!/usr/bin/env python
"""
audit_and_verify.py — Phase 1 (CSV audit) + Phase 2 (WandB verification).

Run this BEFORE migrate_to_db.py.  It produces four output files in
experiment_scripts/outfiles/:

  verification_report.csv      All RA-relevant CSV rows, each labelled
                               'certain' or 'uncertain' with a reason.
  wandb_only.csv               WandB runs that have NO matching local CSV row.
  uncertain_rows.csv           Just the uncertain rows (convenience subset).
  rerun_queue.txt              Distinct experiment configs that need re-running
                               because their CSV rows are uncertain.
  wandb_lookup_cache.pkl       Cached WandB lookup dict (auto-saved after a
                               successful live fetch; used by --use-wandb-cache).

Nothing is written to results.db here — that is migrate_to_db.py's job.

Usage (first run — fetches WandB live and saves cache, ~3 hours):
  conda activate recon_
  mkdir -p experiment_scripts/outfiles
  nohup /home/golobs/miniconda3/envs/recon_/bin/python experiment_scripts/audit_and_verify.py \\
      > experiment_scripts/outfiles/audit.log 2>&1 &
  echo $! > experiment_scripts/outfiles/audit.pid
  tail -f experiment_scripts/outfiles/audit.log

Usage (subsequent runs — load WandB from cache, ~3 minutes):
  nohup /home/golobs/miniconda3/envs/recon_/bin/python experiment_scripts/audit_and_verify.py \\
      --use-wandb-cache \\
      > experiment_scripts/outfiles/audit.log 2>&1 &
  echo $! > experiment_scripts/outfiles/audit.pid
  tail -f experiment_scripts/outfiles/audit.log

Options:
  --use-wandb-cache   Load WandB lookup from outfiles/wandb_lookup_cache.pkl instead
                      of hitting the API.  Use this for all re-runs after the first
                      successful live fetch.  Reduces runtime from ~3 hours to ~3 min.
  --no-save-cache     Skip writing the cache after a live fetch.
  --no-wandb          Skip WandB entirely (rows without all dims → uncertain).
  --workers N         Threads for parallel CSV reading / matching (default 8).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent
CSV_DIR       = SCRIPT_DIR           # CSVs live next to this script
OUTFILES_DIR  = SCRIPT_DIR / "outfiles"
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_CACHE   = OUTFILES_DIR / "wandb_lookup_cache.pkl"

# Tolerance for "scores match" when comparing CSV ra_mean vs WandB RA_mean
SCORE_TOL = 0.05   # generous — rounding differs between run_job and WandB summary

# ---------------------------------------------------------------------------
# Logging helpers (always flush so `tail -f` shows live progress)
# ---------------------------------------------------------------------------

_T0 = time.time()

def log(msg: str, *, indent: int = 0) -> None:
    elapsed = time.time() - _T0
    prefix  = f"[{elapsed:7.1f}s] {'  ' * indent}"
    print(f"{prefix}{msg}", flush=True)

def log_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{bar}", flush=True)


# ---------------------------------------------------------------------------
# CSV family definitions
# ---------------------------------------------------------------------------
#
# Each entry: (filename_regex, family_name, extra_dims_from_match_fn)
# extra_dims_from_match_fn(re.Match) → dict with any of: dataset, dataset_size
# Returns None if the file should be SKIPPED (not an RA-result file).

# Dataset name aliases: CSV filenames use short names; WandB config uses full names.
_DATASET_ALIASES: dict[str, str] = {
    "arizona": "nist_arizona_25feat",
}

def _skip(_m):
    return None

def _dataset_from_group(group_name):
    """Return a function that reads dataset from a named regex group."""
    def fn(m):
        raw = m.group(group_name)
        # normalise: "nist_arizona_25feat" stays; "cdc_diabetes" stays; etc.
        return {"dataset": raw}
    return fn

CSV_FAMILY_PATTERNS = [
    # Old production sweep results (no dataset/size columns in CSV)
    (
        r"sweep_results_(?P<dataset>adult|nist_sbo|nist_arizona_25feat|"
        r"nist_arizona_50feat|nist_arizona_data|cdc_diabetes|california)_\d{8}_\d{6}\.csv",
        "old_sweep",
        _dataset_from_group("dataset"),
    ),
    # Early CDC sweep (has size column, no dataset column)
    (
        r"cdc_sweep_results_\d{8}_\d{6}\.csv",
        "cdc_sweep",
        lambda _m: {"dataset": "cdc_diabetes"},
    ),
    # New-attacks results — early batch (has dataset col but NO size col)
    (
        r"new_attacks_results_\d{8}_\d{6}\.csv",
        "new_attacks",
        lambda _m: {},   # dataset already in CSV; size may be missing
    ),
    # New-attacks fill-in results (has dataset+size — added by previous data-fix)
    (
        r"new_attacks_fill_in_results_\d{8}_\d{6}\.csv",
        "new_attacks_fill_in",
        lambda _m: {},
    ),
    # fill_in_results_adult (no dataset/size columns)
    (
        r"fill_in_results_(?P<dataset>adult|nist_sbo|cdc_diabetes|california|"
        r"nist_arizona_25feat)_\d{8}_\d{6}\.csv",
        "fill_in",
        _dataset_from_group("dataset"),
    ),
    # col_correction and tabpfn results (has dataset+size)
    (
        r"col_correction_results_\d{8}_\d{6}\.csv",
        "col_correction",
        lambda _m: {},
    ),
    (
        r"tabpfn_(?P<dataset>[a-z_]+)_results\.csv",
        "tabpfn",
        _dataset_from_group("dataset"),
    ),
    # Linear sweep (has dataset+size, memorisation splits)
    # NOTE: run_linear_sweep.py uses --dataset arizona but WandB logs "nist_arizona_25feat"
    (
        r"linear_sweep_(?P<dataset>[a-z0-9_]+)_(?P<size>\d+)_\d{8}_\d{6}\.csv",
        "linear_sweep",
        lambda m: {
            "dataset": _DATASET_ALIASES.get(m.group("dataset"), m.group("dataset")),
            "dataset_size": int(m.group("size")),
        },
    ),
    # linear_extended_cdc: QI_joint_Diabetes_HighBP_Stroke_lowcard is hardcoded
    # in run_linear_extended_cdc.py and never appears as a CSV column.
    (
        r"linear_extended_cdc_(?P<size>\d+)_\d{8}_\d{6}\.csv",
        "linear_extended_cdc",
        lambda m: {
            "dataset":      "cdc_diabetes",
            "dataset_size": int(m.group("size")),
            "qi":           "QI_joint_Diabetes_HighBP_Stroke_lowcard",
        },
    ),
    (
        r"linear_extended_(?P<dataset>[a-z0-9_]+)_(?P<size>\d+)_\d{8}_\d{6}\.csv",
        "linear_extended",
        lambda m: {"dataset": m.group("dataset"), "dataset_size": int(m.group("size"))},
    ),
    # Partial MST comparison (no dataset/size/qi — skip for main DB; too ambiguous)
    (r"partial_mst_comparison_\d{8}_\d{6}\.csv",          "partial_mst_comparison", _skip),
    # Ensembling heatmap (different schema — not individual RA; skip for now)
    (r"ensembling_heatmap_results_\d{8}_\d{6}\.csv",      "ensembling_heatmap",     _skip),
    # Quality / other (not RA results — skip)
    (r"(synth_quality|fill_in_quality|quality_results_merged|wasserstein_ohe"
     r"|linear_sweep_cache).*\.csv",                       "non_ra",                 _skip),
]

_COMPILED = [
    (re.compile(pat, re.IGNORECASE), family, fn)
    for pat, family, fn in CSV_FAMILY_PATTERNS
]


def classify_csv(path: Path) -> tuple[str, dict]:
    """Return (family_name, extra_dims) for a CSV file, or ('unknown', {})."""
    name = path.name
    for pattern, family, fn in _COMPILED:
        m = pattern.fullmatch(name)
        if m:
            extra = fn(m)
            return family, extra  # extra=None means skip
    return "unknown", {}


# ---------------------------------------------------------------------------
# SDG label helpers (CSV ↔ WandB)
# ---------------------------------------------------------------------------

def sdg_label(method: str, params: dict | None) -> str:
    """Construct the display label used in CSVs and tables."""
    if not params:
        return method
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{float(eps):g}"
    return method


def parse_sdg_label(label: str) -> tuple[str, dict]:
    """Reverse of sdg_label: 'MST_eps1' → ('MST', {'epsilon': 1.0})."""
    m = re.match(r"^(.+)_eps([\d.]+)$", label)
    if m:
        return m.group(1), {"epsilon": float(m.group(2))}
    return label, {}


# ---------------------------------------------------------------------------
# Phase 1: CSV discovery and reading
# ---------------------------------------------------------------------------

def _read_one_csv(path: Path, family: str, extra_dims: dict) -> pd.DataFrame | None:
    """Read one CSV and normalise to a standard intermediate format."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"  WARNING: could not read {path.name}: {e}", indent=1)
        return None

    if df.empty:
        return None

    # ── Apply extra dimensions from filename ─────────────────────────────
    for k, v in extra_dims.items():
        if k not in df.columns or df[k].isna().all():
            df[k] = v

    # ── Normalise column names ────────────────────────────────────────────
    # 'label' takes priority over 'attack' as the display label
    if "label" in df.columns:
        mask = df["label"].notna() & (df["label"].astype(str).str.strip() != "")
        df.loc[mask, "attack"] = df.loc[mask, "label"]
        df = df.drop(columns=["label"])

    # Rename 'size' → 'dataset_size' if present
    if "size" in df.columns and "dataset_size" not in df.columns:
        df = df.rename(columns={"size": "dataset_size"})

    # Rename 'sample_idx' → 'sample' if needed
    if "sample_idx" in df.columns and "sample" not in df.columns:
        df = df.rename(columns={"sample_idx": "sample"})

    # Detect split for each row
    # ra_mean != NaN and train_mean == NaN → 'standard'
    # train_mean != NaN                    → 'train' (+ separate 'nontraining')
    rows_out = []

    has_ra    = "ra_mean"      in df.columns
    has_train = "train_mean"   in df.columns
    has_nt    = "nontrain_mean" in df.columns

    for _, row in df.iterrows():
        base = {
            "source_file":   path.name,
            "family":        family,
            "dataset":       row.get("dataset"),
            "dataset_size":  row.get("dataset_size"),
            "sample":        row.get("sample"),
            "qi":            row.get("qi"),
            "sdg_method":    row.get("sdg"),
            "attack_label":  row.get("attack"),
        }
        feat_scores = {
            k[3:]: float(v)
            for k, v in row.items()
            if k.startswith("RA_") and pd.notna(v) and k not in ("RA_mean",)
        }
        base["feature_scores_json"] = json.dumps(feat_scores) if feat_scores else None

        ra = row.get("ra_mean") if has_ra else None
        train_ra = row.get("train_mean")  if has_train else None
        nt_ra    = row.get("nontrain_mean") if has_nt   else None

        if has_ra and pd.notna(ra):
            rows_out.append({**base, "split": "standard", "ra_mean": float(ra)})
        if has_train and pd.notna(train_ra):
            rows_out.append({**base, "split": "train",        "ra_mean": float(train_ra),
                             "feature_scores_json": None})
        if has_nt and pd.notna(nt_ra):
            rows_out.append({**base, "split": "nontraining",  "ra_mean": float(nt_ra),
                             "feature_scores_json": None})

    if not rows_out:
        return None

    return pd.DataFrame(rows_out)


def phase1_read_csvs(workers: int) -> pd.DataFrame:
    log_section("PHASE 1 — CSV Discovery & Reading")

    all_csvs = sorted(CSV_DIR.glob("*.csv"))
    log(f"Found {len(all_csvs)} CSV files in {CSV_DIR}")

    # Classify
    tasks = []
    skipped = []
    for path in all_csvs:
        family, extra = classify_csv(path)
        if extra is None:         # _skip function returned None
            skipped.append(path.name)
        elif family == "unknown":
            log(f"  UNRECOGNISED: {path.name} — will skip", indent=1)
            skipped.append(path.name)
        else:
            tasks.append((path, family, extra))

    log(f"  Relevant RA CSVs : {len(tasks)}")
    log(f"  Skipped (non-RA) : {len(skipped)}")

    if skipped:
        log("  Skipped files:", indent=1)
        for fn in skipped:
            log(fn, indent=2)

    # Read in parallel
    frames: list[pd.DataFrame] = []
    done  = 0
    errors = 0

    log(f"\nReading {len(tasks)} CSVs with {workers} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_read_one_csv, path, family, extra): path
                   for path, family, extra in tasks}
        for fut in concurrent.futures.as_completed(futures):
            path = futures[fut]
            done += 1
            try:
                result = fut.result()
                if result is not None and not result.empty:
                    frames.append(result)
                    log(f"  [{done:3d}/{len(tasks)}] {path.name} → {len(result)} rows")
                else:
                    log(f"  [{done:3d}/{len(tasks)}] {path.name} → 0 rows (empty/skipped)")
            except Exception as e:
                errors += 1
                log(f"  [{done:3d}/{len(tasks)}] {path.name} → ERROR: {e}")

    if not frames:
        log("No RA rows found in any CSV. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    log(f"\nPhase 1 complete: {len(combined):,} total rows from {len(frames)} CSVs "
        f"({errors} errors)")
    return combined


# ---------------------------------------------------------------------------
# Phase 2: WandB bulk fetch
# ---------------------------------------------------------------------------

def phase2_fetch_wandb(use_cache: bool = False, save_cache: bool = True) -> dict:
    """Fetch all WandB runs and return a lookup dict.

    Parameters
    ----------
    use_cache : bool
        If True and WANDB_CACHE exists, load the lookup from disk instead of
        hitting the WandB API.  Pass --use-wandb-cache on the CLI.
    save_cache : bool
        If True, save the lookup to WANDB_CACHE after a successful live fetch.

    Returns
    -------
    dict
        {(dataset, dataset_size, sample, qi, sdg_label, attack_method): [run_dict, …]}
    """
    log_section("PHASE 2 — WandB Bulk Fetch")

    # ── Cache load path ────────────────────────────────────────────────────────
    if use_cache:
        if WANDB_CACHE.exists():
            log(f"Loading WandB lookup from cache: {WANDB_CACHE}")
            with open(WANDB_CACHE, "rb") as fh:
                lookup = pickle.load(fh)
            log(f"  Loaded {sum(len(v) for v in lookup.values()):,} indexed runs "
                f"({len(lookup):,} unique keys) from cache.")
            return lookup
        else:
            log(f"Cache file not found ({WANDB_CACHE}); falling back to live WandB fetch.")

    try:
        import wandb
    except ImportError:
        log("wandb not installed — skipping WandB phase.  All rows will be 'uncertain'.")
        return {}

    log(f"Connecting to WandB project '{WANDB_PROJECT}'...")
    api = wandb.Api(timeout=90)

    log("Fetching all runs (per_page=500 — may take several minutes)...")
    lookup: dict[tuple, list[dict]] = {}
    skipped_no_config = 0
    total_fetched = 0
    errors = 0

    try:
        for i, run in enumerate(api.runs(WANDB_PROJECT, per_page=500), 1):
            total_fetched = i
            if i % 100 == 0:
                log(f"  Fetched {i} runs so far "
                    f"(indexed: {sum(len(v) for v in lookup.values()):,}, "
                    f"errors: {errors})...", indent=1)
            try:
                cfg = dict(run.config) if run.config else {}
                if not cfg:
                    skipped_no_config += 1
                    continue

                dataset     = cfg.get("dataset")
                size        = cfg.get("size")
                _idx        = cfg.get("sample_idx")   # 0 is valid — don't use `or`
                sample      = _idx if _idx is not None else cfg.get("sample")
                qi          = cfg.get("qi") or cfg.get("QI")
                method      = cfg.get("sdg_method")
                sdg_params  = cfg.get("sdg_params") or {}
                attack      = cfg.get("attack_method")

                if None in (dataset, sample, qi, method, attack):
                    skipped_no_config += 1
                    continue

                if isinstance(sdg_params, str):
                    try:
                        sdg_params = json.loads(sdg_params)
                    except Exception:
                        sdg_params = {}

                label = sdg_label(method, sdg_params)

                # Collect RA scores from summary
                summary = dict(run.summary) if run.summary else {}
                ra_mean_wb = summary.get("RA_mean")
                feat_scores_wb = {
                    k[3:]: v for k, v in summary.items()
                    if k.startswith("RA_") and k != "RA_mean" and not k.startswith("RA_train")
                    and not k.startswith("RA_nontraining") and not k.startswith("RA_delta")
                    and isinstance(v, (int, float))
                }
                train_mean_wb = summary.get("RA_train_mean")
                nt_mean_wb    = summary.get("RA_nontraining_mean")

                run_dict = {
                    "run_id":        run.id,
                    "run_name":      run.name,
                    "group":         getattr(run, "group", None),
                    "project":       WANDB_PROJECT,
                    "dataset":       dataset,
                    "dataset_size":  int(size) if size is not None else None,
                    "sample":        int(sample),
                    "qi":            qi,
                    "sdg_method":    label,
                    "attack_method": attack,
                    "attack_params": cfg.get("attack_params"),
                    "ra_mean":       float(ra_mean_wb) if ra_mean_wb is not None else None,
                    "train_mean":    float(train_mean_wb) if train_mean_wb is not None else None,
                    "nt_mean":       float(nt_mean_wb)    if nt_mean_wb    is not None else None,
                    "feat_scores":   feat_scores_wb,
                }

                # Key includes size (may be None for old runs without size in config)
                key = (dataset, int(size) if size is not None else None,
                       int(sample), qi, label, attack)
                lookup.setdefault(key, []).append(run_dict)

            except Exception as e:
                errors += 1
                log(f"  WARNING: error processing run {getattr(run, 'id', '?')}: {e}", indent=1)

    except Exception as e:
        log(f"ERROR fetching WandB runs: {e}")
        log("Continuing without WandB — all rows will be 'uncertain'.")
        return {}

    n_indexed = sum(len(v) for v in lookup.values())
    log(f"  Total runs fetched: {total_fetched:,}")
    log(f"  Runs indexed : {n_indexed:,}")
    log(f"  Unique keys  : {len(lookup):,}")
    log(f"  Skipped (incomplete config): {skipped_no_config:,}")
    log(f"  Errors       : {errors:,}")

    # ── Cache save path ────────────────────────────────────────────────────────
    if save_cache and n_indexed > 0:
        OUTFILES_DIR.mkdir(parents=True, exist_ok=True)
        with open(WANDB_CACHE, "wb") as fh:
            pickle.dump(lookup, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"  WandB lookup saved to cache: {WANDB_CACHE}")
    elif save_cache and n_indexed == 0:
        log("  WARNING: 0 runs indexed — NOT saving cache (likely API error).")

    return lookup


# ---------------------------------------------------------------------------
# Phase 3: Matching CSV rows → WandB
# ---------------------------------------------------------------------------

# WandB logs the BASE attack name (e.g. "MarginalRF") for all variant attacks.
# CSV files record the full variant label (e.g. "MarginalRF_mst_local_100").
# _base_attack() maps variant labels back to their WandB base name for fallback lookup.
_VARIANT_BASE_NAMES = ["MarginalRF"]

def _base_attack(label: str) -> str:
    for base in _VARIANT_BASE_NAMES:
        if label.startswith(base + "_"):
            return base
    return label


def _required_dims(row: pd.Series) -> list[str]:
    """Return list of dimension names that are missing or NaN in this row."""
    missing = []
    for dim in ("dataset", "dataset_size", "sample", "qi", "sdg_method", "attack_label"):
        v = row.get(dim)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            missing.append(dim)
    return missing


def _wb_ra_for_split(run_dict: dict, split: str) -> float | None:
    """Return the WandB RA score appropriate for the given split."""
    if split == "train":
        return run_dict.get("train_mean")
    elif split == "nontraining":
        return run_dict.get("nt_mean")
    else:
        return run_dict.get("ra_mean")


def _disambiguate(matches: list[dict], csv_ra: float | None, split: str) -> list[dict]:
    """Try to reduce multiple WandB matches to one using score proximity.

    Strategy: pick the match whose split-appropriate RA is closest to csv_ra,
    provided it is within SCORE_TOL AND is clearly better than the second-best
    (margin > 0.001 absolute).  Returns the original list unchanged if
    disambiguation fails.
    """
    if csv_ra is None or np.isnan(float(csv_ra)):
        return matches  # no score to compare

    scored = []
    for r in matches:
        wb_ra = _wb_ra_for_split(r, split)
        if wb_ra is None:
            # Fall back to standard RA if split-specific is absent
            wb_ra = r.get("ra_mean")
        if wb_ra is not None:
            scored.append((abs(float(wb_ra) - float(csv_ra)), r))

    if not scored:
        return matches  # no WandB scores available

    scored.sort(key=lambda x: x[0])
    best_diff, best_match = scored[0]

    if best_diff >= SCORE_TOL:
        return matches  # best candidate is outside tolerance — can't confirm

    if len(scored) == 1:
        return [best_match]

    second_diff = scored[1][0]
    if second_diff - best_diff > 0.001:
        return [best_match]  # clear winner

    return matches  # too close to call


def _lookup_matches(lookup: dict, dataset: str, dataset_size: int,
                    sample: int, qi: str, sdg: str, attack: str) -> list[dict]:
    """Return WandB candidates for the given key, with size-relaxed fallback."""
    key = (dataset, dataset_size, sample, qi, sdg, attack)
    matches = lookup.get(key, [])

    if not matches:
        key_no_size = (dataset, None, sample, qi, sdg, attack)
        candidates = lookup.get(key_no_size, [])
        if candidates:
            sizes = {r["dataset_size"] for r in candidates}
            if sizes <= {None, dataset_size}:
                matches = candidates

    return matches


def _match_row(row: pd.Series, lookup: dict) -> tuple[str, str, dict | None]:
    """Try to match one CSV row against the WandB lookup.

    Returns
    -------
    (confidence, notes, matched_run_dict | None)
    confidence: 'certain' | 'uncertain'
    """
    missing = _required_dims(row)

    if missing:
        return "uncertain", f"dimensions missing from CSV/filename: {missing}", None

    dataset      = str(row["dataset"])
    dataset_size = int(row["dataset_size"])
    sample       = int(row["sample"])
    qi           = str(row["qi"])
    sdg          = str(row["sdg_method"])
    attack       = str(row["attack_label"])
    split        = str(row.get("split", "standard"))
    csv_ra       = row.get("ra_mean")

    if not lookup:
        return "uncertain", "WandB verification skipped (--no-wandb)", None

    # ── Step 1: exact attack label lookup ──────────────────────────────────
    matches = _lookup_matches(lookup, dataset, dataset_size, sample, qi, sdg, attack)

    # ── Step 2: base-name fallback for variant attacks ─────────────────────
    # WandB logs attack_method = "MarginalRF" for all MarginalRF_* variants.
    # If exact lookup failed, retry with the base name and score-disambiguate.
    base_attack = _base_attack(attack)
    used_base_fallback = False
    if not matches and base_attack != attack:
        matches = _lookup_matches(lookup, dataset, dataset_size, sample, qi, sdg, base_attack)
        if matches:
            used_base_fallback = True

    if not matches:
        return "uncertain", "no matching WandB run found", None

    # ── Step 3: disambiguate if multiple candidates ─────────────────────────
    if len(matches) > 1:
        resolved = _disambiguate(matches, csv_ra, split)
        if len(resolved) == 1:
            matches = resolved
        else:
            run_ids = [r["run_id"] for r in matches]
            return (
                "uncertain",
                f"multiple WandB matches ({len(matches)}) and could not "
                f"disambiguate by score; run_ids={run_ids}",
                None,
            )

    matched = matches[0]

    # ── Step 4: score verification ──────────────────────────────────────────
    wb_ra = _wb_ra_for_split(matched, split)

    if csv_ra is not None and not np.isnan(float(csv_ra)) and wb_ra is not None:
        diff = abs(float(csv_ra) - float(wb_ra))
        if diff > SCORE_TOL:
            return (
                "uncertain",
                f"score mismatch: CSV ra_mean={float(csv_ra):.4f}, "
                f"WandB RA_mean={float(wb_ra):.4f} (diff={diff:.4f})",
                matched,
            )

    suffix = " (base-name fallback)" if used_base_fallback else ""
    notes = f"verified via WandB run {matched['run_id']}{suffix}"
    return "certain", notes, matched



def phase3_match_rows(df: pd.DataFrame, lookup: dict, workers: int) -> pd.DataFrame:
    log_section("PHASE 3 — Matching CSV Rows Against WandB")
    log(f"  {len(df):,} CSV rows to match using {workers} threads...")

    results = [None] * len(df)

    def _match_idx(idx_row):
        idx, row = idx_row
        conf, notes, matched = _match_row(row, lookup)
        wandb_run_id   = matched["run_id"]   if matched else None
        wandb_group    = matched["group"]     if matched else None
        wandb_ra_mean  = matched["ra_mean"]   if matched else None
        wandb_feat_json = json.dumps(matched["feat_scores"]) if matched and matched.get("feat_scores") else None
        return idx, conf, notes, wandb_run_id, wandb_group, wandb_ra_mean, wandb_feat_json

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_match_idx, (i, row)) for i, row in df.iterrows()]
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            if done % 500 == 0 or done == len(futures):
                log(f"  Matched {done:,} / {len(futures):,} rows...")
            try:
                results[futures.index(fut)] = fut.result()
            except Exception as e:
                log(f"  WARNING: match error: {e}", indent=1)

    # Attach results to dataframe
    df = df.copy()
    df["confidence"]       = "uncertain"
    df["confidence_notes"] = ""
    df["wandb_run_id"]     = None
    df["wandb_group"]      = None
    df["wandb_ra_mean"]    = None
    df["wandb_feat_scores_json"] = None

    for r in results:
        if r is None:
            continue
        idx, conf, notes, wid, wgroup, wra, wfeat = r
        df.at[idx, "confidence"]       = conf
        df.at[idx, "confidence_notes"] = notes
        df.at[idx, "wandb_run_id"]     = wid
        df.at[idx, "wandb_group"]      = wgroup
        df.at[idx, "wandb_ra_mean"]    = wra
        df.at[idx, "wandb_feat_scores_json"] = wfeat

    n_certain   = (df["confidence"] == "certain").sum()
    n_uncertain = (df["confidence"] == "uncertain").sum()
    log(f"\n  Certain   : {n_certain:,}")
    log(f"  Uncertain : {n_uncertain:,}")

    # Uncertainty breakdown
    reasons = df[df["confidence"] == "uncertain"]["confidence_notes"].value_counts()
    log("\n  Uncertainty reasons:", indent=1)
    for reason, count in reasons.head(15).items():
        short = reason[:90] + "…" if len(reason) > 90 else reason
        log(f"  {count:5d}×  {short}", indent=2)

    return df


# ---------------------------------------------------------------------------
# Phase 4: Collect WandB-only rows
# ---------------------------------------------------------------------------

def phase4_wandb_only(df_verified: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Find WandB runs not matched to ANY local CSV row."""
    log_section("PHASE 4 — WandB-Only Rows")

    if not lookup:
        log("No WandB data — skipping.")
        return pd.DataFrame()

    # Collect all WandB run_ids that were matched
    matched_ids = set(df_verified["wandb_run_id"].dropna().unique())
    log(f"  WandB runs matched to local CSVs: {len(matched_ids):,}")

    wandb_only_rows = []
    for key, runs in lookup.items():
        for run in runs:
            if run["run_id"] not in matched_ids:
                wandb_only_rows.append({
                    "wandb_run_id":   run["run_id"],
                    "wandb_group":    run["group"],
                    "wandb_project":  run["project"],
                    "dataset":        run["dataset"],
                    "dataset_size":   run["dataset_size"],
                    "sample":         run["sample"],
                    "qi":             run["qi"],
                    "sdg_method":     run["sdg_method"],
                    "attack_label":   run["attack_method"],   # base name only (no variant)
                    "split":          "standard" if run["ra_mean"] is not None else
                                      "train"    if run["train_mean"] is not None else "unknown",
                    "ra_mean":        run["ra_mean"],
                    "train_mean":     run["train_mean"],
                    "nt_mean":        run["nt_mean"],
                    "feat_scores_json": json.dumps(run["feat_scores"]) if run["feat_scores"] else None,
                    "confidence":     "certain",
                    "confidence_notes": f"WandB-only run; all dims from WandB config",
                    "source_file":    "wandb_only",
                })

    df_only = pd.DataFrame(wandb_only_rows)
    log(f"  WandB-only runs  : {len(df_only):,}")
    if not df_only.empty:
        # Brief breakdown
        counts = df_only.groupby(["dataset", "dataset_size"]).size()
        log("  Breakdown:", indent=1)
        for (ds, sz), n in counts.items():
            log(f"    {ds} / size {sz} : {n}", indent=2)
    return df_only


# ---------------------------------------------------------------------------
# Phase 5: Output reports
# ---------------------------------------------------------------------------

def phase5_write_outputs(df_verified: pd.DataFrame, df_wandb_only: pd.DataFrame) -> None:
    log_section("PHASE 5 — Writing Output Files")
    OUTFILES_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Full verification report
    out_report = OUTFILES_DIR / "verification_report.csv"
    df_verified.to_csv(out_report, index=False)
    log(f"  verification_report.csv   ({len(df_verified):,} rows) → {out_report}")

    # Uncertain rows only
    df_uncertain = df_verified[df_verified["confidence"] == "uncertain"].copy()
    out_uncertain = OUTFILES_DIR / "uncertain_rows.csv"
    df_uncertain.to_csv(out_uncertain, index=False)
    log(f"  uncertain_rows.csv        ({len(df_uncertain):,} rows) → {out_uncertain}")

    # WandB-only
    if not df_wandb_only.empty:
        out_only = OUTFILES_DIR / "wandb_only.csv"
        df_wandb_only.to_csv(out_only, index=False)
        log(f"  wandb_only.csv            ({len(df_wandb_only):,} rows) → {out_only}")
    else:
        log("  wandb_only.csv            (0 rows — all WandB runs have local CSVs)")

    # Re-run queue: distinct experiment configs from uncertain rows
    rerun_keys = (
        df_uncertain[["dataset", "dataset_size", "sample", "qi", "sdg_method", "attack_label", "split"]]
        .dropna(subset=["dataset", "sample", "qi", "sdg_method", "attack_label"])
        .drop_duplicates()
        .sort_values(["dataset", "dataset_size", "qi", "sdg_method", "attack_label", "sample"])
    )
    out_queue = OUTFILES_DIR / "rerun_queue.txt"
    with open(out_queue, "w") as f:
        f.write(f"# Re-run queue generated {ts}\n")
        f.write(f"# {len(rerun_keys)} distinct experiment configurations could not be\n")
        f.write("# verified against WandB.  Re-running these will write them directly\n")
        f.write("# to results.db via the updated sweep scripts.\n\n")
        for _, row in rerun_keys.iterrows():
            f.write(
                f"dataset={row.get('dataset', '?')}  "
                f"size={row.get('dataset_size', '?')}  "
                f"sample={row.get('sample', '?')}  "
                f"qi={row.get('qi', '?')}  "
                f"sdg={row.get('sdg_method', '?')}  "
                f"attack={row.get('attack_label', '?')}  "
                f"split={row.get('split', 'standard')}\n"
            )
    log(f"  rerun_queue.txt           ({len(rerun_keys):,} configs) → {out_queue}")

    # Final summary
    n_certain_csv  = (df_verified["confidence"] == "certain").sum()
    n_uncertain    = len(df_uncertain)
    n_wandb_only   = len(df_wandb_only)

    log("\n" + "=" * 60)
    log("  AUDIT COMPLETE")
    log("=" * 60)
    log(f"  Certain CSV rows  : {n_certain_csv:,}  → migration_candidates")
    log(f"  WandB-only rows   : {n_wandb_only:,}   → direct insert")
    log(f"  Uncertain rows    : {n_uncertain:,}    → review & re-run")
    log(f"  Re-run configs    : {len(rerun_keys):,}")
    log(f"\nNext step: review outfiles/uncertain_rows.csv and outfiles/rerun_queue.txt,")
    log("then run:  python experiment_scripts/migrate_to_db.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip WandB API calls.  All rows become 'uncertain'.")
    parser.add_argument("--use-wandb-cache", action="store_true",
                        help=f"Load WandB lookup from {WANDB_CACHE} instead of hitting the API. "
                             "Fast path for re-running matching logic without re-fetching WandB.")
    parser.add_argument("--no-save-cache", action="store_true",
                        help="Do not save the WandB lookup to cache after a live fetch.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads for parallel CSV reading / matching (default: 8).")
    args = parser.parse_args()

    log(f"audit_and_verify.py  started {datetime.now(timezone.utc).isoformat()}")
    log(f"  Script dir : {SCRIPT_DIR}")
    log(f"  Workers    : {args.workers}")
    if args.no_wandb:
        log("  WandB      : DISABLED (--no-wandb)")
    elif args.use_wandb_cache:
        log(f"  WandB      : CACHE ({WANDB_CACHE})")
    else:
        log("  WandB      : LIVE FETCH")

    # Phase 1
    df_csv = phase1_read_csvs(workers=args.workers)

    # Phase 2
    if args.no_wandb:
        lookup = {}
    else:
        lookup = phase2_fetch_wandb(
            use_cache=args.use_wandb_cache,
            save_cache=not args.no_save_cache,
        )

    # Phase 3
    df_verified = phase3_match_rows(df_csv, lookup, workers=args.workers)

    # Phase 4
    df_wandb_only = phase4_wandb_only(df_verified, lookup)

    # Phase 5
    phase5_write_outputs(df_verified, df_wandb_only)

    elapsed = time.time() - _T0
    log(f"\nTotal elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
