#!/usr/bin/env python
"""
migrate_to_db.py — Phase 4: Insert verified results into results.db.

Run this AFTER audit_and_verify.py has produced:
  experiment_scripts/outfiles/verification_report.csv
  experiment_scripts/outfiles/wandb_only.csv          (optional)

What this script does
---------------------
1. Reads verification_report.csv, keeps only rows with confidence='certain'.
2. Reads wandb_only.csv (WandB runs with no local CSV).
3. Inserts all rows into results.db (handling duplicates gracefully).
4. Writes migration_log.csv with every decision made.

Nothing is deleted or overwritten.  The script is idempotent: running it
again on the same input is safe — duplicates are logged, not re-inserted.

Usage (can run in the background):
  nohup python experiment_scripts/migrate_to_db.py \\
      > experiment_scripts/outfiles/migrate.log 2>&1 &
  echo $! > experiment_scripts/outfiles/migrate.pid
  tail -f experiment_scripts/outfiles/migrate.log

Options:
  --dry-run        Print what would be inserted without writing to DB.
  --db PATH        Override the default DB path (experiment_scripts/results.db).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure we can import results_db regardless of working directory
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))  # repo root
sys.path.insert(0, str(SCRIPT_DIR))

from results_db import ResultsDB, DEFAULT_DB_PATH

OUTFILES_DIR = SCRIPT_DIR / "outfiles"

# ---------------------------------------------------------------------------
# Logging
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
# Row normalisation
# ---------------------------------------------------------------------------

def _float_or_none(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _int_or_none(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _feat_scores_from_row(row: pd.Series, feat_json_col: str | None = None) -> dict:
    """Extract feature scores from either a JSON column or RA_* columns."""
    # Prefer pre-serialised JSON (from audit script)
    if feat_json_col and feat_json_col in row.index:
        raw = row.get(feat_json_col)
        if raw and not (isinstance(raw, float) and np.isnan(raw)):
            try:
                return json.loads(raw)
            except Exception:
                pass
    # Fall back to RA_* columns in the row
    return {
        k[3:]: float(v)
        for k, v in row.items()
        if k.startswith("RA_") and k != "RA_mean" and not isinstance(v, float) is False
        or (k.startswith("RA_") and k != "RA_mean" and pd.notna(v) and isinstance(v, (int, float)))
    }


def _clean_feat_scores(row: pd.Series) -> dict:
    return {
        k[3:]: float(v)
        for k, v in row.items()
        if k.startswith("RA_") and k not in ("RA_mean",)
        and pd.notna(v) and isinstance(v, (int, float))
    }


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def _insert_from_verification_row(db: ResultsDB, row: pd.Series, dry_run: bool) -> str:
    """Insert one row from verification_report.csv.  Returns 'inserted'|'conflict'|'error'."""
    # Required fields
    dataset      = row.get("dataset")
    dataset_size = _int_or_none(row.get("dataset_size"))
    sample       = _int_or_none(row.get("sample"))
    qi           = row.get("qi")
    sdg_method   = row.get("sdg_method")
    attack_label = row.get("attack_label")
    split        = row.get("split", "standard") or "standard"
    ra_mean      = _float_or_none(row.get("ra_mean"))

    if None in (dataset, dataset_size, sample, qi, sdg_method, attack_label):
        return "skip_missing_dims"

    # Feature scores
    feat_json_col = "feature_scores_json"
    if feat_json_col in row.index and pd.notna(row.get(feat_json_col)):
        try:
            feat_scores = json.loads(row[feat_json_col])
        except Exception:
            feat_scores = _clean_feat_scores(row)
    else:
        feat_scores = _clean_feat_scores(row)

    wandb_feat_col = "wandb_feat_scores_json"
    if not feat_scores and wandb_feat_col in row.index and pd.notna(row.get(wandb_feat_col)):
        try:
            feat_scores = json.loads(row[wandb_feat_col])
        except Exception:
            pass

    if dry_run:
        return "dry_run"

    run_id = db.insert_run(
        dataset=str(dataset),
        dataset_size=int(dataset_size),
        sample=int(sample),
        qi=str(qi),
        sdg_method=str(sdg_method),
        attack_label=str(attack_label),
        split=str(split),
        ra_mean=ra_mean,
        feature_scores=feat_scores if feat_scores else None,
        source_file=str(row.get("source_file", "")),
        wandb_run_id=str(row["wandb_run_id"]) if pd.notna(row.get("wandb_run_id")) else None,
        wandb_group=str(row["wandb_group"]) if pd.notna(row.get("wandb_group")) else None,
        confidence=str(row.get("confidence", "certain")),
        confidence_notes=str(row.get("confidence_notes", "")) or None,
    )
    return "inserted" if run_id is not None else "conflict_or_duplicate"


def _insert_from_wandb_only_row(db: ResultsDB, row: pd.Series, dry_run: bool) -> str:
    """Insert one row from wandb_only.csv."""
    dataset      = row.get("dataset")
    dataset_size = _int_or_none(row.get("dataset_size"))
    sample       = _int_or_none(row.get("sample"))
    qi           = row.get("qi")
    sdg_method   = row.get("sdg_method")
    attack_label = row.get("attack_label")
    ra_mean      = _float_or_none(row.get("ra_mean"))
    split        = row.get("split", "standard") or "standard"

    if None in (dataset, dataset_size, sample, qi, sdg_method, attack_label):
        return "skip_missing_dims"

    feat_scores = {}
    if pd.notna(row.get("feat_scores_json")):
        try:
            feat_scores = json.loads(row["feat_scores_json"])
        except Exception:
            pass

    if dry_run:
        return "dry_run"

    run_id = db.insert_run(
        dataset=str(dataset),
        dataset_size=int(dataset_size),
        sample=int(sample),
        qi=str(qi),
        sdg_method=str(sdg_method),
        attack_label=str(attack_label),
        split=str(split),
        ra_mean=ra_mean,
        feature_scores=feat_scores if feat_scores else None,
        source_file="wandb_only",
        wandb_run_id=str(row["wandb_run_id"]) if pd.notna(row.get("wandb_run_id")) else None,
        wandb_group=str(row.get("wandb_group", "")) or None,
        wandb_project=row.get("wandb_project"),
        confidence="certain",
        confidence_notes="WandB-only: all dimensions from WandB run config",
    )
    return "inserted" if run_id is not None else "conflict_or_duplicate"

    # Also insert memorisation splits if present
    for split_key, col in [("train", "train_mean"), ("nontraining", "nt_mean")]:
        v = _float_or_none(row.get(col))
        if v is not None:
            db.insert_run(
                dataset=str(dataset),
                dataset_size=int(dataset_size),
                sample=int(sample),
                qi=str(qi),
                sdg_method=str(sdg_method),
                attack_label=str(attack_label),
                split=split_key,
                ra_mean=v,
                source_file="wandb_only",
                wandb_run_id=str(row["wandb_run_id"]) if pd.notna(row.get("wandb_run_id")) else None,
                wandb_group=str(row.get("wandb_group", "")) or None,
                wandb_project=row.get("wandb_project"),
                confidence="certain",
                confidence_notes="WandB-only: memorisation split",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be inserted; do NOT write to DB.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()

    log(f"migrate_to_db.py  started {datetime.now(timezone.utc).isoformat()}")
    log(f"  DB path   : {args.db}")
    log(f"  Dry-run   : {args.dry_run}")

    # ── Load input files ────────────────────────────────────────────────────
    report_path = OUTFILES_DIR / "verification_report.csv"
    wandb_only_path = OUTFILES_DIR / "wandb_only.csv"

    if not report_path.exists():
        log(f"ERROR: {report_path} not found.  Run audit_and_verify.py first.")
        sys.exit(1)

    df_report = pd.read_csv(report_path)
    log(f"Loaded verification_report.csv: {len(df_report):,} total rows")

    df_certain = df_report[df_report["confidence"] == "certain"].copy()
    n_uncertain = (df_report["confidence"] != "certain").sum()
    log(f"  Certain (to migrate) : {len(df_certain):,}")
    log(f"  Uncertain (skipped)  : {n_uncertain:,}")

    df_wandb_only = pd.DataFrame()
    if wandb_only_path.exists():
        df_wandb_only = pd.read_csv(wandb_only_path)
        log(f"Loaded wandb_only.csv        : {len(df_wandb_only):,} rows")
    else:
        log("wandb_only.csv not found — skipping WandB-only inserts.")

    total_to_insert = len(df_certain) + len(df_wandb_only)
    log(f"\nTotal rows to insert: {total_to_insert:,}")

    if args.dry_run:
        log("\n[DRY RUN] No changes will be made to the database.")

    # ── Open DB ─────────────────────────────────────────────────────────────
    db = ResultsDB(db_path=args.db)
    if not args.dry_run:
        log(f"Opened DB: {args.db}")

    # ── Insert verification_report rows ─────────────────────────────────────
    log_section("Inserting CSV-verified rows")
    log_rows = []
    counts = {"inserted": 0, "conflict_or_duplicate": 0,
              "skip_missing_dims": 0, "dry_run": 0, "error": 0}

    for i, (_, row) in enumerate(df_certain.iterrows(), 1):
        if i % 500 == 0 or i == len(df_certain):
            log(f"  [{i:,}/{len(df_certain):,}]  "
                f"inserted={counts['inserted']:,}  "
                f"conflicts={counts['conflict_or_duplicate']:,}  "
                f"skipped={counts['skip_missing_dims']:,}")
        try:
            outcome = _insert_from_verification_row(db, row, args.dry_run)
        except Exception as e:
            outcome = "error"
            log(f"  ERROR on row {i}: {e}", indent=1)

        counts[outcome] = counts.get(outcome, 0) + 1
        log_rows.append({
            "source": "csv",
            "source_file": row.get("source_file", ""),
            "dataset": row.get("dataset"), "dataset_size": row.get("dataset_size"),
            "sample": row.get("sample"), "qi": row.get("qi"),
            "sdg_method": row.get("sdg_method"), "attack_label": row.get("attack_label"),
            "split": row.get("split"), "ra_mean": row.get("ra_mean"),
            "outcome": outcome,
        })

    log(f"\n  CSV rows — inserted: {counts['inserted']:,}  "
        f"conflicts/dups: {counts['conflict_or_duplicate']:,}  "
        f"skipped: {counts['skip_missing_dims']:,}  errors: {counts['error']:,}")

    # ── Insert WandB-only rows ───────────────────────────────────────────────
    if not df_wandb_only.empty:
        log_section("Inserting WandB-only rows")
        wb_counts = {"inserted": 0, "conflict_or_duplicate": 0,
                     "skip_missing_dims": 0, "dry_run": 0, "error": 0}

        for i, (_, row) in enumerate(df_wandb_only.iterrows(), 1):
            if i % 200 == 0 or i == len(df_wandb_only):
                log(f"  [{i:,}/{len(df_wandb_only):,}]  "
                    f"inserted={wb_counts['inserted']:,}")
            try:
                outcome = _insert_from_wandb_only_row(db, row, args.dry_run)
            except Exception as e:
                outcome = "error"
                log(f"  ERROR on row {i}: {e}", indent=1)

            wb_counts[outcome] = wb_counts.get(outcome, 0) + 1
            log_rows.append({
                "source": "wandb_only",
                "source_file": "wandb_only",
                "dataset": row.get("dataset"), "dataset_size": row.get("dataset_size"),
                "sample": row.get("sample"), "qi": row.get("qi"),
                "sdg_method": row.get("sdg_method"), "attack_label": row.get("attack_label"),
                "split": row.get("split"), "ra_mean": row.get("ra_mean"),
                "outcome": outcome,
            })

        log(f"\n  WandB-only — inserted: {wb_counts['inserted']:,}  "
            f"conflicts/dups: {wb_counts['conflict_or_duplicate']:,}  "
            f"skipped: {wb_counts['skip_missing_dims']:,}  errors: {wb_counts['error']:,}")

    # ── Write migration log ──────────────────────────────────────────────────
    log_section("Writing Migration Log")
    log_path = OUTFILES_DIR / "migration_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    log(f"  migration_log.csv → {log_path}")

    # ── Final DB summary ─────────────────────────────────────────────────────
    if not args.dry_run:
        log_section("DB Summary After Migration")
        db.summary()

    db.close()

    elapsed = time.time() - _T0
    log(f"\nTotal elapsed: {elapsed:.1f}s")
    log("Done.")


if __name__ == "__main__":
    main()
