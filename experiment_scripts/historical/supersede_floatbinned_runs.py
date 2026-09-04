#!/usr/bin/env python
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT

"""
supersede_floatbinned_runs.py

Retire every results.db row that was scored against a float-bin-midpoint
synth.csv (see fix_binned_synth_encoding.py).

Those rows are not merely noisy — integer-valued hidden features scored ~0
because a float midpoint can never equal a real integer, so their `ra_mean` is
biased downward by roughly 5-10 points depending on how many integral features
the QI leaves hidden. Deleting them would destroy the audit trail, so instead we
flip `confidence` from 'certain' to 'superseded_encoding' and record why.

ResultsDB.query() filters on confidence='certain' by default, so this single
change removes the stale rows from every LaTeX/table/plot path automatically.
They stay in the file for provenance and can be recovered with confidence=None.

Affected rows are identified by the *.floatbinned.bak files that
fix_binned_synth_encoding.py left behind: a backup at
<dataset>/<size>/<sample>/<SDG>/synth.csv.floatbinned.bak means every run keyed
to that (dataset, size, sample, sdg_method) was computed on bad input.

USAGE
    python experiment_scripts/supersede_floatbinned_runs.py --dry-run
    python experiment_scripts/supersede_floatbinned_runs.py
    python experiment_scripts/supersede_floatbinned_runs.py --revert
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(str(DATA_ROOT))
DB_PATH = Path(__file__).with_name("results.db")
MARK = "superseded_encoding"
NOTE = ("scored against float bin-midpoint synth.csv (integral hidden features "
        "scored ~0); corrected by fix_binned_synth_encoding.py — re-run required")

# Data directory -> (dataset, dataset_size) as keyed in results.db.
DIR_TO_DB = {
    ("adult", "size_1000"):                          ("adult", 1000),
    ("adult", "size_10000"):                         ("adult", 10000),
    ("adult", "size_20000"):                         ("adult", 20000),
    ("cdc_diabetes", "size_1000"):                   ("cdc_diabetes", 1000),
    ("cdc_diabetes", "size_100000"):                 ("cdc_diabetes", 100000),
    ("nist_arizona_data", "size_10000_25feat"):      ("nist_arizona_25feat", 10000),
    ("nist_sbo", "size_1000"):                       ("nist_sbo", 1000),
    ("california", "size_1000"):                     ("california", 1000),
}


def affected_keys():
    """(dataset, size, sample_idx, sdg_method) for every corrected synth file."""
    keys, unmapped = set(), set()
    for bak in DATA_ROOT.glob("*/*/sample_*/*/synth.csv.floatbinned.bak"):
        sdg = bak.parent.name
        sample_idx = int(bak.parent.parent.name.split("_")[1])
        size_dir = bak.parent.parent.parent.name
        ds_dir = bak.parent.parent.parent.parent.name
        mapped = DIR_TO_DB.get((ds_dir, size_dir))
        if mapped is None:
            unmapped.add((ds_dir, size_dir))
            continue
        keys.add((mapped[0], mapped[1], sample_idx, sdg))
    return keys, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true",
                    help="restore superseded rows to confidence='certain'")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.revert:
        n = conn.execute(
            "UPDATE runs SET confidence='certain', confidence_notes=NULL "
            "WHERE confidence=?", (MARK,)).rowcount
        conn.commit()
        print(f"reverted {n} rows to confidence='certain'")
        return

    keys, unmapped = affected_keys()
    if unmapped:
        print("WARNING: unmapped data directories (rows NOT superseded):")
        for u in sorted(unmapped):
            print(f"  {u[0]}/{u[1]}")

    total, by_gen = 0, defaultdict(int)
    for dataset, size, sample_idx, sdg in sorted(keys):
        rows = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE dataset=? AND dataset_size=? "
            "AND sample=? AND sdg_method=? AND confidence='certain'",
            (dataset, size, sample_idx, sdg)).fetchone()[0]
        if not rows:
            continue
        total += rows
        by_gen[f"{dataset} {size} {sdg.split('_eps')[0]}"] += rows
        if not args.dry_run:
            conn.execute(
                "UPDATE runs SET confidence=?, confidence_notes=? "
                "WHERE dataset=? AND dataset_size=? AND sample=? AND sdg_method=? "
                "AND confidence='certain'",
                (MARK, NOTE, dataset, size, sample_idx, sdg))
    if not args.dry_run:
        conn.commit()

    print(f"\n{len(keys)} corrected (dataset,size,sample,sdg) combinations")
    print(f"{'would supersede' if args.dry_run else 'superseded'} {total} runs\n")
    for k in sorted(by_gen):
        print(f"  {k:44} {by_gen[k]:5}")

    remaining = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE confidence='certain'").fetchone()[0]
    print(f"\nrows still 'certain' (visible to queries): {remaining}")
    if args.dry_run:
        print("(dry run — nothing written)")
    conn.close()


if __name__ == "__main__":
    main()
