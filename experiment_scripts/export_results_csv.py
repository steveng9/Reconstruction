"""
export_results_csv.py
─────────────────────
Export results from the SQLite results DB to a CSV suitable for analysis
(e.g. upload to Google Colab for graphing).

Key transforms applied
  • sdg_method labels like "MST_eps1" / "AIM_eps0.3" are split into
    sdg_base = "MST" / "AIM"  and  epsilon = 1.0 / 0.3
  • All other SDG methods keep sdg_base = original label, epsilon = NaN
  • Feature-level scores are optionally included as extra columns

Usage examples
──────────────
  # Adult 1k + 10k, standard split only, no feature scores (default)
  python experiment_scripts/export_results_csv.py

  # Specify dataset / sizes / splits / output path
  python experiment_scripts/export_results_csv.py \\
      --dataset adult \\
      --sizes 1000 10000 \\
      --splits standard train nontraining \\
      --out my_results.csv

  # Include per-feature scores as extra columns
  python experiment_scripts/export_results_csv.py --features

  # Any dataset (e.g. future use)
  python experiment_scripts/export_results_csv.py \\
      --dataset nist_arizona_25feat --sizes 10000

  # Show what would be exported without writing anything
  python experiment_scripts/export_results_csv.py --dry-run
"""

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── defaults ──────────────────────────────────────────────────────────────────
DB_PATH_DEFAULT = Path(__file__).parent / "results.db"
OUT_DIR_DEFAULT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Export results DB rows to a CSV for graphing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--db", default=str(DB_PATH_DEFAULT),
                   help="Path to results.db (default: %(default)s)")
    p.add_argument("--dataset", default="adult",
                   help="Dataset name to filter on (default: %(default)s)")
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000],
                   help="dataset_size values to include (default: 1000 10000)")
    p.add_argument("--splits", nargs="+", default=["standard"],
                   choices=["standard", "train", "nontraining", "unknown"],
                   help="split values to include (default: standard)")
    p.add_argument("--qi", nargs="+", default=None,
                   help="QI variant(s) to include; omit for all")
    p.add_argument("--sdg", nargs="+", default=None,
                   help="sdg_method label(s) to include; omit for all")
    p.add_argument("--attack", nargs="+", default=None,
                   help="attack_label(s) to include; omit for all")
    p.add_argument("--features", action="store_true",
                   help="Add per-feature RA scores as extra columns")
    p.add_argument("--out", default=None,
                   help="Output CSV path. Default: auto-named in experiment_scripts/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print row counts and column list without writing CSV")
    return p.parse_args()


# ── SDG label parsing ─────────────────────────────────────────────────────────
_EPS_RE = re.compile(r"^(MST|AIM)_eps([\d.]+)$", re.IGNORECASE)

def _split_sdg_label(label: str):
    """
    Returns (sdg_base, epsilon).
    For 'MST_eps1'  → ('MST', 1.0)
    For 'AIM_eps0.3' → ('AIM', 0.3)
    For anything else → (label, None)
    """
    m = _EPS_RE.match(label)
    if m:
        return m.group(1).upper(), float(m.group(2))
    return label, None


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"[ERROR] DB not found: {args.db}")

    conn = sqlite3.connect(args.db)

    # ── build WHERE clause dynamically ────────────────────────────────────────
    where_parts = ["dataset = ?", f"dataset_size IN ({','.join('?'*len(args.sizes))})"]
    params      = [args.dataset] + args.sizes

    where_parts.append(f"split IN ({','.join('?'*len(args.splits))})")
    params.extend(args.splits)

    if args.qi:
        where_parts.append(f"qi IN ({','.join('?'*len(args.qi))})")
        params.extend(args.qi)

    if args.sdg:
        where_parts.append(f"sdg_method IN ({','.join('?'*len(args.sdg))})")
        params.extend(args.sdg)

    if args.attack:
        where_parts.append(f"attack_label IN ({','.join('?'*len(args.attack))})")
        params.extend(args.attack)

    where_sql = " AND ".join(where_parts)

    # ── fetch runs ────────────────────────────────────────────────────────────
    runs_df = pd.read_sql_query(
        f"""
        SELECT
            run_id,
            dataset,
            dataset_size,
            sample,
            qi,
            sdg_method,
            attack_label,
            split,
            ra_mean,
            wandb_run_id,
            wandb_group
        FROM runs
        WHERE {where_sql}
        ORDER BY dataset_size, sample, sdg_method, attack_label, qi, split
        """,
        conn,
        params=params,
    )

    if runs_df.empty:
        print("[WARNING] No rows matched the filter criteria. Check --dataset / --sizes / --splits.")
        conn.close()
        return

    # ── parse sdg_base + epsilon ───────────────────────────────────────────────
    parsed = runs_df["sdg_method"].map(_split_sdg_label)
    runs_df.insert(
        runs_df.columns.get_loc("sdg_method") + 1,
        "sdg_base",
        parsed.map(lambda t: t[0]),
    )
    runs_df.insert(
        runs_df.columns.get_loc("sdg_base") + 1,
        "epsilon",
        parsed.map(lambda t: t[1]),
    )

    # ── optionally join feature scores ─────────────────────────────────────────
    if args.features:
        feat_df = pd.read_sql_query(
            f"""
            SELECT fs.run_id, fs.feature, fs.ra_score
            FROM feature_scores fs
            JOIN runs r ON r.run_id = fs.run_id
            WHERE {where_sql}
            """,
            conn,
            params=params,
        )
        if not feat_df.empty:
            feat_wide = feat_df.pivot(index="run_id", columns="feature", values="ra_score")
            feat_wide.columns = [f"feature_{c}" for c in feat_wide.columns]
            runs_df = runs_df.merge(feat_wide, on="run_id", how="left")

    conn.close()

    # ── drop internal run_id from output ──────────────────────────────────────
    runs_df = runs_df.drop(columns=["run_id"])

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nRows matched : {len(runs_df):,}")
    print(f"Columns      : {list(runs_df.columns)}")
    print(f"\nDataset sizes: {sorted(runs_df['dataset_size'].unique())}")
    print(f"SDG methods  : {sorted(runs_df['sdg_base'].unique())}")
    print(f"Epsilon vals : {sorted(runs_df['epsilon'].dropna().unique())}")
    print(f"Attacks      : {len(runs_df['attack_label'].unique())} unique")
    print(f"QI variants  : {sorted(runs_df['qi'].unique())}")
    print(f"Splits       : {sorted(runs_df['split'].unique())}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    # ── auto-name output file ─────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        sizes_str  = "_".join(
            ("1k" if s == 1000 else "10k" if s == 10000 else str(s))
            for s in sorted(args.sizes)
        )
        splits_str = "_".join(sorted(args.splits))
        feat_tag   = "_with_features" if args.features else ""
        out_path   = OUT_DIR_DEFAULT / f"results_{args.dataset}_{sizes_str}_{splits_str}{feat_tag}.csv"

    runs_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
