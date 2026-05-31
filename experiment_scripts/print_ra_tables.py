"""
Terminal-format RA tables for adult 1k (QI1, standard) and memorization deltas.

Usage:
  python print_ra_tables.py          # all three tables
  python print_ra_tables.py --table 1  # just RA_mean
  python print_ra_tables.py --table 2  # just memorization delta (QI_linear)
  python print_ra_tables.py --table 3  # feature-level delta (requires nontraining feature scores)
"""
import argparse
import sqlite3
import pandas as pd
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "results.db"

# SDG display order: MST (ascending epsilon), AIM (ascending epsilon), then others
SDG_ORDER_ALL = [
    "MST_eps0.1", "MST_eps1", "MST_eps10", "MST_eps100", "MST_eps1000",
    "AIM_eps1", "AIM_eps10",
    "ARF", "CTGAN", "CellSuppression", "RankSwap", "Synthpop", "TabDDPM", "TVAE",
]

# Short display names for SDG columns
SDG_LABELS = {
    "MST_eps0.1": "MST\n.1",
    "MST_eps1": "MST\n1",
    "MST_eps10": "MST\n10",
    "MST_eps100": "MST\n100",
    "MST_eps1000": "MST\n1k",
    "AIM_eps1": "AIM\n1",
    "AIM_eps10": "AIM\n10",
    "ARF": "ARF",
    "CTGAN": "CTGAN",
    "CellSuppression": "CellSup",
    "RankSwap": "RankSwp",
    "Synthpop": "Synpop",
    "TabDDPM": "TabDDPM",
    "TVAE": "TVAE",
}

# Attack display order
ATTACK_ORDER = [
    "Random", "Mode", "MeasureDeid",
    "KNN", "NaiveBayes", "LogisticRegression", "RandomForest", "LightGBM", "SVM", "MLP",
    "Attention", "TabPFN",
    "PartialMST", "PartialMSTIndependent", "PartialMSTBounded",
    "TabDDPM", "TabDDPMWithMLP", "ConditionedRePaint",
]

# Minimum number of attacks that must have data for an epsilon SDG column to be included
# (non-epsilon SDGs are always included)
MIN_EPSILON_COVERAGE = 5


def _fmt(x, decimals=1):
    if pd.isna(x):
        return "  -  "
    return f"{x:5.{decimals}f}"


def _print_table(pivot: pd.DataFrame, title: str, decimals: int = 1):
    """Print a pivot table in aligned terminal format."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    attack_col_w = max(len(a) for a in pivot.index) + 2
    cell_w = 7

    # Header
    header = f"{'Attack':{attack_col_w}}"
    for col in pivot.columns:
        label = SDG_LABELS.get(col, col)
        # Use first line of multi-line label for single-row header
        label = label.replace("\n", "/")
        header += f"  {label:>{cell_w-2}}"
    print(header)
    print("-" * len(header))

    for attack in pivot.index:
        row = f"{attack:{attack_col_w}}"
        for col in pivot.columns:
            val = pivot.loc[attack, col]
            row += f"  {_fmt(val, decimals):>{cell_w-2}}"
        print(row)

    print()
    # Coverage note
    n_attacks = len(pivot.index)
    for col in pivot.columns:
        n_present = pivot[col].notna().sum()
        if n_present < n_attacks:
            print(f"  Note: {col} has data for {n_present}/{n_attacks} attacks shown")


def load_standard_pivot(conn, dataset, size, qi, split="standard"):
    """Load avg-over-samples RA_mean pivot for the given setting."""
    df = pd.read_sql_query("""
        SELECT attack_label, sdg_method, AVG(ra_mean) as ra_mean
        FROM runs
        WHERE dataset=? AND dataset_size=? AND qi=? AND split=?
          AND attack_label NOT LIKE '%+%'
          AND attack_label NOT LIKE 'MarginalRF%'
          AND attack_label NOT IN ('FeatSelectorOracle','OracleEnsemble')
        GROUP BY attack_label, sdg_method
    """, conn, params=(dataset, size, qi, split))

    if df.empty:
        return None

    pivot = df.pivot(index="attack_label", columns="sdg_method", values="ra_mean")
    return pivot


def filter_sdg_columns(pivot, min_count=MIN_EPSILON_COVERAGE):
    """Keep SDG columns in canonical order.

    For epsilon-based SDGs (MST_*, AIM_*): require >= min_count attacks to have data.
    For non-epsilon SDGs (ARF, CTGAN, etc.): always include if present.
    """
    threshold = min_count
    epsilon_prefixes = ("MST_eps", "AIM_eps")

    available = []
    for s in SDG_ORDER_ALL:
        if s not in pivot.columns:
            continue
        is_epsilon_sdg = any(s.startswith(p) for p in epsilon_prefixes)
        if is_epsilon_sdg:
            if pivot[s].notna().sum() >= threshold:
                available.append(s)
        else:
            available.append(s)
    return pivot[available]


def filter_attack_rows(pivot):
    """Keep only canonical attacks, in canonical order, that have any data."""
    available = [a for a in ATTACK_ORDER if a in pivot.index]
    return pivot.loc[available]


def table1_ra_mean(conn):
    """Table 1: RA_mean, adult 1k, QI1, standard."""
    pivot = load_standard_pivot(conn, "adult", 1000, "QI1")
    if pivot is None:
        print("No data for adult 1k QI1 standard.")
        return

    pivot = filter_attack_rows(pivot)
    pivot = filter_sdg_columns(pivot)
    _print_table(pivot, "Table 1 — RA_mean: Adult 1k, QI1 (avg over 5 samples)")


def table2_memo_delta(conn):
    """
    Table 2: memorization delta = train_RA - nontraining_RA.
    Uses QI_linear (the linear sweep QI variant) since QI1 has no train/nontraining data.
    Shows adult 1k, adult 10k (QI_linear_lowcard), and CDC 1k.
    """
    settings = [
        ("adult", 1000, "QI_linear", "Adult 1k, QI_linear"),
        ("adult", 10000, "QI_linear_lowcard", "Adult 10k, QI_linear_lowcard"),
        ("cdc_diabetes", 1000, "QI_linear", "CDC Diabetes 1k, QI_linear"),
    ]

    for dataset, size, qi, label in settings:
        train_p = load_standard_pivot(conn, dataset, size, qi, split="train")
        nontraining_p = load_standard_pivot(conn, dataset, size, qi, split="nontraining")

        if train_p is None or nontraining_p is None:
            print(f"\nNo data for {label} (train or nontraining split missing)")
            continue

        # Align on common attacks and SDGs
        common_attacks = train_p.index.intersection(nontraining_p.index)
        common_sdgs = train_p.columns.intersection(nontraining_p.columns)
        delta = train_p.loc[common_attacks, common_sdgs] - nontraining_p.loc[common_attacks, common_sdgs]

        delta = filter_attack_rows(delta)
        delta = filter_sdg_columns(delta)
        _print_table(delta, f"Table 2 — Memorization Delta (train - holdout): {label}")


def table3_feature_delta(conn):
    """
    Table 3: feature-level delta, averaged.
    = mean_i(RA_train_feat_i - RA_holdout_feat_i)

    Requires feature_scores for BOTH train and nontraining splits.
    Currently only standard split has feature_scores — not feasible.
    """
    # Check if nontraining feature_scores exist
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM feature_scores fs
        JOIN runs r ON fs.run_id = r.run_id
        WHERE r.split IN ('train','nontraining')
    """)
    count = cur.fetchone()[0]

    if count == 0:
        print("\n" + "="*80)
        print("  Table 3 — Feature-level Memorization Delta")
        print("="*80)
        print("""
  NOT AVAILABLE: feature_scores table only has 'standard' split rows.
  The train/nontraining runs (from the linear sweep) were ingested with run-level
  ra_mean only — no per-feature scores were stored.

  To generate this table, the linear-sweep runs need to be re-run (or re-ingested)
  with per-feature logging enabled, OR the main-sweep (QI1) needs a memorization
  test pass.

  Table 2 above uses the run-level delta (equivalent when all features are equally
  weighted, which is an approximation for the rarity-weighted adult scoring).
""")
        return

    # If data existed, compute feature-level delta here
    # (placeholder — not reachable currently)
    df = pd.read_sql_query("""
        SELECT r.attack_label, r.sdg_method, r.split, fs.feature, AVG(fs.ra_score) as ra_score
        FROM feature_scores fs
        JOIN runs r ON fs.run_id = r.run_id
        WHERE r.dataset='adult' AND r.dataset_size=1000 AND r.qi='QI_linear'
          AND r.split IN ('train','nontraining')
        GROUP BY r.attack_label, r.sdg_method, r.split, fs.feature
    """, conn)

    train_df = df[df.split == "train"].drop(columns="split")
    nt_df = df[df.split == "nontraining"].drop(columns="split")

    merged = train_df.merge(nt_df, on=["attack_label", "sdg_method", "feature"],
                            suffixes=("_train", "_nt"))
    merged["delta"] = merged["ra_score_train"] - merged["ra_score_nt"]

    avg_delta = (merged.groupby(["attack_label", "sdg_method"])["delta"]
                 .mean().reset_index()
                 .rename(columns={"delta": "ra_mean"}))

    pivot = avg_delta.pivot(index="attack_label", columns="sdg_method", values="ra_mean")
    pivot = filter_attack_rows(pivot)
    pivot = filter_sdg_columns(pivot)
    _print_table(pivot, "Table 3 — Feature-level Delta (avg_feat[train_RA - holdout_RA]): Adult 1k, QI_linear")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, choices=[1, 2, 3], default=None,
                        help="Which table to print (default: all)")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    tables = [args.table] if args.table else [1, 2, 3]

    if 1 in tables:
        table1_ra_mean(conn)
    if 2 in tables:
        print("\n--- Memorization data availability for QI1 ---")
        print("  Adult 1k  QI1: NO train/nontraining split data")
        print("  Adult 10k QI1: NO train/nontraining split data")
        print("  CDC 1k    QI1: NO train/nontraining split data")
        print("  => Using QI_linear (linear sweep) for memorization tables below.")
        print("     These have: adult 1k/10k and CDC 1k, 12 SDGs, ~9 attacks each.")
        table2_memo_delta(conn)
    if 3 in tables:
        table3_feature_delta(conn)

    conn.close()


if __name__ == "__main__":
    main()
