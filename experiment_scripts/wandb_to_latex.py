#!/usr/bin/env python
"""
Fetch RA_mean scores from WandB, average across 5 samples, and write a LaTeX table.

Rows = attack methods (grouped by type, separated by \\midrule).
Cols = SDG methods.

Usage:
    conda activate recon_
    python experiment_scripts/wandb_to_latex.py
    python experiment_scripts/wandb_to_latex.py --group "main attack sweep 1" --qi QI1
    python experiment_scripts/wandb_to_latex.py --out my_table.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


# ── WandB config ───────────────────────────────────────────────────────────────

WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "main attack sweep 1"
#DATASET       = "nist_arizona_25feat"   # filter to this dataset; None = all datasets
DATASET       = "adult"   # filter to this dataset; None = all datasets
#DATASET       = "california"   # filter to this dataset; None = all datasets


# ── Display order & groupings ──────────────────────────────────────────────────

# SDG methods in preferred column order; anything unseen here appended at end.
SDG_ORDER = [
    "RankSwap",
    "CellSuppression",
    "Synthpop",
    "MST_eps0.1",
    # "MST_eps0.3",   # 3×10^i variants — sparse, excluded for now
    "MST_eps1",
    # "MST_eps3",
    "MST_eps10",
    # "MST_eps30",
    "MST_eps100",
    # "MST_eps300",
    "MST_eps1000",
    # "AIM_eps0.3",
    "AIM_eps1",
    # "AIM_eps3",
    "AIM_eps10",
    "TVAE",
    "CTGAN",
    "ARF",
    "TabDDPM",
]

# Maps old attack label names → current canonical names.
# Applied at fetch time so deduplication (keep most-recent) drops older partial runs.
LABEL_REMAP: dict[str, str] = {
    "MarginalRF_global":       "MarginalRF_mst_global",
    "MarginalRF_local_k50":    "MarginalRF_mst_local_50",
    "MarginalRF_local_k100":   "MarginalRF_mst_local_100",
    "MarginalRF_local_k200":   "MarginalRF_mst_local_200",
    "MarginalRF_mst_local":    "MarginalRF_mst_local_100",
    "MarginalRF_complete_local": "MarginalRF_complete_local_100",
    "MarginalRF_topk_local":   "MarginalRF_topk_local_100",
}

# Attack groups control row ordering and \midrule placement.
# Attacks absent from the data are silently skipped.
ATTACK_GROUPS = [
    ("Baselines",      ["Mode", "Random", "MeasureDeid"]),
    ("ML Classifiers", ["KNN", "NaiveBayes", "LogisticRegression", "SVM",
                        "RandomForest", "LightGBM"]),
    ("Neural",         ["MLP", "Attention", "AttentionAutoregressive"]),
    ("Partial SDG",      ["TabDDPM", "TabDDPMWithMLP", "ConditionedRePaint", "RePaint", "PartialMST", "PartialMSTIndependent", "PartialMSTBounded"]),
    ("SOTA",           ["LinearReconstruction"]),
    ("New Attacks",    ["TabPFN",
                        "MarginalRF_mst_global",
                        "MarginalRF_mst_local_50",
                        "MarginalRF_mst_local_100",
                        "MarginalRF_mst_local_200",
                        "MarginalRF_complete_local_100",
                        "MarginalRF_topk_local_100",
                        "MarginalRF_complete_global"]),
]

ATTACK_DISPLAY: dict[str, str] = {
    "Mode":                    "Mode",
    "Random":                  "Random",
    "MeasureDeid":             "MeasureDeid",
    "KNN":                     r"\textsc{knn}",
    "NaiveBayes":              "Naive Bayes",
    "LogisticRegression":      "Logistic Regression",
    "SVM":                     r"\textsc{svm}",
    "RandomForest":            "Random Forest",
    "LightGBM":                "LightGBM",
    "MLP":                     r"\textsc{mlp}",
    "Attention":               "Attention",
    "AttentionAutoregressive": "Attention (AR)",
    "TabDDPM":                 "TabDDPM",
    "ConditionedRePaint":      r"Cond.\ RePaint",
    "TabDDPMWithMLP":         r"TabDDPM+MLP",
    #"RePaint":                 "RePaint",
    "LinearReconstruction":    "Linear Recon.",
    "PartialMST":              "MST",
    "PartialMSTIndependent":   "MST (1 ft./time)",
    "PartialMSTBounded":       "MST, k=3",
    "TabPFN":                      "TabPFN",
    "MarginalRF_mst_global":       r"MarginalRF (MST, global)",
    "MarginalRF_mst_local_50":     r"MarginalRF (MST, local $k$=50)",
    "MarginalRF_mst_local_100":    r"MarginalRF (MST, local $k$=100)",
    "MarginalRF_mst_local_200":    r"MarginalRF (MST, local $k$=200)",
    "MarginalRF_complete_local_100": r"MarginalRF (complete, local $k$=100)",
    "MarginalRF_topk_local_100":   r"MarginalRF (top-$k$, local $k$=100)",
    "MarginalRF_complete_global":  r"MarginalRF (complete, global)",
}

DATASET_DISPLAY: dict[str, str] = {
    "adult":               "Adult",
    "cdc_diabetes":        "CDC Diabetes",
    "california":          "California Housing",
    "nist_arizona_data":   "NIST Arizona (full)",
    "nist_arizona_25feat": "NIST Arizona (25 feat.)",
    "nist_arizona_50feat": "NIST Arizona (50 feat.)",
    "nist_sbo":            "NIST SBO",
}

SDG_DISPLAY: dict[str, str] = {
    "RankSwap":      "RankSwap",
    "CellSuppression": "Cell Supp.",
    "Synthpop":      "Synthpop",
    "MST_eps0.1":    r"MST $\varepsilon{=}0.1$",
    "MST_eps0.3":    r"MST $\varepsilon{=}0.3$",
    "MST_eps1":      r"MST $\varepsilon{=}1$",
    "MST_eps3":      r"MST $\varepsilon{=}3$",
    "MST_eps10":     r"MST $\varepsilon{=}10$",
    "MST_eps30":     r"MST $\varepsilon{=}30$",
    "MST_eps100":    r"MST $\varepsilon{=}100$",
    "MST_eps300":    r"MST $\varepsilon{=}300$",
    "MST_eps1000":   r"MST $\varepsilon{=}1000$",
    "AIM_eps0.3":    r"AIM $\varepsilon{=}0.3$",
    "AIM_eps1":      r"AIM $\varepsilon{=}1$",
    "AIM_eps3":      r"AIM $\varepsilon{=}3$",
    "AIM_eps10":     r"AIM $\varepsilon{=}10$",
    "AIM_eps100":    r"AIM $\varepsilon{=}100$",
    "TVAE":          "TVAE",
    "CTGAN":         "CTGAN",
    "ARF":           "ARF",
    "TabDDPM":       "TabDDPM",
}


# ── WandB helpers ──────────────────────────────────────────────────────────────

def _sdg_label(method: str, params: dict | None) -> str:
    """Reconstruct the canonical SDG label (e.g. 'MST_eps1') from config fields."""
    params = params or {}
    # wandb may store nested dicts flattened; try both access patterns.
    eps = params.get("epsilon") or params.get("eps")
    if eps is None:
        # Fallback: wandb sometimes flattens to "sdg_params.epsilon"
        # (handled upstream in fetch_runs, but guard here too)
        pass
    return f"{method}_eps{float(eps):g}" if eps is not None else method


def fetch_runs(group: str, qi_filter: str | None,
               attack_filter: list[str] | None = None,
               dataset_filter: str | None = None) -> pd.DataFrame:
    """Pull all finished runs from the given WandB group and return a flat DataFrame."""
    api    = wandb.Api(timeout=60)
    entity = api.default_entity
    path   = f"{entity}/{WANDB_PROJECT}"

    # dataset_filter=None means use the module-level DATASET constant.
    effective_dataset = dataset_filter if dataset_filter is not None else DATASET

    server_filters: dict = {"group": group}
    if attack_filter:
        server_filters["config.attack_method"] = {"$in": attack_filter}
    if effective_dataset:
        server_filters["config.dataset"] = effective_dataset

    print(f"Querying {path}  group={group!r} ...")
    if attack_filter:
        print(f"  Attack filter (server-side): {attack_filter}")
    if effective_dataset:
        print(f"  Dataset filter (server-side): {effective_dataset!r}")
    runs = api.runs(path, filters=server_filters)

    rows = []
    skipped = 0
    for run in runs:
        cfg  = run.config
        summ = run.summary

        attack     = cfg.get("attack_label") or cfg.get("attack_method")
        attack     = LABEL_REMAP.get(attack, attack)
        sdg_method = cfg.get("sdg_method")
        sdg_params = cfg.get("sdg_params") or {}
        qi         = cfg.get("qi")
        sample     = cfg.get("sample_idx")
        ra_mean    = summ.get("RA_mean")
        dataset_cfg = cfg.get("dataset") or {}
        dataset = dataset_cfg.get("name") if isinstance(dataset_cfg, dict) else dataset_cfg

        # Skip failed / incomplete runs
        if None in (attack, sdg_method, qi, sample) or ra_mean is None:
            skipped += 1
            continue

        if effective_dataset and dataset != effective_dataset:
            skipped += 1
            continue

        if qi_filter and qi != qi_filter:
            continue

        if attack_filter and attack not in attack_filter:
            skipped += 1
            continue

        rows.append({
            "attack":     attack,
            "sdg":        _sdg_label(sdg_method, sdg_params),
            "qi":         qi,
            "sample":     int(sample),
            "ra_mean":    float(ra_mean),
            "created_at": run.created_at,
        })

    if skipped:
        print(f"  Skipped {skipped} runs (missing RA_mean or config fields).")
    return pd.DataFrame(rows)


# ── Table construction ─────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent run per (attack, sdg, qi, sample)."""
    key    = ["attack", "sdg", "qi", "sample"]
    before = len(df)
    df = (df.sort_values("created_at", ascending=False)
            .drop_duplicates(subset=key)
            .drop(columns=["created_at"])
            .reset_index(drop=True))
    dropped = before - len(df)
    if dropped:
        print(f"  Deduplicated: dropped {dropped} duplicate runs "
              f"(kept most recent per sample).")
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Average RA_mean over samples; return pivot table (index=attack, columns=sdg)."""
    avg   = df.groupby(["attack", "sdg"])["ra_mean"].mean()
    pivot = avg.unstack("sdg")
    return pivot


def _ordered_cols(pivot: pd.DataFrame) -> list[str]:
    """SDG columns in preferred order. SDG_ORDER is authoritative — columns absent
    from it are silently dropped (so commenting out an entry excludes it)."""
    return [c for c in SDG_ORDER if c in pivot.columns]


def _ordered_attack_groups(pivot: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Return [(group_label, [attack, ...]), ...] preserving defined order."""
    seen   = set()
    result = []
    for label, attacks in ATTACK_GROUPS:
        present = [a for a in attacks if a in pivot.index]
        if present:
            result.append((label, present))
            seen.update(present)
    leftover = [a for a in pivot.index if a not in seen]
    if leftover:
        result.append(("Other", sorted(leftover)))
    return result


def _n_samples(df: pd.DataFrame, attack: str, sdg: str) -> int:
    sub = df[(df["attack"] == attack) & (df["sdg"] == sdg)]
    return len(sub)


# ── LaTeX generation ───────────────────────────────────────────────────────────

def to_latex(pivot: pd.DataFrame, df_raw: pd.DataFrame,
             group: str, qi: str, decimals: int = 3,
             dataset: str | None = None, size: int | None = None) -> str:
    cols      = _ordered_cols(pivot)
    atk_groups = _ordered_attack_groups(pivot)

    n_cols   = len(cols)
    col_spec = "l" + "r" * n_cols

    col_headers = [SDG_DISPLAY.get(c, c.replace("_", r"\_")) for c in cols]

    # Pre-compute caption components (caption goes above the table)
    any_flagged = any(
        _n_samples(df_raw, atk, col) < 5
        for _, _atks in atk_groups for atk in _atks
        for col in cols
        if not np.isnan(
            pivot.at[atk, col]
            if (atk in pivot.index and col in pivot.columns) else float("nan")
        )
    )
    dataset_display = DATASET_DISPLAY.get(dataset, dataset) if dataset else None
    effective_size = size
    if effective_size is None and "size" in df_raw.columns:
        sizes = df_raw["size"].dropna().astype(int)
        if not sizes.empty:
            effective_size = int(sizes.mode()[0])
    _ds  = f"\\textit{{{dataset_display}}}" if dataset_display else "unknown dataset"
    _sz  = f" ($N={effective_size:,}$ rows)" if effective_size else ""
    _flg = r" $^*$Fewer than 5 samples available for this cell." if any_flagged else ""
    caption = (
        r"Mean reconstruction accuracy (\texttt{RA\_mean}) averaged over disjoint training samples. "
        f"Dataset: {_ds}{_sz}. "
        f"QI variant: \\texttt{{{qi}}}.{_flg}"
    )

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  % Requires: \usepackage{booktabs,rotating,graphicx}")
    lines.append(f"  % WandB group: {group!r}   QI: {qi}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \label{tab:ra_mean}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    # Column headers — rotated for readability
    header_cells = ["Attack"] + [
        r"\rotatebox{60}{" + h + r"}" for h in col_headers
    ]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    first_group = True
    for _group_label, attacks in atk_groups:
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False

        for attack in attacks:
            display_name = ATTACK_DISPLAY.get(attack, attack.replace("_", r"\_"))
            cells = [display_name]
            for col in cols:
                val = (
                    pivot.at[attack, col]
                    if (attack in pivot.index and col in pivot.columns)
                    else float("nan")
                )
                n = _n_samples(df_raw, attack, col)
                fmt = f".{decimals}f"
                if np.isnan(val):
                    cells.append("---")
                elif n < 5:
                    cells.append(f"{val:{fmt}}$^*$")
                else:
                    cells.append(f"{val:{fmt}}")
            lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WandB → LaTeX RA_mean table.")
    parser.add_argument("--group", default=WANDB_GROUP,
                        help=f"WandB run group (default: {WANDB_GROUP!r}).")
    parser.add_argument("--qi",    default="QI1",
                        help="QI variant to include (default: QI1; use 'all' for all).")
    parser.add_argument("--out",      default=None,
                        help="Output .tex path (default: experiment_scripts/ra_table_<qi>.tex).")
    parser.add_argument("--decimals", type=int, default=3,
                        help="Decimal places for table values (default: 3).")
    parser.add_argument("--attacks", nargs="+", default=None, metavar="ATTACK",
                        help="Only include these attack methods (e.g. --attacks PartialMST PartialMSTBounded).")
    parser.add_argument("--dataset", default=None,
                        help="Filter to this dataset name (overrides DATASET constant at top of file; "
                             "use 'all' to disable filtering).")
    parser.add_argument("--from-csv", nargs="+", default=None, metavar="CSV",
                        help="Load results from one or more local sweep CSVs instead of querying WandB. "
                             "CSVs are concatenated and deduplicated. Expected columns: "
                             "sample, sdg, attack, qi, ra_mean.")
    parser.add_argument("--size", type=int, default=None,
                        help="When using --from-csv, filter to rows where the 'size' column equals this value "
                             "(e.g. --size 1000 or --size 100000).")
    parser.add_argument("--csv-dataset", type=str, default=None, metavar="DATASET",
                        help="When using --from-csv, filter to rows where the 'dataset' column equals this value. "
                             "CSVs without a 'dataset' column are always kept.")
    args = parser.parse_args()

    qi_filter = None if args.qi.lower() == "all" else args.qi

    if args.from_csv:
        frames = []
        for path in args.from_csv:
            frames.append(pd.read_csv(path))
            print(f"Loaded {path}: {len(frames[-1])} rows")
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = df[df["ra_mean"].notna()]
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} rows with missing ra_mean (failed runs).")
        if qi_filter:
            df = df[df["qi"] == qi_filter]
        if args.size is not None and "size" in df.columns:
            # Rows from CSVs that had no 'size' column get NaN after concat — keep them
            # (they came from single-dataset sweeps so they're implicitly the right size).
            df = df[(df["size"] == args.size) | df["size"].isna()]
        if args.csv_dataset is not None and "dataset" in df.columns:
            # Keep rows matching the requested dataset; also keep rows from CSVs that
            # had no 'dataset' column (they're implicitly single-dataset files).
            df = df[(df["dataset"] == args.csv_dataset) | df["dataset"].isna()]
        # Prefer 'label' over 'attack' when the CSV has a label column (e.g. new-attacks
        # sweep where multiple MarginalRF variants share the same attack_method).
        if "label" in df.columns:
            mask = df["label"].notna() & (df["label"].astype(str).str.strip() != "")
            df.loc[mask, "attack"] = df.loc[mask, "label"]
            df = df.drop(columns=["label"])
        if args.attacks:
            df = df[df["attack"].isin(args.attacks)]
        # Dedup: keep last (latest file wins for same key)
        key = ["attack", "sdg", "qi", "sample"]
        df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
        print(f"Combined: {len(df)} rows after dedup")
    else:
        dataset_filter = ("" if args.dataset and args.dataset.lower() == "all" else args.dataset)
        df = fetch_runs(args.group, qi_filter, attack_filter=args.attacks,
                        dataset_filter=dataset_filter)
        if not df.empty:
            df = deduplicate(df)

    if df.empty:
        print("No runs found.")
        return

    print(f"  {len(df)} runs  |  "
          f"{df['attack'].nunique()} attacks  |  "
          f"{df['sdg'].nunique()} SDG methods  |  "
          f"{df['sample'].nunique()} samples")
    print(f"  Attacks : {sorted(df['attack'].unique())}")
    print(f"  SDG     : {sorted(df['sdg'].unique())}")

    pivot  = build_pivot(df)
    dataset_label = args.dataset or None
    latex  = to_latex(pivot, df, args.group, args.qi, decimals=args.decimals,
                      dataset=dataset_label, size=args.size)

    size_suffix = f"_size{args.size}" if args.size is not None else ""
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / f"ra_table_{args.qi}{size_suffix}.tex"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX table written to: {out_path}")

    print("\n" + "─" * 70)
    print(latex)
    print("─" * 70)

    # Also print a plain-text pivot for a quick sanity check
    print("\nPivot (plain text preview):")
    cols = _ordered_cols(pivot)
    print(pivot[cols].to_string(float_format=lambda x: f"{x:.{args.decimals}f}"))


if __name__ == "__main__":
    main()
