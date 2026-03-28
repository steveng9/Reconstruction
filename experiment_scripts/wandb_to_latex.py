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
    "MST_eps1",
    "MST_eps10",
    "MST_eps100",
    "MST_eps1000",
    "AIM_eps1",
    "AIM_eps10",
    "AIM_eps100",
    "TVAE",
    "CTGAN",
    "ARF",
    "TabDDPM",
]

# Attack groups control row ordering and \midrule placement.
# Attacks absent from the data are silently skipped.
ATTACK_GROUPS = [
    ("Baselines",      ["Mode", "Random", "MeasureDeid"]),
    ("ML Classifiers", ["KNN", "NaiveBayes", "LogisticRegression", "SVM",
                        "RandomForest", "LightGBM"]),
    ("Neural",         ["MLP", "Attention", "AttentionAutoregressive"]),
    ("Partial SDG",      ["TabDDPM", "ConditionedRePaint", "RePaint", "PartialMST", "PartialMSTIndependent", "PartialMSTBounded"]),
    ("SOTA",           ["LinearReconstruction"]),
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
    "RePaint":                 "RePaint",
    "LinearReconstruction":    "Linear Recon.",
    "PartialMST":              "MST",
    "PartialMSTIndependent":   "MST (1 ft./time)",
    "PartialMSTBounded":       "MST, k=3",
}

SDG_DISPLAY: dict[str, str] = {
    "RankSwap":      "RankSwap",
    "CellSuppression": "Cell Supp.",
    "Synthpop":      "Synthpop",
    "MST_eps0.1":    r"MST $\varepsilon{=}0.1$",
    "MST_eps1":      r"MST $\varepsilon{=}1$",
    "MST_eps10":     r"MST $\varepsilon{=}10$",
    "MST_eps100":    r"MST $\varepsilon{=}100$",
    "MST_eps1000":   r"MST $\varepsilon{=}1000$",
    "AIM_eps1":      r"AIM $\varepsilon{=}1$",
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

        attack     = cfg.get("attack_method")
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
    """SDG columns in preferred order, with any extras appended alphabetically."""
    ordered = [c for c in SDG_ORDER if c in pivot.columns]
    extra   = sorted(c for c in pivot.columns if c not in ordered)
    return ordered + extra


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
             group: str, qi: str, decimals: int = 3) -> str:
    cols      = _ordered_cols(pivot)
    atk_groups = _ordered_attack_groups(pivot)

    n_cols   = len(cols)
    col_spec = "l" + "r" * n_cols

    col_headers = [SDG_DISPLAY.get(c, c.replace("_", r"\_")) for c in cols]

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  % Requires: \usepackage{booktabs,rotating,graphicx}")
    lines.append(f"  % WandB group: {group!r}   QI: {qi}")
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
                    # Flag cells averaged over fewer than 5 samples
                    cells.append(f"{val:{fmt}}$^*$")
                else:
                    cells.append(f"{val:{fmt}}")
            lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")
    lines.append(r"  \caption{Mean reconstruction accuracy (\texttt{RA\_mean}) "
                 r"averaged over 5 disjoint training samples.")
    lines.append(f"           WandB group: \\textit{{{group}}}. QI variant: {qi}.")
    lines.append(r"           $^*$Fewer than 5 samples available for this cell.}")
    lines.append(r"  \label{tab:ra_mean}")
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
    args = parser.parse_args()

    qi_filter = None if args.qi.lower() == "all" else args.qi
    dataset_filter = ("" if args.dataset and args.dataset.lower() == "all" else args.dataset)

    df = fetch_runs(args.group, qi_filter, attack_filter=args.attacks,
                    dataset_filter=dataset_filter)
    if not df.empty:
        df = deduplicate(df)
    if df.empty:
        print("No runs found — check group name and `wandb login`.")
        return

    print(f"  {len(df)} runs  |  "
          f"{df['attack'].nunique()} attacks  |  "
          f"{df['sdg'].nunique()} SDG methods  |  "
          f"{df['sample'].nunique()} samples")
    print(f"  Attacks : {sorted(df['attack'].unique())}")
    print(f"  SDG     : {sorted(df['sdg'].unique())}")

    pivot  = build_pivot(df)
    latex  = to_latex(pivot, df, args.group, args.qi, decimals=args.decimals)

    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / f"ra_table_{args.qi}.tex"
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
