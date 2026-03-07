#!/usr/bin/env python
"""
Generate LaTeX tables comparing LinearReconstruction vs ML attacks across
binary hidden features, from the linear_sweep WandB runs.

Table 1: Attacks × features, averaged over all SDG methods and samples.
         Shows RA_train (Tr) and RA_nontrain (NT) side by side.
         Best attack per column in bold. LinearReconstruction row shaded.

Table 2: Same columns, one row group per SDG method, averaged over samples.
         Rendered as a longtable (multi-page) for completeness.

Required LaTeX packages:
  \\usepackage{booktabs, longtable, colortbl, xcolor, graphicx, caption}
  \\definecolor{linrecon}{gray}{0.88}

Usage:
    conda activate recon_
    python experiment_scripts/linear_sweep_to_latex.py
    python experiment_scripts/linear_sweep_to_latex.py --decimals 1
    python experiment_scripts/linear_sweep_to_latex.py \\
        --out-table1 tab_summary.tex --out-table2 tab_by_sdg.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


# ── WandB settings ────────────────────────────────────────────────────────────

WANDB_PROJECT = "tabular-reconstruction-attacks"


# ── Column specification ───────────────────────────────────────────────────────
# Each entry: (dataset_key, size, qi_variant, feature_label, group_label)
#   dataset_key   — used to build the WandB group name
#   size          — sample size (1000 or 10000)
#   qi_variant    — stored in run.config["qi"]
#   feature_label — short display name for the hidden feature
#   group_label   — top-level \multicolumn header (consecutive equal = one span)

COLUMNS = [
    ("adult",        1_000,  "QI_linear",             "income",   "Adult 1k"),
    ("adult",        1_000,  "QI_binary_sex",          "sex",      "Adult 1k"),
    ("adult",        10_000, "QI_linear_lowcard",      "income",   "Adult 10k"),
    ("adult",        10_000, "QI_binary_sex_lowcard",  "sex",      "Adult 10k"),
    ("arizona",      1_000,  "QI_binary_SEX_lowcard",  "SEX",      "Arizona 1k"),
    ("cdc_diabetes", 1_000,  "QI_linear",              "Diabetes", "CDC 1k"),
    ("cdc_diabetes", 1_000,  "QI_binary_HighBP",       "HighBP",   "CDC 1k"),
    ("cdc_diabetes", 1_000,  "QI_binary_Stroke",       "Stroke",   "CDC 1k"),
]


# ── Attack display ─────────────────────────────────────────────────────────────

ATTACKS = ["Random", "KNN", "NaiveBayes", "RandomForest", "MLP", "LinearReconstruction"]

ATTACK_LABELS: dict[str, str] = {
    "Random":               "Random",
    "KNN":                  r"\textsc{knn}",
    "NaiveBayes":           "Naive Bayes",
    "RandomForest":         "Random Forest",
    "MLP":                  r"\textsc{mlp}",
    "LinearReconstruction": "Linear Recon.",
}


# ── SDG display ────────────────────────────────────────────────────────────────

SDG_ORDER = [
    "MST_eps1", "MST_eps10", "MST_eps100", "MST_eps1000",
    "AIM_eps1",
    "ARF", "TVAE", "CTGAN", "Synthpop", "TabDDPM",
    "RankSwap", "CellSuppression",
]

SDG_LABELS: dict[str, str] = {
    "MST_eps1":        r"MST $\varepsilon{=}1$",
    "MST_eps10":       r"MST $\varepsilon{=}10$",
    "MST_eps100":      r"MST $\varepsilon{=}100$",
    "MST_eps1000":     r"MST $\varepsilon{=}1000$",
    "AIM_eps1":        r"AIM $\varepsilon{=}1$",
    "ARF":             "ARF",
    "TVAE":            "TVAE",
    "CTGAN":           "CTGAN",
    "Synthpop":        "Synthpop",
    "TabDDPM":         "TabDDPM",
    "RankSwap":        "RankSwap",
    "CellSuppression": "Cell Supp.",
}


# ── WandB fetch ────────────────────────────────────────────────────────────────

def _sdg_label(method: str, params: dict | None) -> str:
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{float(eps):g}" if eps is not None else method


def fetch_runs() -> pd.DataFrame:
    """
    Fetch all linear-sweep runs from WandB across every (dataset_key, size)
    combination referenced in COLUMNS.  Returns a flat DataFrame with one row
    per completed run, deduplicated to the most-recent per unique experiment.
    """
    api    = wandb.Api(timeout=120)
    entity = api.default_entity
    path   = f"{entity}/{WANDB_PROJECT}"

    groups_needed = sorted({
        f"linear-sweep-{dk}-{sz}"
        for dk, sz, *_ in COLUMNS
    })

    rows = []
    for group in groups_needed:
        print(f"  Querying group: {group!r} ...")
        runs = api.runs(path, filters={"group": group})
        n = 0
        for run in runs:
            cfg  = run.config
            summ = run.summary

            attack     = cfg.get("attack_method")
            sdg_method = cfg.get("sdg_method")
            sdg_params = cfg.get("sdg_params") or {}
            qi         = cfg.get("qi")
            sample     = cfg.get("sample_idx")
            dk         = cfg.get("dataset_key")
            sz         = cfg.get("size")

            train_ra    = summ.get("RA_train_mean")
            nontrain_ra = summ.get("RA_nontraining_mean")

            if None in (attack, sdg_method, qi, sample, dk, sz) or train_ra is None:
                continue

            rows.append({
                "dataset_key": str(dk),
                "size":        int(sz),
                "qi":          str(qi),
                "attack":      str(attack),
                "sdg":         _sdg_label(sdg_method, sdg_params),
                "sample":      int(sample),
                "train_ra":    float(train_ra),
                "nontrain_ra": float(nontrain_ra) if nontrain_ra is not None else float("nan"),
                "created_at":  run.created_at,
            })
            n += 1
        print(f"    → {n} valid runs")

    if not rows:
        return pd.DataFrame(columns=["dataset_key", "size", "qi", "attack",
                                     "sdg", "sample", "train_ra", "nontrain_ra"])

    df = pd.DataFrame(rows)
    # Keep most recent run per unique experiment
    df = (df.sort_values("created_at")
            .drop_duplicates(
                subset=["dataset_key", "size", "qi", "attack", "sdg", "sample"],
                keep="last",
            )
            .reset_index(drop=True))
    return df


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _fmt(val: float, d: int) -> str:
    return "--" if np.isnan(val) else f"{val:.{d}f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _build_header(columns: list) -> tuple[str, str, str, str]:
    """
    Build tabular column-spec and three header rows from the COLUMNS spec.

    Returns (col_spec, row1, row2, row3) where each row is a complete table
    row string including the trailing \\\\ and any \\cmidrule lines.
    """
    # Group consecutive columns by group_label
    groups: list[tuple[str, list]] = []
    for col in columns:
        glabel = col[4]
        if groups and groups[-1][0] == glabel:
            groups[-1][1].append(col)
        else:
            groups.append((glabel, [col]))

    col_spec = "l" + " c" * (len(columns) * 2)

    # Row 1: dataset group \multicolumns
    r1_cells = [""]
    cmidrule1 = []
    ci = 2
    for glabel, gcols in groups:
        span = len(gcols) * 2
        r1_cells.append(f"\\multicolumn{{{span}}}{{c}}{{{glabel}}}")
        cmidrule1.append(f"\\cmidrule(lr){{{ci}-{ci+span-1}}}")
        ci += span
    row1 = " & ".join(r1_cells) + r" \\" + "\n" + " ".join(cmidrule1)

    # Row 2: feature \multicolumns
    r2_cells = [""]
    cmidrule2 = []
    ci = 2
    for dk, sz, qi, feat, glabel in columns:
        r2_cells.append(f"\\multicolumn{{2}}{{c}}{{{feat}}}")
        cmidrule2.append(f"\\cmidrule(lr){{{ci}-{ci+1}}}")
        ci += 2
    row2 = " & ".join(r2_cells) + r" \\" + "\n" + " ".join(cmidrule2)

    # Row 3: Tr / NT labels
    r3_cells = ["Attack"] + ["Tr", "NT"] * len(columns)
    row3 = " & ".join(r3_cells) + r" \\"

    return col_spec, row1, row2, row3


def _cell_vals(
    df: pd.DataFrame,
    dataset_key: str,
    size: int,
    qi: str,
    attack: str,
    sdg: str | None = None,
) -> tuple[float, float]:
    """
    Average train_ra and nontrain_ra for a given (dataset_key, size, qi, attack),
    optionally restricted to a single SDG method.
    """
    mask = (
        (df["dataset_key"] == dataset_key)
        & (df["size"] == size)
        & (df["qi"] == qi)
        & (df["attack"] == attack)
    )
    if sdg is not None:
        mask &= df["sdg"] == sdg
    sub = df[mask]
    if sub.empty:
        return float("nan"), float("nan")
    return float(sub["train_ra"].mean()), float(sub["nontrain_ra"].mean(skipna=True))


def _col_max(df, col_keys, attack_list, sdg=None):
    """Per-column best train_ra across attacks (for bolding)."""
    maxes = {}
    for dk, sz, qi in col_keys:
        vals = [_cell_vals(df, dk, sz, qi, atk, sdg)[0] for atk in attack_list]
        finite = [v for v in vals if not np.isnan(v)]
        maxes[(dk, sz, qi)] = max(finite) if finite else float("nan")
    return maxes


def _data_row(df, col_keys, attack, decimals, col_maxes, sdg=None, shade=False):
    """Format a single attack row."""
    lbl = ATTACK_LABELS.get(attack, attack)
    cells = [lbl]
    for dk, sz, qi in col_keys:
        tr, nt = _cell_vals(df, dk, sz, qi, attack, sdg)
        tr_s = _fmt(tr, decimals)
        nt_s = _fmt(nt, decimals)
        if not np.isnan(tr) and not np.isnan(col_maxes[(dk, sz, qi)]):
            if abs(tr - col_maxes[(dk, sz, qi)]) < 1e-9:
                tr_s = _bold(tr_s)
        cells.append(tr_s)
        cells.append(nt_s)
    row = " & ".join(cells) + r" \\"
    if shade:
        row = r"\rowcolor{linrecon}" + row
    return row


def _avg_row(df, col_keys, attack_list, decimals, sdg=None, label=r"\textit{Avg.}"):
    """Macro-average row across all attacks."""
    cells = [label]
    for dk, sz, qi in col_keys:
        trs = [_cell_vals(df, dk, sz, qi, atk, sdg)[0] for atk in attack_list]
        nts = [_cell_vals(df, dk, sz, qi, atk, sdg)[1] for atk in attack_list]
        tr_fin = [v for v in trs if not np.isnan(v)]
        nt_fin = [v for v in nts if not np.isnan(v)]
        cells.append(r"\textit{" + _fmt(np.mean(tr_fin) if tr_fin else float("nan"), decimals) + "}")
        cells.append(r"\textit{" + _fmt(np.mean(nt_fin) if nt_fin else float("nan"), decimals) + "}")
    return " & ".join(cells) + r" \\"


# ── Table 1: averaged over all SDG + samples ──────────────────────────────────

def build_table1(df: pd.DataFrame, decimals: int = 2) -> str:
    col_spec, row1, row2, row3 = _build_header(COLUMNS)
    col_keys = [(dk, sz, qi) for dk, sz, qi, *_ in COLUMNS]
    maxes    = _col_max(df, col_keys, ATTACKS)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{%",
        r"  Reconstruction accuracy on binary hidden features, averaged over all",
        r"  SDG methods and 5 samples.",
        r"  \textbf{Tr} = training-set targets;",
        r"  \textbf{NT} = held-out (non-training) targets.",
        r"  Best \textbf{Tr} per column in \textbf{bold}.",
        r"  \colorbox{linrecon}{Shaded} row = Linear Recon.\ (proposed SOTA).",
        r"}",
        r"\label{tab:linear_sweep_summary}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        row1,
        row2,
        row3,
        r"\midrule",
    ]

    for atk in ATTACKS:
        shade = (atk == "LinearReconstruction")
        lines.append(_data_row(df, col_keys, atk, decimals, maxes, shade=shade))

    lines += [
        r"\midrule",
        _avg_row(df, col_keys, ATTACKS, decimals),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table 2: per SDG method, averaged over samples ────────────────────────────

def build_table2(df: pd.DataFrame, decimals: int = 2) -> str:
    col_spec, row1, row2, row3 = _build_header(COLUMNS)
    col_keys    = [(dk, sz, qi) for dk, sz, qi, *_ in COLUMNS]
    n_total     = 1 + len(COLUMNS) * 2

    # Determine which SDGs have data, in preferred display order
    sdgs_present = set(df["sdg"].unique())
    sdg_order    = [s for s in SDG_ORDER if s in sdgs_present]
    # Append anything not in SDG_ORDER
    for s in sorted(sdgs_present - set(sdg_order)):
        sdg_order.append(s)

    continued = f"\\multicolumn{{{n_total}}}{{r}}{{\\textit{{Continued on next page}}}}"

    lines = [
        f"\\begin{{longtable}}{{{col_spec}}}",
        r"\caption{%",
        r"  Reconstruction accuracy by SDG method, averaged over 5 samples.",
        r"  \textbf{Tr} = training targets; \textbf{NT} = non-training targets.",
        r"  Best \textbf{Tr} per (SDG, column) in \textbf{bold}.",
        r"} \label{tab:linear_sweep_by_sdg} \\",
        r"\toprule",
        row1,
        row2,
        row3,
        r"\midrule \endfirsthead",
        r"\toprule",
        row1,
        row2,
        row3,
        r"\midrule \endhead",
        r"\midrule",
        continued + r" \\",
        r"\endfoot",
        r"\bottomrule \endlastfoot",
    ]

    for i, sdg in enumerate(sdg_order):
        sdg_lbl = SDG_LABELS.get(sdg, sdg)
        maxes   = _col_max(df, col_keys, ATTACKS, sdg=sdg)

        if i > 0:
            lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{{n_total}}}{{l}}"
            r"{\textbf{" + sdg_lbl + r"}} \\"
        )

        for atk in ATTACKS:
            shade = (atk == "LinearReconstruction")
            lines.append(_data_row(df, col_keys, atk, decimals, maxes, sdg=sdg, shade=shade))

        lines.append(_avg_row(df, col_keys, ATTACKS, decimals, sdg=sdg))

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from the linear_sweep WandB runs."
    )
    parser.add_argument("--project",    default=WANDB_PROJECT,
                        help="WandB project name.")
    parser.add_argument("--out-table1", default="linear_sweep_table1.tex",
                        help="Output file for Table 1 (summary, averaged over SDG).")
    parser.add_argument("--out-table2", default="linear_sweep_table2.tex",
                        help="Output file for Table 2 (per SDG, longtable).")
    parser.add_argument("--decimals",   type=int, default=2,
                        help="Decimal places in cells (default 2).")
    args = parser.parse_args()

    global WANDB_PROJECT
    WANDB_PROJECT = args.project

    print("Fetching WandB runs ...")
    df = fetch_runs()
    print(f"Total rows after deduplication: {len(df)}")
    if df.empty:
        print("No data found — check group names and WandB project.")
        return

    # Quick diagnostic: show which (dataset_key, size, qi, attack, sdg) have data
    present = df.groupby(["dataset_key", "size", "qi", "attack", "sdg"]).size()
    print(f"\nData coverage ({len(present)} combos with ≥1 sample):")
    for (dk, sz, qi, atk, sdg), n in present.items():
        print(f"  {dk:<14} {sz:>6}  {qi:<30}  {atk:<22}  {sdg:<16}  n={n}")

    t1 = build_table1(df, decimals=args.decimals)
    Path(args.out_table1).write_text(t1)
    print(f"\nTable 1 → {args.out_table1}")

    t2 = build_table2(df, decimals=args.decimals)
    Path(args.out_table2).write_text(t2)
    print(f"Table 2 → {args.out_table2}")

    print("\nAdd to LaTeX preamble:")
    print(r"  \usepackage{booktabs, longtable, colortbl, xcolor, graphicx, caption}")
    print(r"  \definecolor{linrecon}{gray}{0.88}")


if __name__ == "__main__":
    main()
