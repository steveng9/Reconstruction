#!/usr/bin/env python
"""
Convert synth quality evaluation results (CSV) to a LaTeX table.

Rows    = SDG methods grouped by type (DP / statistical / deep generative)
Columns = all (dataset, size) combinations, grouped by dataset under a shared
          multi-column header.  Sizes are shown as separate sub-columns.

          Example column layout:
              |←── Adult ───→|  Ariz. |  SBO  |←─ CDC ─→|  Calif. |
              | 1k  10k  20k |  10k   |  1k   |  1k  100k|  1k     |

Usage:
    python experiment_scripts/synth_quality_to_latex.py results.csv
    python experiment_scripts/synth_quality_to_latex.py results.csv --metric mean_tvd
    python experiment_scripts/synth_quality_to_latex.py results.csv --metric tstr_ratio --out table.tex

Available --metric values (from the evaluation CSV):
  Fidelity (lower is better):
    mean_tvd            Mean TVD across categorical columns
    mean_jsd            Mean Jensen-Shannon divergence (categorical)
    pairwise_tvd        Mean pairwise 2-way joint TVD (categorical)
    mean_mean_err_pct   Mean relative mean error (continuous)
    mean_std_err_pct    Mean relative std error (continuous)
    mean_wasserstein    Mean normalized Wasserstein distance (continuous)
    corr_diff           Mean absolute correlation matrix difference
    prop_score          Propensity AUC (0.5=indistinguishable, 1.0=perfect)

  Fidelity (higher is better):
    cat_coverage        Fraction of training categories present in synth
    sdv_col_shapes      SDV column shapes quality score
    sdv_col_pairs       SDV column pair trends quality score

  Utility (higher is better):
    tstr_score          TSTR F1-macro (classification) or R² (regression)
    trtr_score          TRTR cross-val baseline (upper bound for TSTR)
    tstr_ratio          TSTR / TRTR (utility retention, 1.0 = full utility)

Default metric: tstr_ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Display configuration ──────────────────────────────────────────────────────

# SDG methods to include (in row order). Absent methods are silently skipped.
SDG_ORDER = [
    "MST_eps0.1",
    "MST_eps0.3",
    "MST_eps1",
    "MST_eps3",
    "MST_eps10",
    "MST_eps30",
    "MST_eps100",
    "MST_eps1000",
    "AIM_eps0.1",
    "AIM_eps0.3",
    "AIM_eps1",
    "AIM_eps3",
    "AIM_eps10",
    "AIM_eps30",
    "AIM_eps100",
    "RankSwap",
    "CellSuppression",
    "Synthpop",
    "TVAE",
    "CTGAN",
    "ARF",
    "TabDDPM",
    "~train_baseline",
]

# Groups control \midrule placement between them.
SDG_GROUPS = [
    ("Differentially private",  ["MST_eps0.1", "MST_eps0.3", "MST_eps1",
                                  "MST_eps3", "MST_eps10", "MST_eps30",
                                  "MST_eps100", "MST_eps1000",
                                  "AIM_eps0.1", "AIM_eps0.3", "AIM_eps1",
                                  "AIM_eps3", "AIM_eps10", "AIM_eps30",
                                  "AIM_eps100"]),
    ("Non-DP statistical",      ["RankSwap", "CellSuppression", "Synthpop"]),
    ("Deep generative",         ["TVAE", "CTGAN", "ARF", "TabDDPM"]),
    ("Baseline",                ["~train_baseline"]),
]

SDG_DISPLAY: dict[str, str] = {
    "MST_eps0.1":      r"MST $(\varepsilon{=}0.1)$",
    "MST_eps0.3":      r"MST $(\varepsilon{=}0.3)$",
    "MST_eps1":        r"MST $(\varepsilon{=}1)$",
    "MST_eps3":        r"MST $(\varepsilon{=}3)$",
    "MST_eps10":       r"MST $(\varepsilon{=}10)$",
    "MST_eps30":       r"MST $(\varepsilon{=}30)$",
    "MST_eps100":      r"MST $(\varepsilon{=}100)$",
    "MST_eps1000":     r"MST $(\varepsilon{=}1000)$",
    "AIM_eps0.1":        r"AIM $(\varepsilon{=}0.1)$",
    "AIM_eps0.3":        r"AIM $(\varepsilon{=}0.3)$",
    "AIM_eps1":        r"AIM $(\varepsilon{=}1)$",
    "AIM_eps3":        r"AIM $(\varepsilon{=}3)$",
    "AIM_eps10":       r"AIM $(\varepsilon{=}10)$",
    "AIM_eps30":        r"AIM $(\varepsilon{=}30)$",
    "RankSwap":        "RankSwap",
    "CellSuppression": r"Cell Supp.",
    "Synthpop":        "Synthpop",
    "TVAE":            "TVAE",
    "CTGAN":           "CTGAN",
    "ARF":             "ARF",
    "TabDDPM":         "TabDDPM",
    "~train_baseline": r"\textit{Train--Train}",
}

# ── Dataset/size column layout ─────────────────────────────────────────────────

# Ordered list of columns: (dataset_name, size_dir, short_size_label)
# dataset_name and size_dir must match the CSV values exactly.
# short_size_label appears in the second header row (e.g. "1k", "10k").
COLUMN_ORDER: list[tuple[str, str, str]] = [
    ("adult",            "size_1000",          "1k"),
    ("adult",            "size_10000",         "10k"),
    ("adult",            "size_20000",         "20k"),
    ("nist_arizona_data","size_10000_25feat",  "10k"),
    ("nist_sbo",         "size_1000",          "1k"),
    ("cdc_diabetes",     "size_1000",          "1k"),
    ("cdc_diabetes",     "size_100000",        "100k"),
    ("california",       "size_1000",          "1k"),
]

# Top-level dataset labels (for \multicolumn header row)
DATASET_DISPLAY: dict[str, str] = {
    "adult":             "Adult",
    "nist_arizona_data": "Arizona",
    "nist_sbo":          "SBO",
    "cdc_diabetes":      r"CDC\,Diabetes",
    "california":        "California",
}

# ── Metric metadata ────────────────────────────────────────────────────────────

METRIC_META: dict[str, dict] = {
    "mean_tvd":          {"dir": "↓", "latex": r"mean TVD"},
    "mean_jsd":          {"dir": "↓", "latex": r"mean JSD"},
    "pairwise_tvd":      {"dir": "↓", "latex": r"pairwise TVD"},
    "cat_coverage":      {"dir": "↑", "latex": r"cat.\ coverage"},
    "mean_mean_err_pct": {"dir": "↓", "latex": r"mean err.\%"},
    "mean_std_err_pct":  {"dir": "↓", "latex": r"std err.\%"},
    "mean_wasserstein":  {"dir": "↓", "latex": r"Wasserstein"},
    "corr_diff":         {"dir": "↓", "latex": r"corr.\ diff"},
    "sdv_col_shapes":    {"dir": "↑", "latex": r"SDV shapes"},
    "sdv_col_pairs":     {"dir": "↑", "latex": r"SDV pairs"},
    "prop_score":        {"dir": "↓", "latex": r"prop.\ AUC"},
    "tstr_score":        {"dir": "↑", "latex": r"TSTR score"},
    "trtr_score":        {"dir": "↑", "latex": r"TRTR score"},
    "tstr_ratio":        {"dir": "↑", "latex": r"TSTR ratio"},
}

MIN_SAMPLES_FLAG = 4   # cells averaged from fewer samples get a $^*$ marker


# ── Data loading & aggregation ─────────────────────────────────────────────────

def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"dataset", "size_dir", "method"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Compound column key used throughout
    df["col_key"] = df["dataset"] + "|" + df["size_dir"]
    return df


def build_pivot(df: pd.DataFrame, metric: str,
                col_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean pivot, count pivot).  Index = method, columns = col_key."""
    if metric not in df.columns:
        available = [c for c in df.columns
                     if c not in {"dataset", "size_dir", "sample", "method",
                                  "n_rows_train", "n_rows_synth", "col_key"}]
        raise ValueError(
            f"Metric {metric!r} not found in CSV.\n"
            f"Available: {available}"
        )

    sub  = df[df["col_key"].isin(col_keys)].copy()
    agg  = sub.groupby(["col_key", "method"])[metric].agg(["mean", "count"])
    means = agg["mean"].unstack("col_key")
    cnts  = agg["count"].unstack("col_key")
    return means, cnts


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _ordered_methods(pivot: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Group and order SDG methods; return [(group_label, [method, ...]), ...]."""
    seen, result = set(), []
    for label, methods in SDG_GROUPS:
        present = [m for m in methods if m in pivot.index]
        if present:
            result.append((label, present))
            seen.update(present)
    leftover = [m for m in SDG_ORDER if m in pivot.index and m not in seen]
    leftover += sorted(m for m in pivot.index if m not in seen
                       and m not in {mm for _, ms in result for mm in ms})
    if leftover:
        result.append(("Other", leftover))
    return result


def _fmt_cell(val: float, count: int, decimals: int) -> str:
    if np.isnan(val):
        return "--"
    s    = f"{val:.{decimals}f}"
    flag = r"$^*$" if count < MIN_SAMPLES_FLAG else ""
    return s + flag


def _best_per_col(pivot_mean: pd.DataFrame, col_keys: list[str],
                  direction: str) -> dict[str, str]:
    """Return {col_key: best_method} for bold highlighting."""
    best: dict[str, str] = {}
    for ck in col_keys:
        if ck not in pivot_mean.columns:
            continue
        col = pivot_mean[ck].dropna()
        if col.empty:
            continue
        best[ck] = col.idxmin() if direction == "↓" else col.idxmax()
    return best


# ── Multi-level column header builder ─────────────────────────────────────────

def _build_header_rows(active_cols: list[tuple[str, str, str]],
                       row_label_col: int = 1) -> tuple[str, str, str]:
    """Return three LaTeX lines: top-level dataset header, size sub-header, cmidrule line.

    active_cols: subset of COLUMN_ORDER that are actually present in the pivot.
    row_label_col: number of left-most label columns (always 1: the SDG method).
    """
    # Group consecutive entries by dataset
    groups: list[tuple[str, list[tuple[str, str, str]]]] = []
    for entry in active_cols:
        ds = entry[0]
        if groups and groups[-1][0] == ds:
            groups[-1][1].append(entry)
        else:
            groups.append((ds, [entry]))

    # Build top header row: \multicolumn{n}{c}{Dataset} or plain cell if n==1
    top_cells: list[str] = []
    cmidrules: list[str] = []
    col_idx = row_label_col + 1   # 1-indexed LaTeX column number

    for ds, entries in groups:
        n = len(entries)
        label = DATASET_DISPLAY.get(ds, ds)
        if n == 1:
            top_cells.append(r"\multicolumn{1}{c}{" + label + r"}")
        else:
            top_cells.append(r"\multicolumn{" + str(n) + r"}{c}{" + label + r"}")
        lo, hi = col_idx, col_idx + n - 1
        cmidrules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
        col_idx += n

    # Build sub-header row: size labels
    sub_cells = [entry[2] for entry in active_cols]   # e.g. "1k", "10k"

    row1 = "    " + r"\textbf{SDG method}" + " & " + " & ".join(top_cells) + r" \\"
    row2 = "    " + " & " + " & ".join(sub_cells) + r" \\"
    row3 = "    " + " ".join(cmidrules)

    return row1, row2, row3


# ── LaTeX table ────────────────────────────────────────────────────────────────

def to_latex(pivot_mean: pd.DataFrame, pivot_cnt: pd.DataFrame,
             metric: str, decimals: int, bold_best: bool,
             active_cols: list[tuple[str, str, str]],
             source_note: str) -> str:

    meta_m    = METRIC_META.get(metric, {"dir": "?", "latex": metric})
    direction = meta_m["dir"]

    # Only keep columns present in the pivot
    active_cols = [(ds, sz, lbl) for (ds, sz, lbl) in active_cols
                   if f"{ds}|{sz}" in pivot_mean.columns]
    col_keys = [f"{ds}|{sz}" for ds, sz, _ in active_cols]

    best_map = _best_per_col(pivot_mean, col_keys, direction) if bold_best else {}

    n_data_cols = len(col_keys)
    col_spec    = "l" + "r" * n_data_cols

    atk_groups = _ordered_methods(pivot_mean)

    header_row1, header_row2, cmidrule_row = _build_header_rows(active_cols)

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs,rotating,graphicx,multirow}")
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    # Two-row column header
    lines.append(header_row1)
    lines.append(cmidrule_row)
    lines.append(header_row2)
    lines.append(r"    \midrule")

    first_group = True
    for _grp_label, methods in atk_groups:
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False

        for method in methods:
            if method not in pivot_mean.index:
                continue
            display = SDG_DISPLAY.get(method, method.replace("_", r"\_"))
            cells   = [display]
            for ck in col_keys:
                val   = float(pivot_mean.at[method, ck]) if ck in pivot_mean.columns else float("nan")
                cnt_val = pivot_cnt.at[method, ck] if ck in pivot_cnt.columns else np.nan
                count = 0 if (cnt_val is None or (isinstance(cnt_val, float) and np.isnan(cnt_val))) else int(cnt_val)
                cell  = _fmt_cell(val, count, decimals)
                if bold_best and best_map.get(ck) == method and not np.isnan(val):
                    cell = r"\textbf{" + cell + r"}"
                cells.append(cell)
            lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")

    dir_str   = "(lower is better)" if direction == "↓" else "(higher is better)"
    ml_note   = (r"For classification datasets TSTR/TRTR use F1-macro; "
                 r"for California (regression) they use R$^2$.  "
                 if "tstr" in metric else "")
    lines.append(
        r"  \caption{Synthetic data quality: \textbf{" + meta_m["latex"] + r"} "
        + dir_str + r".  "
        + r"Rows = SDG methods; column groups = datasets, sub-columns = training-set size.  "
        + r"Values averaged over all available disjoint samples "
        + r"(5--10 depending on dataset/size).  "
        + ml_note
        + r"$^*$ fewer than " + str(MIN_SAMPLES_FLAG) + r" samples.  "
        + r"\textit{Train--Train}: one real sample evaluated against another "
        + r"(same-distribution upper bound).}"
    )
    if source_note:
        lines.append(f"  % Source: {source_note}")
    lines.append(r"  \label{tab:synth_quality_" + metric + r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert synth_quality_results CSV to a grouped LaTeX booktabs table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("results_csv",
                        help="Path to synth_quality_results_*.csv")
    parser.add_argument("--metric",   default="tstr_ratio",
                        help="Metric column to tabulate (default: tstr_ratio)")
    parser.add_argument("--out",      default=None,
                        help="Output .tex path (default: <csv_dir>/synth_quality_<metric>.tex)")
    parser.add_argument("--decimals", type=int, default=3,
                        help="Decimal places (default: 3)")
    parser.add_argument("--no-bold",  action="store_true",
                        help="Disable bold-best-per-column highlighting")
    parser.add_argument("--dataset",  nargs="*",
                        help="Restrict to these dataset names")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = load_results(csv_path)

    # Determine which (dataset, size) columns to include
    active_cols = COLUMN_ORDER
    if args.dataset:
        active_cols = [(ds, sz, lbl) for ds, sz, lbl in active_cols
                       if ds in args.dataset]
    col_keys = [f"{ds}|{sz}" for ds, sz, _ in active_cols]

    pivot_mean, pivot_cnt = build_pivot(df, args.metric, col_keys)

    # Reorder rows into SDG_ORDER (intersect with what's present)
    present_methods = [m for m in SDG_ORDER if m in pivot_mean.index]
    others = [m for m in pivot_mean.index if m not in present_methods]
    pivot_mean = pivot_mean.loc[present_methods + others]
    pivot_cnt  = pivot_cnt.loc[present_methods + others]

    print(f"Metric     : {args.metric}  ({METRIC_META.get(args.metric, {}).get('dir', '?')})")
    print(f"Columns    : {col_keys}")
    print(f"Methods    : {list(pivot_mean.index)}")
    print(f"Samples/cell (min/max): "
          f"{int(pivot_cnt.values[~np.isnan(pivot_cnt.values.astype(float))].min()) if pivot_cnt.notna().any().any() else 0}"
          f" / "
          f"{int(pivot_cnt.values[~np.isnan(pivot_cnt.values.astype(float))].max()) if pivot_cnt.notna().any().any() else 0}")

    latex = to_latex(
        pivot_mean, pivot_cnt,
        metric=args.metric,
        decimals=args.decimals,
        bold_best=not args.no_bold,
        active_cols=active_cols,
        source_note=str(csv_path),
    )

    out_path = (Path(args.out) if args.out else
                csv_path.parent / f"synth_quality_{args.metric}.tex")
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX table written to: {out_path}")

    print("\n" + "─" * 80)
    print(latex)
    print("─" * 80)

    # Plain-text pivot preview
    print(f"\n── Plain-text pivot ({args.metric}) ──")
    # Build readable column labels: "Dataset (size)"
    col_labels = {
        f"{ds}|{sz}": f"{DATASET_DISPLAY.get(ds, ds)} ({lbl})"
        for ds, sz, lbl in COLUMN_ORDER
    }
    disp = pivot_mean[[ck for ck in col_keys if ck in pivot_mean.columns]].copy()
    disp.columns = [col_labels.get(c, c) for c in disp.columns]
    disp.index   = [SDG_DISPLAY.get(m, m) for m in disp.index]
    fmt = lambda x: f"{x:.{args.decimals}f}" if not np.isnan(x) else "--"
    print(disp.to_string(float_format=fmt))


if __name__ == "__main__":
    main()
