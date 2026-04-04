#!/usr/bin/env python
"""
plot_synth_quality_bars.py — multi-page PDF bar-chart report of synth quality.

Page 1  : title page with experiment overview, dataset list, metric descriptions,
          and the full color legend.
Pages 2+: one bar chart per metric (9 total).

X-axis  : every (SDG method, ε) combination listed in SDG_ORDER, grouped into
          families (MST / AIM / Non-DP / Deep Gen. / Baseline) with alternating
          background shading and family labels above the plot.
Sub-bars: one per (dataset, size_dir) in COLUMN_ORDER, color-coded and
          visually clustered by dataset family (Adult 1k/10k/20k share a hue,
          CDC 1k/100k share a hue, etc.).  Missing combinations are silently
          omitted.

Normalization: each metric column is min-max scaled to [0, 1] per (dataset, size)
          independently, so wildly different raw scales (e.g. wasserstein_ohe
          SBO ≈ 68 vs Arizona ≈ 3) remain visually comparable.
          For "lower is better" metrics the scale is then inverted so that
          taller bars always mean better quality.

Usage:
    conda activate recon_
    python experiment_scripts/plot_synth_quality_bars.py \\
        experiment_scripts/quality_results_merged.csv
    python experiment_scripts/plot_synth_quality_bars.py \\
        experiment_scripts/quality_results_merged.csv --out figures/quality.pdf
    python experiment_scripts/plot_synth_quality_bars.py \\
        experiment_scripts/quality_results_merged.csv \\
        --metrics tstr_ratio mean_tvd wasserstein_ohe
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ── SDG configuration (mirrors synth_quality_to_latex.py; comment out rows) ───

SDG_ORDER = [
    "MST_eps0.1",
    "MST_eps0.3",
    "MST_eps1",
    "MST_eps3",
    "MST_eps10",
    "MST_eps30",
    "MST_eps100",
    "MST_eps300",
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

SDG_FAMILIES = [
    ("MST",       ["MST_eps0.1","MST_eps0.3","MST_eps1","MST_eps3","MST_eps10",
                   "MST_eps30","MST_eps100","MST_eps300","MST_eps1000"]),
    ("AIM",       ["AIM_eps0.1","AIM_eps0.3","AIM_eps1","AIM_eps3",
                   "AIM_eps10","AIM_eps30","AIM_eps100"]),
    ("Non-DP",    ["RankSwap","CellSuppression","Synthpop"]),
    ("Deep Gen.", ["TVAE","CTGAN","ARF","TabDDPM"]),
    ("Baseline",  ["~train_baseline"]),
]

SDG_TICK = {
    **{f"MST_eps{e}": f"ε={e}"  for e in ["0.1","0.3","1","3","10","30","100","300","1000"]},
    **{f"AIM_eps{e}": f"ε={e}"  for e in ["0.1","0.3","1","3","10","30","100"]},
    "RankSwap":        "RankSwap",
    "CellSuppression": "Cell\nSupp.",
    "Synthpop":        "Synthpop",
    "TVAE":            "TVAE",
    "CTGAN":           "CTGAN",
    "ARF":             "ARF",
    "TabDDPM":         "TabDDPM",
    "~train_baseline": "Train\nBaseline",
}

# ── Dataset / size columns ─────────────────────────────────────────────────────

# (dataset, size_dir, display_label)
COLUMN_ORDER = [
    ("adult",             "size_1000",         "Adult 1k"),
    ("adult",             "size_10000",        "Adult 10k"),
    ("adult",             "size_20000",        "Adult 20k"),
    ("nist_arizona_data", "size_10000_25feat", "Arizona 10k"),
    ("nist_sbo",          "size_1000",         "SBO 1k"),
    ("cdc_diabetes",      "size_1000",         "CDC 1k"),
    ("cdc_diabetes",      "size_100000",       "CDC 100k"),
    ("california",        "size_1000",         "Calif. 1k"),
]

# Which visual group each column belongs to (for intra-slot spacing)
_COL_GROUP = [0, 0, 0, 1, 2, 3, 3, 4]   # 5 groups: Adult / Arizona / SBO / CDC / Calif.

# Colors: similar hues per dataset family
COLORS = [
    "#aec6e8",   # Adult 1k   — light blue
    "#4e9ac7",   # Adult 10k  — mid blue
    "#1a5fa8",   # Adult 20k  — dark blue
    "#4daf4a",   # Arizona    — green
    "#ff7f00",   # SBO        — orange
    "#f4a4a4",   # CDC 1k     — light red
    "#c0392b",   # CDC 100k   — dark red
    "#9b59b6",   # California — purple
]

# ── Metrics ────────────────────────────────────────────────────────────────────

METRICS = [
    ("tstr_ratio",     "↑",
     "TSTR Ratio",
     "Utility retention: TSTR F1-macro (or R²) divided by TRTR baseline.\n"
     "1.0 = synthetic data is as useful as real data for training a classifier."),
    ("mean_tvd",       "↓",
     "Mean TVD",
     "Mean total variation distance across all categorical columns.\n"
     "0 = perfect match to training distribution."),
    ("wasserstein_ohe","↓",
     "Wasserstein OHE",
     "Custom Wasserstein: one-hot-encode every feature (cats → dummies;\n"
     "continuous → 20 equal-depth bins → dummies), sum |freq_train − freq_synth|.\n"
     "Normalized per dataset before plotting (raw values not comparable across datasets)."),
    ("corr_diff",      "↓",
     "Correlation Diff.",
     "Mean absolute difference between the training and synthetic correlation matrices.\n"
     "0 = perfect correlation structure preservation."),
    ("sdv_col_pairs",  "↑",
     "SDV Column Pairs",
     "SDV column pair trends quality score (pairwise relationship fidelity).\n"
     "1.0 = perfect pairwise fidelity."),
    ("prop_score",     "↓",
     "Propensity AUC",
     "AUC of a logistic-regression classifier trained to distinguish real from synthetic.\n"
     "0.5 = indistinguishable; 1.0 = perfectly distinguishable."),
    ("pairwise_tvd",   "↓",
     "Pairwise TVD",
     "Mean pairwise 2-way joint TVD across categorical column pairs.\n"
     "Only computed for datasets with ≤ 30 categorical columns."),
    ("mean_jsd",       "↓",
     "Mean JSD",
     "Mean Jensen–Shannon divergence across categorical columns.\n"
     "0 = identical marginal distributions."),
    ("sdv_col_shapes", "↑",
     "SDV Column Shapes",
     "SDV column shapes quality score (marginal distribution fidelity).\n"
     "1.0 = perfect marginal fidelity."),
]

# ── Bar geometry ───────────────────────────────────────────────────────────────

BAR_W       = 0.08    # width of each sub-bar
INNER_GAP   = 0.008   # gap between adjacent sub-bars
DATASET_GAP = 0.05    # extra gap when switching dataset family within a slot
METHOD_GAP  = 0.18    # gap between methods in the same SDG family
FAMILY_GAP  = 0.55    # gap between different SDG families
FAMILY_COLORS = ["#f4f4f4", "#ffffff", "#f4f4f4", "#ffffff", "#f4f4f4"]


def _build_bar_offsets() -> list[float]:
    """Offsets of each sub-bar's centre within one method slot."""
    offsets, x, prev_g = [], 0.0, _COL_GROUP[0]
    for i, g in enumerate(_COL_GROUP):
        if i > 0 and g != prev_g:
            x += DATASET_GAP
        offsets.append(x + BAR_W / 2)
        x += BAR_W + INNER_GAP
        prev_g = g
    return offsets


BAR_OFFSETS = _build_bar_offsets()
SLOT_W = BAR_OFFSETS[-1] + BAR_W / 2      # total width of one method slot


def _method_family_index() -> dict[str, int]:
    return {m: fi for fi, (_, ms) in enumerate(SDG_FAMILIES) for m in ms}


M2F = _method_family_index()


# ── Layout computation ─────────────────────────────────────────────────────────

def compute_layout(present_methods: list[str]):
    """
    Returns:
      centers      : {method: x_center}  — x centre of the method slot
      family_spans : [(x_lo, x_hi, label, color), ...]
      total_width  : float
    """
    centers: dict[str, float] = {}
    family_spans = []
    x = 0.0
    prev_fi: int | None = None
    fam_start = 0.0

    for i, method in enumerate(present_methods):
        fi = M2F.get(method, len(SDG_FAMILIES))
        if prev_fi is not None:
            if fi != prev_fi:
                fname = SDG_FAMILIES[prev_fi][0] if prev_fi < len(SDG_FAMILIES) else "Other"
                family_spans.append((fam_start, x, fname, FAMILY_COLORS[prev_fi % len(FAMILY_COLORS)]))
                x += FAMILY_GAP
                fam_start = x
            else:
                x += METHOD_GAP
        centers[method] = x + SLOT_W / 2
        x += SLOT_W
        prev_fi = fi

    if prev_fi is not None:
        fname = SDG_FAMILIES[prev_fi][0] if prev_fi < len(SDG_FAMILIES) else "Other"
        family_spans.append((fam_start, x, fname, FAMILY_COLORS[prev_fi % len(FAMILY_COLORS)]))

    return centers, family_spans, x


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_per_column(pivot: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each column independently to [0, 1]."""
    normed = pivot.copy().astype(float)
    for col in normed.columns:
        vals = normed[col].dropna()
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            normed[col] = (normed[col] - lo) / (hi - lo)
        else:
            normed[col] = 0.5
    return normed


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_metric_page(pdf: PdfPages, pivot_raw: pd.DataFrame, metric_key: str,
                     direction: str, title_short: str, description: str,
                     centers: dict, family_spans: list, total_width: float,
                     present_methods: list[str], col_keys: list[str]):

    normed = normalize_per_column(pivot_raw)
    if direction == "↓":
        normed = 1.0 - normed        # invert: taller bar = better in all cases

    fig, ax = plt.subplots(figsize=(34, 7.5))
    fig.subplots_adjust(left=0.035, right=0.87, bottom=0.20, top=0.84)

    # Background shading per family
    for lo, hi, fname, fc in family_spans:
        ax.axvspan(lo - 0.05, hi + 0.05, color=fc, zorder=0, lw=0)
        ax.text((lo + hi) / 2, 1.025, fname,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

    # Bars
    for method in present_methods:
        if method not in normed.index or method not in centers:
            continue
        cx = centers[method]
        is_baseline = method == "~train_baseline"
        for ci, (ds, sz, _lbl) in enumerate(COLUMN_ORDER):
            ck = f"{ds}|{sz}"
            if ck not in normed.columns:
                continue
            val = float(normed.at[method, ck]) if method in normed.index else np.nan
            if np.isnan(val):
                continue
            bx = cx - SLOT_W / 2 + BAR_OFFSETS[ci]
            ax.bar(bx, val, width=BAR_W,
                   color=COLORS[ci],
                   edgecolor="none",
                   hatch="////" if is_baseline else None,
                   zorder=2)

    # X-axis ticks
    tick_xs     = [centers[m] for m in present_methods if m in centers]
    tick_labels = [SDG_TICK.get(m, m)   for m in present_methods if m in centers]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=45, ha="right", va="top")

    ax.set_xlim(-0.3, total_width + 0.3)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--", zorder=1)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#555", linewidth=0.6)

    dir_note = ("↑ higher = better" if direction == "↑"
                else "↓ lower is better  —  bars are inverted so taller = better")
    ax.set_ylabel("Normalized score (per dataset, taller = better)", fontsize=9)
    ax.set_title(
        f"{title_short}    [{dir_note}]\n"
        + "\n".join(f"  {line}" for line in description.split("\n")),
        fontsize=9.5, loc="left", pad=8, color="#222",
    )

    # Per-page legend (compact, right of plot)
    patches = [mpatches.Patch(color=COLORS[ci], label=COLUMN_ORDER[ci][2])
               for ci, (ds, sz, _) in enumerate(COLUMN_ORDER)
               if f"{ds}|{sz}" in col_keys]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.001, 1.0),
              fontsize=8.5, framealpha=0.95, title="Dataset / size",
              title_fontsize=8.5, edgecolor="#ccc")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{metric_key}] page added.")


# ── Title page ─────────────────────────────────────────────────────────────────

def make_title_page(pdf: PdfPages, metrics_to_plot: list):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    def txt(x, y, s, **kw):
        fig.text(x, y, s, transform=fig.transFigure, **kw)

    txt(0.5, 0.945, "Synthetic Data Quality — Bar Chart Report",
        ha="center", fontsize=19, fontweight="bold", color="#111")
    txt(0.5, 0.910, f"Generated {date.today().isoformat()}",
        ha="center", fontsize=10, color="#666")

    y = 0.870
    txt(0.06, y, "Overview", fontsize=12, fontweight="bold", color="#222")
    y -= 0.030
    overview = (
        "Each chart compares synthetic data generators across 5 datasets and one quality metric.\n"
        "X-axis: SDG methods and ε variants (MST / AIM / Non-DP / Deep Generative / Baseline).\n"
        "Sub-bars within each method slot: one bar per (dataset, training-set size), colour-coded.\n"
        "Values are normalised per (dataset, size) to [0, 1] across all SDG methods independently,\n"
        "so datasets with wildly different raw scales (e.g. wasserstein_ohe SBO ≈ 68 vs Arizona ≈ 3)\n"
        "remain visually comparable.  For lower-is-better metrics bars are inverted so that taller\n"
        "always means better quality.  Missing sub-bars = no synthetic data for that combination."
    )
    txt(0.08, y, overview, fontsize=8.8, va="top", linespacing=1.55, color="#333")

    y -= 0.175
    txt(0.06, y, "Datasets", fontsize=12, fontweight="bold", color="#222")
    y -= 0.028
    for line in [
        "Adult (income, classification)       — 1 k / 10 k / 20 k training rows",
        "Arizona IPUMS 1940 (25-feature subset, classification)  — 10 k rows",
        "NIST SBO (small-business owners, classification)        —  1 k rows",
        "CDC Diabetes (binary classification) —  1 k / 100 k rows",
        "California Housing (regression)      —  1 k rows",
    ]:
        txt(0.09, y, "•  " + line, fontsize=8.8, va="top", color="#333")
        y -= 0.026

    y -= 0.010
    txt(0.06, y, "SDG methods", fontsize=12, fontweight="bold", color="#222")
    y -= 0.028
    for line in [
        "Differentially private:  MST  (ε ∈ {0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000})",
        "                         AIM  (ε ∈ {0.1, 0.3, 1, 3, 10, 30, 100})",
        "Non-DP statistical:      RankSwap · CellSuppression · Synthpop",
        "Deep generative:         TVAE · CTGAN · ARF · TabDDPM",
        "Baseline:                Train–Train  (one real sample vs another;  "
        "same-distribution upper bound for utility metrics)",
    ]:
        txt(0.09, y, line, fontsize=8.8, va="top", color="#333")
        y -= 0.026

    y -= 0.010
    txt(0.06, y, "Metrics in this report", fontsize=12, fontweight="bold", color="#222")
    y -= 0.025
    for _key, direction, short, _desc in metrics_to_plot:
        arrow = "↑ higher" if direction == "↑" else "↓ lower"
        txt(0.09, y, f"•  {short}   ({arrow} is better)", fontsize=8.8, va="top", color="#333")
        y -= 0.025

    # Color legend
    y -= 0.015
    txt(0.06, y, "Sub-bar colour legend", fontsize=12, fontweight="bold", color="#222")
    y -= 0.030
    for ci, (ds, sz, lbl) in enumerate(COLUMN_ORDER):
        col_x = 0.09 + (ci % 4) * 0.22
        col_y = y - (ci // 4) * 0.028
        patch = mpatches.FancyBboxPatch((col_x - 0.001, col_y - 0.008), 0.018, 0.018,
                                        boxstyle="square,pad=0", color=COLORS[ci],
                                        transform=fig.transFigure, clip_on=False, zorder=3)
        fig.add_artist(patch)
        txt(col_x + 0.020, col_y, lbl, fontsize=8.8, va="center", color="#333")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  [title page] added.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-page PDF bar-chart report of synth quality metrics."
    )
    parser.add_argument("results_csv", help="Path to quality_results_merged.csv")
    parser.add_argument("--out",     default=None,
                        help="Output PDF (default: <csv_dir>/synth_quality_bars.pdf)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Subset of metric keys to include (default: all 9)")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_path = Path(args.out) if args.out else csv_path.parent / "synth_quality_bars.pdf"

    df = pd.read_csv(csv_path)
    df["col_key"] = df["dataset"] + "|" + df["size_dir"]

    # Filter metrics to those requested and present in CSV
    metrics_to_plot = [
        (key, direction, short, desc)
        for key, direction, short, desc in METRICS
        if (args.metrics is None or key in args.metrics) and key in df.columns
    ]
    if not metrics_to_plot:
        raise ValueError("None of the requested metrics are present in the CSV.")

    # Which SDG methods are in the data (preserve SDG_ORDER)
    present_methods = [m for m in SDG_ORDER if m in df["method"].values]
    col_keys = [f"{ds}|{sz}" for ds, sz, _ in COLUMN_ORDER
                if f"{ds}|{sz}" in df["col_key"].values]

    # Layout (same for every metric page)
    centers, family_spans, total_width = compute_layout(present_methods)

    print(f"SDG methods : {len(present_methods)}")
    print(f"Dataset cols: {len(col_keys)}")
    print(f"Metrics     : {[k for k,*_ in metrics_to_plot]}")
    print(f"Output      : {out_path}")

    with PdfPages(out_path) as pdf:
        make_title_page(pdf, metrics_to_plot)

        for metric_key, direction, title_short, description in metrics_to_plot:
            # Aggregate: mean over samples per (method, col_key)
            sub = df[df["col_key"].isin(col_keys)].copy()
            agg = (sub.groupby(["col_key", "method"])[metric_key]
                      .mean()
                      .unstack("col_key")
                      .reindex(index=present_methods, columns=col_keys))

            plot_metric_page(
                pdf, agg, metric_key, direction, title_short, description,
                centers, family_spans, total_width,
                present_methods, col_keys,
            )

    print(f"\nDone — {1 + len(metrics_to_plot)} pages written to {out_path}")


if __name__ == "__main__":
    main()
