#!/usr/bin/env python
"""
Plot the ensembling heatmap from the CSV produced by run_ensembling_heatmap.py.

The heatmap is an N×N matrix where:
  - Diagonal cells show the individual attack score (averaged over samples × SDG).
  - Off-diagonal cell [i, j] shows the ensemble score for pair (attack_i, attack_j),
    also averaged over samples × SDG.
  - Each cell is colour-coded by score; the diagonal uses a distinct marker so
    individual baselines stand out from ensemble pairs.

Usage:
    python experiment_scripts/plot_ensembling_heatmap.py results.csv
    python experiment_scripts/plot_ensembling_heatmap.py results.csv --sdg TabDDPM
    python experiment_scripts/plot_ensembling_heatmap.py results.csv --out heatmap.pdf
    python experiment_scripts/plot_ensembling_heatmap.py results.csv --diff
        # show ensemble_score - max(individual_a, individual_b) instead of raw score
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ── Attack display order and short labels ──────────────────────────────────────

# Canonical order for the heatmap axes — edit to match your ATTACKS list.
ATTACK_ORDER = [
    "Mode",
    "Random",
    "KNN",
    "NaiveBayes",
    "LogisticRegression",
    "RandomForest",
    "LightGBM",
    "MLP",
    "CondDDPM",
    "CondMSTBounded",
]

SHORT_LABELS = {
    "Mode":               "Mode",
    "Random":             "Random",
    "KNN":                "KNN",
    "NaiveBayes":         "NaiveBayes",
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "LightGBM":           "LGB",
    "MLP":                "MLP",
    "CondMSTBounded":     "CondMST",
    "CondDDPM":           "CondDDPM",
    "ARFFormer":          "Attn",
    "SVM":                "SVM",
}


def load_results(csv_path: str, sdg_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if sdg_filter:
        df = df[df["sdg"].str.startswith(sdg_filter)]
        if df.empty:
            sys.exit(f"ERROR: no rows matching --sdg '{sdg_filter}'")
    return df


def build_matrix(df: pd.DataFrame, attacks: list[str], diff: bool
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        matrix : N×N array of mean RA scores
                 diagonal = individual attack scores
                 off-diagonal = ensemble pair scores
        delta  : N×N array of (ensemble - max(individual_a, individual_b))
                 diagonal = 0
    """
    n = len(attacks)
    idx = {a: i for i, a in enumerate(attacks)}

    # Average over samples and SDG methods
    indiv_means: dict[str, float] = {}
    for attack in attacks:
        rows = df[(df["type"] == "individual") & (df["attack"] == attack) & df["ra_mean"].notna()]
        indiv_means[attack] = float(rows["ra_mean"].mean()) if not rows.empty else float("nan")

    ens_means: dict[tuple[str, str], float] = {}
    ens_rows = df[(df["type"] == "ensemble") & df["ra_mean"].notna()]
    for _, row in ens_rows.iterrows():
        pair = row["attack"]           # "A+B"
        if "+" not in pair:
            continue
        a, b = pair.split("+", 1)
        if a in idx and b in idx:
            ens_means.setdefault((a, b), []).append(row["ra_mean"])
            ens_means.setdefault((b, a), []).append(row["ra_mean"])  # symmetric

    # Collapse lists to means
    ens_means = {k: float(np.mean(v)) for k, v in ens_means.items()}

    matrix = np.full((n, n), float("nan"))
    delta  = np.zeros((n, n))

    for attack, i in idx.items():
        matrix[i, i] = indiv_means.get(attack, float("nan"))

    for (a, b), mean_ra in ens_means.items():
        i, j = idx[a], idx[b]
        matrix[i, j] = mean_ra
        delta[i, j]  = mean_ra - max(indiv_means.get(a, 0), indiv_means.get(b, 0))

    return matrix, delta


def _heatmap_scales(plot_data: np.ndarray, n: int, diff: bool):
    mask = np.isnan(plot_data)
    if diff:
        off_diag = plot_data[~np.eye(n, dtype=bool) & ~mask]
        actual_min = off_diag.min() if off_diag.size else -1.0
        actual_max = off_diag.max() if off_diag.size else 0.0
        vmin = float(np.floor(actual_min))
        vmax = max(float(np.ceil(actual_max)), 1.0)
        cmap, fmt = "RdYlGn", "+.1f"
        cbar_label = "Gain over best individual (RA pp)"
    else:
        finite = plot_data[~mask]
        vmin = finite.min() if finite.size else 0.0
        vmax = finite.max() if finite.size else 1.0
        cmap, fmt = "YlOrRd", ".1f"
        cbar_label = "Mean RA (%)"
    return vmin, vmax, cmap, fmt, cbar_label


def _draw_panel(ax, cbar_ax, matrix: np.ndarray, delta: np.ndarray,
                attacks: list[str], diff: bool, title: str, show_xlabel: bool,
                show_xticks: bool = True):
    """Draw one heatmap panel onto the provided axes."""
    n = len(attacks)
    labels = [SHORT_LABELS.get(a, a) for a in attacks]
    plot_data = delta if diff else matrix
    mask = np.isnan(plot_data)
    vmin, vmax, cmap, fmt, cbar_label = _heatmap_scales(plot_data, n, diff)

    sns.heatmap(
        plot_data, mask=mask, ax=ax,
        vmin=vmin, vmax=vmax, cmap=cmap,
        annot=True, fmt=fmt, annot_kws={"size": 9},
        linewidths=0.4, linecolor="white", square=True,
        cbar=True, cbar_ax=cbar_ax,
        cbar_kws={"label": cbar_label},
        xticklabels=labels, yticklabels=labels,
    )
    for i in range(n):
        if not np.isnan(plot_data[i, i]):
            ax.add_patch(mpatches.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False,
            ))
    ax.set_title(title, fontsize=22, pad=14)
    ax.set_ylabel("Attack A", fontsize=18)
    ax.set_xlabel("Attack B" if show_xlabel else "", fontsize=18)
    if show_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=16)
    else:
        ax.set_xticklabels([])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    cbar_ax.tick_params(labelsize=14)
    cbar_ax.yaxis.label.set_size(15)

    diag_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                                linewidth=2, label="Diagonal = individual score")
    cbar_ax.legend(handles=[diag_patch], loc="upper left",
                   bbox_to_anchor=(-0.3, -0.04), borderaxespad=0, fontsize=14,
                   frameon=True)


def plot_heatmap(matrix: np.ndarray, delta: np.ndarray, attacks: list[str],
                 diff: bool, title: str, out_path: str | None):
    n = len(attacks)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.85), max(6, n * 0.75)))
    fig.subplots_adjust(right=0.78)
    cbar_ax = fig.add_axes([0.80, 0.15, 0.03, 0.7])
    _draw_panel(ax, cbar_ax, matrix, delta, attacks, diff, title, show_xlabel=True)
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_combined(panels: list[tuple[np.ndarray, np.ndarray, list[str], bool, str]],
                  out_path: str):
    """Stack multiple heatmap panels vertically in a single figure."""
    import matplotlib.gridspec as gridspec

    n_panels = len(panels)
    n_attacks = len(panels[0][2])
    cell_size      = 0.42                     # inches per heatmap cell
    heatmap_side   = n_attacks * cell_size    # e.g. 3.78 in for 9 attacks
    left_margin    = 1.8                      # room for 16pt y-tick labels
    right_margin   = 3.2                      # colorbar + legend
    top_margin     = 0.65                     # title above heatmap
    bot_margin_xt  = 0.95                     # bottom panel: room for rotated x-ticks
    bot_margin_no  = 0.10                     # other panels: no x-tick labels
    gap_inches     = 0.15                     # breathing room between panels (for titles)

    fig_w = left_margin + heatmap_side + right_margin

    # Per-panel heights
    panel_heights = [
        top_margin + heatmap_side + (bot_margin_xt if i == n_panels - 1 else bot_margin_no)
        for i in range(n_panels)
    ]
    total_h = sum(panel_heights) + gap_inches * (n_panels - 1)

    fig = plt.figure(figsize=(fig_w, total_h))

    hm_left  = left_margin / fig_w
    hm_right = (left_margin + heatmap_side) / fig_w
    cb_left  = (left_margin + heatmap_side + 0.25) / fig_w
    cb_w     = 0.28 / fig_w

    # Build bottom positions from the bottom up
    bottoms = []
    y = 0.0
    for i in range(n_panels - 1, -1, -1):
        bottoms.insert(0, y)
        y += panel_heights[i] + gap_inches
    bottoms = [b / total_h for b in bottoms]

    for p_idx, (matrix, delta, attacks, diff, title) in enumerate(panels):
        is_bottom  = (p_idx == n_panels - 1)
        bot_margin = bot_margin_xt if is_bottom else bot_margin_no
        bot        = bottoms[p_idx]
        hm_h       = heatmap_side / total_h
        hm_bot     = bot + bot_margin / total_h

        ax      = fig.add_axes([hm_left, hm_bot, hm_right - hm_left, hm_h])
        cbar_ax = fig.add_axes([cb_left, hm_bot + hm_h * 0.10,
                                cb_w, hm_h * 0.78])

        _draw_panel(ax, cbar_ax, matrix, delta, attacks, diff, title,
                    show_xlabel=is_bottom, show_xticks=is_bottom)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved combined to {out_path}")
    plt.close(fig)


COMBINED_PANELS = [
    # (sdg_filter, title)
    (None,        "All SDG methods (average)"),
    ("TabDDPM",   "TabDDPM"),
    ("MST_eps10", "MST ($\\varepsilon=10$)"),
    ("TVAE",      "TVAE"),
]


def _resolve_attacks(df: pd.DataFrame) -> list[str]:
    present = set(df.loc[df["type"] == "individual", "attack"].unique())
    attacks = [a for a in ATTACK_ORDER if a in present]
    for a in sorted(present - set(attacks)):
        attacks.append(a)
    return attacks


def main():
    parser = argparse.ArgumentParser(description="Plot ensembling heatmap.")
    parser.add_argument("csv", help="CSV produced by run_ensembling_heatmap.py")
    parser.add_argument("--sdg",  default=None,
                        help="Filter to rows whose 'sdg' column starts with this string.")
    parser.add_argument("--diff", action="store_true",
                        help="Show ensemble gain over best individual instead of raw RA.")
    parser.add_argument("--out",  default=None,
                        help="Output file path (PDF/PNG). Omit to show interactively.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    parser.add_argument("--combined", action="store_true",
                        help="Produce one PDF with all 5 SDG panels stacked vertically.")
    args = parser.parse_args()

    full_df = pd.read_csv(args.csv)

    if args.combined:
        if not args.out:
            sys.exit("ERROR: --combined requires --out")
        panels = []
        for sdg_filter, title in COMBINED_PANELS:
            df = load_results(args.csv, sdg_filter)
            attacks = _resolve_attacks(df)
            matrix, delta = build_matrix(df, attacks, diff=args.diff)
            panels.append((matrix, delta, attacks, args.diff, title))
        plot_combined(panels, args.out)
        return

    df = load_results(args.csv, args.sdg)
    attacks = _resolve_attacks(df)

    if len(attacks) < 2:
        sys.exit("ERROR: need at least 2 attacks in the CSV to plot a heatmap.")

    matrix, delta = build_matrix(df, attacks, diff=args.diff)

    sdg_str = f" — {args.sdg}" if args.sdg else " — all SDG methods"
    title = args.title or f"Ensemble heatmap{sdg_str}"
    plot_heatmap(matrix, delta, attacks, diff=args.diff, title=title, out_path=args.out)


if __name__ == "__main__":
    main()
