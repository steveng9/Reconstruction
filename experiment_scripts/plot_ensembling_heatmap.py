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
    "TabDDPM",
    "PartialMSTBounded",
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
    "PartialMSTBounded":  "PartMST",
    "Attention":          "Attn",
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


def plot_heatmap(matrix: np.ndarray, delta: np.ndarray, attacks: list[str],
                 diff: bool, title: str, out_path: str | None):
    n = len(attacks)
    labels = [SHORT_LABELS.get(a, a) for a in attacks]

    plot_data = delta if diff else matrix

    # Mask: NaN cells (missing data) shown in light grey
    mask = np.isnan(plot_data)

    fig, ax = plt.subplots(figsize=(max(7, n * 0.85), max(6, n * 0.75)))

    if diff:
        vmin, vmax = -0.05, 0.05
        cmap = "RdYlGn"
        cbar_label = "Ensemble gain over best individual (RA pp)"
    else:
        finite = plot_data[~mask]
        vmin = finite.min() if finite.size else 0.0
        vmax = finite.max() if finite.size else 1.0
        cmap = "YlOrRd"
        cbar_label = "Mean RA (avg over samples × SDG)"

    sns.heatmap(
        plot_data,
        mask=mask,
        ax=ax,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
    )

    # Highlight diagonal cells (individual attacks) with a border
    for i in range(n):
        if not np.isnan(plot_data[i, i]):
            ax.add_patch(mpatches.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor="black", lw=2.5, clip_on=False,
            ))

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Attack B", fontsize=11)
    ax.set_ylabel("Attack A", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Legend note for diagonal
    diag_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                                linewidth=2, label="Diagonal = individual attack score")
    ax.legend(handles=[diag_patch], loc="upper left",
              bbox_to_anchor=(1.18, 1), borderaxespad=0, fontsize=9)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot ensembling heatmap.")
    parser.add_argument("csv", help="CSV produced by run_ensembling_heatmap.py")
    parser.add_argument("--sdg",  default=None,
                        help="Filter to rows whose 'sdg' column starts with this string "
                             "(e.g. 'TabDDPM', 'MST_eps1'). Default: average over all SDG methods.")
    parser.add_argument("--diff", action="store_true",
                        help="Show ensemble gain over best individual instead of raw RA.")
    parser.add_argument("--out",  default=None,
                        help="Output file path (PDF/PNG). Omit to show interactively.")
    parser.add_argument("--title", default=None,
                        help="Custom plot title.")
    args = parser.parse_args()

    df = load_results(args.csv, args.sdg)

    # Determine which attacks are actually present in the CSV
    present_indiv = set(df.loc[df["type"] == "individual", "attack"].unique())
    attacks = [a for a in ATTACK_ORDER if a in present_indiv]
    # Append any attacks in the CSV but not in ATTACK_ORDER
    for a in sorted(present_indiv - set(attacks)):
        attacks.append(a)

    if len(attacks) < 2:
        sys.exit("ERROR: need at least 2 attacks in the CSV to plot a heatmap.")

    matrix, delta = build_matrix(df, attacks, diff=args.diff)

    sdg_str = f" — {args.sdg}" if args.sdg else " — all SDG methods"
    title = args.title or (
        f"Ensemble heatmap{sdg_str}\n"
        f"{'Gain over best individual (RA pp)' if args.diff else 'Mean RA (avg over samples)'}"
    )

    plot_heatmap(matrix, delta, attacks, diff=args.diff, title=title, out_path=args.out)


if __name__ == "__main__":
    main()
