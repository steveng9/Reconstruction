#!/usr/bin/env python3
"""
Empirical decomposition of reconstruction attack results.

Phase 1: PCA + NMF biplot — attack × SDG condition matrix (adult 10k QI1 by default).
         Reveals attack archetypes and which SDG conditions spread attacks apart.

Phase 2: Variance decomposition — how much does each factor (attack identity,
         SDG method, dataset, epsilon, QI) explain across all data in results.db?

Usage:
    python experiment_scripts/decompose_results.py
    python experiment_scripts/decompose_results.py --phase1-only --dataset adult --size 10000 --qi QI1
    python experiment_scripts/decompose_results.py --phase2-only
    python experiment_scripts/decompose_results.py --out experiment_scripts/decomp
"""

import argparse
import os
import re
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DB_PATH = os.path.join(os.path.dirname(__file__), "results.db")

EXCLUDE_PATTERNS = ["+", "Oracle", "FeatSelector"]

# ── Attack family classification ──────────────────────────────────────────────

FAMILY_MAP = [
    ("marginal_rf", ["marginalrf", "marginal_rf"]),
    ("diffusion",   ["tabddpm", "conditioned", "repaint"]),
    ("partial_mst", ["partialmst"]),
    ("tabpfn",      ["tabpfn"]),
    ("attention",   ["attention"]),
    ("ml",          ["randomforest", "lightgbm", "naivebayes",
                     "logisticregression", "mlp", "knn", "svm", "linear"]),
    ("baseline",    ["mode", "random", "mean", "measuredeid"]),
]
FAMILY_COLORS = {
    "baseline":    "#888888",
    "ml":          "#2196F3",
    "tabpfn":      "#00BCD4",
    "marginal_rf": "#FF5722",
    "diffusion":   "#9C27B0",
    "partial_mst": "#4CAF50",
    "attention":   "#FF9800",
    "other":       "#795548",
}

def attack_family(name):
    n = name.lower()
    for fam, keywords in FAMILY_MAP:
        if any(k in n for k in keywords):
            return fam
    return "other"

# ── SDG family classification ─────────────────────────────────────────────────

SDG_FAMILY_COLORS = {
    "dp_mst":    "#1565C0",
    "dp_aim":    "#42A5F5",
    "deep_gen":  "#7B1FA2",
    "deid":      "#E65100",
    "other":     "#546E7A",
}

def sdg_family(sdg):
    s = str(sdg)
    if "MST" in s:   return "dp_mst"
    if "AIM" in s:   return "dp_aim"
    if any(x in s for x in ["TVAE", "CTGAN", "ARF", "TabDDPM", "Synthpop"]):
        return "deep_gen"
    if any(x in s for x in ["RankSwap", "CellSuppression"]):
        return "deid"
    return "other"

def parse_epsilon(sdg):
    m = re.search(r"eps([\d.]+)", str(sdg))
    return float(m.group(1)) if m else None

# ── Data loading ──────────────────────────────────────────────────────────────

def _filter_attacks(df):
    mask = pd.Series(True, index=df.index)
    for pat in EXCLUDE_PATTERNS:
        mask &= ~df["attack_label"].str.contains(pat, regex=False)
    return df[mask]

def load_pivot(db_path, dataset, dataset_size, qi):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT attack_label, sdg_method, ra_mean
        FROM runs
        WHERE dataset=? AND dataset_size=? AND qi=? AND split='standard'
          AND ra_mean IS NOT NULL
    """, conn, params=(dataset, dataset_size, qi))
    conn.close()
    df = _filter_attacks(df)
    # Average over the 5 samples per (attack, SDG) cell
    df = df.groupby(["attack_label", "sdg_method"])["ra_mean"].mean().reset_index()
    pivot = df.pivot(index="attack_label", columns="sdg_method", values="ra_mean")
    return pivot

def load_all(db_path, exclude_continuous=True):
    """Load all rows for variance decomposition.

    exclude_continuous: drop california (RMSE-based metric, incomparable scale).
    Also clips extreme outliers (SGDRegressor divergence etc.).
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT dataset, dataset_size, sample, qi, sdg_method,
               attack_label, ra_mean
        FROM runs
        WHERE split='standard' AND ra_mean IS NOT NULL
          AND ra_mean BETWEEN 0 AND 200
    """, conn)
    conn.close()
    df = _filter_attacks(df)
    if exclude_continuous:
        df = df[df["dataset"] != "california"]
        print("  (california excluded — uses RMSE, not RA%; incomparable scale)")
    return df

# ── Phase 1: PCA + NMF biplot ─────────────────────────────────────────────────

def _annotate_scatter(ax, xs, ys, labels, fontsize=7):
    """Place labels with simple nudging to reduce overlap."""
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        dx, dy = 3, 3
        ax.annotate(lab, (x, y), fontsize=fontsize,
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))

def _arrow(ax, x, y, label, color, fontsize=7):
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.75))
    ax.annotate(label, (x, y), fontsize=fontsize, color=color,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, lw=0))

def run_phase1(pivot, out_prefix):
    attack_names = pivot.index.tolist()
    sdg_names    = pivot.columns.tolist()
    n_attacks, n_sdg = pivot.shape

    print(f"\n=== Phase 1: Attack × SDG matrix ===")
    print(f"  Attacks: {n_attacks}  |  SDG conditions: {n_sdg}")
    n_missing = pivot.isna().sum().sum()
    print(f"  Missing cells: {n_missing}/{pivot.size} ({100*n_missing/pivot.size:.1f}%)")

    # Impute missing with column means, then standardise
    X_raw = pivot.values
    imp = SimpleImputer(strategy="mean")
    X_imp = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    # ── PCA ──────────────────────────────────────────────────────────────────
    n_comp = min(n_attacks, n_sdg, 10)
    pca = PCA(n_components=n_comp)
    scores   = pca.fit_transform(X)       # (n_attacks, n_comp)
    loadings = pca.components_.T          # (n_sdg, n_comp)
    ev = pca.explained_variance_ratio_

    print(f"\nPCA explained variance:")
    cumulative = 0.0
    for i, v in enumerate(ev[:6]):
        cumulative += v
        print(f"  PC{i+1}: {v*100:5.1f}%   cumulative: {cumulative*100:5.1f}%")

    print(f"\nPC1 attack scores (high → low):")
    for name, s in sorted(zip(attack_names, scores[:, 0]), key=lambda x: -x[1]):
        print(f"  {name:<42} {s:+.3f}")

    print(f"\nPC1 SDG loadings (high → low):")
    for name, l in sorted(zip(sdg_names, loadings[:, 0]), key=lambda x: -x[1]):
        print(f"  {name:<30} {l:+.3f}")

    print(f"\nPC2 attack scores (high → low):")
    for name, s in sorted(zip(attack_names, scores[:, 1]), key=lambda x: -x[1]):
        print(f"  {name:<42} {s:+.3f}")

    print(f"\nPC2 SDG loadings (high → low):")
    for name, l in sorted(zip(sdg_names, loadings[:, 1]), key=lambda x: -x[1]):
        print(f"  {name:<30} {l:+.3f}")

    # ── NMF ──────────────────────────────────────────────────────────────────
    X_nn = np.clip(X_imp, 0, None)
    nmf = NMF(n_components=2, random_state=42, max_iter=1000)
    W = nmf.fit_transform(X_nn)   # (n_attacks, 2)
    H = nmf.components_           # (2, n_sdg)
    print(f"\nNMF reconstruction error: {nmf.reconstruction_err_:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax_idx, (ax, title, atk_x, atk_y, sdg_xs, sdg_ys, xlabel, ylabel) in enumerate([
        (axes[0],
         f"PCA Biplot  (PC1={ev[0]*100:.0f}%, PC2={ev[1]*100:.0f}%)",
         scores[:, 0], scores[:, 1],
         loadings[:, 0] * np.sqrt(pca.explained_variance_[0]),
         loadings[:, 1] * np.sqrt(pca.explained_variance_[0]),
         f"PC1  ({ev[0]*100:.1f}% variance)",
         f"PC2  ({ev[1]*100:.1f}% variance)"),
        (axes[1],
         "NMF Biplot  (component 1 vs 2)",
         W[:, 0], W[:, 1],
         H[0, :] * 0.4, H[1, :] * 0.4,
         "NMF component 1",
         "NMF component 2"),
    ]):
        # SDG arrows
        for j, sdg in enumerate(sdg_names):
            fam = sdg_family(sdg)
            col = SDG_FAMILY_COLORS[fam]
            _arrow(ax, sdg_xs[j], sdg_ys[j], sdg, col, fontsize=6.5)

        # Attack scatter
        for i, name in enumerate(attack_names):
            fam = attack_family(name)
            col = FAMILY_COLORS.get(fam, "#333333")
            ax.scatter(atk_x[i], atk_y[i], c=col, s=70, zorder=5,
                       edgecolors="white", linewidths=0.4)

        _annotate_scatter(ax, atk_x, atk_y, attack_names, fontsize=6.5)

        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax.axvline(0, color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Legends
        atk_patches = [mpatches.Patch(color=c, label=f)
                       for f, c in FAMILY_COLORS.items()
                       if f in {attack_family(n) for n in attack_names}]
        sdg_patches = [mpatches.Patch(color=c, label=f)
                       for f, c in SDG_FAMILY_COLORS.items()
                       if f in {sdg_family(s) for s in sdg_names}]
        leg1 = ax.legend(handles=atk_patches, title="Attack family",
                         fontsize=7, loc="upper left")
        ax.add_artist(leg1)
        ax.legend(handles=sdg_patches, title="SDG family",
                  fontsize=7, loc="lower right")

    plt.tight_layout()
    out = f"{out_prefix}_phase1.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()

# ── Phase 2: Variance decomposition ──────────────────────────────────────────

def run_phase2(df, out_prefix):
    print(f"\n=== Phase 2: Variance decomposition ===")
    print(f"  Rows: {len(df)}")

    df = df.copy()
    df["epsilon"]     = df["sdg_method"].apply(parse_epsilon)
    df["log_epsilon"] = np.log10(df["epsilon"].clip(lower=0.01))
    df["has_dp"]      = df["epsilon"].notna()
    df["attack_fam"]  = df["attack_label"].apply(attack_family)
    df["sdg_fam"]     = df["sdg_method"].apply(sdg_family)
    df["log_size"]    = np.log10(df["dataset_size"])

    # Epsilon bin (for grouped factor)
    def eps_bin(eps):
        if pd.isna(eps): return "none"
        if eps <= 0.3:   return "≤0.3"
        if eps <= 1:     return "0.3–1"
        if eps <= 10:    return "1–10"
        return ">10"
    df["eps_bin"] = df["epsilon"].apply(eps_bin)

    grand_mean = df["ra_mean"].mean()
    ss_total   = ((df["ra_mean"] - grand_mean) ** 2).sum()

    factors = {
        "attack_label": "Attack identity",
        "attack_fam":   "Attack family",
        "sdg_method":   "SDG method (specific)",
        "sdg_fam":      "SDG family",
        "has_dp":       "Uses DP at all",
        "eps_bin":      "DP epsilon tier",
        "dataset":      "Dataset domain",
        "log_size":     "Dataset size (log₁₀)",
        "qi":           "QI variant",
    }

    print(f"\n  Grand mean RA: {grand_mean:.2f}%")
    print(f"\n  {'Factor':<35} {'η²':>8}   {'n groups':>8}")
    print(f"  {'-'*60}")

    results = []
    for col, label in factors.items():
        grp = df.groupby(col)["ra_mean"]
        gm  = grp.mean()
        gc  = grp.count()
        ss_between = sum(gc[g] * (gm[g] - grand_mean) ** 2 for g in gm.index)
        eta2 = ss_between / ss_total
        results.append((label, eta2, len(gm)))
        print(f"  {label:<35} {eta2*100:>7.1f}%   {len(gm):>8d}")

    results.sort(key=lambda x: -x[1])
    labels, eta2s, _ = zip(*results)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.7, len(results)))
    bars = ax.barh(list(labels)[::-1], [e * 100 for e in eta2s[::-1]],
                   color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("η²  —  fraction of total RA variance explained  (%)", fontsize=10)
    ax.set_title("What drives reconstruction attack success?\n(one-way variance decomposition across all results.db rows)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(e * 100 for e in eta2s) * 1.25)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    plt.tight_layout()
    out = f"{out_prefix}_phase2.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db",        default=DB_PATH)
    parser.add_argument("--dataset",   default="adult")
    parser.add_argument("--size",      type=int, default=10000)
    parser.add_argument("--qi",        default="QI1")
    parser.add_argument("--out",       default="experiment_scripts/decomp")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--phase2-only", action="store_true")
    args = parser.parse_args()

    if not args.phase2_only:
        pivot = load_pivot(args.db, args.dataset, args.size, args.qi)
        run_phase1(pivot, args.out)

    if not args.phase1_only:
        df = load_all(args.db)
        run_phase2(df, args.out)


if __name__ == "__main__":
    main()
