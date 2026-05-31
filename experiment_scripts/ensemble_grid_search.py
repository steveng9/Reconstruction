#!/usr/bin/env python
"""
Grid search over ensemble weight combinations using pre-saved attack probas.

Loads pickled (probas, classes, recon) written by save_attack_probas.py, then
sweeps over weight combinations without re-running any attack.

Key features:
  - Default: 4-attack subsets that exclude LGB (avoids overconfidence collapse)
  - Any subset configurable via --attacks
  - Weight grid in 0.1 increments (~286 combos for 4 attacks; ~1001 for 5)
  - Per-(sample, SDG) evaluation → mean RA aggregated across all
  - Per-SDG breakdown for top results
  - Results saved to CSV for further analysis

Usage:
    python experiment_scripts/ensemble_grid_search.py
    python experiment_scripts/ensemble_grid_search.py --step 0.05 --top 30
    python experiment_scripts/ensemble_grid_search.py --attacks MarginalRF MLP KNN NaiveBayes
    python experiment_scripts/ensemble_grid_search.py --attacks MarginalRF LightGBM MLP KNN NaiveBayes
    python experiment_scripts/ensemble_grid_search.py --sdg TabDDPM --out results_tabddpm.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/golobs/Reconstruction")
from scoring import calculate_reconstruction_score
from enhancements.ensembling_wrapper import _soft_voting


PROBAS_DIR   = Path(__file__).parent.parent / "outfiles" / "probas"
DATA_ROOT    = "/home/golobs/data/reconstruction_data/adult/size_10000"
ALL_ATTACKS  = ["MarginalRF", "LightGBM", "NaiveBayes", "KNN", "MLP"]

WEIGHT_CAPS = {
    "NaiveBayes": 0.5,
    "KNN":        0.5,
    "LightGBM":   0.6,
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_probas(
    attacks:       list[str],
    sdg_filter:    str | None = None,
    sample_filter: int | None = None,
) -> dict[tuple[int, str], dict[str, dict]]:
    """Return {(sample_idx, sdg_label): {attack_label: payload}}."""
    data: dict[tuple[int, str], dict] = defaultdict(dict)

    all_pkls = sorted(PROBAS_DIR.glob("*/*/*.pkl"))
    n_pkls   = len(all_pkls)
    for i, pkl_path in enumerate(all_pkls):
        sample_dir = pkl_path.parent.parent.name   # "sample_00"
        sdg_label  = pkl_path.parent.name           # "MST_eps1"
        atk_label  = pkl_path.stem                  # "MarginalRF"

        sample_idx = int(sample_dir.split("_")[1])

        if sdg_filter:
            if sdg_label != sdg_filter and not sdg_label.startswith(sdg_filter):
                continue
        if sample_filter is not None and sample_idx != sample_filter:
            continue
        if atk_label not in attacks:
            continue

        print(f"  loading [{i+1:>3}/{n_pkls}] {sample_dir}/{sdg_label}/{atk_label}.pkl",
              flush=True)
        with open(pkl_path, "rb") as f:
            data[(sample_idx, sdg_label)][atk_label] = pickle.load(f)

    return dict(data)


# ── Weight grid ────────────────────────────────────────────────────────────────

def weight_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """All weight n-tuples summing to 1.0 in increments of step.

    Uses integer arithmetic to avoid floating-point drift, then converts.
    """
    k = round(1.0 / step)
    combos = []
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) == k:
            combos.append(tuple(round(x * step, 10) for x in c))
    return combos


# ── Ensemble scoring ───────────────────────────────────────────────────────────

def score_ensemble(
    job_data:   dict[str, dict],
    attacks:    list[str],
    weights:    tuple[float, ...],
    train_df:   pd.DataFrame,
) -> float | None:
    """Weighted soft-vote RA for one (sample, SDG) pair. Returns None on failure."""
    for atk in attacks:
        if atk not in job_data:
            return None

    hidden_features = job_data[attacks[0]]["hidden_features"]
    norm_w = np.array(weights, dtype=float)
    norm_w = norm_w / norm_w.sum()

    all_recons  = [job_data[a]["recon"]   for a in attacks]
    all_probas  = [job_data[a]["probas"]  for a in attacks]
    all_classes = [job_data[a]["classes"] for a in attacks]

    recon = all_recons[0].copy()
    for feat_idx, feat in enumerate(hidden_features):
        feat_preds = [r[feat] for r in all_recons]
        recon[feat] = _soft_voting(feat_preds, all_probas, all_classes, feat_idx, norm_w)

    scores = calculate_reconstruction_score(train_df, recon, hidden_features)
    return float(np.mean(scores))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over ensemble weights from saved attack probas.")
    parser.add_argument("--attacks", nargs="+",
                        default=["MarginalRF", "MLP", "KNN", "NaiveBayes"],
                        choices=ALL_ATTACKS,
                        help="Attacks to include (default excludes LGB).")
    parser.add_argument("--step",   type=float, default=0.1,
                        help="Weight step size (default 0.1).")
    parser.add_argument("--top",    type=int,   default=20,
                        help="Top-K results to print.")
    parser.add_argument("--sdg",    type=str,   default=None,
                        help="Filter to one SDG method.")
    parser.add_argument("--sample", type=int,   default=None,
                        help="Filter to one sample index.")
    parser.add_argument("--out",    type=str,   default=None,
                        help="Save full results to CSV.")
    args = parser.parse_args()

    attacks = args.attacks
    n       = len(attacks)

    print(f"\nAttacks ({n}):  {attacks}")
    print(f"Weight step:  {args.step}")

    # Load saved probas
    print(f"\nLoading probas from {PROBAS_DIR} ...")
    all_data = load_all_probas(
        attacks=attacks,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )
    if not all_data:
        print("  No pickles found. Run save_attack_probas.py first.")
        return

    # Report coverage
    sdg_labels = sorted({sdg for _, sdg in all_data})
    samples    = sorted({s for s, _ in all_data})
    print(f"  Loaded {len(all_data)} (sample, SDG) pairs")
    print(f"  SDGs:    {sdg_labels}")
    print(f"  Samples: {samples}")

    # Per-job coverage — report which attacks are present
    coverage: dict[str, int] = defaultdict(int)
    for job_data in all_data.values():
        for atk in attacks:
            if atk in job_data:
                coverage[atk] += 1
    print(f"  Coverage: " + ", ".join(f"{a}={coverage[a]}/{len(all_data)}" for a in attacks))

    # Load train data (per sample index)
    train_cache: dict[int, pd.DataFrame] = {}
    for sample_idx in samples:
        p = Path(DATA_ROOT) / f"sample_{sample_idx:02d}" / "train.csv"
        train_cache[sample_idx] = pd.read_csv(p)

    # Generate weight grid (exclude all-zero configs and cap per-attack weights)
    default_cap = 0.8
    caps = tuple(WEIGHT_CAPS.get(a, default_cap) for a in attacks)

    def within_caps(w: tuple[float, ...]) -> bool:
        return all(wi <= cap + 1e-9 for wi, cap in zip(w, caps))

    raw_grid = [w for w in weight_grid(n, args.step) if max(w) > 0]
    grid     = [w for w in raw_grid if within_caps(w)]
    print(f"\nCaps: " + ", ".join(f"{a}≤{c:.1f}" for a, c in zip(attacks, caps)))
    print(f"Sweeping {len(grid)}/{len(raw_grid)} weight combinations after cap filter ...")

    # Score every weight combo across all jobs
    results: list[dict] = []
    best_ra   = -float("inf")
    best_desc = ""
    n_grid    = len(grid)
    import time as _time
    t0 = _time.monotonic()

    for gi, weight_combo in enumerate(grid):
        per_job:  list[float] = []
        per_sdg:  dict[str, list[float]] = defaultdict(list)

        for (sample_idx, sdg_label), job_data in all_data.items():
            ra = score_ensemble(job_data, attacks, weight_combo, train_cache[sample_idx])
            if ra is not None:
                per_job.append(ra)
                per_sdg[sdg_label].append(ra)

        if not per_job:
            continue

        mean_ra = round(float(np.mean(per_job)), 4)
        row: dict = {
            "weights_str": " ".join(f"{a}={w:.2f}" for a, w in zip(attacks, weight_combo)),
            "mean_ra":     mean_ra,
            "n_jobs":      len(per_job),
        }
        for a, w in zip(attacks, weight_combo):
            row[f"w_{a}"] = w
        for sdg, vals in per_sdg.items():
            row[f"ra_{sdg}"] = round(float(np.mean(vals)), 4)
        results.append(row)

        if mean_ra > best_ra:
            best_ra   = mean_ra
            best_desc = row["weights_str"]

        done    = gi + 1
        elapsed = _time.monotonic() - t0
        eta     = (elapsed / done) * (n_grid - done) if done > 1 else float("nan")
        pct     = 100.0 * done / n_grid
        w_str   = row["weights_str"]
        eta_str = f"ETA {int(eta)}s" if not (eta != eta) else "ETA --"
        print(f"  [{done:>{len(str(n_grid))}}/{n_grid}]  {pct:5.1f}%  "
              f"this={mean_ra:.4f}  best={best_ra:.4f}  ({best_desc})  {eta_str}",
              flush=True)

    if not results:
        print("No results scored.")
        return

    # Sort by mean RA
    results.sort(key=lambda r: r["mean_ra"], reverse=True)

    # Individual baselines (weight=1.0 for that attack, 0 for others)
    baselines = {}
    for i, atk in enumerate(attacks):
        w = tuple(1.0 if j == i else 0.0 for j in range(n))
        for r in results:
            if all(abs(r[f"w_{a}"] - w[j]) < 1e-9 for j, a in enumerate(attacks)):
                baselines[atk] = r["mean_ra"]
                break

    # Print top-K
    print(f"\n{'='*78}")
    print(f"  Top {min(args.top, len(results))} ensembles")
    print(f"{'='*78}")

    # Column header
    w_cols = "  ".join(f"{a[:5]:>7}" for a in attacks)
    print(f"  {'Mean RA':>8}  {w_cols}")
    print(f"  {'-'*74}")

    for r in results[:args.top]:
        w_vals = "  ".join(f"{r[f'w_{a}']:>7.2f}" for a in attacks)
        print(f"  {r['mean_ra']:>8.4f}  {w_vals}")

    # Individual baselines for reference
    print(f"\n  Individual baselines:")
    for atk, ra in sorted(baselines.items(), key=lambda x: -x[1]):
        print(f"    {atk:<15}  {ra:.4f}")

    # Per-SDG breakdown for top 5
    sdg_keys = sorted({k for r in results for k in r if k.startswith("ra_")})
    if sdg_keys:
        print(f"\n  Per-SDG breakdown for top 5:")
        sdg_header = "  ".join(f"{k[3:]:>12}" for k in sdg_keys)
        print(f"  {'Weights':<45}  {sdg_header}")
        print(f"  {'-'*80}")
        for r in results[:5]:
            w_str  = r["weights_str"]
            sdg_vals = "  ".join(f"{r.get(k, float('nan')):>12.4f}" for k in sdg_keys)
            print(f"  {w_str:<45}  {sdg_vals}")

    # Save CSV
    if args.out:
        fieldnames = (
            ["weights_str", "mean_ra", "n_jobs"]
            + [f"w_{a}" for a in attacks]
            + sorted({k for r in results for k in r if k.startswith("ra_")})
        )
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Full results ({len(results)} rows) → {args.out}")


if __name__ == "__main__":
    main()
