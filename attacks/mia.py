"""
Membership Inference Attacks (MIA) for synthetic tabular data.

Attacks
-------
SynthDistance
    score = -min_dist(target, synth).
    Members tend to be closer to synthetic data because the generative model
    was trained on them. No reference dataset required.

NNDR  (Nearest-Neighbour Distance Ratio)
    score = dist(target, train_LOO) / dist(target, synth).
    Normalises synth proximity by proximity to the training distribution,
    correcting for records that are inherently "common" in the population.
    Uses leave-one-out on training targets to avoid trivial self-matches.

Distance metric
---------------
Gower distance — per-column values in [0, 1], then averaged:
  Continuous : |x - y| / column_range
  Categorical: 0 if equal, 1 if different

Metrics reported
----------------
MIA_auc, MIA_advantage, MIA_tpr_at_fpr0001, MIA_balanced_acc
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

_CHUNK = 500  # reference records per chunk (controls peak memory)


# ── Distance helpers ──────────────────────────────────────────────────────────

def _col_ranges(df: pd.DataFrame, num_cols: list) -> dict:
    """Per-column range fitted on df, used for Gower normalisation."""
    return {
        c: max(float(df[c].max() - df[c].min()), 1e-10)
        for c in num_cols if c in df.columns
    }


def _gower_block(A: pd.DataFrame, B: pd.DataFrame,
                 cat_cols: list, num_cols: list, ranges: dict) -> np.ndarray:
    """Return (n_A, n_B) Gower distance matrix."""
    n_cols = len(cat_cols) + len(num_cols)
    if n_cols == 0:
        return np.zeros((len(A), len(B)))

    total = np.zeros((len(A), len(B)))

    for col in cat_cols:
        if col not in A.columns or col not in B.columns:
            continue
        a = A[col].to_numpy(dtype=object)
        b = B[col].to_numpy(dtype=object)
        total += (a[:, None] != b[None, :]).astype(float)

    for col in num_cols:
        if col not in A.columns or col not in B.columns:
            continue
        r = ranges.get(col, 1.0)
        a = A[col].to_numpy(dtype=float)
        b = B[col].to_numpy(dtype=float)
        total += np.abs(a[:, None] - b[None, :]) / r

    return total / n_cols


def _min_gower_dist(targets: pd.DataFrame, reference: pd.DataFrame,
                    cat_cols: list, num_cols: list, ranges: dict) -> np.ndarray:
    """Min Gower distance from each target row to any reference row (chunked)."""
    min_d = np.full(len(targets), np.inf)
    for start in range(0, len(reference), _CHUNK):
        block = reference.iloc[start:start + _CHUNK]
        d = _gower_block(targets, block, cat_cols, num_cols, ranges)
        min_d = np.minimum(min_d, d.min(axis=1))
    return min_d


def _min_gower_dist_loo(train_targets: pd.DataFrame, train_full: pd.DataFrame,
                        cat_cols: list, num_cols: list, ranges: dict) -> np.ndarray:
    """
    Leave-one-out min Gower distance: each train_target is assumed to appear
    in train_full, so its exact self-match (distance ≈ 0) is excluded.
    """
    result = np.full(len(train_targets), np.inf)
    for start in range(0, len(train_full), _CHUNK):
        block = train_full.iloc[start:start + _CHUNK]
        d = _gower_block(train_targets, block, cat_cols, num_cols, ranges)
        d[d < 1e-10] = np.inf   # mask self-matches
        result = np.minimum(result, d.min(axis=1))
    # Degenerate case: every neighbour was masked (all-duplicate training set)
    result = np.where(np.isinf(result), 0.0, result)
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def _mia_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Standard MIA evaluation metrics.
    scores : higher → more likely member
    labels : 1 = member (train), 0 = non-member (holdout)
    """
    auc = roc_auc_score(labels, scores)
    fprs, tprs, _ = roc_curve(labels, scores)

    advantage = float(np.max(tprs - fprs))

    # TPR at the highest threshold where FPR ≤ 0.1 %
    valid = np.where(fprs <= 0.001)[0]
    tpr_at_low_fpr = float(tprs[valid[-1]]) if len(valid) > 0 else 0.0

    balanced_acc = float(np.max((tprs + (1 - fprs)) / 2))

    return {
        "MIA_auc":            round(auc, 4),
        "MIA_advantage":      round(advantage, 4),
        "MIA_tpr_at_fpr0001": round(tpr_at_low_fpr, 4),
        "MIA_balanced_acc":   round(balanced_acc, 4),
    }


# ── Shared setup ──────────────────────────────────────────────────────────────

def _setup(synth_df, train_df, holdout_df, meta, cfg):
    """Resolve columns, ranges, and balanced target sample."""
    cat_cols = [c for c in meta.get("categorical", []) if c in synth_df.columns]
    num_cols = [c for c in meta.get("continuous", [])  if c in synth_df.columns]
    ranges   = _col_ranges(train_df, num_cols)

    n_each = cfg.get("mia_params", {}).get("n_targets", 500)
    n_each = min(n_each, len(train_df), len(holdout_df))
    seed   = cfg.get("mia_params", {}).get("seed", 42)

    train_tgts   = train_df.sample(n=n_each,   random_state=seed).reset_index(drop=True)
    holdout_tgts = holdout_df.sample(n=n_each, random_state=seed).reset_index(drop=True)
    labels = np.array([1] * n_each + [0] * n_each)

    return cat_cols, num_cols, ranges, train_tgts, holdout_tgts, labels


# ── Attack functions ──────────────────────────────────────────────────────────

def synth_distance_mia(cfg, synth_df, train_df, holdout_df, meta):
    """
    SynthDistance MIA.
    Score = -min_dist(target, synth).  Higher → closer to synth → more likely member.
    """
    cat_cols, num_cols, ranges, train_tgts, holdout_tgts, labels = \
        _setup(synth_df, train_df, holdout_df, meta, cfg)

    all_tgts = pd.concat([train_tgts, holdout_tgts], ignore_index=True)
    d_synth  = _min_gower_dist(all_tgts, synth_df, cat_cols, num_cols, ranges)
    scores   = -d_synth

    print(f"  [SynthDistance] d_synth  members: {d_synth[:len(train_tgts)].mean():.4f}"
          f"  non-members: {d_synth[len(train_tgts):].mean():.4f}")
    return _mia_metrics(scores, labels)


def nndr_mia(cfg, synth_df, train_df, holdout_df, meta):
    """
    NNDR MIA.
    Score = dist(target, train_LOO) / dist(target, synth).
    Higher → closer to synth relative to training distribution → more likely member.
    """
    cat_cols, num_cols, ranges, train_tgts, holdout_tgts, labels = \
        _setup(synth_df, train_df, holdout_df, meta, cfg)

    all_tgts = pd.concat([train_tgts, holdout_tgts], ignore_index=True)
    d_synth  = _min_gower_dist(all_tgts, synth_df, cat_cols, num_cols, ranges)

    # Leave-one-out for training targets; standard NN for holdout targets
    d_train_m = _min_gower_dist_loo(train_tgts,   train_df, cat_cols, num_cols, ranges)
    d_train_h = _min_gower_dist    (holdout_tgts, train_df, cat_cols, num_cols, ranges)
    d_train   = np.concatenate([d_train_m, d_train_h])

    scores = d_train / np.maximum(d_synth, 1e-10)

    print(f"  [NNDR] ratio  members: {scores[:len(train_tgts)].mean():.4f}"
          f"  non-members: {scores[len(train_tgts):].mean():.4f}")
    return _mia_metrics(scores, labels)


# ── Registry ──────────────────────────────────────────────────────────────────

MIA_REGISTRY = {
    "SynthDistance": synth_distance_mia,
    "NNDR":          nndr_mia,
}
