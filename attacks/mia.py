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
    """Resolve columns, ranges, and balanced target sample.

    If mia_params.n_targets is None, uses the full train and holdout DataFrames
    (no sampling).  Otherwise samples n_targets rows from each (balanced).
    """
    cat_cols = [c for c in meta.get("categorical", []) if c in synth_df.columns]
    num_cols = [c for c in meta.get("continuous", [])  if c in synth_df.columns]
    ranges   = _col_ranges(train_df, num_cols)

    n_each = cfg.get("mia_params", {}).get("n_targets", 500)
    seed   = cfg.get("mia_params", {}).get("seed", 42)

    if n_each is None:
        train_tgts   = train_df.reset_index(drop=True)
        holdout_tgts = holdout_df.reset_index(drop=True)
    else:
        n_each       = min(n_each, len(train_df), len(holdout_df))
        train_tgts   = train_df.sample(n=n_each,   random_state=seed).reset_index(drop=True)
        holdout_tgts = holdout_df.sample(n=n_each, random_state=seed).reset_index(drop=True)

    labels = np.array([1] * len(train_tgts) + [0] * len(holdout_tgts))

    return cat_cols, num_cols, ranges, train_tgts, holdout_tgts, labels


# ── Attack functions ──────────────────────────────────────────────────────────

def synth_distance_mia(cfg, synth_df, train_df, holdout_df, meta, return_raw=False):
    """
    SynthDistance MIA.
    Score = -min_dist(target, synth).  Higher → closer to synth → more likely member.

    Parameters
    ----------
    return_raw : bool
        If True, returns (metrics_dict, scores, labels, all_targets_df) instead of
        just metrics_dict.  Default False for backward compatibility.
    """
    cat_cols, num_cols, ranges, train_tgts, holdout_tgts, labels = \
        _setup(synth_df, train_df, holdout_df, meta, cfg)

    all_tgts = pd.concat([train_tgts, holdout_tgts], ignore_index=True)
    d_synth  = _min_gower_dist(all_tgts, synth_df, cat_cols, num_cols, ranges)
    scores   = -d_synth

    print(f"  [SynthDistance] d_synth  members: {d_synth[:len(train_tgts)].mean():.4f}"
          f"  non-members: {d_synth[len(train_tgts):].mean():.4f}")
    metrics = _mia_metrics(scores, labels)
    if return_raw:
        return metrics, scores, labels, all_tgts
    return metrics


def nndr_mia(cfg, synth_df, train_df, holdout_df, meta, return_raw=False):
    """
    NNDR MIA.
    Score = dist(target, train_LOO) / dist(target, synth).
    Higher → closer to synth relative to training distribution → more likely member.

    Parameters
    ----------
    return_raw : bool
        If True, returns (metrics_dict, scores, labels, all_targets_df) instead of
        just metrics_dict.  Default False for backward compatibility.
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
    metrics = _mia_metrics(scores, labels)
    if return_raw:
        return metrics, scores, labels, all_tgts
    return metrics


def ra_as_mia(attack_fn, cfg, synth_df, train_df, holdout_df, qi, hidden_features,
              n_targets=500, seed=42):
    """
    RA-as-MIA: use reconstruction accuracy as the membership inference signal.

    The hypothesis: the synthetic data reconstructs hidden features better for
    training members (the model memorised them) than for holdout non-members.
    Higher reconstruction quality → higher membership score.

    Parameters
    ----------
    attack_fn       : callable with signature (cfg, synth, targets, qi, hidden)
                      → (reconstructed_df, probas, classes)
    cfg             : experiment config dict
    synth_df        : synthetic data DataFrame
    train_df        : training records (members, label=1)
    holdout_df      : holdout records (non-members, label=0)
    qi              : list of quasi-identifier column names
    hidden_features : list of hidden feature column names
    n_targets       : number of records to sample from each group
    seed            : random seed for reproducible sampling

    Returns
    -------
    metrics    : dict with keys RA_as_MIA_{auc,advantage,tpr_at_fpr0001,balanced_acc}
    scores     : np.ndarray of per-record membership scores (higher = more likely member)
    labels     : np.ndarray of ground-truth labels (1=member, 0=non-member)
    all_targets: pd.DataFrame of sampled records (train first, then holdout)
    """
    if n_targets is None:
        train_tgts   = train_df.reset_index(drop=True)
        holdout_tgts = holdout_df.reset_index(drop=True)
    else:
        n_each       = min(n_targets, len(train_df), len(holdout_df))
        train_tgts   = train_df.sample(n=n_each,   random_state=seed).reset_index(drop=True)
        holdout_tgts = holdout_df.sample(n=n_each, random_state=seed).reset_index(drop=True)
    all_targets  = pd.concat([train_tgts, holdout_tgts], ignore_index=True)
    labels       = np.array([1] * len(train_tgts) + [0] * len(holdout_tgts))

    # Run the reconstruction attack on all targets
    reconstructed, _, _ = attack_fn(cfg, synth_df, all_targets, qi, hidden_features)

    # Compute per-row membership score inline (no scoring.py import to stay portable)
    dataset_type = cfg.get("dataset", {}).get("type", "categorical")

    if dataset_type == "continuous":
        # Per-row mean normalized abs error; negate so higher = better reconstruction = member
        per_row_errors = []
        for feat in hidden_features:
            real = all_targets[feat].values.astype(float)
            pred = reconstructed[feat].values.astype(float)
            rng = float(real.max() - real.min())
            if rng == 0:
                per_row_errors.append(np.zeros(len(real)))
            else:
                per_row_errors.append(np.abs(real - pred) / rng)
        mean_err = np.mean(np.stack(per_row_errors, axis=1), axis=1)
        scores   = -mean_err  # negate: lower error = higher membership score
    else:
        # Rarity-weighted correctness; already higher = better = more likely member
        n_total = len(train_df)  # use full train for rarity estimation
        weighted_sum = np.zeros(len(all_targets))
        total_weight = np.zeros(len(all_targets))

        for feat in hidden_features:
            counts  = train_df[feat].value_counts()
            rarity  = (n_total / counts).to_dict()
            r_vals  = np.array([rarity.get(v, 1.0) for v in all_targets[feat]])
            correct = (all_targets[feat].values == reconstructed[feat].values).astype(float)
            weighted_sum += correct * r_vals
            total_weight += r_vals

        denom  = np.where(total_weight > 0, total_weight, 1.0)
        scores = weighted_sum / denom

    print(f"  [RA-as-MIA] score  members: {scores[:n_each].mean():.4f}"
          f"  non-members: {scores[n_each:].mean():.4f}")

    raw_metrics = _mia_metrics(scores, labels)
    # Re-prefix keys as RA_as_MIA_*
    metrics = {k.replace("MIA_", "RA_as_MIA_"): v for k, v in raw_metrics.items()}

    return metrics, scores, labels, all_targets


# ── Registry ──────────────────────────────────────────────────────────────────

MIA_REGISTRY = {
    "SynthDistance": synth_distance_mia,
    "NNDR":          nndr_mia,
}
