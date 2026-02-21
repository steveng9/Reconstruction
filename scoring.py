from pathlib import Path

import numpy as np
import pandas as pd
import pickle



def calculate_reconstruction_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        value_counts = df_original[col].value_counts()
        rarity_scores = df_original[col].map(total_records / value_counts)
        max_score = rarity_scores.sum()

        score = ( (df_original[col].values == df_reconstructed[col].values) * rarity_scores ).sum()
        scores.append(round(score / max_score * 100, 1))
    return scores


def calculate_continuous_vals_reconstruction_score(train, reconstruction, hidden_features):
    results = {}
    for hidden_feature in hidden_features:
        real = train[hidden_feature].values
        recon = reconstruction[hidden_feature].values

        # Normalize by range of real data
        data_range = real.max() - real.min()

        if data_range == 0:
            # Constant column
            normalized_error = 0 if np.allclose(real, recon) else np.inf
        else:
            # Normalized absolute error
            normalized_error = np.abs(real - recon) / data_range

        results[hidden_feature] = {
            'mean_abs_error': np.mean(np.abs(real - recon)),
            'normalized_mae': np.mean(normalized_error),
            'mse': np.mean((real - recon) ** 2),
            'rmse': np.sqrt(np.mean((real - recon) ** 2)),
            'normalized_rmse': np.sqrt(np.mean(normalized_error ** 2)),
            'max_error': np.max(np.abs(real - recon))
        }

    return pd.DataFrame(results).T



def simple_accuracy_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        score = (df_original[col].values == df_reconstructed[col].values).sum()
        scores.append(round(score / total_records * 100, 1))
    return scores


# ── Row-level scoring ──────────────────────────────────────────────────────────

def calculate_row_level_scores(original, reconstructed, hidden_features, dataset_type):
    """
    Compute per-row reconstruction accuracy scores.

    Returns a DataFrame with n_rows × score columns:
      - RA_row_{feat}: per-cell score for each hidden feature
      - RA_row_mean:   aggregate row-level score

    For categorical: per-cell is 0/1 correctness; mean is rarity-weighted.
    For continuous:  per-cell is normalized abs error in [0,1] (lower=better);
                     mean is the average across features.
    """
    if dataset_type == "continuous":
        return _row_level_continuous(original, reconstructed, hidden_features)
    else:
        return _row_level_categorical(original, reconstructed, hidden_features)


def _row_level_categorical(original, reconstructed, hidden_features):
    """
    Per-row categorical reconstruction scores.

    Per-cell column RA_row_{feat}: binary 0/1 correctness.
    RA_row_mean: rarity-weighted row score.
      score_i = Σ_j [correct_ij × rarity(val_ij)] / Σ_j rarity(val_ij)
      where rarity(val) = total_records / count(val).
    """
    n = len(original)
    out = pd.DataFrame(index=original.index)

    # Rarity map per feature: value → rarity weight
    rarity_maps = {}
    for feat in hidden_features:
        counts = original[feat].value_counts()
        rarity_maps[feat] = n / counts  # Series: val → rarity

    # Per-cell correctness columns
    for feat in hidden_features:
        correct = (original[feat].values == reconstructed[feat].values).astype(float)
        out[f"RA_row_{feat}"] = correct

    # Rarity-weighted mean across features
    weighted_sum = pd.Series(0.0, index=original.index)
    total_weight = pd.Series(0.0, index=original.index)

    for feat in hidden_features:
        rarity = original[feat].map(rarity_maps[feat]).fillna(1.0).values
        correct = out[f"RA_row_{feat}"].values
        weighted_sum += correct * rarity
        total_weight += rarity

    denom = total_weight.replace(0, np.nan)
    out["RA_row_mean"] = (weighted_sum / denom).fillna(0.0)

    return out


def _row_level_continuous(original, reconstructed, hidden_features):
    """
    Per-row continuous reconstruction scores.

    Per-cell column RA_row_{feat}: normalized absolute error |real - pred| / range
    in [0, 1] (lower = better reconstruction).
    RA_row_mean: mean across features.
    """
    out = pd.DataFrame(index=original.index)

    for feat in hidden_features:
        real = original[feat].values.astype(float)
        pred = reconstructed[feat].values.astype(float)
        rng = float(real.max() - real.min())
        if rng == 0:
            norm_err = np.zeros(len(real))
        else:
            norm_err = np.abs(real - pred) / rng
        out[f"RA_row_{feat}"] = norm_err

    feat_cols = [f"RA_row_{f}" for f in hidden_features]
    out["RA_row_mean"] = out[feat_cols].mean(axis=1)

    return out


def compute_outlier_scores(df, qi_cols, cat_cols, num_cols,
                           method='isolation_forest', percentile=90):
    """
    Compute outlier scores in QI feature space.

    Operates on the QI columns (public feature space) — uniqueness in QI space
    is the relevant privacy risk dimension.

    Parameters
    ----------
    df          : DataFrame of target records
    qi_cols     : list of QI column names to use
    cat_cols    : categorical columns (subset of qi_cols)
    num_cols    : continuous columns (subset of qi_cols)
    method      : 'isolation_forest' or 'gower_knn'
    percentile  : rows above this percentile of score are flagged is_outlier

    Returns
    -------
    outlier_score : pd.Series (higher = more outlier-like), indexed like df
    is_outlier    : pd.Series of bool, indexed like df
    """
    qi_cat = [c for c in cat_cols if c in qi_cols and c in df.columns]
    qi_num = [c for c in num_cols if c in qi_cols and c in df.columns]

    if method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler

        X_parts = []
        if qi_cat:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_parts.append(enc.fit_transform(df[qi_cat].astype(str)))
        if qi_num:
            scaler = StandardScaler()
            X_parts.append(scaler.fit_transform(df[qi_num].astype(float)))

        if not X_parts:
            scores = pd.Series(0.0, index=df.index)
        else:
            X = np.hstack(X_parts)
            iso = IsolationForest(random_state=42, contamination='auto')
            iso.fit(X)
            # -score_samples() → higher = more outlier-like
            scores = pd.Series(-iso.score_samples(X), index=df.index)

    elif method == 'gower_knn':
        import sys, os
        _attacks_dir = os.path.join(os.path.dirname(__file__), 'attacks')
        if _attacks_dir not in sys.path:
            sys.path.insert(0, _attacks_dir)
        from mia import _min_gower_dist_loo, _col_ranges

        ranges = _col_ranges(df, qi_num)
        dist = _min_gower_dist_loo(df[qi_cols], df[qi_cols], qi_cat, qi_num, ranges)
        scores = pd.Series(dist, index=df.index)

    else:
        raise ValueError(
            f"Unknown outlier method: {method!r}. Use 'isolation_forest' or 'gower_knn'."
        )

    threshold = np.percentile(scores.values, percentile)
    is_outlier = scores >= threshold

    return scores, is_outlier
