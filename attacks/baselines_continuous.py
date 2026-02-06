import pandas as pd
import numpy as np


def simply_measure_deid_itself_baseline_cont(cfg, deid, targets, qi, hidden_features):
    """
    Baseline that cycles through the deid dataset to fill targets.
    Works for both categorical and continuous data.
    """
    reconstructed_targets = targets.copy()
    full_cycles = len(deid) // len(targets)
    remainder = len(deid) % len(targets)

    result_parts = [deid] * full_cycles
    if remainder > 0:
        result_parts.append(deid.iloc[:remainder])

    reconstructed_targets[hidden_features] = pd.concat(result_parts, ignore_index=True)[hidden_features]

    return reconstructed_targets, None, None


def random_baseline_cont(cfg, deid, targets, qi, hidden_features):
    """
    Randomly sample from the deid distribution for each hidden feature.
    For continuous data, this samples actual values from the deid dataset.
    """
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        # Sample with replacement from deid for each target row
        reconstructed_targets[hidden_feature] = deid[hidden_feature].sample(
            n=len(targets),
            replace=True
        ).values
    return reconstructed_targets, None, None


def mean_baseline_cont(cfg, deid, targets, qi, hidden_features):
    """
    Fill all hidden features with the mean value from deid.
    Simple baseline for continuous data.
    """
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        reconstructed_targets[hidden_feature] = deid[hidden_feature].mean()
    return reconstructed_targets, None, None


def median_baseline_cont(cfg, deid, targets, qi, hidden_features):
    """
    Fill all hidden features with the median value from deid.
    More robust to outliers than mean.
    """
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        reconstructed_targets[hidden_feature] = deid[hidden_feature].median()
    return reconstructed_targets, None, None


# TODO: fix these two conditional baselines
# def conditional_mean_baseline_cont(cfg, deid, targets, qi, hidden_features):
#     """
#     Fill hidden features with conditional mean based on quasi-identifiers (QI).
#     Analogous to slightly_better_mean_baseline but for continuous data.
#     Falls back to global mean when QI combination not found.
#     """
#     reconstructed_targets = targets.copy()
# 
#     # Compute conditional means grouped by QI
#     lookup = deid.groupby(qi)[hidden_features].mean().reset_index()
# 
#     # Merge with targets
#     recon = pd.merge(
#         reconstructed_targets,
#         lookup,
#         on=qi,
#         how='left',
#         suffixes=('', '_predicted')
#     )
# 
#     # Fill in predictions and handle missing values
#     for feature in hidden_features:
#         predicted_col = f'{feature}_predicted'
#         if predicted_col in recon.columns:
#             # Use predicted values where available
#             recon[feature] = recon[predicted_col]
#             # Fill missing with global mean
#             if recon[feature].isna().any():
#                 global_mean = deid[feature].mean()
#                 recon[feature] = recon[feature].fillna(global_mean)
#         else:
#             # If no predicted column, fill with global mean
#             global_mean = deid[feature].mean()
#             recon[feature] = global_mean
# 
#     # Keep only original columns
#     return recon[reconstructed_targets.columns], None, None
# 
# 
# def conditional_median_baseline_cont(cfg, deid, targets, qi, hidden_features):
#     """
#     Fill hidden features with conditional median based on quasi-identifiers (QI).
#     More robust version of conditional_mean_baseline.
#     Falls back to global median when QI combination not found.
#     """
#     reconstructed_targets = targets.copy()
# 
#     # Compute conditional medians grouped by QI
#     lookup = deid.groupby(qi)[hidden_features].median().reset_index()
# 
#     # Merge with targets
#     recon = pd.merge(
#         reconstructed_targets,
#         lookup,
#         on=qi,
#         how='left',
#         suffixes=('', '_predicted')
#     )
# 
#     # Fill in predictions and handle missing values
#     for feature in hidden_features:
#         predicted_col = f'{feature}_predicted'
#         if predicted_col in recon.columns:
#             # Use predicted values where available
#             recon[feature] = recon[predicted_col]
#             # Fill missing with global median
#             if recon[feature].isna().any():
#                 global_median = deid[feature].median()
#                 recon[feature] = recon[feature].fillna(global_median)
#         else:
#             # If no predicted column, fill with global median
#             global_median = deid[feature].median()
#             recon[feature] = global_median
# 
#     # Keep only original columns
#     return recon[reconstructed_targets.columns], None, None


def random_normal_baseline_cont(cfg, deid, targets, qi, hidden_features):
    """
    Sample from a normal distribution fitted to each hidden feature.
    Uses mean and std from deid dataset.
    """
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        mean = deid[hidden_feature].mean()
        std = deid[hidden_feature].std()
        # Generate random samples from normal distribution
        reconstructed_targets[hidden_feature] = np.random.normal(
            loc=mean,
            scale=std,
            size=len(targets)
        )
    return reconstructed_targets, None, None


# def nearest_neighbor_baseline_cont_OLD(cfg, deid, targets, qi, hidden_features):
#     """
#     For each target, find the nearest neighbor in deid based on QI features,
#     and use that neighbor's hidden feature values.
#     Simple distance-based baseline (uses first QI feature for simplicity).
#     """
#     reconstructed_targets = targets.copy()
#
#     for idx, target_row in targets.iterrows():
#         # Find closest match based on first QI feature (simple version)
#         # For more sophisticated version, could use all QI features with proper distance metric
#         if len(qi) > 0:
#             qi_feature = qi[0]
#             distances = (deid[qi_feature] - target_row[qi_feature]).abs()
#             closest_idx = distances.idxmin()
#
#             for hidden_feature in hidden_features:
#                 reconstructed_targets.loc[idx, hidden_feature] = deid.loc[closest_idx, hidden_feature]
#         else:
#             # If no QI, just use random sample
#             for hidden_feature in hidden_features:
#                 reconstructed_targets.loc[idx, hidden_feature] = deid[hidden_feature].sample(1).values[0]
#
#     return reconstructed_targets, None, None


def nearest_neighbor_baseline_cont(cfg, deid, targets, qi, hidden_features):

    reconstructed_targets = targets.copy()

    assert len(qi) > 0
    # Normalize QI features for fair distance calculation
    # Compute mean and std from deid
    qi_means = deid[qi].mean()
    qi_stds = deid[qi].std()

    # Avoid division by zero for constant features
    qi_stds[qi_stds == 0] = 1.0

    # Normalize deid QI features
    deid_qi_normalized = (deid[qi] - qi_means) / qi_stds

    # For each target, find nearest neighbor
    for idx, target_row in targets.iterrows():
        # Normalize target QI features
        target_qi_normalized = (target_row[qi] - qi_means) / qi_stds

        # Calculate Euclidean distance to all deid records
        distances = np.sqrt(((deid_qi_normalized - target_qi_normalized) ** 2).sum(axis=1))

        # Find closest neighbor
        closest_idx = distances.idxmin()

        # Copy hidden features from nearest neighbor
        for hidden_feature in hidden_features:
            reconstructed_targets.loc[idx, hidden_feature] = deid.loc[closest_idx, hidden_feature]

    return reconstructed_targets, None, None
