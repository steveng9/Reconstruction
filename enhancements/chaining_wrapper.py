"""
Generic chaining wrapper for any reconstruction attack.
Wraps existing attacks without modifying their code.

Chaining predicts hidden features sequentially, adding each predicted
feature to the known features for subsequent predictions.
"""

import wandb
import numpy as np
from scoring import calculate_reconstruction_score, calculate_continuous_vals_reconstruction_score
from get_data import get_meta_data_for_diffusion


def apply_chaining(attack_fn, cfg, synth, targets, qi, hidden_features):
    """
    Wrapper that applies chaining to any attack function.

    Args:
        attack_fn: Original attack function with signature (cfg, synth, targets, qi, hidden_features)
        cfg: Configuration dictionary
        synth: Synthetic/de-identified data
        targets: Training data with known features (QI)
        qi: List of quasi-identifier (known) features
        hidden_features: List of features to reconstruct

    Returns:
        reconstructed, probas, classes (same as attack_fn)
    """
    chaining_cfg = cfg["attack_params"].get("chaining", {})

    if not chaining_cfg.get("enabled", False):
        # No chaining, call attack directly
        return attack_fn(cfg, synth, targets, qi, hidden_features)

    # Determine prediction order
    order_strategy = chaining_cfg.get("order_strategy", "default")
    order = _get_chaining_order(chaining_cfg, hidden_features, synth, targets, qi)

    # Log chaining configuration to WandB
    wandb.config.update({
        "chaining_enabled": True,
        "chaining_order_strategy": order_strategy,
        "chaining_order": order
    })

    print(f"\n{'='*60}")
    print(f"CHAINING ENABLED")
    print(f"  Strategy: {order_strategy}")
    print(f"  Order: {order}")
    print(f"{'='*60}\n")

    # Get feature type information (continuous vs discrete)
    _, domain = get_meta_data_for_diffusion(cfg)

    # Initialize reconstruction
    reconstructed = targets.copy()
    known_features = qi.copy()
    all_probas = []
    all_classes = []

    # Sequentially predict each feature
    for idx, feature in enumerate(order):
        print(f"\n  Chaining step {idx+1}/{len(order)}: predicting {feature}")

        # Predict single feature using current known features
        recon_step, probas, classes = attack_fn(
            cfg, synth, reconstructed[known_features], known_features, [feature]
        )

        # Update reconstruction
        reconstructed[feature] = recon_step[feature]

        # Log intermediate accuracy if enabled
        if chaining_cfg.get("log_intermediate", True):
            _log_intermediate_score(cfg, targets, reconstructed, feature, idx, domain)

        # Add predicted feature to known features for next iteration
        known_features.append(feature)

        # Store probabilities/classes if provided
        if probas is not None:
            all_probas.extend(probas if isinstance(probas, list) else [probas])
        if classes is not None:
            all_classes.extend(classes if isinstance(classes, list) else [classes])

    return reconstructed, all_probas if all_probas else None, all_classes if all_classes else None


def _log_intermediate_score(cfg, targets, reconstructed, feature, step_idx, domain):
    """
    Log intermediate reconstruction score for a single feature.
    Handles both continuous and categorical features.
    """
    # Determine if feature is continuous or discrete
    feature_type = domain.get(feature, {}).get("type", "discrete")

    if feature_type == "continuous":
        # Use continuous scoring
        scores_df = calculate_continuous_vals_reconstruction_score(
            targets, reconstructed, [feature]
        )

        # Log multiple continuous metrics
        metrics = scores_df.loc[feature].to_dict()
        log_dict = {
            f"chain_step_{step_idx+1:02d}_{feature}_mae": metrics['mean_abs_error'],
            f"chain_step_{step_idx+1:02d}_{feature}_normalized_mae": metrics['normalized_mae'],
            f"chain_step_{step_idx+1:02d}_{feature}_rmse": metrics['rmse'],
            f"chain_step_{step_idx+1:02d}_{feature}_normalized_rmse": metrics['normalized_rmse'],
        }
        wandb.log(log_dict)

        print(f"    → Normalized MAE: {metrics['normalized_mae']:.4f}, "
              f"Normalized RMSE: {metrics['normalized_rmse']:.4f}")

    else:
        # Use categorical scoring (rarity-weighted accuracy)
        score = calculate_reconstruction_score(targets, reconstructed, [feature])[0]
        wandb.log({f"chain_step_{step_idx+1:02d}_{feature}_accuracy": score})

        print(f"    → Rarity-weighted accuracy: {score:.1f}%")


def _get_chaining_order(chaining_cfg, hidden_features, synth, targets, qi):
    """Determine the order to predict features."""
    strategy = chaining_cfg.get("order_strategy", "default")

    if strategy == "manual":
        order = chaining_cfg.get("order", hidden_features)
        # Validate that manual order contains all hidden features
        if set(order) != set(hidden_features):
            raise ValueError(
                f"Manual order must contain all hidden features.\n"
                f"Expected: {set(hidden_features)}\n"
                f"Got: {set(order)}"
            )
        return order

    elif strategy == "default":
        # Use the order they appear in hidden_features
        return hidden_features

    elif strategy == "random":
        import random
        order = hidden_features.copy()
        random.seed(chaining_cfg.get("random_seed", 42))
        random.shuffle(order)
        return order

    elif strategy == "correlation":
        # Predict features most correlated with QI first
        return _order_by_correlation(synth, qi, hidden_features)

    elif strategy == "mutual_info":
        # Predict features with highest mutual information with QI first
        return _order_by_mutual_info(synth, qi, hidden_features)

    elif strategy == "reverse_correlation":
        # Predict features LEAST correlated with QI first
        return _order_by_correlation(synth, qi, hidden_features, ascending=True)

    else:
        raise ValueError(f"Unknown chaining order strategy: {strategy}")


def _order_by_correlation(synth, qi, hidden_features, ascending=False):
    """
    Order features by average correlation with QI features.

    Args:
        synth: Synthetic data
        qi: Known features
        hidden_features: Features to order
        ascending: If True, order from least to most correlated
    """
    import pandas as pd

    # Compute correlation matrix
    corr_matrix = synth[qi + hidden_features].corr().abs()

    # For each hidden feature, compute average correlation with QI
    scores = {}
    for hf in hidden_features:
        scores[hf] = corr_matrix.loc[hf, qi].mean()

    # Sort by descending correlation (predict most correlated first) unless ascending=True
    return sorted(hidden_features, key=lambda f: scores[f], reverse=(not ascending))


def _order_by_mutual_info(synth, qi, hidden_features):
    """
    Order features by mutual information with QI.
    Predict features with highest mutual information first.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    # Convert all features to numeric if needed
    synth_encoded = synth.copy()
    for col in synth_encoded.columns:
        if synth_encoded[col].dtype == 'object':
            le = LabelEncoder()
            synth_encoded[col] = le.fit_transform(synth_encoded[col].astype(str))

    scores = {}
    for hf in hidden_features:
        # Calculate mutual information between this feature and all QI features
        mi = mutual_info_classif(
            synth_encoded[qi],
            synth_encoded[hf],
            discrete_features=True,
            random_state=42
        )
        scores[hf] = mi.mean()

    # Sort by descending mutual information
    return sorted(hidden_features, key=lambda f: scores[f], reverse=True)
