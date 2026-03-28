"""
Generic ensembling wrapper for combining multiple reconstruction attacks.
Wraps attacks without modifying their code.

Ensembling combines predictions from multiple attack methods using various
aggregation strategies (voting, averaging, weighted, stacking).
"""

import wandb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def apply_ensembling(attack_fn, cfg):
    """
    Wrapper that creates an ensemble of multiple attack methods.

    Args:
        attack_fn: Primary attack function (can be ignored if include_primary=False)
        cfg: Configuration dictionary

    Returns:
        attack_function: Either the original attack_fn or a wrapped ensemble function
    """
    ensembling_cfg = cfg["attack_params"].get("ensembling", {})

    if not ensembling_cfg.get("enabled", False):
        # No ensembling, return original attack
        return attack_fn

    # Import here to avoid circular imports
    from attacks import get_attack

    # Get configuration
    method_names = ensembling_cfg.get("methods", [])
    aggregation = ensembling_cfg.get("aggregation", "voting")
    weights = ensembling_cfg.get("weights", None)
    include_primary = ensembling_cfg.get("include_primary", False)

    if not method_names:
        print("WARNING: Ensembling enabled but no methods specified. Using primary attack only.")
        return attack_fn

    # Get data type from config
    data_type = cfg.get("data_type", "agnostic")

    # Get attack functions for all methods
    ensemble_methods = []
    ensemble_names = []

    if include_primary:
        primary_name = cfg.get("attack_method", "Unknown")
        ensemble_methods.append(attack_fn)
        ensemble_names.append(primary_name)

    for method_name in method_names:
        # Skip if it's the same as primary and we already included it
        if include_primary and method_name == cfg.get("attack_method"):
            continue

        try:
            method_fn = get_attack(method_name, data_type)
            ensemble_methods.append(method_fn)
            ensemble_names.append(method_name)
        except KeyError as e:
            print(f"WARNING: Could not load attack method '{method_name}': {e}")
            continue

    if len(ensemble_methods) == 0:
        print("WARNING: No valid methods for ensemble. Using primary attack only.")
        return attack_fn

    # Log ensembling configuration to WandB
    wandb.config.update({
        "ensembling_enabled": True,
        "ensembling_methods": ensemble_names,
        "ensembling_aggregation": aggregation,
        "ensembling_num_methods": len(ensemble_methods)
    })

    print(f"\n{'='*60}")
    print(f"ENSEMBLING ENABLED")
    print(f"  Methods: {', '.join(ensemble_names)}")
    print(f"  Aggregation: {aggregation}")
    print(f"  Number of models: {len(ensemble_methods)}")
    print(f"{'='*60}\n")

    # Create and return ensemble wrapper function
    def ensemble_attack(cfg_inner, synth, targets, qi, hidden_features):
        """Ensemble attack function that combines multiple methods."""
        return _run_ensemble(
            ensemble_methods,
            ensemble_names,
            cfg_inner,
            cfg,  # Original config with all method params
            synth,
            targets,
            qi,
            hidden_features,
            aggregation,
            weights,
            data_type
        )

    return ensemble_attack


def _run_ensemble(methods, method_names, cfg_inner, cfg_full, synth, targets, qi, hidden_features,
                  aggregation, weights, data_type):
    """
    Run ensemble of multiple attack methods and aggregate predictions.

    Args:
        methods: List of attack functions
        method_names: List of method names (for logging)
        cfg_inner: Config passed to ensemble (may have merged params)
        cfg_full: Original full config with all method-specific params
        synth: Synthetic/de-identified data
        targets: Training data with known features
        qi: List of quasi-identifier features
        hidden_features: List of features to reconstruct
        aggregation: Aggregation strategy
        weights: Optional weights for weighted aggregation
        data_type: "categorical", "continuous", or "agnostic"

    Returns:
        reconstructed, probas, classes (aggregated predictions)
    """
    from master_experiment_script import _prepare_config

    # Collect predictions from all methods
    all_predictions = []
    all_probas = []
    all_classes = []

    print(f"\nRunning ensemble of {len(methods)} methods...")

    for idx, (method_fn, method_name) in enumerate(zip(methods, method_names)):
        print(f"  [{idx+1}/{len(methods)}] Running {method_name}...")

        # Prepare config with method-specific params
        method_cfg = cfg_full.copy()
        method_cfg["attack_method"] = method_name
        method_cfg = _prepare_config(method_cfg)

        # Run attack
        try:
            recon, probas, classes = method_fn(method_cfg, synth, targets, qi, hidden_features)
            all_predictions.append(recon)
            all_probas.append(probas)
            all_classes.append(classes)
        except Exception as e:
            print(f"    WARNING: {method_name} failed with error: {e}")
            continue

    if len(all_predictions) == 0:
        raise RuntimeError("All ensemble methods failed!")

    print(f"  ✓ Successfully ran {len(all_predictions)} methods\n")

    # Aggregate predictions
    reconstructed = _aggregate_predictions(
        all_predictions,
        all_probas,
        all_classes,
        hidden_features,
        aggregation,
        weights,
        data_type
    )

    # For now, return None for probas/classes (could implement aggregated probas later)
    return reconstructed, None, None


def _aggregate_predictions(predictions, all_probas, all_classes, hidden_features,
                           aggregation, weights, data_type):
    """
    Aggregate predictions from multiple models.

    Args:
        predictions: List of DataFrames with predictions
        all_probas: List of probability arrays (may be None)
        all_classes: List of class arrays (may be None)
        hidden_features: List of features being predicted
        aggregation: Aggregation strategy
        weights: Optional weights for each model
        data_type: Type of data being predicted

    Returns:
        DataFrame with aggregated predictions
    """
    n_models = len(predictions)

    # Set default weights if not provided
    if weights is None:
        weights = [1.0] * n_models
    else:
        if len(weights) != n_models:
            print(f"WARNING: {len(weights)} weights provided but {n_models} models. Using equal weights.")
            weights = [1.0] * n_models

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Initialize result with first prediction
    result = predictions[0].copy()

    # Aggregate each feature
    for feat_idx, feature in enumerate(hidden_features):
        feature_predictions = [pred[feature] for pred in predictions]

        if aggregation == "voting" or aggregation == "hard_voting":
            # Hard voting: most common prediction
            result[feature] = _hard_voting(feature_predictions)

        elif aggregation == "soft_voting" or aggregation == "weighted_voting":
            result[feature] = _soft_voting(
                feature_predictions, all_probas, all_classes, feat_idx, weights
            )

        elif aggregation == "averaging" or aggregation == "mean":
            # Averaging: for continuous values
            result[feature] = _averaging(feature_predictions, weights)

        elif aggregation == "median":
            # Median: robust to outliers
            result[feature] = _median(feature_predictions)

        elif aggregation == "weighted":
            # Weighted average
            result[feature] = _averaging(feature_predictions, weights)

        elif aggregation == "confidence_routing":
            # Per-row: pick the attack with the highest max predicted probability.
            # Attacks without real probas get confidence=0.0 and only win if no
            # other attack in the combo has probas.
            result[feature] = _confidence_routing(
                feature_predictions, all_probas, all_classes, feat_idx
            )

        else:
            print(f"WARNING: Unknown aggregation strategy '{aggregation}'. Using voting.")
            result[feature] = _hard_voting(feature_predictions)

    return result


def _hard_voting(predictions):
    """
    Hard voting: return most common prediction for each row.

    Args:
        predictions: List of Series with predictions

    Returns:
        Series with majority vote
    """
    # Stack predictions and get mode (most common value)
    stacked = pd.DataFrame(predictions).T
    return stacked.mode(axis=1)[0]


def _soft_voting(predictions, all_probas, all_classes, feat_idx, weights):
    """
    Soft voting: aggregate predicted probability distributions, take argmax.

    For models that return probas + classes (RF, LGB, NaiveBayes, SVM): uses their
    probability distribution directly, aligned to a common class space.
    For models without probas (KNN, Mode, Random, etc.): converts the hard prediction
    to a one-hot distribution (probability 1.0 on predicted class).
    Models with probas but no classes (NaiveBayes classes=None): treated as no-proba
    and one-hotted from hard predictions.

    Args:
        predictions: List of Series with hard predictions (one per model)
        all_probas:  List of proba-lists (all_probas[i][feat_idx] = (n_targets, n_classes))
        all_classes: List of classes-lists (all_classes[i][feat_idx] = class label array)
        feat_idx:    Index of this feature in hidden_features
        weights:     Per-model weights (already normalised to sum=1)

    Returns:
        Series with soft-voted predictions
    """
    n_targets = len(predictions[0])

    # Gather per-model (proba_array, class_labels) or one-hot from hard pred
    model_probas = []
    model_classes = []

    for i, hard_pred in enumerate(predictions):
        feat_probas  = all_probas[i][feat_idx]  if (all_probas[i]  is not None) else None
        feat_classes = all_classes[i][feat_idx] if (all_classes[i] is not None) else None

        if feat_probas is not None and feat_classes is not None:
            model_probas.append(feat_probas)
            model_classes.append(np.asarray(feat_classes))
        else:
            # One-hot fallback: probability 1.0 on the hard-predicted class
            unique_vals = np.array(sorted(hard_pred.unique()))
            val_to_idx  = {v: j for j, v in enumerate(unique_vals)}
            ohe = np.zeros((n_targets, len(unique_vals)))
            for row, val in enumerate(hard_pred):
                ohe[row, val_to_idx[val]] = 1.0
            model_probas.append(ohe)
            model_classes.append(unique_vals)

    # Build union of all class labels seen across models (preserve original dtype)
    all_unique = sorted(set(c for cls in model_classes for c in cls),
                        key=lambda x: (str(type(x)), x))
    n_classes  = len(all_unique)
    cls_to_idx = {c: j for j, c in enumerate(all_unique)}

    # Align each model's proba to the common class space
    aligned = []
    for p, cls in zip(model_probas, model_classes):
        a = np.zeros((n_targets, n_classes))
        for j, c in enumerate(cls):
            if c in cls_to_idx:
                a[:, cls_to_idx[c]] = p[:, j]
        aligned.append(a)

    # Weighted sum over models → argmax → class label
    agg = sum(w * a for w, a in zip(weights, aligned))
    pred_indices = np.argmax(agg, axis=1)
    predicted    = np.array([all_unique[k] for k in pred_indices])

    return pd.Series(predicted, index=predictions[0].index)


def _confidence_routing(predictions, all_probas, all_classes, feat_idx):
    """
    Per-row routing: for each target record, use the prediction from whichever
    attack has the highest max predicted probability on this feature.

    Attacks without real probas (KNN, Mode, etc.) are assigned confidence=0.0
    so they only win if every attack in the combo lacks probas, in which case
    we fall back to hard voting.

    Args:
        predictions: List of Series with hard predictions (one per model)
        all_probas:  List of proba-lists (all_probas[i][feat_idx] = (n_targets, n_classes))
        all_classes: List of classes-lists (all_classes[i][feat_idx] = class label array)
        feat_idx:    Index of this feature in hidden_features

    Returns:
        Series with confidence-routed predictions
    """
    n_targets = len(predictions[0])
    n_models  = len(predictions)

    # conf[i, row] = max predicted probability for model i on this row
    conf = np.zeros((n_models, n_targets))
    any_real_probas = False

    for i in range(n_models):
        feat_probas  = all_probas[i][feat_idx]  if (all_probas[i]  is not None) else None
        feat_classes = all_classes[i][feat_idx] if (all_classes[i] is not None) else None

        if feat_probas is not None and feat_classes is not None:
            conf[i] = np.max(feat_probas, axis=1)
            any_real_probas = True
        # else: conf[i] stays 0.0 — lowest priority

    if not any_real_probas:
        return _hard_voting(predictions)

    # For each row, pick the attack with the highest confidence
    chosen = np.argmax(conf, axis=0)               # (n_targets,)
    stacked = np.stack([p.values for p in predictions])  # (n_models, n_targets)
    result_vals = stacked[chosen, np.arange(n_targets)]

    return pd.Series(result_vals, index=predictions[0].index)


def _averaging(predictions, weights):
    """
    Weighted averaging for continuous values.

    Args:
        predictions: List of Series with predictions
        weights: Model weights

    Returns:
        Series with weighted average
    """
    weighted_sum = sum(w * pred for w, pred in zip(weights, predictions))
    return weighted_sum


def _median(predictions):
    """
    Median aggregation: robust to outliers.

    Args:
        predictions: List of Series with predictions

    Returns:
        Series with median values
    """
    stacked = pd.DataFrame(predictions).T
    return stacked.median(axis=1)
