"""
Attack registry for reconstruction attacks.

This module provides a single source of truth for attack method names
and their corresponding function references. This prevents spelling
inconsistencies and simplifies configuration.

Attacks are organized by data type (categorical vs continuous).
"""

# Categorical attacks
from .NN_classifier import mlp_classification_reconstruction
from .ML_classifiers import (
    KNN_reconstruction,
    lgboost_reconstruction,
    SVM_classification_reconstruction,
    random_forest_reconstruction,
    logistic_regression_reconstruction,
    naive_bayes_reconstruction
)
from .attention_classifier import attention_reconstruction, attention_autoregressive_reconstruction
from .baselines_classifiers import (
    mode_baseline,
    random_baseline,
    mean_baseline,
    simply_measure_deid_itself_baseline
)

# Continuous attacks
from .NN_regression import mlp_repression_reconstruction
from .ML_regression import (
    KNN_reconstruction_continuous,
    linear_regression_reconstruction,
    ridge_regression_reconstruction,
    lasso_regression_reconstruction,
    elastic_net_reconstruction,
    random_forest_regression_reconstruction,
    lgboost_regression_reconstruction,
    SVM_regression_reconstruction,
    polynomial_regressor_reconstruction,
    bayesian_ridge_reconstruction,
    huber_regressor_reconstruction,
    ransac_regressor_reconstruction,
    sdgregressor_reconstruction
)
from .baselines_continuous import (
    mean_baseline_cont,
    median_baseline_cont,
    random_baseline_cont,
    random_normal_baseline_cont,
    nearest_neighbor_baseline_cont,
    simply_measure_deid_itself_baseline_cont
)

# Data-type agnostic attacks (work on both)
from .partialDiffusion import repaint_reconstruction, partial_tabddpm_reconstruction

# SOTA attacks from published papers
import sys
import os
sota_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SOTA_attacks')
if sota_path not in sys.path:
    sys.path.insert(0, sota_path)
from linear_reconstruction import linear_reconstruction_attack


# =============================================================================
# ATTACK REGISTRY
# Two-tier structure: data_type -> attack_name -> function
# =============================================================================

ATTACK_REGISTRY = {
    "categorical": {
        # Machine Learning Classifiers
        "KNN": KNN_reconstruction,
        "RandomForest": random_forest_reconstruction,
        "LightGBM": lgboost_reconstruction,
        "SVM": SVM_classification_reconstruction,
        "LogisticRegression": logistic_regression_reconstruction,
        "NaiveBayes": naive_bayes_reconstruction,

        # Neural Networks
        "MLP": mlp_classification_reconstruction,
        "Attention": attention_reconstruction,
        "AttentionAutoregressive": attention_autoregressive_reconstruction,

        # Baselines
        "Mode": mode_baseline,
        "Random": random_baseline,
        "Mean": mean_baseline,
        "MeasureDeid": simply_measure_deid_itself_baseline,

        # SOTA attacks (published methods)
        "LinearReconstruction": linear_reconstruction_attack,
    },

    "continuous": {
        # Machine Learning Regressors
        "KNN": KNN_reconstruction_continuous,
        "RandomForest": random_forest_regression_reconstruction,
        "LightGBM": lgboost_regression_reconstruction,
        "SVM": SVM_regression_reconstruction,
        "LinearRegression": linear_regression_reconstruction,
        "Ridge": ridge_regression_reconstruction,
        "Lasso": lasso_regression_reconstruction,
        "ElasticNet": elastic_net_reconstruction,
        "PolynomialRegression": polynomial_regressor_reconstruction,
        "BayesianRidge": bayesian_ridge_reconstruction,
        "HuberRegressor": huber_regressor_reconstruction,
        "RANSACRegressor": ransac_regressor_reconstruction,
        "SGDRegressor": sdgregressor_reconstruction,

        # Neural Networks
        "MLP": mlp_repression_reconstruction,

        # Baselines
        "Mean": mean_baseline_cont,
        "Median": median_baseline_cont,
        "Random": random_baseline_cont,
        "RandomNormal": random_normal_baseline_cont,
        "NearestNeighbor": nearest_neighbor_baseline_cont,
        "MeasureDeid": simply_measure_deid_itself_baseline_cont,
    },

    # Data-type agnostic (work on both categorical and continuous)
    "agnostic": {
        "TabDDPM": partial_tabddpm_reconstruction,
        "RePaint": repaint_reconstruction,
    }
}


def get_attack(attack_name, data_type="agnostic"):
    """
    Get attack function by name and data type from registry.

    Args:
        attack_name: Name of the attack (e.g., "RePaint", "RandomForest")
        data_type: Type of data - "categorical", "continuous", or "agnostic" (default)

    Returns:
        Attack function with signature (cfg, synth, targets, qi, hidden_features)

    Raises:
        KeyError: If attack_name is not found in registry for given data_type
        ValueError: If data_type is not valid
    """
    valid_types = ["categorical", "continuous", "agnostic"]
    if data_type not in valid_types:
        raise ValueError(
            f"Invalid data_type '{data_type}'. Must be one of: {valid_types}"
        )

    # Check agnostic attacks first
    if attack_name in ATTACK_REGISTRY.get("agnostic", {}):
        return ATTACK_REGISTRY["agnostic"][attack_name]

    # Then check data-type specific attacks
    if data_type != "agnostic" and attack_name in ATTACK_REGISTRY.get(data_type, {}):
        return ATTACK_REGISTRY[data_type][attack_name]

    # Attack not found - provide helpful error message
    available = []
    for dtype in valid_types:
        if dtype in ATTACK_REGISTRY:
            attacks = sorted(ATTACK_REGISTRY[dtype].keys())
            available.append(f"  {dtype}: {', '.join(attacks)}")

    raise KeyError(
        f"Attack '{attack_name}' not found for data_type '{data_type}'.\n"
        f"Available attacks:\n" + "\n".join(available)
    )


def list_attacks(data_type=None):
    """
    Return list of available attack names.

    Args:
        data_type: Optional filter by "categorical", "continuous", or "agnostic"
                  If None, returns all attacks with their data types

    Returns:
        List of attack names, or dict mapping data_type -> [attack_names]
    """
    if data_type is None:
        # Return all attacks organized by type
        return {
            dtype: sorted(attacks.keys())
            for dtype, attacks in ATTACK_REGISTRY.items()
        }
    elif data_type in ATTACK_REGISTRY:
        return sorted(ATTACK_REGISTRY[data_type].keys())
    else:
        raise ValueError(f"Invalid data_type '{data_type}'")


__all__ = [
    'ATTACK_REGISTRY',
    'get_attack',
    'list_attacks',
]
