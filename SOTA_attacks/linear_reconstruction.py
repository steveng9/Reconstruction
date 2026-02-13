"""
Linear Reconstruction Attack wrapper.

Paper: "A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data"
Authors: M.S.M.S. Annamalai, A. Gadotti, L. Rocher
Conference: USENIX Security 2024
Source: https://github.com/Filienko/recon-synth (forked to github.com/steveng9/recon-synth)

This attack uses k-way marginal queries and LP solving (Gurobi) to reconstruct
secret attributes. It only works on:
- Single binary features (one feature at a time)
- Smaller datasets (memory constraints due to query matrix size)

The wrapper converts our standard interface to the format expected by recon-synth.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add recon-synth to path
RECON_SYNTH_PATH = os.path.expanduser('~/recon-synth')
if RECON_SYNTH_PATH not in sys.path:
    sys.path.insert(0, RECON_SYNTH_PATH)

# Import recon-synth attack components
from attacks import query_attack
from attacks.simple_kway_queries import gen_all_simple_kway, get_result_simple_kway, simple_kway
from load_data import process_data


def linear_reconstruction_attack(cfg, synth, targets, qi, hidden_features):
    """
    Linear reconstruction attack for single binary features.

    Uses k-way marginal queries and LP solving to reconstruct a secret binary attribute.
    Based on the USENIX Security 2024 paper implementation.

    Args:
        cfg: Configuration dict with attack_params containing:
            - k: k-way queries (default: 3)
            - n_procs: Number of processors for Gurobi (default: 4)
        synth: Synthetic DataFrame
        targets: Training DataFrame (with QI columns)
        qi: List of QI column names
        hidden_features: List of hidden feature names (MUST be length 1 for this attack)

    Returns:
        reconstructed: DataFrame with QI + reconstructed hidden feature
        None, None: Placeholders for compatibility with standard interface

    Raises:
        ValueError: If hidden_features is not length 1 or if feature is not binary
    """
    # Validate: only works for single features
    if len(hidden_features) != 1:
        raise ValueError(
            f"Linear reconstruction attack only works on single features. "
            f"Got {len(hidden_features)} features: {hidden_features}"
        )

    secret_feature = hidden_features[0]

    # Get parameters from config
    attack_params = cfg.get("attack_params", {})
    k = attack_params.get("k", 3)
    n_procs = attack_params.get("n_procs", 4)

    # Use QI features for query generation
    print(f"  [Linear Attack] Using {len(qi)} QI features: {qi}")

    if len(qi) < 10:
        print(f"  [Linear Attack] WARNING: Only {len(qi)} features")
        print(f"  [Linear Attack] Performance improves with 15+ features")

    # Extract QI features from training and synthetic data
    train_attrs = targets[qi].to_numpy()

    # Synthetic data: QI + secret feature
    synth_qi_secret = synth[qi + [secret_feature]].copy()
    synth_attrs, synth_secret_values = process_data(synth_qi_secret, secret_feature)

    # Check if feature is binary and map to {0, 1} if needed
    synth_unique = np.unique(synth_secret_values)

    if len(synth_unique) != 2:
        raise ValueError(
            f"Linear reconstruction attack only works on binary features. "
            f"Feature '{secret_feature}' has {len(synth_unique)} unique values: {synth_unique}"
        )

    # Map to {0, 1} if not already
    if not (set(synth_unique) == {0, 1}):
        print(f"  [Linear Attack] Mapping {synth_unique[0]} -> 0, {synth_unique[1]} -> 1")
        synth_secret_bits = (synth_secret_values == synth_unique[1]).astype(int)
        value_mapping = {0: synth_unique[0], 1: synth_unique[1]}  # For reverse mapping later
    else:
        synth_secret_bits = synth_secret_values
        value_mapping = {0: 0, 1: 1}

    # Generate all k-way queries
    all_queries = gen_all_simple_kway(train_attrs, k)

    # CRITICAL: Paper only uses queries with target_val=1
    queries = [
        (attr_inds, attr_vals, target_val)
        for (attr_inds, attr_vals, target_val) in all_queries
        if target_val == 1
    ]

    print(f"  [Linear Attack] Generated {len(queries)} queries (k={k}, target_val=1 only)")

    # Build query matrix A from TRAINING data (what we're reconstructing)
    A = simple_kway(queries, train_attrs)
    train_n_users = np.sum(A, axis=1)

    # Evaluate queries on SYNTHETIC data
    synth_n_users, synth_results = get_result_simple_kway(
        synth_attrs, synth_secret_bits, queries
    )

    # Apply conditional scaling (paper's method)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        scale = np.where(
            synth_n_users > 0,
            train_n_users / synth_n_users,
            len(targets) / len(synth)
        )
    synth_results_scaled = synth_results * scale

    print(f"  [Linear Attack] Query matrix: {A.shape}, avg scale: {np.mean(scale):.2f}")

    # Run LP attack with Gurobi
    print(f"  [Linear Attack] Running LP solver (Gurobi, {n_procs} procs)...")
    est_secret_bits, scores, success = query_attack(A, synth_results_scaled, n_procs)

    if not success:
        print(f"  [Linear Attack] WARNING: LP solver failed, predictions may be random")
    else:
        print(f"  [Linear Attack] LP solver succeeded")

    # Convert back to original encoding if needed
    est_secret_values = np.array([value_mapping[bit] for bit in est_secret_bits])

    # Construct reconstructed DataFrame
    reconstructed = targets[qi].copy()
    reconstructed[secret_feature] = est_secret_values

    return reconstructed, None, None
