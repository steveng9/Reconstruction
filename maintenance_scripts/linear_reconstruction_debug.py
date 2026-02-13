"""
Linear Reconstruction Attack wrapper - DEBUG VERSION

Adds extensive logging to understand what's happening inside the attack.
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
    Linear reconstruction attack - DEBUG VERSION with extensive logging.
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

    print(f"\n{'='*70}")
    print(f"LINEAR RECONSTRUCTION ATTACK - DEBUG MODE")
    print(f"{'='*70}")
    print(f"Secret feature: {secret_feature}")
    print(f"QI features: {len(qi)} features")
    print(f"  QI: {qi}")
    print(f"k-way queries: k={k}")
    print(f"Training data: {len(targets)} rows")
    print(f"Synthetic data: {len(synth)} rows")

    # Extract QI features from training and synthetic data
    train_attrs = targets[qi].to_numpy()

    # Synthetic data: QI + secret feature
    synth_qi_secret = synth[qi + [secret_feature]].copy()
    synth_attrs, synth_secret_values = process_data(synth_qi_secret, secret_feature)

    print(f"\nData shapes:")
    print(f"  train_attrs: {train_attrs.shape}")
    print(f"  synth_attrs: {synth_attrs.shape}")
    print(f"  synth_secret_values: {synth_secret_values.shape}")

    # Check if feature is binary and map to {0, 1} if needed
    synth_unique = np.unique(synth_secret_values)
    train_unique = np.unique(targets[secret_feature])

    print(f"\nSecret feature statistics:")
    print(f"  Train unique values: {train_unique}")
    print(f"  Train distribution: {np.bincount(targets[secret_feature].astype(int))}")
    print(f"  Synth unique values: {synth_unique}")
    print(f"  Synth distribution: {np.bincount(synth_secret_values.astype(int))}")

    if len(synth_unique) != 2:
        raise ValueError(
            f"Linear reconstruction attack only works on binary features. "
            f"Feature '{secret_feature}' has {len(synth_unique)} unique values: {synth_unique}"
        )

    # Map to {0, 1} if not already
    if not (set(synth_unique) == {0, 1}):
        print(f"\nMapping values: {synth_unique[0]} -> 0, {synth_unique[1]} -> 1")
        synth_secret_bits = (synth_secret_values == synth_unique[1]).astype(int)
        value_mapping = {0: synth_unique[0], 1: synth_unique[1]}
    else:
        synth_secret_bits = synth_secret_values
        value_mapping = {0: 0, 1: 1}

    # Generate all k-way queries
    print(f"\n{'='*70}")
    print(f"GENERATING QUERIES")
    print(f"{'='*70}")
    all_queries = gen_all_simple_kway(train_attrs, k)
    print(f"Total queries generated: {len(all_queries)}")

    # CRITICAL: Paper only uses queries with target_val=1
    queries = [
        (attr_inds, attr_vals, target_val)
        for (attr_inds, attr_vals, target_val) in all_queries
        if target_val == 1
    ]
    print(f"Queries with target_val=1: {len(queries)}")

    # Show example queries
    print(f"\nExample queries (first 5):")
    for i, (attr_inds, attr_vals, target_val) in enumerate(queries[:5]):
        print(f"  Query {i}: attrs {attr_inds} = {attr_vals}, target={target_val}")

    # Build query matrix A from TRAINING data
    print(f"\n{'='*70}")
    print(f"BUILDING QUERY MATRIX FROM TRAINING DATA")
    print(f"{'='*70}")
    A = simple_kway(queries, train_attrs)
    train_n_users = np.sum(A, axis=1)

    print(f"Query matrix A: {A.shape} (queries x users)")
    print(f"  Min users per query: {train_n_users.min()}")
    print(f"  Max users per query: {train_n_users.max()}")
    print(f"  Mean users per query: {train_n_users.mean():.1f}")
    print(f"  Queries with 0 users: {np.sum(train_n_users == 0)}")

    # Evaluate queries on SYNTHETIC data
    print(f"\n{'='*70}")
    print(f"EVALUATING QUERIES ON SYNTHETIC DATA")
    print(f"{'='*70}")
    synth_n_users, synth_results = get_result_simple_kway(
        synth_attrs, synth_secret_bits, queries
    )

    print(f"Synthetic query results:")
    print(f"  Min users per query: {synth_n_users.min()}")
    print(f"  Max users per query: {synth_n_users.max()}")
    print(f"  Mean users per query: {synth_n_users.mean():.1f}")
    print(f"  Queries with 0 users: {np.sum(synth_n_users == 0)}")
    print(f"  Min result: {synth_results.min():.1f}")
    print(f"  Max result: {synth_results.max():.1f}")
    print(f"  Mean result: {synth_results.mean():.1f}")

    # Apply conditional scaling (paper's method)
    print(f"\n{'='*70}")
    print(f"SCALING RESULTS")
    print(f"{'='*70}")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        scale = np.where(
            synth_n_users > 0,
            train_n_users / synth_n_users,
            len(targets) / len(synth)
        )
    synth_results_scaled = synth_results * scale

    print(f"Scaling factors:")
    print(f"  Min scale: {scale.min():.2f}")
    print(f"  Max scale: {scale.max():.2f}")
    print(f"  Mean scale: {scale.mean():.2f}")
    print(f"  Median scale: {np.median(scale):.2f}")
    print(f"  Fallback scale (len ratio): {len(targets) / len(synth):.2f}")
    print(f"  Times fallback used: {np.sum(synth_n_users == 0)}")

    print(f"\nScaled results:")
    print(f"  Min: {synth_results_scaled.min():.1f}")
    print(f"  Max: {synth_results_scaled.max():.1f}")
    print(f"  Mean: {synth_results_scaled.mean():.1f}")

    # Run LP attack with Gurobi
    print(f"\n{'='*70}")
    print(f"RUNNING LP SOLVER (GUROBI)")
    print(f"{'='*70}")
    print(f"Using {n_procs} processors...")

    est_secret_bits, scores, success = query_attack(A, synth_results_scaled, n_procs)

    print(f"\nLP Solver result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Scores statistics:")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")

    # Show score distribution
    print(f"\nPredicted bits distribution:")
    print(f"  Predicted 0: {np.sum(est_secret_bits == 0)}")
    print(f"  Predicted 1: {np.sum(est_secret_bits == 1)}")

    if not success:
        print(f"\n⚠ WARNING: LP solver failed, predictions may be random")

    # Convert back to original encoding if needed
    est_secret_values = np.array([value_mapping[bit] for bit in est_secret_bits])

    # Compute accuracy
    true_values = targets[secret_feature].values
    accuracy = np.mean(est_secret_values == true_values)
    baseline = max(np.mean(true_values == train_unique[0]),
                   np.mean(true_values == train_unique[1]))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Baseline (majority class): {baseline:.2%}")
    print(f"Improvement over baseline: {accuracy - baseline:+.2%}")
    print(f"{'='*70}\n")

    # Construct reconstructed DataFrame
    reconstructed = targets[qi].copy()
    reconstructed[secret_feature] = est_secret_values

    return reconstructed, None, None
