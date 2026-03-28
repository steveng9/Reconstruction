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

import itertools
import sys
import os
import numpy as np
import pandas as pd
import warnings

from l1_solve import l1_solve as query_attack, categorical_l1_solve
from simple_kway_queries import (
    gen_all_simple_kway, get_result_simple_kway, simple_kway,
    gen_all_simple_kway_categorical, get_result_simple_kway_categorical,
)
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


def linear_reconstruction_attack_categorical(cfg, synth, targets, qi, hidden_features):
    """
    Linear reconstruction attack for a single multi-valued (C >= 2) hidden feature.

    Extends the binary LP attack (Annamalai et al., USENIX Security 2024) to handle
    C categories via an n*C variable LP: t_im = P(record i has category m), with
    simplex constraints sum_m(t_im) = 1.

    Args:
        cfg: config dict; attack_params may contain k (default 3) and n_procs (default 4)
        synth: synthetic DataFrame
        targets: training DataFrame
        qi: list of QI column names
        hidden_features: list of length 1

    Returns:
        reconstructed DataFrame, None, None
    """
    if len(hidden_features) != 1:
        raise ValueError(
            f"LinearReconstructionCategorical requires exactly 1 hidden feature. "
            f"Got {len(hidden_features)}: {hidden_features}"
        )

    secret_feature = hidden_features[0]
    attack_params = cfg.get("attack_params", {})
    k = attack_params.get("k", 3)
    n_procs = attack_params.get("n_procs", 4)

    # Enumerate categories from synth (sorted for determinism)
    synth_unique = np.array(sorted(synth[secret_feature].unique()))
    num_categories = len(synth_unique)
    val_to_cat = {v: i + 1 for i, v in enumerate(synth_unique)}  # original value → 1-indexed
    cat_to_val = {i + 1: v for i, v in enumerate(synth_unique)}  # 1-indexed → original value

    print(f"  [LinearCategorical] feature='{secret_feature}', C={num_categories}, "
          f"{len(qi)} QI features, k={k}")

    # QI attribute matrices
    train_attrs = targets[qi].to_numpy()
    synth_attrs, synth_secret_raw = process_data(synth[qi + [secret_feature]].copy(), secret_feature)
    # Map synth secret values to 1-indexed categories (fall back to cat 1 for unseen values)
    synth_secret_cats = np.array([val_to_cat.get(v, 1) for v in synth_secret_raw], dtype=int)

    # Generate all k-way categorical queries; extract per-query target category from tuples
    all_queries = gen_all_simple_kway_categorical(train_attrs, k, num_categories)
    target_categories = np.array([q[2] for q in all_queries], dtype=int)
    print(f"  [LinearCategorical] {len(all_queries)} queries generated")

    # Query matrix over train records (used as LP variable-selection matrix)
    A = simple_kway(all_queries, train_attrs)
    train_n_users = np.sum(A, axis=1)

    # Evaluate same queries on synth records
    synth_n_users, synth_results = get_result_simple_kway_categorical(
        synth_attrs, synth_secret_cats, all_queries
    )

    # Scale synth counts to train population size (paper's conditional scaling)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scale = np.where(
            synth_n_users > 0,
            train_n_users / synth_n_users,
            len(targets) / len(synth),
        )
    synth_results_scaled = synth_results * scale

    print(f"  [LinearCategorical] Query matrix: {A.shape}, avg scale: {np.mean(scale):.2f}")
    print(f"  [LinearCategorical] Running LP solver (Gurobi, {n_procs} procs)...")

    est_cats, scores, success = categorical_l1_solve(
        A, synth_results_scaled, num_categories, target_categories, n_procs
    )

    status = "succeeded" if success else "WARNING: failed — predictions may be unreliable"
    print(f"  [LinearCategorical] LP solver {status}")

    # Decode 1-indexed LP output back to original feature values
    est_values = np.array([cat_to_val.get(int(c), synth_unique[0]) for c in est_cats])

    reconstructed = targets[qi].copy()
    reconstructed[secret_feature] = est_values
    return reconstructed, None, None


def linear_reconstruction_attack_joint(cfg, synth, targets, qi, hidden_features):
    """
    Joint linear reconstruction of multiple hidden features simultaneously.

    Encodes the joint value of all hidden features as a single categorical variable
    with C = C1 * C2 * ... * Cn categories (e.g. two binary features → C=4, three
    binary features → C=8).  One LP is solved over this joint space, capturing
    cross-feature correlations — unlike running independent per-feature LPs.

    Extends Annamalai et al. (USENIX Security 2024) to the multi-feature setting.

    Args:
        cfg: config dict; attack_params may contain k (default 3) and n_procs (default 4)
        synth: synthetic DataFrame
        targets: training DataFrame
        qi: list of QI column names
        hidden_features: list of >= 2 feature names

    Returns:
        reconstructed DataFrame with all hidden features filled in, None, None
    """
    if len(hidden_features) < 2:
        raise ValueError(
            f"LinearReconstructionJoint requires at least 2 hidden features. "
            f"Got {len(hidden_features)}: {hidden_features}"
        )

    attack_params = cfg.get("attack_params", {})
    k = attack_params.get("k", 3)
    n_procs = attack_params.get("n_procs", 4)

    # Unique values per feature, sorted for determinism
    feat_uniques = [np.array(sorted(synth[f].unique())) for f in hidden_features]
    cardinalities = [len(u) for u in feat_uniques]
    num_categories = 1
    for c in cardinalities:
        num_categories *= c

    print(f"  [LinearJoint] features={hidden_features}, "
          f"joint C={num_categories} ({'×'.join(str(c) for c in cardinalities)}), "
          f"{len(qi)} QI features, k={k}")

    # Enumerate all joint value combinations (Cartesian product, ordered)
    joint_combos = list(itertools.product(*feat_uniques))  # list of (v1, v2, ...) tuples
    combo_to_cat = {combo: i + 1 for i, combo in enumerate(joint_combos)}  # 1-indexed
    cat_to_combo = {i + 1: combo for i, combo in enumerate(joint_combos)}

    # Encode each synth record's joint hidden-feature value as a single category
    synth_joint = list(zip(*[synth[f].to_numpy() for f in hidden_features]))
    synth_secret_cats = np.array(
        [combo_to_cat.get(combo, 1) for combo in synth_joint], dtype=int
    )

    # QI attribute matrices (no hidden columns)
    train_attrs = targets[qi].to_numpy()
    synth_attrs = synth[qi].to_numpy()

    # Generate queries: (k-1) QI conditions + 1 joint-category target
    all_queries = gen_all_simple_kway_categorical(train_attrs, k, num_categories)
    target_categories = np.array([q[2] for q in all_queries], dtype=int)
    print(f"  [LinearJoint] {len(all_queries)} queries generated")

    # Query matrix over train records
    A = simple_kway(all_queries, train_attrs)
    train_n_users = np.sum(A, axis=1)

    # Evaluate same queries on synth
    synth_n_users, synth_results = get_result_simple_kway_categorical(
        synth_attrs, synth_secret_cats, all_queries
    )

    # Scale synth counts to train population size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scale = np.where(
            synth_n_users > 0,
            train_n_users / synth_n_users,
            len(targets) / len(synth),
        )
    synth_results_scaled = synth_results * scale

    print(f"  [LinearJoint] Query matrix: {A.shape}, avg scale: {np.mean(scale):.2f}")
    print(f"  [LinearJoint] Running LP solver (Gurobi, {n_procs} procs)...")

    est_cats, scores, success = categorical_l1_solve(
        A, synth_results_scaled, num_categories, target_categories, n_procs
    )

    status = "succeeded" if success else "WARNING: failed — predictions may be unreliable"
    print(f"  [LinearJoint] LP solver {status}")

    # Decode joint category back to per-feature values
    reconstructed = targets[qi].copy()
    fallback_combo = joint_combos[0]
    for feat_idx, feat in enumerate(hidden_features):
        est_values = np.array(
            [cat_to_combo.get(int(c), fallback_combo)[feat_idx] for c in est_cats]
        )
        reconstructed[feat] = est_values

    return reconstructed, None, None
