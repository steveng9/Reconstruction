#!/usr/bin/env python3
"""
Debug script to inspect all intermediate variables in the linear reconstruction
attack wrapper, comparing with what the paper's original code does.
"""
import sys
import os
import numpy as np
import pandas as pd

# Import from recon-synth FIRST (before Reconstruction/attacks shadows it)
sys.path.insert(0, os.path.expanduser('~/recon-synth'))
from attacks import query_attack
from attacks.simple_kway_queries import gen_all_simple_kway, get_result_simple_kway, simple_kway
from load_data import process_data

# Now add Reconstruction to path for get_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from get_data import QIs, minus_QIs

# ============================================================
# Load data exactly as master_experiment_script would
# ============================================================
print("=" * 70)
print("LOADING DATA (same as master_experiment_script)")
print("=" * 70)

train = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
synth = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')

qi = QIs["nist_arizona_data"]["QI1"]
hidden_features = minus_QIs["nist_arizona_data"]["QI1"]

print(f"Train shape: {train.shape}")
print(f"Synth shape: {synth.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"QI ({len(qi)}): {qi}")
print(f"Hidden features ({len(hidden_features)}): {hidden_features}")

# Pick a single hidden feature to debug (first binary one)
# Let's find which hidden features are binary
print(f"\nHidden feature unique value counts:")
for hf in hidden_features:
    nuniq = train[hf].nunique()
    vals = sorted(train[hf].unique())
    if nuniq <= 5:
        print(f"  {hf}: {nuniq} values -> {vals}")
    else:
        print(f"  {hf}: {nuniq} values -> {vals[:5]}...")

# Pick a binary hidden feature for debugging
binary_features = [hf for hf in hidden_features if train[hf].nunique() == 2]
print(f"\nBinary hidden features: {binary_features}")

if not binary_features:
    print("No binary hidden features found!")
    sys.exit(1)

secret_feature = binary_features[0]
print(f"\n{'=' * 70}")
print(f"DEBUGGING WITH secret_feature = {secret_feature}")
print(f"{'=' * 70}")

# ============================================================
# Now trace through the wrapper step by step
# ============================================================
k = 3

# Step 1: train_attrs (what the wrapper does)
print(f"\n--- Step 1: train_attrs = targets[qi].to_numpy() ---")
train_attrs = train[qi].to_numpy()
print(f"train_attrs shape: {train_attrs.shape}")
print(f"train_attrs dtype: {train_attrs.dtype}")
print(f"First 3 rows:\n{train_attrs[:3]}")
for i, col in enumerate(qi):
    uniq = np.unique(train_attrs[:, i])
    print(f"  Column {i} ({col}): {len(uniq)} unique values, range [{uniq.min()}, {uniq.max()}]")

# Step 2: synth processing (what the wrapper does)
print(f"\n--- Step 2: synth_attrs via process_data ---")
synth_qi_secret = synth[qi + [secret_feature]].copy()
print(f"synth_qi_secret shape: {synth_qi_secret.shape}")
print(f"synth_qi_secret columns: {list(synth_qi_secret.columns)}")

synth_attrs, synth_secret_values = process_data(synth_qi_secret, secret_feature)
print(f"synth_attrs shape: {synth_attrs.shape}")
print(f"synth_attrs dtype: {synth_attrs.dtype}")
print(f"synth_secret_values shape: {synth_secret_values.shape}")
print(f"synth_secret_values unique: {np.unique(synth_secret_values)}")
print(f"synth_secret_values distribution: {dict(zip(*np.unique(synth_secret_values, return_counts=True)))}")

# Step 2b: secret bit mapping
synth_unique = np.unique(synth_secret_values)
print(f"\nsynth_unique: {synth_unique}")
if not (set(synth_unique) == {0, 1}):
    print(f"Mapping {synth_unique[0]} -> 0, {synth_unique[1]} -> 1")
    synth_secret_bits = (synth_secret_values == synth_unique[1]).astype(int)
    value_mapping = {0: synth_unique[0], 1: synth_unique[1]}
else:
    synth_secret_bits = synth_secret_values
    value_mapping = {0: 0, 1: 1}

print(f"synth_secret_bits unique: {np.unique(synth_secret_bits)}")
print(f"synth_secret_bits distribution: {dict(zip(*np.unique(synth_secret_bits, return_counts=True)))}")
print(f"value_mapping: {value_mapping}")

# Step 3: Generate queries
print(f"\n--- Step 3: gen_all_simple_kway(train_attrs, k={k}) ---")
all_queries = gen_all_simple_kway(train_attrs, k)
print(f"Total all_queries: {len(all_queries)}")
print(f"First 5 all_queries:")
for q in all_queries[:5]:
    print(f"  attr_inds={q[0]}, attr_vals={q[1]}, target_val={q[2]}")
print(f"Last 5 all_queries:")
for q in all_queries[-5:]:
    print(f"  attr_inds={q[0]}, attr_vals={q[1]}, target_val={q[2]}")

# Target val distribution
target_vals = [q[2] for q in all_queries]
print(f"Target val distribution: 0={target_vals.count(0)}, 1={target_vals.count(1)}")

# Step 4: Filter to target_val=1
print(f"\n--- Step 4: Filter queries to target_val=1 ---")
queries = [
    (attr_inds, attr_vals, target_val)
    for (attr_inds, attr_vals, target_val) in all_queries
    if target_val == 1
]
print(f"Filtered queries: {len(queries)}")
print(f"First 5 filtered queries:")
for q in queries[:5]:
    print(f"  attr_inds={q[0]}, attr_vals={q[1]}, target_val={q[2]}")

# Step 5: Build A matrix
print(f"\n--- Step 5: A = simple_kway(queries, train_attrs) ---")
A = simple_kway(queries, train_attrs)
print(f"A shape: {A.shape} (queries x training_records)")
print(f"A dtype: {A.dtype}")
print(f"A density (fraction of 1s): {A.mean():.6f}")
print(f"A row sums (train_n_users) stats:")
train_n_users = np.sum(A, axis=1)
print(f"  min={train_n_users.min()}, max={train_n_users.max()}, mean={train_n_users.mean():.2f}, median={np.median(train_n_users):.2f}")
print(f"  zeros (queries matching no train records): {np.sum(train_n_users == 0)}")
print(f"A col sums (queries per record) stats:")
col_sums = np.sum(A, axis=0)
print(f"  min={col_sums.min()}, max={col_sums.max()}, mean={col_sums.mean():.2f}")

# Step 6: Evaluate on synth
print(f"\n--- Step 6: get_result_simple_kway(synth_attrs, synth_secret_bits, queries) ---")
synth_n_users, synth_results = get_result_simple_kway(synth_attrs, synth_secret_bits, queries)
print(f"synth_n_users shape: {synth_n_users.shape}")
print(f"synth_n_users stats: min={synth_n_users.min()}, max={synth_n_users.max()}, mean={synth_n_users.mean():.2f}")
print(f"synth_n_users zeros: {np.sum(synth_n_users == 0)}")
print(f"synth_results shape: {synth_results.shape}")
print(f"synth_results stats: min={synth_results.min():.4f}, max={synth_results.max():.4f}, mean={synth_results.mean():.4f}")
print(f"synth_results zeros: {np.sum(synth_results == 0)}")

# Step 7: Scaling
print(f"\n--- Step 7: Conditional scaling ---")
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    scale = np.where(
        synth_n_users > 0,
        train_n_users / synth_n_users,
        len(train) / len(synth)
    )
synth_results_scaled = synth_results * scale
print(f"scale stats: min={scale.min():.4f}, max={scale.max():.4f}, mean={scale.mean():.4f}")
print(f"synth_results_scaled stats: min={synth_results_scaled.min():.4f}, max={synth_results_scaled.max():.4f}, mean={synth_results_scaled.mean():.4f}")

# ============================================================
# NOW: Compare with paper's method (process_data on full df)
# ============================================================
print(f"\n{'=' * 70}")
print(f"COMPARISON: Paper's method (ALL features, not just QI)")
print(f"{'=' * 70}")

# Paper uses ALL non-secret columns
paper_train_attrs, paper_train_secret = process_data(train, secret_feature)
paper_synth_attrs, paper_synth_secret = process_data(synth, secret_feature)

print(f"Paper train_attrs shape: {paper_train_attrs.shape} (all {paper_train_attrs.shape[1]} non-secret cols)")
print(f"Wrapper train_attrs shape: {train_attrs.shape} (only {train_attrs.shape[1]} QI cols)")

# Map paper secret to {0,1}
paper_unique = np.unique(paper_synth_secret)
if not (set(paper_unique) == {0, 1}):
    paper_synth_bits = (paper_synth_secret == paper_unique[1]).astype(int)
    paper_train_bits = (paper_train_secret == paper_unique[1]).astype(int)
else:
    paper_synth_bits = paper_synth_secret
    paper_train_bits = paper_train_secret

print(f"\nPaper train secret distribution: {dict(zip(*np.unique(paper_train_bits, return_counts=True)))}")

# Generate paper queries
paper_all_queries = gen_all_simple_kway(paper_train_attrs, k)
paper_queries = [q for q in paper_all_queries if q[2] == 1]
print(f"Paper total queries: {len(paper_all_queries)}")
print(f"Paper filtered queries (target_val=1): {len(paper_queries)}")
print(f"Wrapper filtered queries: {len(queries)}")
print(f"RATIO: paper has {len(paper_queries)/max(len(queries),1):.1f}x more queries")

# Paper A matrix (don't build it if too large)
if len(paper_queries) < 100000:
    paper_A = simple_kway(paper_queries, paper_train_attrs)
    print(f"\nPaper A shape: {paper_A.shape}")
    print(f"Paper A density: {paper_A.mean():.6f}")
    paper_train_n = np.sum(paper_A, axis=1)
    print(f"Paper train_n_users: min={paper_train_n.min()}, max={paper_train_n.max()}, mean={paper_train_n.mean():.2f}")
else:
    print(f"\nPaper queries too large ({len(paper_queries)}) to build A matrix - skipping")

# ============================================================
# Ground truth check: what SHOULD the attack predict?
# ============================================================
print(f"\n{'=' * 70}")
print(f"GROUND TRUTH")
print(f"{'=' * 70}")
train_secret = train[secret_feature].values
print(f"Train secret feature '{secret_feature}' distribution: {dict(zip(*np.unique(train_secret, return_counts=True)))}")
baseline = max(np.mean(train_secret == v) for v in np.unique(train_secret))
print(f"Baseline (majority class): {baseline:.4f} = {baseline*100:.2f}%")

print(f"\nDone. Run the actual LP solver separately if needed.")
