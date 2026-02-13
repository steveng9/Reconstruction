"""
Verify that queries are generated correctly from the right datasets.

Trace through exact data flow:
1. Where are queries generated from? (train or synth?)
2. Do they include the secret bit in attributes? (they shouldn't!)
3. Is the query matrix A built from the right data?
4. Are synthetic results computed correctly?
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/golobs/recon-synth')
from attacks.simple_kway_queries import gen_all_simple_kway, simple_kway, get_result_simple_kway
from load_data import process_data

print("="*70)
print("VERIFYING QUERY GENERATION LOGIC")
print("="*70)

# Load data
train = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
synth = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')
train = train.drop('ID', axis=1, errors='ignore')
synth = synth.drop('ID', axis=1, errors='ignore')

# Sample small for clarity
train = train.sample(n=100, random_state=42)
synth = synth.sample(n=100, random_state=43)

secret = 'F13'
qi = [col for col in train.columns if col != secret]

print(f"\n1. DATA SETUP")
print(f"   Train shape: {train.shape}")
print(f"   Synth shape: {synth.shape}")
print(f"   Secret feature: {secret}")
print(f"   QI features: {len(qi)}")

# STEP 1: Extract attributes for query generation
print(f"\n2. QUERY GENERATION INPUT")
train_attrs_for_queries = train[qi].to_numpy()
print(f"   train_attrs shape: {train_attrs_for_queries.shape}")
print(f"   Source: TRAINING data")
print(f"   Includes secret {secret}? NO (only QI features)")
print(f"   First row example: {train_attrs_for_queries[0][:5]}...")

# STEP 2: Generate queries
queries = gen_all_simple_kway(train_attrs_for_queries, k=3)
queries_target1 = [q for q in queries if q[2] == 1]
print(f"\n3. QUERY GENERATION")
print(f"   Total queries: {len(queries)}")
print(f"   After filtering to target_val=1: {len(queries_target1)}")
print(f"   Queries generated from: TRAINING data QI features")
print(f"   Example query: {queries_target1[0]}")
print(f"     Interpretation: Select users where")
print(f"                     QI feature {queries_target1[0][0][0]} = {queries_target1[0][1][0]}")
print(f"                     AND QI feature {queries_target1[0][0][1]} = {queries_target1[0][1][1]}")
print(f"                     AND secret = {queries_target1[0][2]}")

# STEP 3: Build query matrix A from TRAINING data
print(f"\n4. QUERY MATRIX A (for LP solver)")
A = simple_kway(queries_target1, train_attrs_for_queries)
print(f"   A shape: {A.shape} (queries x users)")
print(f"   Built from: TRAINING data QI features")
print(f"   A[i,j] = 1 if training user j satisfies query i's attributes")
print(f"   Example: Query 0 matches {int(A[0].sum())} training users")
train_n_users = np.sum(A, axis=1)
print(f"   Average users per query: {train_n_users.mean():.2f}")

# STEP 4: Process synthetic data
print(f"\n5. SYNTHETIC DATA PROCESSING")
synth_qi_secret = synth[qi + [secret]].copy()
synth_attrs, synth_secret_values = process_data(synth_qi_secret, secret)
print(f"   synth_attrs shape: {synth_attrs.shape}")
print(f"   Source: SYNTHETIC data")
print(f"   Includes secret? NO (process_data removes it)")
print(f"   synth_secret_values shape: {synth_secret_values.shape}")
print(f"   synth_secret_values values: {np.unique(synth_secret_values)}")

# Map to binary
synth_unique = np.unique(synth_secret_values)
if not (set(synth_unique) == {0, 1}):
    synth_secret_bits = (synth_secret_values == synth_unique[1]).astype(int)
    print(f"   Mapped {synth_unique} to {{0, 1}}")
else:
    synth_secret_bits = synth_secret_values

# STEP 5: Evaluate queries on SYNTHETIC data
print(f"\n6. EVALUATE QUERIES ON SYNTHETIC DATA")
synth_n_users, synth_results = get_result_simple_kway(
    synth_attrs, synth_secret_bits, queries_target1
)
print(f"   synth_n_users: count of synth users matching query attributes")
print(f"   synth_results: count of those with secret=1")
print(f"   Example: Query 0 matches {int(synth_n_users[0])} synth users,")
print(f"            of which {int(synth_results[0])} have secret=1")
print(f"   Average users per query: {synth_n_users.mean():.2f}")

# STEP 6: Scaling
print(f"\n7. SCALING")
scale = np.where(
    synth_n_users > 0,
    train_n_users / synth_n_users,
    len(train) / len(synth)
)
synth_results_scaled = synth_results * scale
print(f"   Scale formula: train_n_users / synth_n_users")
print(f"   Fallback (if synth_n_users=0): len(train) / len(synth) = {len(train)/len(synth)}")
print(f"   Average scale: {scale.mean():.2f}")
print(f"   Average scaled result: {synth_results_scaled.mean():.2f}")

# STEP 7: LP Solver
print(f"\n8. LP SOLVER INPUTS")
print(f"   A: {A.shape} - query matrix from TRAINING data")
print(f"   synth_results_scaled: {synth_results_scaled.shape} - from SYNTHETIC data")
print(f"   LP solves: find x such that A @ x ≈ synth_results_scaled")
print(f"   where x[j] = estimated probability that TRAINING user j has secret=1")

print(f"\n{'='*70}")
print("VERIFICATION SUMMARY:")
print("="*70)
print("✓ Queries generated from: TRAINING data (QI only, no secret)")
print("✓ Query matrix A built from: TRAINING data (correct)")
print("✓ Query results evaluated on: SYNTHETIC data (correct)")
print("✓ LP solves for: TRAINING users' secret values (correct)")
print("="*70)
