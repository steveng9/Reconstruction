#!/usr/bin/env python3
"""
1. Verify ACS NonPrivate results match paper's reported ~0.99 AUC (using scores, not just accuracy)
2. Test NIST CRC data through the wrapper (dropping F21, F22)
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.expanduser('~/recon-synth'))
from attacks import query_attack
from attacks.simple_kway_queries import gen_all_simple_kway, get_result_simple_kway, simple_kway
from load_data import process_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SOTA_attacks'))
from linear_reconstruction import linear_reconstruction_attack


def run_paper_method_with_auc(train_df, synth_df, secret_bit, k=3, n_procs=4):
    """Run paper's method and return accuracy, AUC, and scores."""
    train_attrs, train_secret = process_data(train_df, secret_bit)
    synth_attrs, synth_secret = process_data(synth_df, secret_bit)

    all_q = gen_all_simple_kway(train_attrs, k)
    queries = [q for q in all_q if q[2] == 1]

    A = simple_kway(queries, train_attrs)
    train_n = np.sum(A, axis=1)
    synth_n, synth_res = get_result_simple_kway(synth_attrs, synth_secret, queries)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        scale = np.where(synth_n > 0, train_n / synth_n, len(train_df) / len(synth_df))
    synth_res_scaled = synth_res * scale

    est_bits, scores, success = query_attack(A, synth_res_scaled, n_procs)

    acc = np.mean(est_bits == train_secret)
    baseline = max(np.mean(train_secret == 0), np.mean(train_secret == 1))

    # AUC using continuous scores
    try:
        auc = roc_auc_score(train_secret, scores)
    except:
        auc = None

    return acc, auc, baseline, est_bits, scores, train_secret


# ============================================================
# PART 1: Verify ACS NonPrivate matches paper (~0.99 AUC)
# ============================================================
print("=" * 70)
print("PART 1: ACS + NonPrivate — Verify against paper's ~0.99 AUC")
print("=" * 70)

acs = pd.read_csv(os.path.expanduser('~/recon-synth/datasets/acs.csv'))
acs_train = acs.sample(n=1000, random_state=42)
acs_synth = acs_train.sample(n=1000, replace=True, random_state=43)

acc, auc, baseline, est_bits, scores, true_secret = run_paper_method_with_auc(
    acs_train, acs_synth, 'SEX')

print(f"Accuracy: {acc*100:.2f}% (baseline: {baseline*100:.2f}%, lift: {(acc-baseline)*100:.2f}pp)")
print(f"AUC: {auc:.4f}")
print(f"Scores stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
print(f"Paper reports: ~0.99 AUC for NonPrivate")

# Now through the ACTUAL wrapper
print(f"\nSame thing through wrapper:")
qi_acs = [c for c in acs_train.columns if c != 'SEX']
cfg = {"attack_params": {"k": 3, "n_procs": 4}}
recon_acs, _, _ = linear_reconstruction_attack(cfg, acs_synth, acs_train, qi_acs, ['SEX'])
acc_wrap = np.mean(recon_acs['SEX'].values == acs_train['SEX'].values)
print(f"Wrapper accuracy: {acc_wrap*100:.2f}%")
# NOTE: wrapper doesn't return scores — it only returns binary predictions
# This means we CANNOT compute AUC from the wrapper as-is

# ============================================================
# PART 2: NIST CRC through the wrapper (drop F21, F22)
# ============================================================
print(f"\n{'=' * 70}")
print("PART 2: NIST CRC through wrapper (F21, F22 dropped)")
print("=" * 70)

train = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
synth = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')

if 'ID' in train.columns:
    train = train.drop(columns=['ID'])
for c in ['F21', 'F22']:
    if c in train.columns:
        train = train.drop(columns=[c])
    if c in synth.columns:
        synth = synth.drop(columns=[c])

print(f"Train: {train.shape}, Synth: {synth.shape}")

for n_train in [1000]:
    train_sub = train.sample(n=n_train, random_state=42)

    for secret in ['F13', 'F43', 'F41']:
        if secret not in train_sub.columns:
            print(f"  {secret} not in columns, skipping")
            continue

        qi = [c for c in train_sub.columns if c != secret]

        train_dist = dict(zip(*np.unique(train_sub[secret], return_counts=True)))
        baseline = max(np.mean(train_sub[secret] == v) for v in train_sub[secret].unique())

        print(f"\n  secret={secret}, n_train={n_train}, n_qi={len(qi)}")
        print(f"  Train dist: {train_dist}, baseline: {baseline*100:.2f}%")

        # Check if binary
        if train_sub[secret].nunique() != 2:
            print(f"  SKIPPING: {secret} has {train_sub[secret].nunique()} unique values (not binary)")
            continue

        cfg = {"attack_params": {"k": 3, "n_procs": 4}}
        try:
            reconstructed, _, _ = linear_reconstruction_attack(
                cfg, synth, train_sub, qi, [secret]
            )
            acc = np.mean(reconstructed[secret].values == train_sub[secret].values)
            print(f"  Accuracy: {acc*100:.2f}%, Lift: {(acc - baseline)*100:.2f}pp")
        except Exception as e:
            print(f"  FAILED: {e}")
