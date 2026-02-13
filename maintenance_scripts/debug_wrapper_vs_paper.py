#!/usr/bin/env python3
"""
Compare: call the actual wrapper function vs paper's direct method on ACS data.
"""
import sys, os
import numpy as np
import pandas as pd

# recon-synth first to avoid import conflicts
sys.path.insert(0, os.path.expanduser('~/recon-synth'))
from attacks import query_attack
from attacks.simple_kway_queries import gen_all_simple_kway, get_result_simple_kway, simple_kway
from load_data import process_data

# Now import the actual wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SOTA_attacks'))
from linear_reconstruction import linear_reconstruction_attack

SECRET_BIT = 'SEX'
K = 3
N_TRAIN = 1000
N_SYNTH = 1000
N_PROCS = 4

# Load ACS data
train_df = pd.read_csv(os.path.expanduser('~/recon-synth/datasets/acs.csv'))
train_df = train_df.sample(n=N_TRAIN, random_state=42)
synth_df = train_df.sample(n=N_SYNTH, replace=True, random_state=43)

qi = [c for c in train_df.columns if c != SECRET_BIT]

# ============================================================
# PATH A: Paper's exact method
# ============================================================
print("=" * 70)
print("PATH A: Paper's exact method (test_paper_method.py)")
print("=" * 70)
train_attrs, train_secret = process_data(train_df, SECRET_BIT)
synth_attrs, synth_secret = process_data(synth_df, SECRET_BIT)

all_q = gen_all_simple_kway(train_attrs, K)
queries = [q for q in all_q if q[2] == 1]

A = simple_kway(queries, train_attrs)
train_n = np.sum(A, axis=1)
synth_n, synth_res = get_result_simple_kway(synth_attrs, synth_secret, queries)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    scale = np.where(synth_n > 0, train_n / synth_n, N_TRAIN / N_SYNTH)
synth_res_scaled = synth_res * scale

est_paper, scores_paper, success_paper = query_attack(A, synth_res_scaled, N_PROCS)
acc_paper = np.mean(est_paper == train_secret)
baseline = max(np.mean(train_secret == 0), np.mean(train_secret == 1))
print(f"Paper accuracy: {acc_paper*100:.2f}%")
print(f"Baseline: {baseline*100:.2f}%")
print(f"Lift: {(acc_paper - baseline)*100:.2f}pp")

# ============================================================
# PATH B: Actual wrapper function call
# ============================================================
print("\n" + "=" * 70)
print("PATH B: Calling linear_reconstruction_attack() wrapper directly")
print("=" * 70)

cfg = {
    "attack_params": {
        "k": K,
        "n_procs": N_PROCS,
    }
}

reconstructed, _, _ = linear_reconstruction_attack(
    cfg, synth_df, train_df, qi, [SECRET_BIT]
)

acc_wrap = np.mean(reconstructed[SECRET_BIT].values == train_df[SECRET_BIT].values)
print(f"\nWrapper accuracy: {acc_wrap*100:.2f}%")
print(f"Baseline: {baseline*100:.2f}%")
print(f"Lift: {(acc_wrap - baseline)*100:.2f}pp")

print(f"\n{'=' * 70}")
print(f"Paper: {acc_paper*100:.2f}%  |  Wrapper: {acc_wrap*100:.2f}%")
print(f"{'=' * 70}")
