#!/usr/bin/env python3
"""
Compute RA-as-MIA AUC for CellSuppression and RankSwap
to update the MIA comparison table.
"""
import sys
sys.path.insert(0, '/home/golobs/Reconstruction')
sys.path.append('/home/golobs/MIA_on_diffusion/')
sys.path.append('/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM')
sys.path.append('/home/golobs/recon-synth')
sys.path.append('/home/golobs/recon-synth/attacks')
sys.path.append('/home/golobs/recon-synth/attacks/solvers')

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from master_experiment_script import _prepare_config
from attacks import get_attack

DATASET = 'adult'
DATA_ROOT = '/home/golobs/data/reconstruction_data/'
SAMPLE_SIZE = 10_000
SAMPLE_DIR = f'{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01'
HOLDOUT_DIR = f'{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_02'

# Use the same restricted hidden features as compare_mia_ra.py
RA_HIDDEN_OVERRIDE = ['workclass', 'occupation', 'relationship', 'income']
QI = 'QI1'
DATA_TYPE = 'categorical'

SDG_METHODS = [
    ('CellSuppression', {}),
    ('RankSwap', {}),
    ('MST', {'epsilon': 1.0}),
]


def sdg_dirname(method, params=None):
    params = params or {}
    eps = params.get('epsilon')
    return f'{method}_eps{eps:g}' if eps is not None else method


def get_ra_scores(target_df, cfg, synth_df, qi_feats, hidden_all, ra_hidden):
    """Run RF attack on target_df, return per-row mean accuracy for ra_hidden features."""
    attack_fn = get_attack('RandomForest', data_type='categorical')

    # Filter targets to ra_hidden features
    hidden = [f for f in ra_hidden if f in hidden_all]

    recon, probas, classes = attack_fn(cfg, synth_df, target_df, qi_feats, hidden_all)

    scores = []
    for i in range(len(target_df)):
        row_scores = []
        for feat in hidden:
            true_val = str(target_df.iloc[i][feat])
            pred_val = str(recon[feat].iloc[i])
            row_scores.append(1.0 if true_val == pred_val else 0.0)
        scores.append(np.mean(row_scores) if row_scores else 0.0)
    return np.array(scores)


def run_ra_as_mia(sdg_method, sdg_params):
    dirname = sdg_dirname(sdg_method, sdg_params)

    cfg = {
        'dataset': {'name': DATASET, 'dir': SAMPLE_DIR, 'size': SAMPLE_SIZE, 'type': DATA_TYPE},
        'QI': QI, 'data_type': DATA_TYPE,
        'sdg_method': sdg_method,
        'sdg_params': sdg_params if sdg_params else None,
        'attack_method': 'RandomForest',
        'memorization_test': {'enabled': False},
        'attack_params': {
            'chaining': {'enabled': False},
            'ensembling': {'enabled': False},
            'RandomForest': {'max_depth': 15, 'num_estimators': 25},
        },
    }
    prepared = _prepare_config(cfg)

    from get_data import load_data
    train, synth, qi_feats, hidden_all, holdout = load_data(prepared)

    # Use holdout from sample_02
    holdout_df = pd.read_csv(f'{HOLDOUT_DIR}/train.csv')

    print(f'  Scoring {len(train)} members...', flush=True)
    ra_train = get_ra_scores(train, prepared, synth, qi_feats, hidden_all, RA_HIDDEN_OVERRIDE)

    print(f'  Scoring {len(holdout_df)} non-members...', flush=True)
    ra_holdout = get_ra_scores(holdout_df, prepared, synth, qi_feats, hidden_all, RA_HIDDEN_OVERRIDE)

    ra_scores = np.concatenate([ra_train, ra_holdout])
    labels = np.array([1]*len(train) + [0]*len(holdout_df))

    auc = roc_auc_score(labels, ra_scores)
    print(f'  RA-as-MIA: member mean={ra_train.mean():.4f}, nonmember mean={ra_holdout.mean():.4f}')
    return auc


print(f"\n{'SDG':<25} {'RA-as-MIA (QI1) AUC':>22}")
print('-' * 50)

for method, params in SDG_METHODS:
    dirname = sdg_dirname(method, params)
    print(f'\n--- {dirname} ---')
    auc = run_ra_as_mia(method, params)
    print(f'{dirname:<25} {auc:>22.3f}')
