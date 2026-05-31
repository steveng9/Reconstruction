#!/usr/bin/env python3
"""
Quick MIA comparison for Table update.
Runs SynthDistance and NNDR for RankSwap, CellSuppression, and MST eps=1.
Reports AUC for each.
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

from attacks.mia import synth_distance_mia, nndr_mia

DATASET = 'adult'
DATA_ROOT = '/home/golobs/data/reconstruction_data/'
SAMPLE_SIZE = 10_000
SAMPLE_DIR = f'{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01'
HOLDOUT_DIR = f'{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_02'

with open(f'{DATA_ROOT}{DATASET}/meta.json') as f:
    meta = json.load(f)

train_df = pd.read_csv(f'{SAMPLE_DIR}/train.csv')
holdout_df = pd.read_csv(f'{HOLDOUT_DIR}/train.csv')

SDG_METHODS = [
    ('RankSwap', {}),
    ('CellSuppression', {}),
    ('MST', {'epsilon': 1.0}),
    ('MST', {'epsilon': 1000.0}),
]


def sdg_dirname(method, params=None):
    params = params or {}
    eps = params.get('epsilon')
    return f'{method}_eps{eps:g}' if eps is not None else method


cfg = {'mia_params': {'n_targets': None}}  # use full sets

print(f"\n{'SDG':<25} {'SynthDist AUC':>15} {'NNDR AUC':>12}")
print('-' * 55)

for method, params in SDG_METHODS:
    dirname = sdg_dirname(method, params)
    synth_path = f'{SAMPLE_DIR}/{dirname}/synth.csv'
    synth_df = pd.read_csv(synth_path)

    sd_m = synth_distance_mia(cfg, synth_df, train_df, holdout_df, meta)
    nn_m = nndr_mia(cfg, synth_df, train_df, holdout_df, meta)

    print(f'{dirname:<25} {sd_m["MIA_auc"]:>15.3f} {nn_m["MIA_auc"]:>12.3f}')
