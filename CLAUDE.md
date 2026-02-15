# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Survey-style experiment framework for **reconstruction attacks** (attribute inference) on synthetic tabular data. Given a synthetic dataset and partial knowledge of a target record (quasi-identifiers), attacks attempt to reconstruct unknown attribute values. The framework supports multiple SDG methods, attack algorithms, and composable enhancements (chaining, ensembling).

## Running Experiments

```bash
# Activate conda environment
conda activate recon_

# Run experiment (reads config from CONFIG_PATH_default)
python master_experiment_script.py --n_runs 1 --on_server T

# Override config path
CONFIG_PATH_default=/path/to/config.yaml python master_experiment_script.py --n_runs 3 --on_server T

# Run SDG smoke tests
python -m sdg.test_sdg              # all methods
python -m sdg.test_sdg MST TVAE     # specific methods
```

There is no formal test suite (no pytest). Test/debug scripts live in `maintenance_scripts/`.

## Architecture

### Execution Flow

`master_experiment_script.py` orchestrates everything:
1. Load YAML config → `load_config()`
2. Init WandB tracking
3. Per run: `load_data()` → `_prepare_config()` → `_run_attack()` → `_score_reconstruction()` → `wandb.log()`
4. Optional memorization test: runs attack on both training and holdout targets, logs `RA_train_*`, `RA_nontraining_*`, `RA_delta_*`

### Registry Pattern

Both attacks and SDG methods use a registry dict → `get_X(name)` lookup:

- **`attacks/__init__.py`**: `ATTACK_REGISTRY` maps `{data_type: {name: fn}}` where data_type is `categorical`, `continuous`, or `agnostic`. All attack functions share signature: `(cfg, synth, targets, qi, hidden_features) → (reconstructed_df, probas, classes)`
- **`sdg/__init__.py`**: `SDG_REGISTRY` maps `{name: fn}`. All SDG functions share signature: `generate(train_df, meta, **config) → synthetic_df` where `meta` has keys `categorical`, `continuous`, `ordinal`.

### Enhancement Wrappers (`enhancements/`)

Composable wrappers applied in order: ensembling first, then chaining.

- **Ensembling**: Runs multiple attack methods, aggregates predictions (voting, soft_voting, averaging, median)
- **Chaining**: Sequentially predicts hidden features one-at-a-time, adding each prediction to known features. Order strategies: `default`, `manual`, `random`, `correlation`, `reverse_correlation`, `mutual_info`

### Config Structure (YAML)

```yaml
wandb: {project, name, tags, group}
dataset: {name, dir, size, type}  # type: categorical or continuous
QI: "QI1"                         # selects from predefined sets in get_data.py
sdg_method: "MST"
sdg_params: {epsilon: 1.0}
memorization_test: {enabled: false}
attack_method: "RandomForest"
data_type: "categorical"
attack_params:
  chaining: {enabled, order_strategy, ...}
  ensembling: {enabled, methods, aggregation, ...}
  RandomForest: {max_depth: 15, ...}  # method-specific params
```

`_prepare_config()` flattens method-specific params (e.g., `attack_params.RandomForest.*`) into `attack_params` directly, keeping enhancement sub-dicts (those with `enabled` key) intact.

### Data Loading (`get_data.py`)

- `load_data(config) → (train, synth, qi, hidden_features, holdout)`
- Reads `train.csv`, `synth.csv`, optionally `holdout.csv` from `config.dataset.dir`
- QI definitions are hardcoded: `QIs` and `minus_QIs` dicts keyed by dataset name then QI variant
- Datasets: `nist_arizona_data` (categorical, features F1-F50), `california_housing_data` (continuous)

### Scoring (`scoring.py`)

- Categorical: rarity-weighted accuracy (rarer values weighted higher)
- Continuous: normalized RMSE

### External Dependencies

The framework imports from sibling repos via `sys.path` manipulation (set by `--on_server` flag):
- **`MIA_on_diffusion/`**: TabDDPM and RePaint attack implementations
- **`recon-synth/`**: Linear reconstruction attack (LP + Gurobi solver)

Import of `recon-synth` modules uses direct function imports (not package imports) to avoid collision with local `attacks/` package.

## Key Conventions

- NIST CRC binary features use values `{1, 2}` not `{0, 1}` — attacks that need binary encoding must handle this mapping
- `hidden_features = all_features - qi` (complementary sets)
- SDG methods are run separately to produce `synth.csv`; the master script only consumes pre-generated synthetic data
- WandB metric keys: `RA_{feature}` (standard), `RA_train_*`/`RA_nontraining_*`/`RA_delta_*` (memorization test)
- Gurobi academic license required for LinearReconstruction attack
