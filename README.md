# Reconstruction Attack Experiment Suite

Survey-style framework for **reconstruction attacks** (attribute inference on synthetic tabular data) and **membership inference attacks (MIA)**. Given a synthetic dataset and partial knowledge of a target record (quasi-identifiers), attacks attempt to reconstruct unknown attribute values or determine membership.

## Quick Start

```bash
conda activate recon_

# Run reconstruction experiment
python master_experiment_script.py --n_runs 1 --on_server T

# Run MIA experiment
python master_experiment_script.py --mode mia --n_runs 1 --on_server T
```

## Generating Synthetic Data

Data generation is a two-step process managed by `sdg/generate_synth.py`. Set `DATASET`, `SAMPLE_SIZE`, and other options at the top of the script.

```bash
conda activate sdg

python sdg/generate_synth.py sample   # Step 1: create training samples
python sdg/generate_synth.py sdg      # Step 2: generate synthetic data (parallelized)
python sdg/generate_synth.py count    # Count generated synth.csv files
```

### Key Configuration Options

| Variable | Default | Description |
|---|---|---|
| `DATASET` | `"nist_sbo"` | Dataset name (must have entry in `_SDG_JOBS_BY_DATASET`) |
| `SAMPLE_SIZE` | `1000` | Rows per training sample |
| `NUM_SAMPLES` | `5` | Number of training samples to create |
| `DISJOINT` | `True` | Whether samples must be non-overlapping |
| `SAMPLES_TO_GENERATE` | `range(5)` | Which sample indices to run SDG for |
| `MAX_WORKERS` | `8` | Max parallel SDG jobs |

### Disjoint vs. Non-Disjoint Sampling

**`DISJOINT=True`** (default): Samples are non-overlapping slices of the shuffled dataset. Requires `NUM_SAMPLES × SAMPLE_SIZE ≤ total rows`. Safe to use as holdout sets in memorization tests.

**`DISJOINT=False`**: Each sample is drawn independently — samples may share rows. Use when the dataset isn't large enough for fully disjoint splits, or when you don't need memorization tests. A `NO_HOLDOUT` marker file is written into each sample directory; `get_data.py` will refuse to load these as holdout sets (both the train dir and holdout dir are checked).

### Per-Dataset SDG Job Configs

SDG jobs are defined in `_SDG_JOBS_BY_DATASET` — a dict keyed by dataset name. Setting `DATASET` at the top automatically selects the right job list with all dataset-specific parameters pre-configured. No more commenting/uncommenting.

To add a new dataset, add an entry to `_SDG_JOBS_BY_DATASET`. Missing entries raise a clear `ValueError`.

#### Dataset-specific notes

- **california**: No MST/AIM (all-continuous dataset, binning loses fidelity). No CellSuppression (no categorical QIs).
- **nist_sbo**: No AIM (130 cols, combinatorially infeasible). MST jobs use `bin_continuous_as_ordinal=True` (see below). TabDDPM uses wider architecture (`d_layers=[1024,2048,2048,2048,1024]`, `iterations=300000`) for the high-dimensional one-hot encoded input.

## SDG Methods

| Method | Type | Notes |
|---|---|---|
| MST | DP (marginal) | CPU. `epsilon` param. On wide datasets with skewed continuous cols, use `bin_continuous_as_ordinal=True` |
| AIM | DP (marginal) | CPU. Infeasible for 50+ columns |
| TVAE | Deep generative | GPU |
| CTGAN | Deep generative | GPU |
| ARF | Deep generative | GPU |
| TabDDPM | Diffusion | GPU. Supports `d_layers`, `iterations`, `num_timesteps` |
| Synthpop | R-based | CPU |
| RankSwap | De-identification | CPU. `swap_features` = continuous cols only |
| CellSuppression | De-identification | CPU. `key_vars` = categorical QI cols, must be non-empty |

### MST `bin_continuous_as_ordinal`

SmartNoise's private bound estimation (`approx_bounds`) fails when `preprocessor_eps` is too small for the data's range (e.g., `epsilon=0.1` → `preprocessor_eps=0.03` on financial columns spanning 0–140k). Setting `bin_continuous_as_ordinal=True` pre-bins continuous columns using the actual data range before fitting, bypassing `BinTransformer` entirely. MST bins continuous data internally anyway — this is equivalent and gives the full epsilon budget to synthesis.

## Evaluating Synthetic Data Quality

```bash
conda activate recon_

python sdg/evaluate_synth.py              # all datasets
python sdg/evaluate_synth.py nist_sbo    # one dataset
python sdg/evaluate_synth.py --verbose nist_sbo  # + per-column TVD/error tables
```

Metrics: `mean_tvd↓`, `mean_jsd↓`, `pairwise_tvd↓`, `mean_mean_err%↓`, `mean_std_err%↓`, `corr_diff↓`, `sdv_col_shapes↑`, `sdv_col_pairs↑`. A `~train_baseline` row (train vs. train round-robin) is shown when ≥2 samples exist.

## Data Layout

```
/home/golobs/data/reconstruction_data/
  {dataset}/
    meta.json
    full_data.csv
    size_{N}/
      sdg_log.txt
      sample_{XX}/
        train.csv
        [NO_HOLDOUT]          ← present if sample is non-disjoint
        {Method_params}/
          synth.csv
          sdg.log
```

## Memorization / Holdout Tests

Memorization tests compare attack performance on training members vs. held-out non-members. The holdout is a separate disjoint sample directory.

`get_data.py` enforces safety: if **either** the train dir or the holdout dir contains a `NO_HOLDOUT` marker, loading raises a `ValueError`. This prevents accidentally using overlapping samples as holdout.

## Architecture Overview

See `CLAUDE.md` for full architecture details (attack registry, config structure, scoring, external dependencies).
