# SoK: Reconstruction Attacks on Synthetic Tabular Data

This repository is the artifact for:

> **SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC)**
> Submitted to *Privacy Enhancing Technologies Symposium (PoPETs) 2027*

The framework systematically evaluates **reconstruction attacks** (attribute inference) on de-identified and synthetic tabular data. Given a synthetic dataset and a target record's quasi-identifying features (e.g., age, sex, race), each attack attempts to reconstruct the target's hidden attribute values. We benchmark **13 attack algorithms** against **9 synthetic data generation (SDG) methods** across **5 benchmark datasets**, and introduce six new attacks: **MarginalRF**, **PartialMST**, **PartialTabDDPM**, **ConditionedRePaint**, **Attention**, and **JointMLP**.

The same methodology placed **first among all red teams** in the 2025 NIST Privacy Collaborative Research Cycle (CRC).

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Datasets](#datasets)
4. [Quick Start](#quick-start)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Repository Structure](#repository-structure)
7. [Attack Taxonomy](#attack-taxonomy)
8. [SDG Methods](#sdg-methods)
9. [Configuration Reference](#configuration-reference)
10. [Citation](#citation)

---

## Overview

### What this repo does

The central workflow is:

```
Raw data → Disjoint training samples → Synthetic data (via SDG) → Reconstruction attack → Scored results
```

- **`sdg/generate_synth.py`**: Generates synthetic data from training samples using any of 9 SDG methods.
- **`master_experiment_script.py`**: Runs a reconstruction (or MIA) attack for a single configuration and logs results to [Weights & Biases](https://wandb.ai).
- **`experiment_scripts/run_production_sweep.py`**: Parallelized sweep over all samples × SDG methods × attacks for a given dataset.

### Scoring

We use **rarity-weighted reconstruction advantage** ($R_{adv}$): a correct prediction of a rare attribute value earns proportionally more credit than predicting a common one. Under this metric, a mode-baseline attack scores exactly $100/C$% for a $C$-valued feature regardless of the distribution; a perfect attack scores 100%. This was also the official scoring metric for the NIST CRC competition.

---

## Installation

### Step 1: Clone this repository

```bash
git clone <repo-url>
cd Reconstruction
```

### Step 2: Create the experiment environment (`recon_`)

This environment is used to run attacks and experiments.

```bash
conda env create -f environment.yaml
conda activate recon_
```

Additional packages not in `environment.yaml`:

```bash
pip install wandb lightgbm tabpfn

# For PartialMST attack (private-pgm / mbi library):
pip install git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca
```

### Step 3: Create the SDG environment (`sdg`)

A separate environment is used for generating synthetic data, since some SDG libraries (e.g., SmartNoise, SDV, SynthCity) conflict with the attack dependencies.

```bash
conda create -n sdg python=3.10
conda activate sdg
pip install smartnoise-synth sdv synthcity pandas numpy scikit-learn tqdm
```

For R-based methods (Synthpop, RankSwap, CellSuppression via sdcMicro):
```bash
# Install R ≥ 4.2 and the following packages in R:
Rscript -e "install.packages(c('synthpop', 'sdcMicro'))"
```

For GPU-accelerated methods (TVAE, CTGAN, ARF, TabDDPM), a CUDA-enabled GPU is recommended but not required.

### Step 4: Install external attack dependencies

Two sibling repositories are required for the **PartialTabDDPM / ConditionedRePaint** and **LinearReconstruction** attacks.

**TabDDPM / RePaint attacks** (must be a sibling directory of `Reconstruction/`):
```bash
cd ..
git clone <MIA_on_diffusion_repo_url>
# Expected path: ../MIA_on_diffusion/midst_models/single_table_TabDDPM/
```

**Linear Reconstruction attack** (LP + Gurobi, must be a sibling directory):
```bash
git clone https://github.com/steveng9/recon-synth
# Expected path: ../recon-synth/
```

The LinearReconstruction attack also requires a [Gurobi](https://www.gurobi.com/) academic license (free for academics). Install via `pip install gurobipy` and activate with your license key.

### Verifying the installation

```bash
conda activate recon_

# Smoke-test all SDG methods (in sdg env):
conda run -n sdg python -m sdg.test_sdg

# Run a minimal reconstruction experiment:
python master_experiment_script.py --n_runs 1 --on_server T
```

---

## Datasets

All experiments use pre-generated data stored under a root data directory. The expected layout is:

```
/path/to/data/reconstruction_data/
  {dataset}/
    meta.json          ← feature metadata (categorical / continuous / ordinal)
    full_data.csv      ← full raw dataset
    size_{N}/
      sample_{XX}/
        train.csv      ← training sample (N rows)
        holdout.csv    ← holdout for memorization tests (optional)
        {SDG_Method_params}/
          synth.csv    ← synthetic data generated from train.csv
```

Update `DATA_ROOT` at the top of `experiment_scripts/run_production_sweep.py` to point to your data directory.

### Dataset 1: Adult (Census Income)

**Size**: 47,621 rows, 15 features  
**Type**: Categorical  
**Source**: UCI Machine Learning Repository

```bash
# Download via UCI ML Repo
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=2); d.data.original.to_csv('adult/full_data.csv', index=False)"
```

Or download directly from [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult).

**Quasi-identifiers (QI1 / demo)**: age, sex, race, native-country, education, marital-status  
**Hidden features**: workclass, fnlwgt, education-num, occupation, relationship, capital-gain, capital-loss, hours-per-week, income

### Dataset 2: CDC Diabetes Health Indicators

**Size**: 253,680 rows, 22 features  
**Type**: Categorical (all binary or low-cardinality)  
**Source**: Kaggle / CDC BRFSS 2015

Download `diabetes_binary_health_indicators_BRFSS2015.csv` from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

**Quasi-identifiers (QI1)**: Sex, Age, Education, Income, BMI, PhysActivity, Fruits, Veggies, HvyAlcoholConsump  
**Hidden features**: Diabetes_binary, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysHlth, MentHlth, DiffWalk, AnyHealthcare, NoDocbcCost, GenHlth

### Dataset 3: California Housing

**Size**: 20,640 rows, 9 features  
**Type**: Continuous  
**Source**: scikit-learn built-in

```bash
python -c "
from sklearn.datasets import fetch_california_housing
import pandas as pd
d = fetch_california_housing(as_frame=True)
df = d.frame
df.to_csv('california/full_data.csv', index=False)
"
```

**Quasi-identifiers (QI1)**: Latitude, Longitude, HouseAge, AveRooms  
**Hidden features**: AveBedrms, Population, AveOccup, MedInc, MedHouseVal

### Dataset 4: NIST Arizona (IPUMS 1940 Census)

**Size**: 293,999 rows, up to 98 features  
**Type**: Categorical (integer-coded IPUMS variables)  
**Source**: IPUMS USA (registration required)

Register at [https://usa.ipums.org/](https://usa.ipums.org/) and request the 1940 Arizona census extract with the 25-column subset described below. Rename or link the downloaded file to `nist_arizona_data/full_data.csv`.

We study three feature-subset variants:
- **`nist_arizona_25feat`**: 25 columns (mirrors the NIST CRC competition format)
- **`nist_arizona_50feat`**: 50 columns
- **`nist_arizona_data`**: all 98 columns

See `NIST_code/README.md` for the full feature list and QI definitions.

### Dataset 5: NIST Survey of Business Owners (SBO)

**Size**: ~123,892 rows (after cleaning), 130 features  
**Type**: Categorical  
**Source**: Available from NIST upon request as part of the Privacy CRC

The SBO dataset was provided to red teams in the 2025 NIST Privacy CRC. Place the cleaned CSV at `nist_sbo/full_data.csv`. Contact NIST or the CRC organizers for access.

### Setting up the data directory

After acquiring the datasets, create the directory structure and generate training samples:

```bash
conda activate sdg

# Edit DATASET, SAMPLE_SIZE, NUM_SAMPLES at the top of sdg/generate_synth.py
# then:
python sdg/generate_synth.py sample   # Step 1: carve out disjoint training samples
python sdg/generate_synth.py sdg      # Step 2: generate synthetic data for each sample
python sdg/generate_synth.py count    # Verify expected synth.csv files exist
```

---

## Quick Start

### Run a single experiment

```bash
conda activate recon_

# Edit configs/example_cfg.yaml to point to your data and set the attack/SDG method
python master_experiment_script.py --n_runs 1 --on_server T
```

Results are logged to WandB under the project and group specified in the config YAML. Set `WANDB_MODE=offline` to run without a WandB account.

### Run the MarginalRF attack on Adult (our best attack)

```bash
# Create a config pointing to adult/size_10000/sample_00, SDG_method=MST_eps1
# attack_method: MarginalRF
# data_type: categorical
python master_experiment_script.py --n_runs 1 --on_server T
```

---

## Reproducing Paper Results

The paper reports results averaged over **5 disjoint training samples** × **9 SDG methods** (MST at 5 ε values + AIM, TVAE, CTGAN, ARF, TabDDPM, Synthpop, RankSwap, CellSuppression). The full production sweep is run via:

```bash
conda activate recon_

# Edit dataset/paths/workers at the top of the script
python experiment_scripts/run_production_sweep.py \
    --workers 8 \
    --progress-log outfiles/progress.log
```

Monitor progress:
```bash
tail -f outfiles/progress.log
```

### Specific paper experiments

| Paper section | Script |
|---|---|
| Main attack × SDG table (Table 1) | `experiment_scripts/run_production_sweep.py` |
| Epsilon sweep (MST, AIM over ε) | `experiment_scripts/run_mst_epsilon_rf_nb_sweep.py` |
| LinearReconstruction comparison | `experiment_scripts/run_linear_sweep.py` |
| Memorization test (train vs. holdout) | `experiment_scripts/run_memorization_sweep.py` |
| MIA comparison (SynthDistance vs. NNDR vs. RA-as-MIA) | `experiment_scripts/compare_mia_ra.py` |
| QI composition/size analysis | `experiment_scripts/qi_analysis/run_qi_sweep.py` |
| Subgroup / disparate-impact analysis | `experiment_scripts/analyze_ra_subgroups.py` |
| Ensembling heatmap | `experiment_scripts/run_ensembling_heatmap.py` + `plot_ensembling_heatmap.py` |
| Chaining analysis | `experiment_scripts/run_chaining_analysis.py` |
| Synthetic data quality evaluation | `experiment_scripts/evaluate_synth_quality.py` |

### Generating LaTeX tables from WandB results

```bash
# Main results table
python experiment_scripts/wandb_to_latex.py --dataset adult --size 10000 --qi QI1

# Epsilon sweep table
python experiment_scripts/wandb_to_latex_epsilon_sweep.py --dataset adult

# Per-feature breakdown
python experiment_scripts/wandb_to_latex_by_feature.py --dataset adult

# LinearReconstruction binary-feature table
python experiment_scripts/linear_sweep_to_latex.py

# Synthetic data quality table
python experiment_scripts/synth_quality_to_latex.py experiment_scripts/synth_quality_results_*.csv
```

---

## Repository Structure

```
Reconstruction/
├── master_experiment_script.py   ← Main entry point: runs one experiment, logs to WandB
├── attack_defaults.py            ← Default hyperparameters for all 13 attacks
├── get_data.py                   ← Data loading: load_data(), load_mia_data(), QI definitions
├── scoring.py                    ← Rarity-weighted accuracy (categorical), NRMSE (continuous)
├── util.py                       ← Shared utilities
├── environment.yaml              ← Conda environment for running experiments (recon_ env)
│
├── attacks/                      ← All attack implementations
│   ├── __init__.py               ← ATTACK_REGISTRY: maps {data_type: {name: fn}}
│   ├── baselines_classifiers.py  ← Mode, Random, Copy baselines (categorical)
│   ├── baselines_continuous.py   ← Mean, Median baselines (continuous)
│   ├── ML_classifiers.py         ← KNN, NaiveBayes, LogisticRegression, SVM, RandomForest, LightGBM
│   ├── ML_regression.py          ← KNN, LinearRegression, Ridge, Lasso, ElasticNet, SGD, RandomForest, LightGBM (continuous)
│   ├── NN_classifier.py          ← MLP (categorical)
│   ├── NN_regression.py          ← MLP (continuous)
│   ├── attention_classifier.py   ← Attention, AttentionAutoregressive (new)
│   ├── joint_mlp.py              ← JointMLP: single network over all hidden features (new)
│   ├── marginal_rf.py            ← MarginalRF: RF posteriors + belief propagation (new, strongest attack)
│   ├── partialMST.py             ← PartialMST, PartialMSTIndep, PartialMSTBounded, PartialMSTHub (new)
│   ├── partialDiffusion.py       ← PartialTabDDPM, ConditionedRePaint, RePaint, TabDDPMWithMLP (new)
│   ├── tabpfn_attack.py          ← TabPFN: pre-trained transformer attack
│   └── mia.py                    ← Membership inference: SynthDistance, NNDR, RA-as-MIA
│
├── enhancements/                 ← Composable attack wrappers
│   ├── chaining_wrapper.py       ← Sequential feature prediction (any order strategy)
│   └── ensembling_wrapper.py     ← Multi-attack aggregation (voting, soft_voting, averaging)
│
├── sdg/                          ← Synthetic data generation
│   ├── generate_synth.py         ← Main script: sample / sdg / count subcommands
│   ├── evaluate_synth.py         ← Fidelity metrics (TVD, JSD, pairwise TVD, SDV scores)
│   ├── smartnoise_methods.py     ← MST and AIM (differentially private)
│   ├── tvae_method.py            ← TVAE (SDV)
│   ├── ctgan_method.py           ← CTGAN (SDV)
│   ├── arf_method.py             ← ARF (SynthCity)
│   ├── tabddpm_method.py         ← TabDDPM
│   ├── r_methods.py              ← Synthpop, RankSwap, CellSuppression (via R)
│   └── test_sdg.py               ← Smoke tests for all SDG methods
│
├── experiment_scripts/           ← Paper-specific experiment drivers
│   ├── run_production_sweep.py   ← Main sweep: all samples × SDG × attacks (parallelized)
│   ├── run_linear_sweep.py       ← LinearReconstruction comparison
│   ├── run_memorization_sweep.py ← Memorization test: train vs. holdout targets
│   ├── run_mst_epsilon_rf_nb_sweep.py  ← MST/AIM epsilon sweep
│   ├── run_chaining_analysis.py  ← Chaining order strategy analysis
│   ├── run_ensembling_heatmap.py ← All-pairs ensembling heatmap
│   ├── compare_mia_ra.py         ← RA vs. MIA comparison (SynthDistance, NNDR, RA-as-MIA)
│   ├── analyze_ra_subgroups.py   ← Disparate impact by subgroup
│   ├── evaluate_synth_quality.py ← Fidelity + utility quality evaluation (multiprocessed)
│   ├── qi_analysis/              ← QI composition and size sweep
│   ├── wandb_to_latex.py         ← WandB results → LaTeX tables (main results)
│   ├── wandb_to_latex_epsilon_sweep.py  ← ε-sweep LaTeX table
│   ├── wandb_to_latex_by_feature.py     ← Per-feature breakdown table
│   ├── linear_sweep_to_latex.py  ← LinearReconstruction LaTeX tables
│   ├── synth_quality_to_latex.py ← Synth quality LaTeX table
│   ├── results_db.py             ← SQLite results database utilities
│   ├── audit_and_verify.py       ← Audit WandB runs against DB for completeness
│   └── plot_ensembling_heatmap.py ← Heatmap visualization
│
├── configs/
│   └── example_cfg.yaml          ← Fully annotated example configuration
│
├── SOTA_attacks/                 ← Wrapper for LinearReconstruction (Annamalai et al. 2024)
│
├── NIST-CRC_leaderboardScripts/  ← Our NIST CRC competition submission scripts
│
├── incomplete_attacks/           ← Exploratory/unfinished attack variants (not used in paper)
└── maintenance_scripts/          ← Debug and integration test scripts
```

---

## Attack Taxonomy

Attacks are organized by how much joint structure they exploit, following the taxonomy in Figure 1 of the paper.

### Reference points (not attacks)

| Name | Description |
|---|---|
| Mode | Predict the most common value in the synthetic column |
| Random | Sample randomly from the synthetic column's distribution |
| Copy | Copy the synthetic row at the same index |

### Each feature in isolation

These attacks treat each hidden feature independently, training one model per feature on synthetic data (QI → hidden feature) and applying it to the target's QI.

| Name | Class | Notes |
|---|---|---|
| KNN | `ML_classifiers.py` | k-nearest neighbors (default k=1) |
| NaiveBayes | `ML_classifiers.py` | Gaussian/categorical NB |
| LogisticRegression | `ML_classifiers.py` | L2-regularized logistic regression |
| SVM | `ML_classifiers.py` | RBF kernel SVM; excluded for n≥10k (O(n²)) |
| RandomForest | `ML_classifiers.py` | 25 trees, max depth 25 |
| LightGBM | `ML_classifiers.py` | 100 estimators |
| MLP | `NN_classifier.py` | PyTorch MLP, [300] hidden dims |
| TabPFN | `tabpfn_attack.py` | Pre-trained transformer, single forward pass |
| LinearReconstruction | `SOTA_attacks/` | LP attack (Annamalai et al. 2024); binary features only, requires Gurobi |

### Each feature in isolation — continuous data

For continuous datasets (California Housing), classifiers are replaced by regressors. All are registered under `data_type: "continuous"`.

| Name | Class | Notes |
|---|---|---|
| Mean | `baselines_continuous.py` | Predict the synthetic column mean |
| Median | `baselines_continuous.py` | Predict the synthetic column median |
| KNN | `ML_regression.py` | k-nearest-neighbor regressor |
| RandomForest | `ML_regression.py` | Random forest regressor |
| LightGBM | `ML_regression.py` | Gradient-boosted regressor |
| LinearRegression | `ML_regression.py` | Polynomial (degree 2) linear regression |
| Ridge | `ML_regression.py` | Ridge regression |
| Lasso | `ML_regression.py` | Lasso regression |
| ElasticNet | `ML_regression.py` | Elastic net regression |
| SGDRegressor | `ML_regression.py` | Stochastic gradient descent regressor |
| MLP | `NN_regression.py` | PyTorch MLP regressor |

Scoring for continuous features uses normalized RMSE (lower is better), inverted so higher is still better, consistent with the categorical $R_{adv}$ direction.

### Feature-correlated: autoregressive

| Name | Description |
|---|---|
| Attention | Multi-head self-attention encoder predicting hidden features in parallel |
| AttentionAutoregressive | Same but predicts autoregressively |
| Chaining (enhancement) | Wraps any base attack to predict features sequentially, each prediction fed forward |

### Feature-correlated: row-wise message passing

| Name | Description |
|---|---|
| **MarginalRF** | **Our strongest attack.** Per-feature RF posteriors refined by belief propagation on an MI-weighted MST of hidden features. Local PMI tables from synth nearest-neighbors capture residual hidden-feature dependencies after conditioning on QI. Optional variants: `qi_in_graph` (QI as observed nodes), `entropy_weighted` (message damping by sender confidence). |

### Feature-correlated: joint generative conditioning

| Name | Description |
|---|---|
| JointMLP | Single neural network predicting all hidden features jointly |
| PartialMST | MST graphical model fit on synth, sampled conditional on QI |
| PartialMST (indep.) | Independent per-feature MST models |
| PartialMST (k=3) | Bounded-clique variant with up to k-1 QI cols per clique |
| PartialTabDDPM | Tabular diffusion model with QI-conditioned forward process |
| ConditionedRePaint | Same diffusion model; RePaint sampling (QI re-noised at each step) |
| TabDDPMWithMLP | Two-stage: diffusion hints → MLP stacker trained on synthetic pseudo-targets |

### Enhancements (composable)

| Enhancement | Description |
|---|---|
| Ensembling | Combine any set of attacks by voting, soft voting (probability-weighted), averaging, or median |
| Chaining | Predict hidden features sequentially in any order (default, correlation, mutual_info, random) |

Enhancements are composable: ensembling first, then chaining.

---

## SDG Methods

| Method | Type | Key parameters | Notes |
|---|---|---|---|
| MST | DP (marginal) | ε ∈ {0.1, 1, 10, 100, 1000} | SmartNoise. Use `bin_continuous_as_ordinal=True` for wide datasets with skewed continuous cols |
| AIM | DP (marginal) | ε ∈ {1, 10, 100} | SmartNoise. Infeasible for 50+ columns |
| TVAE | Deep generative | — | SDV library, GPU recommended |
| CTGAN | Deep generative | — | SDV library, GPU recommended |
| ARF | Deep generative | — | SynthCity, GPU recommended |
| TabDDPM | Diffusion | `iterations`, `num_timesteps` | GPU recommended |
| Synthpop | Sequential regression | — | R `synthpop` package; CPU |
| RankSwap | De-identification | `swap_features` (continuous only) | R `sdcMicro`; CPU |
| CellSuppression | De-identification | `key_vars` (categorical QIs, ≥1) | R `sdcMicro`; CPU |

---

## Configuration Reference

Experiments are configured via YAML files. See `configs/example_cfg.yaml` for a fully annotated example. Key fields:

```yaml
wandb:
  project: "my-project"
  group: "main-sweep"

dataset:
  name: "adult"         # dataset key (matches QI definitions in get_data.py)
  size: 10000           # training sample size
  dir: "/path/to/data/adult/size_10000/sample_00"

QI: "QI1"              # quasi-identifier variant (see get_data.py for available options)

sdg_method: "MST"
sdg_params:
  epsilon: 1.0

attack_method: "MarginalRF"
data_type: "categorical"   # "categorical", "continuous", or "agnostic"

memorization_test:
  enabled: true
  holdout_dir: "/path/to/data/adult/size_10000/sample_01"

attack_params:
  chaining:
    enabled: false
  ensembling:
    enabled: false
  MarginalRF:
    num_estimators: 25
    max_depth: 25
    knn_k: 100
    graph_type: "mst"
    col_correction_alpha: 0.5
```

### Data type conventions

- `"categorical"`: classification-based attacks (RF, LightGBM, MLP, MarginalRF, PartialMST, …)
- `"continuous"`: regression-based attacks (RF regressor, MLP regressor, …)
- `"agnostic"`: works for both (PartialTabDDPM, ConditionedRePaint, RePaint)

### Adding a new attack

1. Implement `my_attack(cfg, synth, targets, qi, hidden_features) → (reconstructed_df, probas, classes)` in `attacks/`
2. Register in `attacks/__init__.py` ATTACK_REGISTRY under the appropriate data_type key
3. Add default parameters to `attack_defaults.py`

---

## Experiment Tracking

All experiments log to [Weights & Biases](https://wandb.ai). Set your project/group in the config YAML or `run_production_sweep.py`.

To run without internet access:
```bash
WANDB_MODE=offline python master_experiment_script.py ...
```

Key WandB metrics logged per run:
- `RA_{feature}` — per-feature reconstruction accuracy
- `RA_mean` — mean across all hidden features (primary metric)
- `RA_train_*` / `RA_nontraining_*` / `RA_delta_*` — memorization test split
- `MIA_auc`, `MIA_advantage`, `MIA_tpr_at_fpr001` — MIA metrics (when `--mode mia`)

---

## Citation

If you use this code or the MarginalRF / PartialMST / PartialTabDDPM / ConditionedRePaint / JointMLP / Attention attacks in your research, please cite:

```bibtex
@article{golob2027sok,
  title     = {{SoK}: Reconstruction Attacks on Synthetic Tabular Data
               (Insights from Winning the {NIST CRC})},
  author    = {Golob, Steven and Pentyala, Sikha and De Cock, Martine},
  journal   = {Proceedings on Privacy Enhancing Technologies},
  year      = {2027},
}
```

This work also uses the LinearReconstruction attack from:

```bibtex
@inproceedings{annamalai2024linear,
  title     = {A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data},
  author    = {Annamalai, M.S.M.S. and Gadotti, A. and Rocher, L.},
  booktitle = {33rd USENIX Security Symposium},
  year      = {2024}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The `SOTA_attacks/linear_reconstruction.py` is adapted from [synthetic-society/recon-synth](https://github.com/synthetic-society/recon-synth) (our fork: [steveng9/recon-synth](https://github.com/steveng9/recon-synth)); see that repository for its license.
