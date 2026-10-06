# SoK: Reconstruction Attacks on Synthetic Tabular Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22701232.svg)](https://doi.org/10.5281/zenodo.22701232)

This is the code and data for

> Steven Golob, Sikha Pentyala, Martine De Cock.
> *SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC).*
> Proceedings on Privacy Enhancing Technologies (PoPETs), 2027.

A reconstruction attack (also called attribute inference) starts from a synthetic
dataset and a few things known about a target person, such as age, sex and race
(the quasi-identifiers, or QIs), and tries to recover that person's other
attributes. This repository measures how well such attacks work. It holds
fourteen attacks, thirteen synthetic-data generators, the scoring code, and the
50,826 scored attack runs behind the paper. The same approach placed first among
the red teams in the 2025 NIST Privacy Collaborative Research Cycle (CRC).

You can use it three ways:

- run an attack against a synthetic release, your own or one of ours
  ([Quick start](#quick-start));
- rebuild the paper's tables and figures
  ([Reproducing the paper](#reproducing-the-paper));
- add your own attack, generator or dataset, or query our results directly
  ([Extending](#extending), [The results database](#the-results-database)).

[`ARTIFACT-APPENDIX.md`](ARTIFACT-APPENDIX.md) is the PoPETs artifact appendix. It
lists the requirements and maps each claim in the paper to an experiment.

## Quick start

You need Docker. Everything else is in the image.

```bash
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
cd Reconstruction
docker build -f docker/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t recon-artifact:latest .
docker run --rm -it -v "${PWD}":/workspace -w /workspace recon-artifact:latest bash
```

The build takes 10 to 15 minutes. On Windows, run these in a WSL2 shell, or in
PowerShell without the two `--build-arg` flags. On an Apple Silicon Mac they work
unchanged, under emulation, so everything runs more slowly.

You are now in a shell inside the container, at the repository root. Check the
installation:

```bash
./test.sh
```

It takes about two minutes and ends with `18 passed, 0 failed`.

Now run an attack:

```bash
python master_experiment_script.py --n_runs 1
```

This runs the paper's strongest attack, CoBP-RA, against a small bundled dataset
(`data/dummy/`) that was synthesized with the MST generator at ε = 10. The attack
knows four attributes of each person (age band, region, sex, education) and
predicts the other six. Near the end of the output are the score for each of
the six and their mean:

```
[47.5 26.3 51.1 48.9 40.8 26.9]
ave: 40.25
```

The score is the reconstruction advantage $R_{adv}$ described under
[Scoring](#scoring). For comparison, always guessing the most common value scores
33.88 here. To see that, open `configs/demo_dummy.yaml`, change `attack_method`
from `"CoBP-RA"` to `"Mode"`, and run the command again. (CoBP-RA's number moves
by a few tenths between runs.)

### The same thing on real data

The bundled dataset is machine-generated. To attack a release of the Adult census
dataset instead, download Adult, synthesize it with MST at ε = 1, and run the
attack. This takes about two minutes on CPU.

```bash
python experiment_scripts/fetch_dataset.py adult
SDG_DATASET=adult SDG_SAMPLE_SIZE=10000 SDG_NUM_SAMPLES=1 SDG_JOBS=MST:1 SDG_BIN_CONTINUOUS=1 python sdg/generate_synth.py sdg
python master_experiment_script.py --config configs/demo_adult.yaml --n_runs 1
```

The first command downloads Adult and cuts it into disjoint training samples of
10,000 rows. The second trains MST on the first sample and writes the synthetic
data to `data/adult/size_10000/sample_00/MST_eps1/synth.csv`
(`SDG_BIN_CONTINUOUS=1` bins Adult's continuous columns first, as the paper
does). The third runs CoBP-RA against it, knowing six attributes and predicting
nine, and scores about 24, against 10.4 for `Mode`.

`configs/demo_adult.yaml` is short and commented. Change `attack_method` to try
another attack, `sdg_params.epsilon` to attack a different release (generate it
first by changing `SDG_JOBS`, for example `SDG_JOBS=MST:10,AIM:1`), or `QI` to
change what the attacker knows.

## Reproducing the paper

Inside the container:

```bash
python reproduce.py
```

This rebuilds the paper's tables and figures from the scored runs in
`experiment_scripts/results.db` and writes them to `expected_output/tables/`. It
takes about a minute. `python reproduce.py --list` shows which file is which
table.

The runs themselves took CPU-weeks, so `reproduce.py` does not repeat them.
[`ARTIFACT-APPENDIX.md`](ARTIFACT-APPENDIX.md) describes two reduced experiments
that do rerun attacks from raw data, one for the main attack-by-generator table
and one for the privacy-budget sweep.

To rerun a full experiment, these are the scripts that produced each part of the
paper. Each needs the dataset in place (see [Datasets](#datasets)), and
[`experiment_scripts/README.md`](experiment_scripts/README.md) describes every
script in that directory.

| Part of the paper | Script in `experiment_scripts/` |
|---|---|
| Attack × generator tables | `run_production_sweep.py` |
| Privacy-budget (ε) sweep | `generate_new_dp_sweep.py`, then `run_new_dp_epsilon_sweep.py` |
| Memorization test (training vs. held-out targets) | `run_memorization_sweep.py` |
| Membership inference comparison | `compare_mia_ra.py` |
| QI size and composition | `qi_analysis/run_qi_sweep.py` |
| Disparate impact by subgroup | `analyze_ra_subgroups.py`, `run_per_attack_disparity.py` |
| Ensembling and chaining | `run_ensembling_heatmap.py`, `run_chaining_analysis.py` |
| LinearReconstruction comparison | `run_linear_sweep.py` |
| Synthetic-data quality metrics | `evaluate_synth_quality.py` |

## How it works

An experiment has four steps.

```
dataset  →  disjoint training samples  →  synthetic data  →  attack  →  score
```

`sdg/generate_synth.py` does the first two arrows: it cuts a dataset into
training samples and runs a generator on each. `master_experiment_script.py`
does the last two for one configuration: it loads a synthetic dataset, takes the
training records as targets, hides every attribute outside the QI set, runs the
attack, and scores the predictions. `experiment_scripts/run_production_sweep.py`
repeats that over many samples, generators and attacks in parallel.

### Scoring

The score is rarity-weighted reconstruction advantage, $R_{adv}$, on a 0 to 100
scale. A correct guess of a rare value earns more than a correct guess of a
common one. The weights are set so that always guessing the most common value
of an attribute with $C$ distinct values scores exactly $100/C$, whatever the
distribution, and a perfect attack scores 100. NIST used the same metric in the
CRC. Continuous attributes are scored by normalized RMSE instead. `scoring.py`
implements both.

### Where data lives

All data sits under one root, `data/` by default. Set `RECON_DATA_ROOT` to keep
it somewhere else. `python paths.py` prints the locations in use.

```
<data root>/<dataset>/
    full_data.csv                  the source dataset
    meta.json                      which columns are categorical, continuous, ordinal
    size_<N>/sample_<KK>/
        train.csv                  one training sample of N rows
        <GENERATOR>_eps<E>/synth.csv   one synthetic release of that sample
```

Generators without a privacy budget use the bare name, for example
`TVAE/synth.csv`. `data/dummy/` is a complete example of this layout.

### Configuration

One YAML file describes one experiment. This is `configs/demo_adult.yaml`:

```yaml
dataset:
  name: "adult"                      # selects the QI definitions in get_data.py
  size: 10000
  dir: "adult/size_10000/sample_00"  # relative to the data root

QI: "QI1"                            # which attributes the attacker knows

sdg_method: "MST"
sdg_params:
  epsilon: 1                         # together these select MST_eps1/synth.csv

attack_method: "CoBP-RA"
data_type: "categorical"             # "categorical", "continuous" or "agnostic"

attack_params:
  ensembling:
    enabled: false
  chaining:
    enabled: false
```

`data_type` says which family the attack belongs to. Classifiers such as
RandomForest and CoBP-RA are `categorical`, regressors are `continuous`, and the
CondMST and diffusion attacks are `agnostic`. `attacks/__init__.py` lists every
attack under its family.

To change an attack's hyperparameters, add a block named after the attack under
`attack_params`. Anything you leave out takes its default from
`attack_defaults.py`.

```yaml
attack_params:
  RandomForest:
    num_estimators: 100
    max_depth: 15
```

Two wrappers combine attacks, and each is switched on in `attack_params`.
Ensembling runs several attacks and merges their predictions (`aggregation` is
`voting`, `soft_voting`, `averaging` or `median`). Chaining predicts the hidden
attributes one at a time and feeds each prediction to the next (`order_strategy`
is `default`, `correlation`, `mutual_info`, `random` or `manual`).
`configs/example_cfg.yaml` shows every option, including the memorization test,
which scores the attack separately on training records and on records the
generator never saw.

Runs are logged with [Weights & Biases](https://wandb.ai). The Docker images set
`WANDB_MODE=offline`, so no account is needed and nothing is uploaded. Outside
Docker, set that variable yourself, or log in to W&B to have runs uploaded to
the project named in the config.

## Attacks

The paper sorts attacks by how much of the relationship between attributes they
use. The names below are the ones to put in `attack_method`.

**Reference points.** These are not attacks. They show what a score means.

| Name | What it predicts |
|---|---|
| `Mode` | the most common value in the synthetic column |
| `Random` | a random draw from the synthetic column |
| `MeasureDeid` (Copy in the paper) | the value in the synthetic row at the target's own index |

**Each attribute on its own.** One model per hidden attribute, trained on the
synthetic data to predict it from the QIs.

| Name | Model |
|---|---|
| `KNN` | nearest neighbour (k = 1) |
| `NaiveBayes` | naive Bayes |
| `LogisticRegression` | logistic regression |
| `SVM` | support vector machine |
| `RandomForest` | random forest (25 trees) |
| `LightGBM` | gradient-boosted trees (100 rounds) |
| `MLP` | neural network, one hidden layer of 300 units |
| `TabPFN` | pre-trained tabular transformer |
| `LinearReconstruction` | linear program of Annamalai et al. (2024); needs Gurobi |

For continuous data, `KNN`, `RandomForest`, `LightGBM`, `MLP` and `SVM` select
regressors, and `Mean` and `Median` are the reference points.

**Attributes together.** These use the dependence between hidden attributes.
All six were introduced in the paper.

| Name | Idea |
|---|---|
| `CoBP-RA` | random-forest predictions for each attribute, reconciled by belief propagation over a graph of the hidden attributes. The strongest attack in the paper. `CoBP-RA_graphQI_entropyBP` is a variant that adds the QIs to the graph and weights each message by its sender's confidence. |
| `ARFFormerAutoregressive` | transformer that predicts the hidden attributes one after another. This is the paper's ARFFormer; `ARFFormer` is an earlier version that predicts them in parallel. |
| `MultiHeadMLP` | one network with an output head per hidden attribute |
| `CondMST` | fits MST's graphical model to the synthetic data and samples the hidden attributes given the QIs. `CondMSTBounded` and `CondMSTIndependent` are variants. |
| `CondDDPM` | tabular diffusion model that denoises the hidden attributes with the QIs held fixed |
| `CondRePaint` | the same model with RePaint sampling |

## Generators

`sdg/generate_synth.py` runs any of these. Choose them with `SDG_JOBS`, writing
`NAME:EPSILON` for the differentially private (DP) ones, for example
`SDG_JOBS=MST:1,PrivBayes:10,TVAE`.

| Name | Kind | Runs in |
|---|---|---|
| `MST`, `AIM` | DP, marginals and a graphical model (SmartNoise) | light image |
| `PrivBayes` | DP, Bayesian network | light image |
| `MWEMPGM` | DP, marginals and a graphical model | light image |
| `PrivSyn` | DP, marginals | light image |
| `PrivateGSD` | DP, genetic algorithm; slow above about 1,000 rows | light image |
| `TabDDPM` | diffusion; a GPU helps | light image |
| `TVAE`, `CTGAN` | deep generative (SDV) | full image |
| `ARF` | adversarial random forest (SynthCity) | full image |
| `Synthpop` | sequential regression (R) | full image |
| `RankSwap`, `CellSuppression` | de-identification (R `sdcMicro`) | full image |

The light image is the one built in the quick start. The full image adds the
generators that need SDV, SynthCity or R:

```bash
docker build -f docker/Dockerfile.full --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t recon-artifact:full .
```

It is about 23 GB. Use it the same way as the light one.

## Datasets

No dataset is redistributed here except the machine-generated `data/dummy/`.

| Dataset | Name in configs | Rows | How to get it |
|---|---|---|---|
| Adult (census income) | `adult` | 47,621 | `python experiment_scripts/fetch_dataset.py adult` |
| CDC Diabetes Health Indicators | `cdc_diabetes` | 253,680 | `python experiment_scripts/fetch_dataset.py cdc_diabetes` |
| California Housing | `california` | 20,640 | `sklearn.datasets.fetch_california_housing(as_frame=True).frame`, saved as `california/full_data.csv` |
| NIST Arizona (1940 census) | `nist_arizona_25feat` | 293,999 | free registration at [IPUMS USA](https://usa.ipums.org/) |
| NIST Survey of Business Owners | `nist_sbo` | 123,892 | from NIST on request, as part of the Privacy CRC |

`fetch_dataset.py` downloads the data, writes `meta.json`, and cuts the training
samples. For the other three, put `full_data.csv` and a `meta.json` in the
dataset's directory and cut the samples yourself:

```bash
SDG_DATASET=california SDG_SAMPLE_SIZE=1000 SDG_NUM_SAMPLES=5 python sdg/generate_synth.py sample
```

The QI sets for each dataset are in the `QIs` dictionary in `get_data.py`. `QI1`
is the default set, and the others (`QI_large`, `QI_behavioral`, …) vary how much
the attacker knows.

## Extending

**Add an attack.** An attack is one function:

```python
def my_attack(cfg, synth, targets, qi, hidden_features):
    ...
    return reconstructed_df, probas, classes
```

`synth` is the synthetic data, `targets` holds the target records, `qi` and
`hidden_features` are lists of column names, and `reconstructed_df` is `targets`
with the hidden columns filled in. Register the function in `ATTACK_REGISTRY` in
`attacks/__init__.py` and give it defaults in `attack_defaults.py`. It then works
in every sweep script and with both wrappers.

`examples/add_your_own_attack.py` does all of this in one file that you can run
and copy:

```bash
python examples/add_your_own_attack.py
```

It defines a small attack, registers it, and scores it next to `Mode` and CoBP-RA
on the bundled dataset in about 30 seconds.

**Add a generator.** Write `generate(train_df, meta, **config)`, returning the
synthetic DataFrame, and register it in `SDG_REGISTRY` in `sdg/__init__.py`.

**Add a dataset.** Put `full_data.csv` and `meta.json` in a new directory under
the data root, and add its QI sets to `QIs` and the matching hidden attributes
to `minus_QIs` in `get_data.py`. Then cut samples and generate synthetic data
with `sdg/generate_synth.py`, as for Adult above, with `SDG_DATASET` set to the
directory's name. `data/dummy/` is a dataset set up this way, and
`data/dummy/make_dummy_data.py` is the script that built it.

**Membership inference.** `master_experiment_script.py --mode mia` runs a
membership inference attack instead. The config then also names a `mia_method`
and a held-out sample (`configs/example_cfg.yaml` shows both), and
`experiment_scripts/compare_mia_ra.py` implements the paper's comparison of
membership inference with reconstruction.

## The results database

`experiment_scripts/results.db` is a SQLite file with one row per scored run:
dataset, training sample, QI set, generator, attack, and score. You can use it
without installing anything here.

```bash
python examples/query_results_db.py
```

That runs five example queries using only the Python standard library.
[`DATABASE.md`](DATABASE.md) documents the tables and the naming conventions,
some of which are easy to trip over (several attacks are stored under earlier
names).

## Repository layout

```
master_experiment_script.py   run one experiment
reproduce.py                  rebuild the paper's tables and figures
test.sh                       check the installation
paths.py                      where data and results are looked up
get_data.py                   data loading and QI definitions
scoring.py                    the R_adv and NRMSE scores
attack_defaults.py            default hyperparameters for every attack

attacks/                      the attacks; __init__.py holds the registry
enhancements/                 the chaining and ensembling wrappers
sdg/                          the generators and generate_synth.py
configs/                      demo_dummy.yaml, demo_adult.yaml, example_cfg.yaml
examples/                     add_your_own_attack.py, query_results_db.py
data/dummy/                   the bundled example dataset
docker/                       the two Dockerfiles and pinned requirements

experiment_scripts/           the scripts behind each experiment in the paper
    results.db                the scored runs
    README.md                 what each script does
expected_output/tables/       reference copies of the rebuilt tables and figures
external/                     two dependencies, as git submodules

ARTIFACT-APPENDIX.md          requirements, claims and experiments
DATABASE.md                   guide to results.db
TABLE-COVERAGE.md             which paper tables and figures reproduce.py rebuilds
```

## Installing without Docker

The Docker images are the tested route. To install natively on Linux, use
Python 3.9 and the same pinned requirements the light image uses:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r docker/requirements-attacks.txt
export WANDB_MODE=offline
./test.sh
```

This covers every attack and the seven generators of the light image. The other
generators need a second environment (Python 3.10,
`docker/requirements-sdg.txt`) and, for Synthpop, RankSwap and CellSuppression,
R with the `synthpop` and `sdcMicro` packages; `docker/Dockerfile.full` is the
exact recipe.

Two attacks have extra requirements. The diffusion attacks (CondDDPM,
CondRePaint) and LinearReconstruction use the two submodules under `external/`,
so clone with `--recurse-submodules` or run
`git submodule update --init --recursive`. LinearReconstruction also needs
[Gurobi](https://www.gurobi.com/academia/), which is free for academic use.
After activating a licence, `pip install gurobipy==11.0.3`.

## Citation

```bibtex
@article{golob2027sok,
  title   = {{SoK}: Reconstruction Attacks on Synthetic Tabular Data
             (Insights from Winning the {NIST CRC})},
  author  = {Golob, Steven and Pentyala, Sikha and De Cock, Martine},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2027},
}
```

To cite the software itself, use the Zenodo DOI
[10.5281/zenodo.22701232](https://doi.org/10.5281/zenodo.22701232), which always
points to the latest archived release. [`CITATION.cff`](CITATION.cff) holds the
same information in machine-readable form.

## License

MIT; see [`LICENSE`](LICENSE). Third-party components are listed in
[`LICENSES/THIRD-PARTY-NOTICES.md`](LICENSES/THIRD-PARTY-NOTICES.md). Please read
it before reusing `SOTA_attacks/linear_reconstruction.py`, which is adapted from
the reference implementation of Annamalai et al. (2024), or the PrivateGSD
wrapper.
