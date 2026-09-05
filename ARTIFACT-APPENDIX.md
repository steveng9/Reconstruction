# Artifact Appendix

Paper title: **SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC)**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

This artifact is the complete experiment framework behind:

> Steven Golob, Sikha Pentyala, Martine De Cock.
> *SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC).*
> Proceedings on Privacy Enhancing Technologies (PoPETs), 2027.

The paper systematises **reconstruction attacks** (equivalently, attribute
inference) on de-identified and synthetic tabular data: given a synthetic
release and a target record's quasi-identifiers (QIs), an attack tries to
recover that record's hidden attribute values.

The artifact contains everything used to produce the paper's results:

- implementations of all attacks in the taxonomy, including the six new ones
  introduced by the paper (**CoBP-RA**, **CondMST**, **CondDDPM**,
  **CondRePaint**, **ARFFormer**, **MultiHeadMLP**);
- wrappers for the two composable enhancements (chaining, ensembling);
- the synthetic-data-generation layer covering all nine SDG methods evaluated,
  plus the four extra DP mechanisms used for the ε sweep;
- the scoring code for rarity-weighted reconstruction advantage ($R_{adv}$),
  which was also the official NIST CRC metric;
- the driver scripts for every experiment in the paper;
- **`experiment_scripts/results.db`**, a 68 MB SQLite database holding all
  **49,126** scored runs behind the paper, and
  `reproduce.py`, which regenerates the paper's tables and figures from it,
  driven by a manifest naming every table and figure in the paper.

### What `results.db` does and does not contain

`results.db` is the system of record for the paper's *reconstruction* results.
One row per (dataset, size, sample, QI, SDG method, attack, split), holding the
mean $R_{adv}$ plus the attack and SDG parameters; per-feature scores live in a
companion table.

| | |
|---|---|
| `runs` | 49,126 scored runs |
| `feature_scores` | 413,318 per-feature scores |
| `runs_superseded` | 17,108 runs invalidated by the 2026-08 encoding repair, retained for audit |
| Datasets | Adult (29,822), CDC Diabetes (14,378), California (2,267), NIST Arizona (1,428), NIST SBO (1,231) |
| SDG methods | 59 distinct configurations — the nine evaluated methods plus MST, AIM, PrivBayes, PrivSyn, MWEM-PGM and PrivateGSD across nine ε values |
| Attacks | 184 labels — every attack in the taxonomy, plus its ablation variants and the pairwise ensembles/chains |
| Splits | `standard`, and `train` / `nontraining` for the memorization test |

**Three things are deliberately outside it:**

1. **Membership-inference results.** The MIA comparison
   (`tab:mia_comparison`) is a different measurement with a different output
   shape, and was never migrated into the schema. It is committed separately as
   `experiment_scripts/mia_comparison_results.csv` (plus the two
   `mia_rebuttal_sweep_*.csv` files and the source logs in
   `experiment_scripts/raw_logs/`), and Experiment 1 regenerates its table from
   there.
2. **Synthetic-data quality metrics.** TVD, JSD, Wasserstein, propensity and
   TSTR/TRTR scores describe *datasets*, not attack runs. They are in
   `experiment_scripts/quality_results_merged.csv`, which is committed.
3. **The NIST CRC scoreboard** (`tab:nist_results`) is transcribed from NIST's
   official published results, not computed here.

Everything needed by Experiment 1 — the database plus four committed CSVs — is
in the repository. `experiment_scripts/README.md` maps every script to the paper
object it produces and states what each one needs to run.

### How experiments find their data

Nothing in the repository hardcodes a path. All roots resolve through
`paths.py`, and each is overridable by an environment variable:

| Variable | Default | Holds |
|---|---|---|
| `RECON_DATA_ROOT` | `<repo>/data` | datasets and generated synthetic data |
| `RECON_EXTERNAL_ROOT` | `<repo>/external` | the two submodules |
| `RECON_RESULTS_DB` | `<repo>/experiment_scripts/results.db` | scored results |
| `RECON_TABLE_OUT` | `<repo>/expected_output/tables` | regenerated tables |

Under `RECON_DATA_ROOT`, each dataset follows one layout:

```
<dataset>/
  full_data.csv                     # the source dataset
  meta.json                         # column types
  size_<N>/sample_<KK>/
      train.csv                     # disjoint training sample
      holdout.csv                   # non-training targets (memorization test)
      <SDG>_eps<E>/synth.csv        # one synthetic release per generator
```

`data/dummy/` is a complete, committed instance of this layout, so the pipeline
runs on a fresh clone with no downloads and no environment variables. To add a
real dataset, create `<dataset>/full_data.csv` and run `sdg/generate_synth.py`;
Experiment 3 walks through this for Adult.

On the authors' machine `RECON_DATA_ROOT` points at a separate ~200 GB tree
outside the repository (`data/` is a gitignored symlink to it), which is why the
generated synthetic data is not shipped: it is far past what a repository can
carry. Nothing about that arrangement is required — it is only what the
environment variable is for.

### Security/Privacy Issues and Ethical Concerns

**No risk to the reviewer's machine.** Nothing here disables a security
mechanism, runs an exploit, or requires elevated privileges. The code trains
ordinary ML models and solves linear programs on tabular data. It makes no
network connections at run time except optional Weights & Biases logging, which
is disabled by default (`WANDB_MODE=offline` is set in both Docker images).

**This is attack code, and that is the point.** The repository implements
privacy attacks against synthetic-data releases. It is published so that
defenders and SDG authors can measure the exposure of their own systems, which
is the same purpose it served in the NIST Collaborative Research Cycle. The
attacks operate on datasets that are already public or already released for
research, and target records within them; they do not attack a live service.

**No human-subjects component.** The paper reports no user study, so no IRB
process, consent flow, or participant compensation applies. No personal data is
redistributed by this artifact: `data/dummy/` is entirely machine-generated by
`data/dummy/make_dummy_data.py` and describes no real person, and every real
dataset must be obtained by the reviewer from its own source (see below).
`results.db` contains only aggregate per-configuration scores — no record-level
data, no reconstructed values.

## Basic Requirements

### Hardware Requirements

**To run the artifact (Functional, and the reduced-scale Reproduced path): no
special hardware.** A laptop with 4 CPU cores and 8 GB RAM is enough for
`./test.sh`, for attacks on the dummy dataset, and for regenerating every table
and figure in the paper from `results.db`. No GPU is required.

**Hardware the paper's full sweeps were run on**, for reference: a single
workstation with an AMD Ryzen Threadripper PRO 5965WX (24 physical cores, 48
threads) and 125 GB RAM, no GPU used for the reported attack results. Sweeps
were parallelised across 8–12 worker processes with `OMP_NUM_THREADS=1` per
worker; thread oversubscription was the dominant performance problem and the
scripts pin it deliberately.

Reproducing the *complete* 49,126-run sweep from scratch needs on the order of
several CPU-weeks, which is why the scored results ship in `results.db` and the
reduced-scale experiments below are provided instead.

### Software Requirements

- **OS**: developed and run on Ubuntu 22.04 (kernel 5.15). The Docker images are
  Debian-based and the artifact is not OS-specific; any Linux, macOS, or Windows
  host that can run Docker will work.
- **Container runtime**: Docker Engine 28.0 or later. Everything else is supplied
  by the images. The pinned dependency set has been verified to install and run
  end to end in a clean Python 3.9.23 virtual environment (`./test.sh` passes
  10/10); the container build itself uses the same pinned requirements.
- **Language**: Python 3.9.23 for the attack/analysis environment; Python 3.10.19
  for the synthetic-data-generation environment. The two are separate because
  SDV/SynthCity/SmartNoise require numpy and torch versions that conflict with
  the attack stack.
- **R** ≥ 4.2 with `synthpop` and `sdcMicro`, needed only for the Synthpop,
  RankSwap and CellSuppression generators. Included in the full image.
- **Python packages**: pinned exactly in `docker/requirements-attacks.txt` and
  `docker/requirements-sdg.txt`. Both include VCS dependencies pinned to specific
  commits (`private-pgm` at `01f02f17`, `genetic_sd` at `f6150d7`).
- **Proprietary software**: the LinearReconstruction attack — and only that
  attack — needs **Gurobi** with a (free) academic licence. It is deliberately
  not installed in the images. See *Limitations*.
- **ML models**: no pre-trained model is required. TabPFN downloads its
  pre-trained checkpoint on first use; every other attack trains from scratch on
  the synthetic data it is given.
- **Datasets**: none are redistributed. Three of the five are freely scriptable
  (Adult, CDC Diabetes, California Housing); two are access-restricted (NIST
  Arizona needs an IPUMS registration, NIST SBO is available from NIST on
  request). `data/dummy/` is shipped so the whole pipeline can be exercised with
  no downloads at all.

### Estimated Time and Storage Consumption

| Task | Human time | Compute time | Disk |
|---|---|---|---|
| Clone (with submodules) + build the light Docker image | 5 min | 10–15 min | ~3 GB |
| `./test.sh` (full smoke test) | 1 min | ~2 min | negligible |
| Experiment 1 — regenerate all paper tables and figures | 2 min | ~1 min | <10 MB |
| Experiment 2 — attack vs. baseline on the dummy dataset | 2 min | ~1 min | negligible |
| Experiment 3 — reduced-scale attack × SDG grid (Adult) | 10 min | ~40 min | ~100 MB |
| Experiment 4 — reduced-scale ε sweep (Adult) | 10 min | ~50 min | ~200 MB |

The repository itself is ~93 MB tracked (dominated by the 68 MB `results.db`),
plus ~90 MB for the two submodules. Budget **under 5 GB total** including the
Docker image. Nothing here requires the 50 GB-class hosting discussed in the
PoPETs FAQ.

## Environment

### Accessibility

The artifact is a single public Git repository:

**<https://github.com/steveng9/Reconstruction>**

Everything is reachable from that one link. The two external dependency
repositories are wired in as git submodules pinned to exact commits:

| Submodule path | Repository | Pinned commit |
|---|---|---|
| `external/recon-synth` | <https://github.com/steveng9/recon-synth> | `2ec60ec` |
| `external/MIA_on_diffusion` | <https://github.com/steveng9/MIA_on_diffusion> | `1c4a21a` |

Licence: MIT (`LICENSE`), with third-party components and two upstream
repositories that carry no explicit licence documented candidly in
`LICENSES/THIRD-PARTY-NOTICES.md`.

### Set Up the Environment

```bash
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
cd Reconstruction
docker build -f docker/Dockerfile -t recon-artifact:latest .
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Then start a container with the repository mounted:

```bash
docker run --rm -it -v "${PWD}":/workspace -w /workspace recon-artifact:latest bash
```

Everything below runs inside that container. No paths need editing: all roots are
resolved by `paths.py` relative to the repository, and can be overridden with the
`RECON_DATA_ROOT`, `RECON_EXTERNAL_ROOT` and `RECON_RESULTS_DB` environment
variables. Run `python paths.py` at any time to print the resolved locations.

The light image runs **every attack in the registry**, including the
diffusion-based CondDDPM / CondRePaint / RePaint, and the six CPU DP generators.
Attacks whose optional dependencies are absent (LinearReconstruction, which needs
Gurobi) are disabled with an explanatory message rather than breaking the
registry, so the rest of the artifact is unaffected.

Only if you need the GPU-oriented generators (TVAE, CTGAN, ARF, TabDDPM) or the
R-based ones (Synthpop, RankSwap, CellSuppression) do you need the larger image:

```bash
docker build -f docker/Dockerfile.full -t recon-artifact:full .
```

### Testing the Environment

Inside the container:

```bash
./test.sh
```

This checks four things: that the environment imports and the paths and the
paper-object manifest resolve; that the dummy dataset is present; that the
attack pipeline runs end to end and a real attack **beats the mode baseline**;
and that the paper's tables regenerate from `results.db` byte-for-byte
identically to the committed reference copies. The list of tables it verifies
is read from the manifest, so it cannot fall behind as coverage grows.

Expected output (final lines):

```
=== 3/4  Attack pipeline (dummy dataset, MST epsilon=10) ===
  mode baseline : 33.88333333333333
  CoBP-RA       : 40.38333333333333
  PASS  CoBP-RA beats the mode baseline by >2 points

=== 4/4  Paper tables regenerate from results.db ===
  PASS  table1_ra_mean_adult.tex matches the committed reference
  PASS  table2_quality_overview.tex matches the committed reference
  PASS  table4_eps_sweep_compact.tex matches the committed reference
  PASS  table4_eps_sweep_full.tex matches the committed reference
  PASS  STATS_eps_curve.md matches the committed reference
  PASS  table6_mia_comparison.tex matches the committed reference
  PASS  table7_memorization.tex matches the committed reference
  PASS  STATS_memorization.md matches the committed reference
  PASS  table9_disparate_impact.tex matches the committed reference

============================================
  15 passed, 0 failed
============================================
```

The CoBP-RA number varies by a few tenths between runs (the attack's random
forests are not seeded identically across platforms); anything comfortably above
the 33.88 baseline is a pass.

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: The SDG method governs risk far more than the attack does

Across the attack × SDG grid, the spread in reconstruction advantage between SDG
methods is much larger than the spread between attacks on any fixed SDG method.
The independent variable is the SDG method (columns) and the attack (rows); the
dependent variable is mean $R_{adv}$. Reported in **Table 1**. Supported by
[Experiment 1](#experiment-1-regenerate-every-paper-table-and-figure) and, at
reduced scale, [Experiment 3](#experiment-3-reduced-scale-attack--sdg-grid).

#### Main Result 2: De-identification methods are the most exposed

Cell Suppression, RankSwap and Synthpop sit at the high-risk end of Table 1,
well above the DP generators. Same table, same experiments as Result 1.

#### Main Result 3: CoBP-RA is the strongest attack measured

The newly introduced CoBP-RA attack attains the highest mean $R_{adv}$ of all
attacks evaluated. Reported in **Table 1**; visible directly in the smoke test,
where CoBP-RA beats the mode baseline on the dummy dataset.

#### Main Result 4: DP reduces reconstruction up to ε≈10, above which risk flattens

Sweeping ε from 0.1 to 1000 across **six** DP mechanisms (MST, AIM, PrivBayes,
PrivSyn, MWEM-PGM, PrivateGSD), reconstruction advantage falls steadily as ε
decreases below about 10, and is close to flat above it. Independent variable:
ε; dependent variable: mean $R_{adv}$. Reported in **Table 4** and
`fig_eps_curves.pdf`, with paired significance tests in `STATS_eps_curve.md`.
Supported by [Experiment 1](#experiment-1-regenerate-every-paper-table-and-figure)
and, at reduced scale, [Experiment 4](#experiment-4-reduced-scale-epsilon-sweep).

Note the claim is *flattening*, not saturation: PrivateGSD is still climbing
slightly at ε=1000, and the stats report quantifies this.

#### Main Result 5: Most reconstruction is distributional, not memorization

The memorization test scores each attack on training targets and on held-out
non-training targets. The gap between them is small for most SDG methods, so
attack success mostly reflects population structure rather than memorized
records; where the gap is large it concentrates on atypical records. Reported in
**Table 7** (memorization) and **Table 9** (disparate impact). Supported by
[Experiment 1](#experiment-1-regenerate-every-paper-table-and-figure).

### Experiments

All commands run inside the light Docker container from the repository root.

#### Experiment 1: Regenerate every paper table and figure

- Time: 2 human-minutes + ~1 compute-minute
- Storage: <10 MB

This is the primary reproduction path, and it covers Main Results 1, 2, 4 and 5
at full scale, plus the membership-inference comparison. It reads the shipped `results.db` — all 49,126 scored runs — and
regenerates the paper's tables and figures. No number is copied from the
manuscript; everything is recomputed from the database.

```bash
python reproduce.py
```

`reproduce.py` is the single entry point for this artifact. It reads the manifest
in `experiment_scripts/paper_objects.py` — one row per table and figure in the
paper, naming its source, its generator and its output file — and builds
everything that can be built from committed data. To see the coverage, including
what is *not* yet covered and why:

```bash
python reproduce.py --list
```

Outputs land in `expected_output/tables/`:

| File | Paper object |
|---|---|
| `table1_ra_mean_adult.tex` | Table 1 — attack × SDG grid (Results 1, 2, 3) |
| `table2_quality_overview.tex` | Table 2 — synthetic data quality |
| `table4_eps_sweep_compact.tex`, `table4_eps_sweep_full.tex` | Table 4 — ε sweep (Result 4) |
| `table6_mia_comparison.tex` | Table 6 — MIA vs. RA-as-MIA |
| `table7_memorization.tex` | Table 7 — memorization test (Result 5) |
| `table9_disparate_impact.tex` | Table 9 — disparate impact (Result 5) |
| `fig_eps_curves.pdf`, `fig_eps_curves_perattack.pdf` | ε-curve figures |
| `STATS_eps_curve.md`, `STATS_memorization.md` | paired significance tests |

Because the reference copies of these files are committed, you can verify
reproduction exactly rather than by eye:

```bash
python reproduce.py --out /tmp/check
diff -r expected_output/tables /tmp/check   # .tex and .md files should be identical
```

`./test.sh` performs exactly this diff on the nine text outputs. The two PDFs and
their PNG companions will differ: matplotlib embeds a creation timestamp, and
glyph rasterisation varies with the freetype build. The numbers plotted in them
are verified through `STATS_eps_curve.md`, which is byte-compared.

Compare any output against the corresponding table in the paper PDF. This
comparison is exact, not within 5%.

Nine of the paper's 33 labelled tables and figures regenerate this way, covering
every table that carries a main claim. Nine more are not derived from this
repository at all (the NIST scoreboard, the hand-written dataset and QI-definition
tables, the two diagrams). The remaining 15 are supporting tables whose data is
committed but whose generators are not yet written; `python reproduce.py --list`
names each one, and `TODO-TABLE-COVERAGE.md` tracks the work.

#### Experiment 2: Attack versus baseline on the dummy dataset

- Time: 2 human-minutes + ~1 compute-minute

Demonstrates the attack pipeline end to end with no downloads, and shows a real
attack beating the mode baseline (Main Result 3, in miniature).

```bash
python master_experiment_script.py --n_runs 1     # uses configs/demo_dummy.yaml
```

Edit `attack_method` in `configs/demo_dummy.yaml` to try others (`Mode`,
`RandomForest`, `KNN`, `NaiveBayes`, `MultiHeadMLP`, `CoBP-RA`, …); the registry
is listed in `attacks/__init__.py`. On the dummy dataset with MST ε=10, expect
the mode baseline at **33.88** and CoBP-RA around **40**.

#### Experiment 3: Reduced-scale attack × SDG grid

- Time: 10 human-minutes + ~40 compute-minutes
- Storage: ~100 MB

Regenerates a small corner of Table 1 from raw data, to confirm the numbers in
`results.db` are actually produced by this code rather than merely stored. This
needs the Adult dataset, which downloads freely:

```bash
mkdir -p data/adult
python -c "from ucimlrepo import fetch_ucirepo; \
  fetch_ucirepo(id=2).data.original.to_csv('data/adult/full_data.csv', index=False)"

# carve training samples and generate synthetic data (CPU generators only)
python sdg/generate_synth.py sample
python sdg/generate_synth.py sdg

# run the attacks
python experiment_scripts/run_production_sweep.py --workers 4
```

Expect the ordering of attacks and of SDG methods to match Table 1, with
individual cells within a few points of the published values — five disjoint
samples are averaged in the paper, and a reduced run averages fewer.

#### Experiment 4: Reduced-scale epsilon sweep

- Time: 10 human-minutes + ~50 compute-minutes
- Storage: ~200 MB

Reproduces the shape of Main Result 4 on one dataset and a subset of mechanisms,
using the same 9-point ε grid as the paper (0.1, 0.3, 1, 3, 10, 30, 100, 300,
1000):

```bash
bash experiment_scripts/run_cdc_dp_sweep_pipeline.sh
```

Expect $R_{adv}$ to rise steeply from ε=0.1 to ε≈10 and to be close to flat
above it, matching the curve shape in `fig_eps_curves.pdf`. The absolute values
depend on how many trials complete; the *shape* is the claim.

## Limitations

Experiment 1 regenerates Tables 1, 2, 4, 6, 7 and 9 and the ε-curve figures —
the tables carrying all five main claims — exactly, from the shipped data. The
limitations below concern everything beyond that.

1. **Two datasets cannot be redistributed.** NIST Arizona requires a free IPUMS
   registration; NIST SBO is available from NIST on request as part of the
   Privacy CRC. Their rows in Tables 1 and 4 therefore cannot be regenerated from
   scratch without the reviewer obtaining that access. Adult, CDC Diabetes and
   California Housing are freely scriptable and cover the majority of results.
   `data/dummy/` stands in for format and pipeline purposes.

2. **LinearReconstruction needs Gurobi.** The LP solver requires a Gurobi licence
   (free for academics, but it must be requested and activated), so `gurobipy` is
   not in the Docker images. The LinearReconstruction rows of the comparison
   tables cannot be regenerated without it. Every other attack is unaffected.

3. **The full sweep is far too long for an evaluation window.** Reproducing all
   49,126 runs from raw data takes CPU-weeks. This is why the scored results ship
   in the database and Experiments 3 and 4 are reduced in scale, as the PoPETs
   guidance for long-running experiments recommends.

4. **GPU generators are not in the light image.** TVAE, CTGAN, ARF and TabDDPM run
   on CPU but slowly. They are in `docker/Dockerfile.full`. Their *attack* results
   are still fully reproducible from `results.db`.

5. **Small run-to-run variance.** Several attacks (random forests, MLPs, the
   diffusion samplers) are stochastic and not seeded bit-identically across
   platforms. Individual cells move by a few tenths of a point between runs;
   the paper's tables average five disjoint samples. Table *regeneration*
   (Experiment 1) is deterministic and exact.

6. **Not every supporting table regenerates yet.** Experiment 1 rebuilds the
   tables and figures behind the five main claims. The paper also contains
   supporting tables — per-dataset results for CDC and NIST SBO, the per-dataset
   quality profiles, the QI-composition analysis — whose underlying data is
   committed but which do not yet have a generator. `python reproduce.py --list`
   names every one of the paper's 33 labelled objects and reports which of them
   rebuild; `TODO-TABLE-COVERAGE.md` tracks the remaining work. No main claim
   depends on an object outside Experiment 1.

7. **`environment.yaml` is superseded.** The historical conda file in the
   repository root does not reflect the versions actually used. The authoritative,
   pinned dependency sets are `docker/requirements-attacks.txt` and
   `docker/requirements-sdg.txt`.

## Notes on Reusability

This is a framework, not a single experiment, and it was built to be extended.

**Adding an attack** takes three steps: implement
`my_attack(cfg, synth, targets, qi, hidden_features) -> (reconstructed_df, probas, classes)`
in `attacks/`, register it in `ATTACK_REGISTRY` in `attacks/__init__.py` under
the right `data_type`, and add defaults to `attack_defaults.py`. It is then
immediately usable by every sweep script, and composable with the chaining and
ensembling wrappers without further work.

**Adding an SDG method** is the same shape: implement
`generate(train_df, meta, **config) -> synthetic_df` and register it in
`SDG_REGISTRY` in `sdg/__init__.py`.

**Adding a dataset** needs a directory of the documented layout
(`meta.json` plus `size_{N}/sample_{XX}/train.csv`) and QI definitions in the
`QIs` / `minus_QIs` dicts in `get_data.py`. `data/dummy/make_dummy_data.py` is a
short, self-contained worked example of exactly this.

**Reusing the results rather than the code.** `results.db` is plain SQLite with a
`runs` table keyed by (dataset, size, sample, QI, SDG method, attack, split) and
a `feature_scores` table for per-feature breakdowns. It is a usable secondary
dataset in its own right for anyone studying reconstruction risk, independent of
this framework — 49,126 scored runs spanning 184 attack labels and 59 SDG
configurations.

**Beyond reconstruction.** The same harness runs membership inference
(`--mode mia`), and `experiment_scripts/compare_mia_ra.py` implements the paper's
reduction placing reconstruction and MIA on one comparable scale, which should
transfer to other attack settings.
