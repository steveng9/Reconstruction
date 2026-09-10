# Artifact Appendix

Paper title: **SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC)**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Quick start for a reviewer

The whole evaluation is four commands. Everything below this box is detail you
only need if one of them does not do what it says.

```bash
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
cd Reconstruction
docker build -f docker/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t recon-artifact:latest .
docker run --rm -it -v "${PWD}":/workspace -w /workspace recon-artifact:latest bash
```

Then, inside the container:

| Command | What it shows | Time |
|---|---|---|
| `./test.sh` | the environment works end to end | ~2 min |
| `python reproduce.py` | every paper table and figure, rebuilt from committed data | ~1 min |
| `python master_experiment_script.py --n_runs 1` | a real attack beating the baseline | ~1 min |

`./test.sh` ending in **`18 passed, 0 failed`** is the single check that the
artifact is functional. On Windows use a WSL2 shell, or PowerShell with the two
`--build-arg` flags dropped. No `--platform` flag is needed on any host,
including Apple Silicon.

Two things worth knowing before you compare a regenerated number against the
printed paper: the artifact was frozen six days before the camera-ready, and a
database repair in August 2026 means some printed cells are pre-repair. Both are
covered under *Experiment 1, continued*, and every difference we know of
is listed cell-by-cell in `TABLE-COVERAGE.md`.

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
| Splits | `standard`, and `train` / `nontraining` for the memorization test. A fourth value, `unknown` (1,569 rows), marks runs migrated from Weights & Biases logs that predate the split field; no table in the paper reads them. |

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

`data/dummy/` is a committed instance of this layout, so the pipeline runs on a
fresh clone with no downloads and no environment variables. It carries two
samples; `sample_00` is complete, and `sample_01` has no `holdout.csv` because
nothing shipped here runs the memorization test on it. To add a
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
threads), 125 GB RAM, and two NVIDIA RTX 6000 Ada GPUs (48 GB each). Sweeps
were parallelised across 8–12 worker processes with `OMP_NUM_THREADS=1` per
worker; thread oversubscription was the dominant performance problem and the
scripts pin it deliberately.

Where the GPUs were used, and why it does not change what you need: the
torch-based attacks select CUDA when it is available and fall back to CPU when
it is not, so on that workstation MultiHeadMLP, ARFFormer, the diffusion attacks
(CondDDPM, CondRePaint, RePaint) and the neural-network classifiers ran on GPU
(`experiment_scripts/run_joint_mlp_adult10k.py` round-robins its jobs across the
two cards explicitly). Everything else -- the classical attacks, CoBP-RA,
CondMST, TabPFN, all scoring, and the DP generators -- is CPU-only, and TabPFN is
pinned to CPU in `attack_defaults.py`.

None of that is a requirement for this artifact. The light image installs a
CPU-only build of torch, so the same attacks run there on CPU; a GPU changes how
long they take, not what they compute. Every number in the paper is already
scored in `results.db`, and Experiment 1 regenerates the tables from it without
touching torch at all.

Reproducing the *complete* 49,126-run sweep from scratch needs on the order of
several CPU-weeks, which is why the scored results ship in `results.db` and the
reduced-scale experiments below are provided instead.

### Software Requirements

- **OS**: developed and run on Ubuntu 22.04 (kernel 5.15). The Docker images are
  Debian-based and the artifact is not OS-specific; any Linux, macOS, or Windows
  host that can run Docker will work. Both images are built for `linux/amd64`,
  pinned in the Dockerfiles themselves, so an Apple Silicon Mac builds and runs
  them under Docker Desktop's emulation -- correctly, but slower than a native
  amd64 host. On Windows, either a WSL2 shell or PowerShell works; the *Set Up
  the Environment* section gives the exact command for each.
- **Container runtime**: Docker Engine. Developed and tested on 28.x, and
  independently built and run on 24.0.6; anything 24.0 or newer is expected to
  work, and nothing in either Dockerfile requires a newer feature. Everything
  else is supplied by the images. The pinned dependency set has also been
  verified to install and run end to end in a clean Python 3.9.23 virtual
  environment (`./test.sh` passes 18/18); the container build itself uses the
  same pinned requirements.
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
- **GPU libraries**: none. The light image installs `torch 1.13.1+cpu` from
  PyTorch's CPU index rather than the default PyPI build, which would pull ~900 MB
  of CUDA libraries (cuBLAS, cuDNN, nvrtc) that a CPU-only image can never use.
  The CPU kernels are the same ones the CUDA build runs on a machine without a
  GPU, so this changes the image size and nothing else.
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
| Experiment 4 — reduced-scale ε sweep (CDC Diabetes) | 10 min | ~50 min | ~200 MB |

The repository itself is ~93 MB tracked (dominated by the 68 MB `results.db`),
plus ~90 MB for the two submodules. Two numbers get quoted for a Docker image and
they differ: the light image is 1.1 GB of layers -- what a registry transfers --
but `docker images` reports about 2.9 GB, which is what it occupies unpacked on
disk.
The build cache adds several GB more while the build runs, so budget **about 15
GB of free disk** for the light image, or **about 40 GB** if you also build the
full one. Nothing here requires the 50 GB-class hosting discussed in the PoPETs
FAQ.

These figures are measured, not estimated: the light image was built and
`./test.sh` was run inside it on Ubuntu 22.04 (kernel 5.15), 24 physical cores,
passing all 18 checks. The build took ~6 minutes of step time there; 10-15
minutes is a fairer estimate on a laptop with a slower network, since most of
the time is spent downloading wheels.

## Environment

### Accessibility

The artifact is a single public Git repository:

**<https://github.com/steveng9/Reconstruction>**

Everything is reachable from that one link, and cloning it is the recommended
way to obtain the artifact.

It is additionally archived on Zenodo, which gives it a permanent DOI that
resolves even if the repository is ever moved or renamed:

**<https://doi.org/10.5281/zenodo.XXXXXXX>**

That is the *concept* DOI: it always resolves to the most recent archived
version, so it stays valid if the artifact is revised during evaluation. Each
individual release also receives its own version DOI, visible from that page, if
you would rather pin the exact snapshot you reviewed.

**One caveat if you take the Zenodo archive rather than cloning.**
Automatically generated source archives — Zenodo's and GitHub's alike — do not
include git submodule contents, so `external/recon-synth` and
`external/MIA_on_diffusion` arrive empty. We have verified what that costs:
`./test.sh` still passes 18/18, `reproduce.py` still rebuilds every paper table
and figure, and Experiments 1 and 2 are unaffected. Only the diffusion attacks
(CondDDPM, CondRePaint, RePaint) and LinearReconstruction need the submodules,
and they report a clear message naming the missing dependency rather than
failing obscurely. To get them, clone instead:

```bash
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
```

The two external dependency repositories are wired in as git submodules pinned
to exact commits:

| Submodule path | Repository | Pinned commit |
|---|---|---|
| `external/recon-synth` | <https://github.com/steveng9/recon-synth> | `2ec60ec` |
| `external/MIA_on_diffusion` | <https://github.com/steveng9/MIA_on_diffusion> | `1c4a21a` |

Licence: MIT (`LICENSE`), with third-party components and two upstream
repositories that carry no explicit licence documented candidly in
`LICENSES/THIRD-PARTY-NOTICES.md`. Citation metadata is in `CITATION.cff`
(machine-readable) and `README.md` (BibTeX).

### Set Up the Environment

**Linux and macOS** (including Apple Silicon), in a normal shell:

```bash
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
cd Reconstruction
docker build -f docker/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t recon-artifact:latest .
```

**Windows.** The simplest path is a WSL2 shell, where the commands above and
everywhere below work unchanged. If you would rather use PowerShell, drop the two
`--build-arg` flags -- Windows bind mounts do not carry Unix ownership, so the
default user inside the image is already correct -- and keep `${PWD}` as written
in the `docker run` commands:

```powershell
git clone --recurse-submodules https://github.com/steveng9/Reconstruction.git
cd Reconstruction
docker build -f docker/Dockerfile -t recon-artifact:latest .
```

The two `--build-arg` flags matter on Linux and macOS. The image runs as a
non-root user, and the repository is mounted from the host, so that user has to
be *you* or it cannot write the files the experiments produce. Without them the
user is 1000:1000, which is right only if your own ids happen to be 1000.
(Experiment 1 is unaffected either way -- it opens `results.db` read-only -- but
Experiments 3 and 4 write generated data and results, and would fail.)

You do not need a `--platform` flag on any host. Both Dockerfiles pin
`linux/amd64` in their own `FROM` lines, because two of the pinned dependencies
(`pac-synth` and `torch 1.13.1`) publish no arm64 wheel at all; left to the host
architecture, the build fails part-way through `pip` on an Apple Silicon Mac with
an error about a missing Rust compiler that says nothing about the real cause.
With the pin, Docker Desktop emulates amd64 there and the build completes -- more
slowly than on an amd64 machine, but with the same results.

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
docker build -f docker/Dockerfile.full --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t recon-artifact:full .
```

That image takes considerably longer to build, and `docker images` reports it as
23.1 GB (the sum of its layers, which is what a registry transfers, is ~16 GB).
It was verified by building and running it: R 4.5.3 with `synthpop` and
`sdcMicro` reachable through `rpy2`, all 13 SDG methods generating, and the
attack environment unaffected. The build itself now checks the second of those
-- it loads R through `rpy2` and `torch` in the attack environment before the
image is finished -- so a build that would produce a broken image fails instead.

Start it the same way, with `recon-artifact:full` in place of
`recon-artifact:latest`; every command in this document, `./test.sh` included,
works unchanged there (it passes 18/18 in both images). It holds two conda environments: `recon_` (Python 3.9)
for the attacks, scoring and table regeneration, and `sdg` (Python 3.10) for the
GPU-oriented and R generators. They are separate because they cannot be merged:
the DP generator stack pins `numpy < 2` and the SDV / SynthCity stack pins
`numpy >= 2`, and installing both in one environment upgrades numpy and breaks
`torch`. You do not have to keep track of which generator lives where --
`sdg/generate_synth.py` detects when the environment it was started in cannot
provide a generator and runs that job in the one that can:

```bash
conda run -n recon_ python sdg/generate_synth.py sdg     # any generator
```

Three notes on it. R comes from conda-forge rather than the distribution, because
Debian bookworm ships R 4.2.2 and `rpy2` 3.6.4 requires R >= 4.5. As a result the
R packages are conda-forge's (`synthpop` 1.9.2, `sdcMicro` 5.8.2), which are
newer than the versions used when the paper's synthetic data was generated; no
number in the paper comes from this image, but synthetic data regenerated with it
will not be bit-identical to ours. Both images are `linux/amd64`, pinned in the
`FROM` line rather than left to the host, so they build and run on an Apple
Silicon Mac too -- under emulation, and slowly. And none of the evaluated
experiments need this image -- Experiments 1 through 4 all run in the light one.

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
  PASS  table_ra_mean_cdc.tex matches the committed reference
  PASS  table_cdc_100k.tex matches the committed reference
  PASS  table_ra_mean_nist_sbo.tex matches the committed reference

============================================
  18 passed, 0 failed
============================================
```

The CoBP-RA number varies by a few tenths between runs (the attack's random
forests are not seeded identically across platforms); anything comfortably above
the 33.88 baseline is a pass.

### Checking what this environment actually supports

`test.sh` runs two attacks. To see which of the attacks and generators genuinely
work in the environment you have — rather than which merely import — run:

```bash
python experiment_scripts/verify_registries.py          # both registries
python experiment_scripts/verify_registries.py attacks  # attacks only
```

Every attack is run against the dummy dataset and every generator is asked for a
small synthetic frame, with training budgets cut to the minimum that still
exercises each code path. In the light image this takes about twenty minutes and
ends with **25 of the 28 attacks and 7 of the 12 generators** running (the rebuilt
full image runs the same 25 attacks). Whatever does not run reports an
ImportError naming the dependency it needs, which for the light image means:

* the three `LinearReconstruction` variants need a Gurobi licence (see
  *Limitations* for how to enable them if you have one);
* `TVAE`, `CTGAN` and `ARF` need SDV / SynthCity, and `Synthpop`, `RankSwap` and
  `CellSuppression` need R — all six are in the full image.

Two further things are reported as skipped rather than failed, because they are
facts about the dummy dataset rather than about the environment. The `Mean`
baseline averages the released column, which needs numbers, and the dummy
dataset spells its categories out as words; every dataset in the paper codes
them as integers, where `Mean` runs. `RankSwap` swaps values between adjacent
ranks and so needs at least one continuous column, which the dummy dataset does
not have. Thirteen attacks are continuous-data-only for the same reason, and the
script lists them instead of counting them as failures.

In the full image, run the generator half once in each environment:

```bash
conda run -n recon_ python experiment_scripts/verify_registries.py
conda run -n sdg    python experiment_scripts/verify_registries.py sdg
```

Each environment holds part of the set — `recon_` runs AIM, MST, MWEM-PGM,
PrivBayes, PrivSyn, Private-GSD and TabDDPM; `sdg` runs ARF, CTGAN, TVAE,
TabDDPM, Synthpop and CellSuppression — and between them every generator runs.
This split matters only to this script: `sdg/generate_synth.py` routes each job
to the environment that can run it, so real generation needs one command.

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: The SDG method governs risk far more than the attack does

Across the attack × SDG grid, the spread in reconstruction advantage between SDG
methods is much larger than the spread between attacks on any fixed SDG method.
The independent variable is the SDG method (columns) and the attack (rows); the
dependent variable is mean $R_{adv}$. Reported in the attack × SDG grid for
Adult (`tab:ra_mean_adult`). Supported by
[Experiment 1](#experiment-1-regenerate-every-paper-table-and-figure) and, at
reduced scale, [Experiment 3](#experiment-3-reduced-scale-attack--sdg-grid).

#### Main Result 2: De-identification methods are the most exposed

Cell Suppression, RankSwap and Synthpop sit at the high-risk end of
`tab:ra_mean_adult`, well above the DP generators. Same table, same experiments
as Result 1.

#### Main Result 3: CoBP-RA is the strongest attack measured

The newly introduced CoBP-RA attack attains the highest mean $R_{adv}$ of all
attacks evaluated. Reported in `tab:ra_mean_adult`; visible directly in the smoke test,
where CoBP-RA beats the mode baseline on the dummy dataset.

#### Main Result 4: DP reduces reconstruction up to ε≈10, above which risk flattens

Sweeping ε from 0.1 to 1000 across **six** DP mechanisms (MST, AIM, PrivBayes,
PrivSyn, MWEM-PGM, PrivateGSD), reconstruction advantage falls steadily as ε
decreases below about 10, and is close to flat above it. Independent variable:
ε; dependent variable: mean $R_{adv}$. Reported in the ε-sweep tables
(`tab:eps_sweep`, `tab:eps_full`) and
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
`tab:memorization_and_ds_risk` (memorization) and `tab:disparate_impact`
(disparate impact). Supported by
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

| File | Paper object | Find it in the PDF by its caption |
|---|---|---|
| `table1_ra_mean_adult.tex` | `tab:ra_mean_adult` (Results 1, 2, 3) | attack × SDG grid, Adult 10k |
| `table2_quality_overview.tex` | `tab:quality_overview` | synthetic data quality profile, Adult |
| `table4_eps_sweep_compact.tex`, `table4_eps_sweep_full.tex` | `tab:eps_sweep`, `tab:eps_full` (Result 4) | ε sweep, compact and full |
| `table6_mia_comparison.tex` | `tab:mia_comparison` | MIA baselines vs. RA-as-MIA |
| `table7_memorization.tex` | `tab:memorization_and_ds_risk` (Result 5) | memorization gap and disclosure risk |
| `table9_disparate_impact.tex` | `tab:disparate_impact` (Result 5) | disparate impact by race and sex |
| `table_ra_mean_cdc.tex`, `table_cdc_100k.tex` | `tab:ra_mean_cdc`, `tab:cdc_100k` | attack × SDG, CDC Diabetes at 1k and 100k |
| `table_ra_mean_nist_sbo.tex` | `tab:ra_mean_nist_sbo` | attack × SDG, NIST SBO |
| `fig_eps_curves.pdf`, `fig_eps_curves_perattack.pdf` | `fig:eps_curves` | reconstruction risk vs. privacy budget |
| `STATS_eps_curve.md`, `STATS_memorization.md` | `tab:eps_stats` and its memorization counterpart | paired significance tests |

**On table numbers.** The middle column gives each object's LaTeX `\label`, not
its printed number, and that is deliberate: table numbers move whenever a float
is added or reordered during copy-editing, so a number written down here could
be wrong by the time you read the final PDF. The labels do not move. Where a
number does appear elsewhere in this document -- "Table 1", say -- read it as a
convenience for the version submitted with this artifact, and match by caption if
it disagrees with the PDF in front of you. `python reproduce.py --list` prints the
same label-to-file mapping straight from the manifest in
`experiment_scripts/paper_objects.py`, which is the authority.

**Expected output.** `reproduce.py` first lists the sixteen files it wrote, and
then prints two coverage reports: Table 7 rows that have no `QI_linear`
memorization runs, and Table 1 cells backed by fewer than five disjoint samples
(many showing `n=0`, for the `CondDDPM` and `CondRePaint` rows that were excluded
as broken). Those lists are long -- around forty lines -- and they are *not*
errors: they are provenance notes recording how many samples back each cell, and
they are printed precisely so that no cell's sample count is taken on trust. The
run ends with:

```
done: 16 files written to expected_output/tables
The lists above are notes on how many samples back each cell, not errors.
```

Because the reference copies of these files are committed, you can verify
reproduction exactly rather than by eye:

```bash
python reproduce.py --out /tmp/check
diff -r expected_output/tables /tmp/check   # .tex and .md files should be identical
```

`./test.sh` performs exactly this diff on the twelve text outputs. The two PDFs and
their PNG companions will differ: matplotlib embeds a creation timestamp, and
glyph rasterisation varies with the freetype build. The numbers plotted in them
are verified through `STATS_eps_curve.md`, which is byte-compared.

#### Experiment 1, continued: comparing against the paper PDF

Two things to know before you diff a regenerated table against the printed one.

**The artifact is the system of record, and it was frozen before the paper was.**
This artifact is submitted at the artifact deadline; the camera-ready manuscript
is still being finalised and is due six days later. Every number the artifact
produces is regenerated, in front of you, from committed data by a script whose
output `./test.sh` checks byte-for-byte -- so where a printed cell and a
regenerated cell disagree, the regenerated one is the value that is actually
traceable to the data in this repository, and the paper is being corrected toward
it rather than the other way round. A handful of cells may therefore still differ
in the published PDF from what you see here. None of them changes a claim: the
five main results in the previous section are each supported by margins far
larger than any of these differences, and the table below lists every one we
know of.

**Some printed cells predate a database repair.** The database was repaired in
August 2026 (a float-binned encoding bug, described in `README.md`), and some
printed tables still carry pre-repair numbers -- most visibly the five MST columns
of `tab:ra_mean_adult`. The artifact regenerates from the repaired database, so
those cells differ from the PDF. Outside the MST columns, every classical-attack
row of that table matches the printed table exactly. Two rows (`CondDDPM`,
`CondRePaint`) print `---` because their runs were flagged as broken and excluded
rather than silently kept.

The complete, cell-level list of differences -- which table, how many cells, and
the largest move in each -- is in `TABLE-COVERAGE.md` under "Known
discrepancies between the regenerated tables and the printed paper". It is worth
opening: it is the audit behind the summary here, and it is the document that
tells you, for any cell you find that disagrees, whether we already knew. In
brief, as of this submission:

| Paper object | What differs | Size |
|---|---|---|
| `tab:ra_mean_adult` | the five MST columns; `CondDDPM` and `CondRePaint` rows blanked | 128 of 252 cells, largest move 13.2 to 21.7 |
| `tab:ra_mean_cdc` | MST(ε=0.1) and AIM(ε=3) columns | 9 of 75 cells, largest move 0.9 pp |
| `tab:ra_mean_nist_sbo` | MST(ε=0.1) and MST(ε=1) print `---`; other MST budgets rest on 2 samples, not 5 | up to 0.4 pp |
| `tab:mia_comparison` | printed table mixes pre- and post-repair runs | 7 cells |
| `tab:disparate_impact` | printed from a run older than any committed CSV; regenerated adds an `AIM (ε=1)` row | 32 of 45 cells move >5% relative, but the largest absolute move is 1.68 pp |
| `tab:cdc_100k`, `tab:quality_overview`, `tab:eps_sweep`, `tab:eps_full`, `tab:eps_stats`, `tab:memorization_and_ds_risk` | nothing | exact match |

Where a table is unaffected by the repair the agreement is exact rather than
within 5% -- `tab:cdc_100k`, for instance, reproduces all 52 of its cells.

Twelve of the paper's 33 labelled tables and figures regenerate this way,
covering every table that carries a main claim. Nine more are not derived from this
repository at all (the NIST scoreboard, the hand-written dataset and QI-definition
tables, the two diagrams). The remaining 12 are supporting tables whose data is
committed but whose generators are not yet written; `python reproduce.py --list`
names each one, and `TABLE-COVERAGE.md` tracks the work. Eleven of those twelve
have their data committed and need only a generator; the twelfth
(`fig:heatmap_ensemble`) also needs a provenance decision about which of several
candidate result files produced the printed figure, and is the one object here
whose source data is not committed.

#### Experiment 2: Attack versus baseline on the dummy dataset

- Time: 2 human-minutes + ~1 compute-minute

Demonstrates the attack pipeline end to end with no downloads, and shows a real
attack beating the mode baseline (Main Result 3, in miniature).

```bash
python master_experiment_script.py --n_runs 1
```

That runs `configs/demo_dummy.yaml`, the default, which needs no downloads. Pass
`--config <path>` to run any other config; `configs/example_cfg.yaml` is the
Adult equivalent, and needs the dataset fetched first (Experiment 3).

Edit `attack_method` in `configs/demo_dummy.yaml` to try others (`Mode`,
`RandomForest`, `KNN`, `NaiveBayes`, `MultiHeadMLP`, `CoBP-RA`, …); the registry
is listed in `attacks/__init__.py`. On the dummy dataset with MST ε=10, expect
the mode baseline at **33.88** and CoBP-RA around **40**.

#### Experiment 3: Reduced-scale attack × SDG grid

- Time: 10 human-minutes + ~40 compute-minutes
- Storage: ~100 MB

Regenerates a small corner of `tab:ra_mean_adult` from raw data, to confirm the numbers in
`results.db` are actually produced by this code rather than merely stored. This
needs the Adult dataset, which downloads freely:

```bash
# download Adult, write its schema, and carve the training samples
python experiment_scripts/fetch_dataset.py adult

# generate the synthetic data and run the attacks
bash experiment_scripts/run_artifact_experiment3.sh
```

The second script fixes the slice being reproduced: four generator settings
that are `tab:ra_mean_adult` columns (MST at ε=1, 10 and 100, and AIM at ε=1)
attacked by five that are its rows (Mode, KNN, RandomForest, NaiveBayes,
CoBP-RA), over
two training samples. Every one of them runs on CPU in the light image — no
GPU, no R, no Gurobi. The ~40-minute estimate assumes 12 worker processes; on
a 4-core laptop budget about 2 hours. Pass `WORKERS=<n>` to match the cores you
have.

The script ends by printing a mean $R_{adv}$ per attack, averaged over the four
generator settings and two samples. Running it as documented gives:

| Attack | Mean $R_{adv}$ here | Same four columns of `tab:ra_mean_adult` |
|---|---|---|
| CoBP-RA | 24.7 | 23.8 |
| Random Forest | 23.6 | 23.1 |
| Naive Bayes | 20.9 | 20.8 |
| KNN | 20.8 | 19.7 |
| Mode (baseline) | 10.4 | 10.3 |

The right-hand column is the published table restricted to the same four
generator settings, so the two are directly comparable. Every attack lands
within about a point, and the **ordering is identical** — which is the claim:
CoBP-RA on top (Main Result 3), every real attack far above the mode baseline.
The residual gap is expected, since the paper averages five training samples and
this averages two.

Wall-clock on the authors' 24-core machine at `WORKERS=12` was 13 minutes, most
of it in the synthetic-data generation step; the ~40-minute figure above is
deliberately conservative for a slower host.

`fetch_dataset.py adult` carves four samples, not five. The samples are disjoint
slices and Adult holds 47,621 rows once rows with missing values are dropped, so
a fifth slice of 10,000 does not fit; the paper's fifth Adult sample was drawn
separately and overlaps the others. Four is more than this experiment uses.

#### Experiment 4: Reduced-scale epsilon sweep

- Time: 10 human-minutes + ~50 compute-minutes at `SWEEP_WORKERS=12`; budget
  about 2.5 hours on the 4-core laptop the Hardware section describes
- Storage: ~200 MB

Reproduces the shape of Main Result 4 on one dataset and a subset of mechanisms,
using the same 9-point ε grid as the paper (0.1, 0.3, 1, 3, 10, 30, 100, 300,
1000):

```bash
# download CDC Diabetes, write its schema, and carve five training samples
python experiment_scripts/fetch_dataset.py cdc_diabetes

# generate the synthetic data and run the sweep
bash experiment_scripts/run_cdc_dp_sweep_pipeline.sh
```

The script writes its runs to `experiment_scripts/results_reproduction.db`, not
to the shipped `results.db`, so re-running it cannot move the numbers that
Experiment 1 and `test.sh` check against the committed tables. Set
`RECON_RESULTS_DB` yourself if you want them somewhere else. On a machine with
fewer cores than the authors', pass `SWEEP_WORKERS` to match what you have, e.g.
`SWEEP_WORKERS=4`.

Expect $R_{adv}$ to rise steeply from ε=0.1 to ε≈10 and to be close to flat
above it, matching the curve shape in `fig_eps_curves.pdf`. The absolute values
depend on how many trials complete; the *shape* is the claim.

## Limitations

Experiment 1 regenerates the tables carrying all five main claims —
`tab:ra_mean_adult`, `tab:quality_overview`, `tab:eps_sweep`, `tab:eps_full`,
`tab:mia_comparison`, `tab:memorization_and_ds_risk`, `tab:disparate_impact` —
and the ε-curve figures, exactly, from the shipped data. The
limitations below concern everything beyond that.

1. **Two datasets cannot be redistributed.** NIST Arizona requires a free IPUMS
   registration; NIST SBO is available from NIST on request as part of the
   Privacy CRC. Their rows in `tab:ra_mean_adult` and the ε-sweep tables therefore cannot be regenerated from
   scratch without the reviewer obtaining that access. Adult, CDC Diabetes and
   California Housing are freely scriptable and cover the majority of results.
   `data/dummy/` stands in for format and pipeline purposes.

2. **LinearReconstruction needs Gurobi.** This attack solves the reconstruction
   as a linear program, and the solver it uses is commercial. Gurobi is free for
   academics, but the licence has to be requested and activated per machine, and
   it cannot be redistributed inside a container image — so `gurobipy` is not
   installed and the three `LinearReconstruction` variants report that they are
   unavailable. Nothing else in the artifact depends on it, and every other
   attack is unaffected.

   A reviewer who already has, or obtains, an academic licence can enable it
   without rebuilding anything:

   ```
   # 1. Request a free academic licence at https://www.gurobi.com/academia/
   # 2. Install the licence on the host; this writes ~/gurobi.lic
   grbgetkey <your-licence-key>
   # 3. Install the Python bindings and mount the licence into the container
   docker run --rm -it -v "${PWD}":/workspace -w /workspace -v "$HOME/gurobi.lic":/opt/gurobi/gurobi.lic:ro -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic recon-artifact:latest bash
   pip install --user gurobipy==11.0.3
   python experiment_scripts/verify_registries.py attacks
   ```

   The three `LinearReconstruction` rows then run, and the attack becomes
   available to `master_experiment_script.py` and the sweep scripts like any
   other. Note that the academic licence is single-machine and does not permit
   use on a shared cluster node.

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
   the paper's tables average five training samples. Table *regeneration*
   (Experiment 1) is deterministic and exact.

6. **Not every supporting table regenerates yet.** Experiment 1 rebuilds the
   tables and figures behind the five main claims. The paper also contains
   supporting tables — per-dataset results for CDC and NIST SBO, the per-dataset
   quality profiles, the QI-composition analysis — whose underlying data is
   committed but which do not yet have a generator. `python reproduce.py --list`
   names every one of the paper's 33 labelled objects and reports which of them
   rebuild; `TABLE-COVERAGE.md` tracks the remaining work. No main claim
   depends on an object outside Experiment 1.

7. **`environment.yaml` is superseded.** The historical conda file in the
   repository root does not reflect the versions actually used. The authoritative,
   pinned dependency sets are `docker/requirements-attacks.txt` and
   `docker/requirements-sdg.txt`. `README.md`'s Installation section describes
   that same historical native setup and is flagged there as not being the
   evaluation path.

8. **The camera-ready paper was not yet final when this artifact was frozen.**
   The artifact deadline falls six days before the camera-ready deadline, so a
   small number of printed cells may still differ from what the artifact
   regenerates, and table *numbers* may shift as floats are reordered. Two
   consequences for a reviewer. First, identify tables by caption or `\label`
   rather than by number -- the mapping is in the Experiment 1 section above and
   in `python reproduce.py --list`. Second, where a printed value and a
   regenerated value disagree, the regenerated one is the one traceable to
   committed data; every difference we know of is enumerated cell-by-cell in
   `TABLE-COVERAGE.md`, and none of them changes a claim. The known cases as
   of this submission are summarised in the Experiment 1 section.

## Notes on Reusability

This is a framework, not a single experiment, and it was built to be extended.

**Adding an attack** takes three steps: implement
`my_attack(cfg, synth, targets, qi, hidden_features) -> (reconstructed_df, probas, classes)`
in `attacks/`, register it in `ATTACK_REGISTRY` in `attacks/__init__.py` under
the right `data_type`, and add defaults to `attack_defaults.py`. It is then
immediately usable by every sweep script, and composable with the chaining and
ensembling wrappers without further work.

`examples/add_your_own_attack.py` does all three steps in one runnable file, so
this claim can be checked rather than taken on trust:

```bash
python examples/add_your_own_attack.py
```

It writes a small conditional-mode attack, registers it, and scores it against
the mode baseline and against CoBP-RA on identical data with the paper's own
metric — about 30 seconds, on the shipped dummy dataset, no downloads. The new
attack should land between the two:

```
Attack                            mean R_adv
--------------------------------------------
NearestQIMatch (this example)           39.5
Mode baseline                           33.9
CoBP-RA (paper's strongest)             40.4
```

The first two rows are deterministic. CoBP-RA is not — its random forests are
unseeded, so it moves by a few tenths between runs (40.1–40.7 observed over
repeated runs here), which is the same tolerance noted for Experiment 2 above.
Anything comfortably above the 33.9 baseline is the expected result.

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

[`DATABASE.md`](DATABASE.md) documents it for that use: the schema, the two
label conventions that will otherwise catch you out (six attacks are stored
under their pre-2026 names, so a query for `ARFFormer` silently returns nothing;
and `attack_label` also holds ablation variants, chains, ensembles and oracles),
five worked queries, and the caveats to check before drawing conclusions from
uneven grid coverage. `python examples/query_results_db.py` runs those queries
and prints the results — standard library only, so it works without installing
this repository at all.

**Beyond reconstruction.** The same harness runs membership inference
(`--mode mia`), and `experiment_scripts/compare_mia_ra.py` implements the paper's
reduction placing reconstruction and MIA on one comparable scale, which should
transfer to other attack settings.
