# Third-party code and licensing notices

The code in this repository is released under the MIT License (see `LICENSE`),
with the exceptions and qualifications recorded below. This file exists so that
anyone reusing the artifact can see exactly what is ours, what is adapted, and
where the provenance is imperfect.

---

## 1. `sdg/_privsyn_vendor.py` — Apache License 2.0

Three core algorithm classes (`View`, `Consistenter`, and the GUM record
synthesizer) are vendored, with only import-path adjustments, from the PrivSyn
reference implementation bundled in the SynMeter benchmark:

- Upstream: <https://github.com/zealscott/SynMeter>
- Upstream licence: Apache License 2.0 (full text in `LICENSES/Apache-2.0.txt`)
- Original paper: Zhang, Wang, Li, Honorio, Backes, He, Chen, Zhang.
  *PrivSyn: Differentially Private Data Synthesis*, USENIX Security 2021.

The DP marginal *measurement* logic (noise calibration, marginal selection,
budget splitting) in `sdg/privsyn_method.py` is **not** from SynMeter; it is
implemented independently in this repository and is MIT-licensed.

As required by Apache-2.0 §4, the licence text is retained and the vendored file
carries a header identifying its origin and the modifications made.

---

## 2. `SOTA_attacks/linear_reconstruction.py` — adapted, upstream licence absent

This file adapts the linear reconstruction attack of:

> Annamalai, Ganev, De Cristofaro. *"What You See Is What You Get":
> Reconstruction Attacks on Synthetic Data.* 2024.

by way of <https://github.com/Filienko/recon-synth>, which we forked to
<https://github.com/steveng9/recon-synth> (pinned in this repository as the
submodule `external/recon-synth`).

**Neither the upstream repository nor our fork carries an explicit licence
file.** We record this plainly rather than assert terms we cannot substantiate:

- The MIT licence in this repository's `LICENSE` covers **our own** code only.
  It does not, and cannot, grant rights over the upstream material.
- Our modifications to the solver (correcting `categorical_l1_solve` to take
  per-query target categories) live in the `external/recon-synth` submodule, not
  in this repository, and are offered back to the upstream project.
- Anyone wishing to reuse this specific attack beyond the terms of academic fair
  use should contact the upstream authors for clarification.

The remainder of the artifact does not depend on this file: excluding
LinearReconstruction leaves every other attack and every other paper table
intact.

---

## 3. `genetic_sd` (PrivateGSD) — upstream licence absent

The PrivateGSD generator (`sdg/private_gsd_method.py`) is a thin wrapper around
the reference implementation:

- Upstream: <https://github.com/giusevtr/private_gsd>, pinned at commit `f6150d7`
- Original paper: Liu, Vietri, Wu. *Generating Private Synthetic Data with
  Genetic Algorithms*, ICML 2023.

This upstream repository likewise carries **no licence file**. It is installed
from source as a dependency (see `docker/requirements-attacks.txt`) rather than
vendored into this repository, so no upstream code is redistributed here.

---

## 4. `external/MIA_on_diffusion` — MIT

The TabDDPM / RePaint implementations are derived from the MIDST models and are
distributed under the MIT License; see `external/MIA_on_diffusion/LICENSE.md`.

---

## 5. Dependencies installed from PyPI

Every remaining dependency is installed unmodified from PyPI under its own
licence; see `docker/requirements-attacks.txt` and
`docker/requirements-sdg.txt` for the pinned set. None of them are
redistributed in this repository.

---

## 6. Datasets

No dataset is redistributed in this repository. `data/dummy/` contains only
machine-generated data produced by `data/dummy/make_dummy_data.py`, describes no
real person, and is covered by this repository's MIT licence. See
`ARTIFACT-APPENDIX.md` for how to obtain each real dataset and the access
restrictions that apply to two of them.
