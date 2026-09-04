#!/usr/bin/env python
"""Generate the `dummy` demonstration dataset shipped with this artifact.

The PoPETs artifact guidelines ask that, where a required dataset cannot be
redistributed, the artifact "provide a synthetic dataset that showcases the
expected data format". Two of our five datasets are access-restricted (NIST
Arizona needs an IPUMS registration; NIST SBO is available from NIST on
request), so this script produces a small, fully synthetic, freely
redistributable stand-in with the same directory layout and file formats.

The data is entirely machine-generated and describes no real person. Features
carry deliberate statistical dependencies so that a reconstruction attack
scores meaningfully above the mode baseline, which makes the smoke test in
test.sh a genuine check rather than a trivial one.

Reproducible: fixed seed, no network access, runs in a few seconds.

    python data/dummy/make_dummy_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20270115
N_FULL, N_TRAIN, N_HOLDOUT = 4000, 1000, 1000
HERE = Path(__file__).resolve().parent

AGE_BANDS = ["18-29", "30-39", "40-49", "50-64", "65+"]
REGIONS = ["north", "south", "east", "west"]
SEXES = ["F", "M"]
EDUCATION = ["none", "secondary", "college", "graduate"]
EMPLOYMENT = ["unemployed", "part-time", "full-time", "retired"]
INCOME_BANDS = ["<20k", "20-40k", "40-70k", "70-120k", "120k+"]
CHRONIC = ["none", "one", "multiple"]
VISIT_BANDS = ["0", "1-2", "3-5", "6+"]


def _pick(rng, options, weights):
    w = np.asarray(weights, dtype=float)
    return options[rng.choice(len(options), p=w / w.sum())]


def generate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        age = _pick(rng, AGE_BANDS, [22, 24, 20, 20, 14])
        region = _pick(rng, REGIONS, [26, 28, 24, 22])
        sex = _pick(rng, SEXES, [51, 49])
        ai = AGE_BANDS.index(age)

        # education: younger cohorts skew more educated
        edu_w = [[6, 34, 40, 20], [7, 35, 38, 20], [10, 40, 34, 16],
                 [16, 46, 27, 11], [24, 50, 19, 7]][ai]
        education = _pick(rng, EDUCATION, edu_w)
        ei = EDUCATION.index(education)

        # employment: driven by age (retirement) and education
        if age == "65+":
            employment = _pick(rng, EMPLOYMENT, [4, 10, 6, 80])
        else:
            emp_w = [[22, 30, 46, 2], [14, 26, 58, 2], [8, 20, 70, 2], [5, 15, 78, 2]][ei]
            employment = _pick(rng, EMPLOYMENT, emp_w)

        # income: education is the dominant driver, employment modulates it
        inc_w = np.array([[46, 30, 15, 7, 2], [24, 33, 27, 13, 3],
                          [10, 22, 32, 27, 9], [4, 12, 26, 38, 20]][ei], dtype=float)
        if employment in ("unemployed", "retired"):
            inc_w *= np.array([3.2, 1.9, 0.8, 0.35, 0.15])
        elif employment == "part-time":
            inc_w *= np.array([1.8, 1.5, 0.9, 0.5, 0.3])
        income = _pick(rng, INCOME_BANDS, inc_w)
        ii = INCOME_BANDS.index(income)

        insured = _pick(rng, ["no", "yes"], [[42, 58], [30, 70], [18, 82], [9, 91], [4, 96]][ii])
        smoker = _pick(rng, ["no", "yes"], [[68, 32], [72, 28], [80, 20], [88, 12]][ei])

        chronic_w = np.array([[76, 19, 5], [70, 23, 7], [58, 29, 13],
                              [41, 36, 23], [26, 39, 35]][ai], dtype=float)
        if smoker == "yes":
            chronic_w *= np.array([0.65, 1.25, 1.75])
        chronic = _pick(rng, CHRONIC, chronic_w)

        visit_w = np.array([[58, 30, 9, 3], [22, 40, 27, 11],
                            [8, 26, 38, 28]][CHRONIC.index(chronic)], dtype=float)
        if insured == "no":
            visit_w *= np.array([2.1, 1.2, 0.6, 0.35])
        visits = _pick(rng, VISIT_BANDS, visit_w)

        rows.append((age, region, sex, education, employment,
                     income, insured, smoker, chronic, visits))

    return pd.DataFrame(rows, columns=[
        "age_band", "region", "sex", "education", "employment",
        "income_band", "insured", "smoker", "chronic_cond", "visits_band"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    full = generate(N_FULL, rng)
    full.to_csv(HERE / "full_data.csv", index=False)

    meta = {
        "categorical": ["region", "sex", "education", "employment",
                        "insured", "smoker", "chronic_cond"],
        "continuous": [],
        "ordinal": ["age_band", "income_band", "visits_band"],
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=4) + "\n")

    # Two disjoint samples, mirroring the size_{N}/sample_{XX} layout used by
    # every real dataset in this repo. sample_01 doubles as the memorization
    # holdout for sample_00.
    shuffled = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    for idx, start in enumerate([0, N_TRAIN]):
        out = HERE / f"size_{N_TRAIN}" / f"sample_{idx:02d}"
        out.mkdir(parents=True, exist_ok=True)
        shuffled.iloc[start:start + N_TRAIN].to_csv(out / "train.csv", index=False)
    holdout = shuffled.iloc[2 * N_TRAIN:2 * N_TRAIN + N_HOLDOUT]
    holdout.to_csv(HERE / f"size_{N_TRAIN}" / "sample_00" / "holdout.csv", index=False)

    print(f"wrote {HERE}/full_data.csv  ({len(full)} rows)")
    print(f"wrote {HERE}/meta.json")
    for idx in range(2):
        print(f"wrote {HERE}/size_{N_TRAIN}/sample_{idx:02d}/train.csv  ({N_TRAIN} rows)")
    print(f"wrote {HERE}/size_{N_TRAIN}/sample_00/holdout.csv  ({len(holdout)} rows)")


if __name__ == "__main__":
    main()
