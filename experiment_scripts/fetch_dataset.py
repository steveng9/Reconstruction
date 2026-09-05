#!/usr/bin/env python
"""Download a public dataset and lay it out the way the pipeline expects.

The experiment scripts assume a dataset directory that looks like this:

    <data root>/<dataset>/full_data.csv
    <data root>/<dataset>/meta.json
    <data root>/<dataset>/size_<N>/sample_00/train.csv
    ...

Nothing in the repository creates that layout -- on the authors' machine it is
a symlink into a large external tree -- so this script builds it from scratch
for the two freely-downloadable datasets the artifact's experiments use.

    python experiment_scripts/fetch_dataset.py adult
    python experiment_scripts/fetch_dataset.py cdc_diabetes

Both are fetched from the UCI repository via `ucimlrepo`. The column schemas
below are the ones the paper's results were produced with, so they are recorded
here verbatim rather than inferred -- inferring them would risk silently
producing a differently-typed dataset than the one behind the published tables.
"""

import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT, REPO_ROOT

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path


DATASETS = {
    "adult": {
        "uci_id": 2,
        "default_size": 10_000,
        "default_samples": 5,
        # Exact column order of the authors' full_data.csv. UCI's own ordering
        # differs, and column order is part of what the pipeline reproduces.
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income",
        ],
        "meta": {
            "categorical": [
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country", "income",
            ],
            "continuous": [
                "age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week",
            ],
            "ordinal": [],
        },
    },
    "cdc_diabetes": {
        "uci_id": 891,
        "default_size": 1_000,
        "default_samples": 5,
        "columns": [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
            "Income", "Diabetes_binary",
        ],
        "meta": {
            "categorical": [
                "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
                "Sex", "Diabetes_binary",
                "GenHlth", "Age", "Education", "Income",
            ],
            "continuous": ["BMI", "MentHlth", "PhysHlth"],
            "ordinal": [],
        },
    },
}


def fetch(name, size, n_samples, force=False):
    spec = DATASETS[name]
    ds_dir = Path(str(DATA_ROOT)) / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    full_csv = ds_dir / "full_data.csv"

    if full_csv.exists() and not force:
        print(f"  {full_csv} already exists, keeping it (--force to re-download)")
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            sys.exit("ucimlrepo is not installed:  pip install ucimlrepo")
        print(f"  downloading UCI dataset {spec['uci_id']} ...")
        data = fetch_ucirepo(id=spec["uci_id"]).data.original
        # UCI ships CDC Diabetes with a leading row-ID column that is not part
        # of the data and is absent from the authors' copy; dropping it makes
        # the download match the file the published results were produced from.
        drop = [c for c in data.columns if c.lower() in ("id", "unnamed: 0")]
        if drop:
            print(f"  dropping non-data column(s): {drop}")
            data = data.drop(columns=drop)
        missing = [c for c in spec["columns"] if c not in data.columns]
        extra = [c for c in data.columns if c not in spec["columns"]]
        if missing or extra:
            sys.exit(f"UCI dataset {spec['uci_id']} does not have the expected "
                     f"columns (missing={missing}, unexpected={extra}). The "
                     f"upstream dataset may have changed.")
        data = data[spec["columns"]]
        data.to_csv(full_csv, index=False)
        print(f"  wrote {full_csv}  ({len(data):,} rows, {len(data.columns)} columns)")

    # Never clobber an existing schema. On the authors' machine this path is a
    # symlink into the real data tree, and silently rewriting a schema there
    # would change what every future run generates.
    meta_path = ds_dir / "meta.json"
    meta = spec["meta"]
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing == meta:
            print(f"  {meta_path} already matches the expected schema")
        else:
            print(f"  {meta_path} exists and differs from the expected schema -- keeping it")
            print("    (delete it and re-run if you want the schema this script ships)")
            meta = existing
    else:
        meta_path.write_text(json.dumps(meta, indent=4) + "\n")
        print(f"  wrote {meta_path}")

    # Sanity check: every column named in the schema must exist, and vice versa.
    import pandas as pd
    cols = set(pd.read_csv(full_csv, nrows=1).columns)
    named = set(meta["categorical"]) | set(meta["continuous"]) | set(meta["ordinal"])
    if cols != named:
        print(f"  WARNING: schema and CSV disagree.")
        if named - cols:
            print(f"    named but absent from the CSV: {sorted(named - cols)}")
        if cols - named:
            print(f"    in the CSV but unnamed:        {sorted(cols - named)}")
        return 1

    print(f"  carving {n_samples} training samples of {size:,} rows ...")
    env = dict(os.environ,
               SDG_DATASET=name,
               SDG_SAMPLE_SIZE=str(size),
               SDG_NUM_SAMPLES=str(n_samples))
    rc = subprocess.call(
        [sys.executable, str(Path(str(REPO_ROOT)) / "sdg" / "generate_synth.py"), "sample"],
        env=env)
    if rc != 0:
        print("  sampling failed")
        return rc

    print(f"\nDone. {ds_dir}/size_{size}/ is ready.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--size", type=int, default=None,
                    help="rows per training sample (default: what the paper used)")
    ap.add_argument("--samples", type=int, default=None,
                    help="number of disjoint training samples (default: 5)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if full_data.csv is already present")
    args = ap.parse_args(argv)

    spec = DATASETS[args.dataset]
    size = args.size or spec["default_size"]
    n_samples = args.samples or spec["default_samples"]

    print(f"Preparing {args.dataset} (size {size:,}, {n_samples} samples) under {DATA_ROOT}")
    return fetch(args.dataset, size, n_samples, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
