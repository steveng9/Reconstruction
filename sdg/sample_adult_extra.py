#!/usr/bin/env python
"""
One-off script: create a 5th 10k training sample for adult data.

The adult dataset has ~47k rows, so only 4 disjoint 10k samples fit.
This script samples 10k rows WITHOUT the disjointness guarantee — some
rows will overlap with samples 00-03. A NO_HOLDOUT marker file is placed
in the sample directory so that memorization experiments will refuse to
use it as holdout.

Usage (from Reconstruction/):
    python sdg/sample_adult_extra.py          # create the sample
    python sdg/generate_synth.py sdg          # then generate SDG (configure SDG_JOBS / SAMPLES_TO_GENERATE first)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("/home/golobs/data/reconstruction_data")
DATASET = "adult"
SAMPLE_SIZE = 10000

full_path = DATA_ROOT / DATASET / "full_data.csv"
df = pd.read_csv(full_path)

# Same cleaning as generate_synth.py
df["income"] = df["income"].str.strip().str.rstrip(".")
df = df.dropna().reset_index(drop=True)

print(f"Full dataset: {len(df)} rows")

# Use a different seed than the main script (which uses 42) to get fresh rows
rng = np.random.RandomState(999)
sample_df = df.sample(n=SAMPLE_SIZE, random_state=rng).reset_index(drop=True)

sample_dir = DATA_ROOT / DATASET / f"size_{SAMPLE_SIZE}" / "sample_04"
sample_dir.mkdir(parents=True, exist_ok=True)
sample_df.to_csv(sample_dir / "train.csv", index=False)

# Marker file: prevents this sample from being used as holdout
marker = sample_dir / "NO_HOLDOUT"
marker.write_text(
    "This sample overlaps with samples 00-03 (not disjoint).\n"
    "Do NOT use as holdout in memorization experiments.\n"
)

print(f"Saved {len(sample_df)} rows to {sample_dir / 'train.csv'}")
print(f"Created {marker}")
print(f"\nNext: configure SDG_JOBS and SAMPLES_TO_GENERATE in sdg/generate_synth.py, then run sdg step.")
