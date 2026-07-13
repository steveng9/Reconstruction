"""
PrivBayes synthetic data generation, via the DataSynthesizer package
(https://github.com/DataResponsibly/DataSynthesizer), which implements PrivBayes's
"correlated attribute mode" (Zhang et al. 2017) — a Bayesian network fit with
differentially private structure learning (greedy_bayes) and noisy conditional
distributions.
"""

import os
import tempfile
from multiprocessing.pool import Pool as _MPPool

import numpy as np
import pandas as pd

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

# DataSynthesizer's PrivBayes.py does a bare `with Pool() as pool:` inside its
# greedy Bayesian-network construction loop (once per remaining attribute),
# which defaults to os.cpu_count() (e.g. 48) worker processes every time --
# regardless of any outer parallelism already in flight. Under our own
# ProcessPoolExecutor-based job sweeps this multiplies out (N outer workers x
# 48 inner workers each) and can spawn hundreds of processes per host. Cap it
# to a small fixed pool size; override via PRIVBAYES_INNER_POOL_SIZE if needed.
import DataSynthesizer.lib.PrivBayes as _privbayes_lib

_INNER_POOL_SIZE = int(os.environ.get("PRIVBAYES_INNER_POOL_SIZE", "2"))
_privbayes_lib.Pool = lambda *a, **k: _MPPool(processes=_INNER_POOL_SIZE)


def privbayes_generate(train_df, meta, **config):
    """Generate synthetic data using PrivBayes (correlated attribute mode).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: epsilon (float, default 1.0) — DP privacy budget.
                  k (int, default 2) — max parents per node in the Bayesian network
                      (the classical PrivBayes degree parameter).
                  num_rows (int, default len(train_df)).
                  category_threshold (int, default 40) — columns with <= this many
                      unique values are auto-detected as categorical by DataSynthesizer.
                  bin_continuous_as_ordinal (bool, default True) — pre-bin continuous
                      columns into n_bins discrete bins and pass them to DataSynthesizer
                      as categorical, decoding bin index -> bin midpoint ourselves after
                      sampling. Bypasses DataSynthesizer's own continuous-attribute
                      handling, which samples *uniformly within a bin* — badly biased
                      for zero-inflated/skewed columns (e.g. capital-gain, capital-loss:
                      mostly 0 but bins span thousands, so uniform-in-bin sampling
                      inflates the synthetic mean by 5-10x). Mirrors the identical
                      trick already used for MST/AIM in sdg/smartnoise_methods.py.
                  n_bins (int, default 20) — bins for pre-binned continuous columns.
                  seed (int, default 0).
    Returns:
        Synthetic DataFrame with the same columns/dtypes as train_df.
    """
    epsilon = config.get("epsilon", 1.0)
    k = config.get("k", 2)
    num_rows = config.get("num_rows", len(train_df))
    category_threshold = config.get("category_threshold", 40)
    bin_continuous_as_ordinal = config.get("bin_continuous_as_ordinal", True)
    n_bins = config.get("n_bins", 20)
    seed = config.get("seed", 0)

    categorical = set(meta.get("categorical", [])) | set(meta.get("ordinal", []))
    categorical = {c for c in categorical if c in train_df.columns}
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]

    df = train_df.copy()
    bin_edges = {}  # col -> bin edge array, for inverse-transform after sampling

    if bin_continuous_as_ordinal and continuous:
        for col in continuous:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                bin_edges[col] = np.array([lo, hi])
                df[col] = 0
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                bin_edges[col] = edges
                df[col] = np.searchsorted(edges[1:-1], df[col].values)
        categorical = categorical | set(continuous)

    attribute_to_is_categorical = {c: (c in categorical) for c in df.columns}
    attribute_to_is_candidate_key = {c: False for c in df.columns}

    with tempfile.TemporaryDirectory(prefix="privbayes_") as tmpdir:
        dataset_file = os.path.join(tmpdir, "train.csv")
        description_file = os.path.join(tmpdir, "description.json")
        df.to_csv(dataset_file, index=False)

        describer = DataDescriber(category_threshold=category_threshold)
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=dataset_file,
            k=k,
            epsilon=epsilon,
            attribute_to_is_categorical=attribute_to_is_categorical,
            attribute_to_is_candidate_key=attribute_to_is_candidate_key,
            seed=seed,
        )
        describer.save_dataset_description_to_file(description_file)

        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_rows, description_file, seed=seed)
        synth_df = generator.synthetic_dataset.copy()

    # Inverse-transform pre-binned columns back to approximate original values
    # (bin midpoint, not a random point in the bin — see docstring above).
    for col, edges in bin_edges.items():
        if col not in synth_df.columns:
            continue
        if len(edges) == 2:
            synth_df[col] = float(edges[0])
        else:
            midpoints = (edges[:-1] + edges[1:]) / 2
            bin_idx = pd.to_numeric(synth_df[col], errors="coerce").fillna(0).clip(0, n_bins - 1).astype(int)
            synth_df[col] = midpoints[bin_idx]

    # Reorder to match input; cast types back to match train_df dtypes.
    synth_df = synth_df[[c for c in train_df.columns if c in synth_df.columns]]
    pre_binned = set(bin_edges.keys())
    for col in synth_df.columns:
        if col in pre_binned:
            continue
        dtype = train_df[col].dtype
        if dtype in (np.int64, np.int32, int):
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").round().fillna(0).astype(np.int64)
        elif dtype == float:
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").astype(float)

    return synth_df.reset_index(drop=True)
