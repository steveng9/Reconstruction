"""
Private-GSD synthetic data generation (genetic-algorithm-based DP synthesizer).

Paper: "Generating Private Synthetic Data with Genetic Algorithms" (ICML 2023),
https://arxiv.org/abs/2306.03257

Library: https://github.com/giusevtr/private_gsd (installed separately, NOT
vendored into this repo — see /home/golobs/private_gsd). Built on jax + the
snsynth `Synthesizer` base class (same base class MST/AIM use), so the
fit/sample conventions mirror sdg/smartnoise_methods.py closely: pass
categorical/continuous/ordinal column lists directly, and the library's
internal LabelTransformer / OrdinalTransformer / MinMaxTransformer handle the
string<->int round trip automatically (sample() already returns decoded
values in the original dtype/category space, no manual decode needed here).

GSD privately measures a workload of marginal statistics (degree-2 by
default here, i.e. all column pairs, following the library's own default in
GSDSynthesizer.fit) and then evolves a synthetic population with a genetic
algorithm to match them under DP noise.
"""

import numpy as np
import pandas as pd

from genetic_sd import GSDSynthesizer


def private_gsd_generate(train_df, meta, epsilon=1.0, delta=1e-5,
                          num_rows=None, preprocessor_eps_ratio=0.1,
                          tree_query_depth=3, num_discretization_intervals=64,
                          continuous_data_granularity=0.001,
                          early_stop_threshold=0.0001,
                          bin_continuous_as_ordinal=True, n_bins=20,
                          verbose=False, seed=None, **config):
    """Generate synthetic data using Private-GSD (genetic-algorithm DP synthesizer).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        epsilon: DP privacy budget (float, default 1.0).
        delta: DP failure probability (float, default 1e-5).
        num_rows: Size of the synthetic dataset to generate (default len(train_df)).
            Passed through as GSD's `N_prime` (the genetic algorithm's population
            size == output row count), so larger values increase runtime.
        preprocessor_eps_ratio: Fraction of epsilon spent on privately fitting
            column transformers (bounds for continuous columns) when continuous
            columns are present; the remainder goes to marginal measurement +
            the genetic search. Mirrors sdg/smartnoise_methods.py's convention.
        tree_query_depth: GSD's internal marginal-tree query depth (default 3,
            matches library default).
        num_discretization_intervals: Number of DP-chosen bin edges used to
            discretize continuous/high-cardinality-ordinal columns before
            building the marginal workload (default 64, library default).
        continuous_data_granularity: Minimum meaningful resolution for continuous
            columns when computing DP bin thresholds (default 0.001, library default).
        early_stop_threshold: Genetic algorithm stops once max statistic error
            drops below this threshold (default 0.0001, library default).
        bin_continuous_as_ordinal: pre-bin continuous columns into n_bins discrete
            bins (using actual, non-private data bounds) and pass them as ordinal,
            bypassing GSD/snsynth's private MinMaxTransformer bound estimation
            (approx_bounds), which can raise "could not find bounds" on small
            samples or skewed data. Mirrors sdg/smartnoise_methods.py's identical
            workaround for MST/AIM. Default True.
        n_bins: Number of bins used when bin_continuous_as_ordinal is True (default 20).
        verbose: Print GSD's internal progress/budget breakdown.
        seed: RNG seed (default: random).
        **config: accepted but unused extra kwargs (for interface parity with
            other sdg methods, e.g. stray `num_rows`-adjacent keys).

    Returns:
        Synthetic DataFrame with the same columns/dtypes as train_df.
    """
    n = num_rows or len(train_df)

    categorical = [c for c in meta.get("categorical", []) if c in train_df.columns]
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]
    ordinal = [c for c in meta.get("ordinal", []) if c in train_df.columns]

    df = train_df.copy()
    bin_edges = {}  # col -> bin edge array, for inverse-transform after sampling
    meta_data_bounds = {}  # col -> {'lower', 'upper'} for pre-binned ordinal cols,
                           # so GSD's OrdinalTransformer doesn't also need epsilon

    if bin_continuous_as_ordinal and continuous:
        for col in continuous:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                bin_edges[col] = np.array([lo, hi])
                df[col] = 0
                meta_data_bounds[col] = {"lower": 0, "upper": 0}
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                bin_edges[col] = edges
                df[col] = np.searchsorted(edges[1:-1], df[col].values)
                meta_data_bounds[col] = {"lower": 0, "upper": n_bins - 1}
        ordinal = ordinal + continuous
        continuous = []

    # Same convention as _smartnoise_generate: only spend preprocessor budget
    # when there are continuous columns whose bounds need private estimation.
    # Pre-binned ordinal columns get explicit bounds above instead (free, no
    # DP budget spent, and avoids approx_bounds failing on small samples).
    preprocessor_eps = epsilon * preprocessor_eps_ratio if continuous else 0.0

    synth = GSDSynthesizer(epsilon=epsilon, delta=delta, verbose=verbose)
    synth.fit(
        df,
        meta_data=meta_data_bounds or None,
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=ordinal,
        preprocessor_eps=preprocessor_eps,
        N_prime=n,
        tree_query_depth=tree_query_depth,
        num_discretization_intervals=num_discretization_intervals,
        continuous_data_granularity=continuous_data_granularity,
        early_stop_threshold=early_stop_threshold,
        seed=seed,
    )

    sample = synth.sample()

    # Inverse-transform pre-binned columns back to approximate original values.
    for col, edges in bin_edges.items():
        if col not in sample.columns:
            continue
        if len(edges) == 2:
            sample[col] = float(edges[0])
        else:
            midpoints = (edges[:-1] + edges[1:]) / 2
            bin_idx = pd.to_numeric(sample[col], errors="coerce").clip(0, n_bins - 1).astype(int)
            sample[col] = midpoints[bin_idx]

    # Cast types back to match training data dtypes (GSD's transformer already
    # decodes categoricals to original string labels; numeric columns can come
    # back as float64 and need rounding/casting back to the source dtype).
    pre_binned = set(bin_edges.keys())
    for col in sample.columns:
        if col in pre_binned or col not in train_df.columns:
            continue
        if train_df[col].dtype in (np.int64, np.int32, int):
            sample[col] = pd.to_numeric(sample[col], errors="coerce").round().astype(np.int64)
        elif train_df[col].dtype == np.float64 and sample[col].dtype != np.float64:
            sample[col] = pd.to_numeric(sample[col], errors="coerce").astype(np.float64)

    return sample
