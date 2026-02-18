"""
SmartNoise-based synthetic data generation methods: MST and AIM.

Both are differentially private methods using the smartnoise-synth library.
"""

import numpy as np
import pandas as pd
from snsynth import Synthesizer


def _smartnoise_generate(method_name, train_df, meta, epsilon=1.0,
                         preprocessor_eps_ratio=0.3, num_rows=None,
                         bin_continuous_as_ordinal=False, n_bins=20):
    """Shared implementation for MST and AIM.

    bin_continuous_as_ordinal: pre-bin continuous columns into n_bins discrete
        bins and pass them as ordinal. Bypasses SmartNoise's private BinTransformer
        bound estimation, which fails with very small epsilon on skewed data.
        MST/AIM are discrete-domain methods that bin continuous data internally
        anyway — this just does it with actual (non-private) data bounds.
    """
    n = num_rows or len(train_df)

    categorical = [c for c in meta.get("categorical", []) if c in train_df.columns]
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]
    ordinal = [c for c in meta.get("ordinal", []) if c in train_df.columns]

    df = train_df.copy()
    bin_edges = {}  # col -> bin edge array, for inverse-transform after sampling

    if bin_continuous_as_ordinal and continuous:
        for col in continuous:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                # Zero-variance column — map to a single bin
                bin_edges[col] = np.array([lo, hi])
                df[col] = 0
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                bin_edges[col] = edges
                # Assign each value to a bin index in [0, n_bins-1]
                df[col] = np.searchsorted(edges[1:-1], df[col].values)
        ordinal = ordinal + continuous
        continuous = []

    # When all continuous columns are pre-binned, no epsilon is needed for
    # the preprocessor — give the full budget to synthesis.
    preprocessor_eps = 0.0 if not continuous else epsilon * preprocessor_eps_ratio

    synthesizer = Synthesizer.create(method_name, epsilon=epsilon)
    synthesizer.fit(
        df,
        preprocessor_eps=preprocessor_eps,
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=ordinal,
    )

    sample = synthesizer.sample(n)

    # Inverse-transform pre-binned columns back to approximate original values
    for col, edges in bin_edges.items():
        if col not in sample.columns:
            continue
        if len(edges) == 2:
            sample[col] = float(edges[0])
        else:
            midpoints = (edges[:-1] + edges[1:]) / 2
            bin_idx = sample[col].clip(0, n_bins - 1).astype(int)
            sample[col] = midpoints[bin_idx]

    # Cast types to match training data dtypes (skip inverse-transformed cols)
    pre_binned = set(bin_edges.keys())
    for col in sample.columns:
        if col in pre_binned:
            continue
        if col in train_df.columns and train_df[col].dtype in [np.int64, np.int32, int]:
            sample[col] = sample[col].round().astype(np.int64)
        elif sample[col].dtype == np.float64:
            sample[col] = sample[col].astype(np.int64)

    return sample


def mst_generate(train_df, meta, **config):
    """Generate synthetic data using MST (Maximum Spanning Tree) with differential privacy.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: epsilon (float, default 1.0), preprocessor_eps_ratio (float, default 0.3),
                  num_rows (int, default len(train_df)).
    Returns:
        Synthetic DataFrame.
    """
    return _smartnoise_generate("mst", train_df, meta, **config)


def aim_generate(train_df, meta, **config):
    """Generate synthetic data using AIM with differential privacy.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: epsilon (float, default 1.0), preprocessor_eps_ratio (float, default 0.3),
                  num_rows (int, default len(train_df)).
    Returns:
        Synthetic DataFrame.
    """
    return _smartnoise_generate("aim", train_df, meta, **config)
