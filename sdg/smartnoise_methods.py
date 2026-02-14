"""
SmartNoise-based synthetic data generation methods: MST and AIM.

Both are differentially private methods using the smartnoise-synth library.
"""

import numpy as np
import pandas as pd
from snsynth import Synthesizer


def _smartnoise_generate(method_name, train_df, meta, epsilon=1.0,
                         preprocessor_eps_ratio=0.3, num_rows=None):
    """Shared implementation for MST and AIM."""
    n = num_rows or len(train_df)

    categorical = [c for c in meta.get("categorical", []) if c in train_df.columns]
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]
    ordinal = [c for c in meta.get("ordinal", []) if c in train_df.columns]

    synthesizer = Synthesizer.create(method_name, epsilon=epsilon)
    synthesizer.fit(
        train_df,
        preprocessor_eps=epsilon * preprocessor_eps_ratio,
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=ordinal,
    )

    sample = synthesizer.sample(n)

    # Convert float columns to int where the original was integer-typed
    for col in sample.columns:
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
