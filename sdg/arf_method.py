"""
ARF (Adaptive Random Forest) synthetic data generation using the Synthcity library.
"""

import pandas as pd
from synthcity.plugins import Plugins


def arf_generate(train_df, meta, **config):
    """Generate synthetic data using ARF.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: epsilon (float or None), n_iter (int or None),
                  num_rows (int, default len(train_df)).
    Returns:
        Synthetic DataFrame.
    """
    epsilon = config.get("epsilon", None)
    n_iter = config.get("n_iter", None)
    num_rows = config.get("num_rows", len(train_df))
    device = config.get("device", "cuda")

    # Build kwargs for plugin creation
    plugin_kwargs = {"device": device}
    if epsilon is not None:
        plugin_kwargs["epsilon"] = epsilon
    if n_iter is not None:
        plugin_kwargs["n_iter"] = n_iter

    # Cast categorical columns to str to avoid mixed-type issues
    df = train_df.copy()
    categorical_cols = meta.get("categorical", []) + meta.get("ordinal", [])
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    model = Plugins().get("arf", **plugin_kwargs)
    model.fit(df)
    syn_data = model.generate(count=num_rows).dataframe()

    # Convert categorical columns back from str to int where possible
    for col in syn_data.columns:
        if col in train_df.columns and train_df[col].dtype in (int, "int64", "int32"):
            try:
                syn_data[col] = pd.to_numeric(syn_data[col], errors="coerce").fillna(0).astype(int)
            except (ValueError, TypeError):
                pass

    return syn_data
