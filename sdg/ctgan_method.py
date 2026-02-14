"""
CTGAN synthetic data generation using the SDV library.
"""

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


def ctgan_generate(train_df, meta, **config):
    """Generate synthetic data using CTGAN (Conditional Tabular GAN).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: num_rows (int, default len(train_df)),
                  epochs (int, default 300),
                  cuda (bool, default True).
    Returns:
        Synthetic DataFrame.
    """
    num_rows = config.get("num_rows", len(train_df))
    epochs = config.get("epochs", 300)
    cuda = config.get("cuda", True)

    metadata = Metadata.detect_from_dataframe(data=train_df, table_name="train")

    # Override column types from meta
    for col in meta.get("categorical", []) + meta.get("ordinal", []):
        if col in train_df.columns:
            metadata.update_column(column_name=col, sdtype="categorical")

    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, cuda=cuda, verbose=True)
    synthesizer.fit(train_df)

    sample = synthesizer.sample(num_rows=num_rows)
    return sample
