"""
TabDDPM synthetic data generation using the ClaVa/TabDDPM pipeline.

Wraps the pipeline from MIA_on_diffusion to match the sdg/ interface.
"""

import sys
import os
import json
import tempfile
import pickle

import numpy as np
import pandas as pd

# Add the MIA_on_diffusion repo so we can import the pipeline
_MIA_ROOT = "/home/golobs/MIA_on_diffusion"
if _MIA_ROOT not in sys.path:
    sys.path.insert(0, _MIA_ROOT)

from midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_synthesizing,
)
from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table_CUSTOM


# Minimal default config for single-table TabDDPM
_DEFAULT_CONFIG = {
    "general": {
        "sample_prefix": "",
        "exp_name": "sdg_run",
    },
    "clustering": {
        "parent_scale": 1.0,
        "num_clusters": 50,
        "clustering_method": "both",
    },
    "diffusion": {
        "d_layers": [512, 1024, 1024, 1024, 1024, 512],
        "dropout": 0.1,
        "num_timesteps": 1000,
        "model_type": "mlp",
        "iterations": 10000,
        "batch_size": 4096,
        "lr": 0.0006,
        "gaussian_loss_type": "mse",
        "weight_decay": 1e-05,
        "scheduler": "cosine",
    },
    "classifier": {
        "d_layers": [128, 256, 512, 1024, 512, 256, 128],
        "lr": 0.0001,
        "dim_t": 128,
        "batch_size": 4096,
        "iterations": 5000,
    },
    "sampling": {
        "batch_size": 20000,
        "classifier_scale": 1.0,
    },
    "matching": {
        "num_matching_clusters": 1,
        "matching_batch_size": 1000,
        "unique_matching": True,
        "no_matching": True,
    },
}


def _build_domain(train_df, meta):
    """Build a TabDDPM domain dict from the sdg meta dict."""
    domain = {}
    cat_cols = set(meta.get("categorical", []) + meta.get("ordinal", []))
    for col in train_df.columns:
        n_unique = train_df[col].nunique()
        if col in cat_cols:
            domain[col] = {"type": "discrete", "size": n_unique}
        else:
            domain[col] = {"type": "discrete", "size": n_unique}
    return domain


def _build_dataset_meta(table_name):
    """Build the dataset_meta dict for a single table."""
    return {
        "relation_order": [[None, table_name]],
        "tables": {
            table_name: {"children": [], "parents": []}
        },
    }


def tabddpm_generate(train_df, meta, **config):
    """Generate synthetic data using TabDDPM (denoising diffusion for tabular data).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: iterations (int, default 10000),
                  num_timesteps (int, default 1000),
                  batch_size (int, default 4096),
                  num_rows (int, default len(train_df)),
                  workspace_dir (str, default tempdir).
    Returns:
        Synthetic DataFrame.
    """
    num_rows = config.get("num_rows", len(train_df))
    sample_scale = num_rows / len(train_df)

    # Build pipeline config from defaults + overrides
    configs = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
    if "iterations" in config:
        configs["diffusion"]["iterations"] = config["iterations"]
    if "num_timesteps" in config:
        configs["diffusion"]["num_timesteps"] = config["num_timesteps"]
    if "batch_size" in config:
        configs["diffusion"]["batch_size"] = config["batch_size"]

    # Workspace dir for model artifacts
    workspace_dir = config.get("workspace_dir", None)
    use_tempdir = workspace_dir is None
    if use_tempdir:
        tmpdir = tempfile.mkdtemp(prefix="tabddpm_")
        workspace_dir = tmpdir
    configs["general"]["workspace_dir"] = workspace_dir

    save_dir = os.path.join(workspace_dir, configs["general"]["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_matching"), exist_ok=True)

    # Build domain and metadata for the pipeline
    table_name = "data"
    domain = _build_domain(train_df, meta)
    dataset_meta = _build_dataset_meta(table_name)

    # Add a placeholder target column (required by the pipeline)
    df = train_df.copy()
    df["placeholder"] = 0
    domain["placeholder"] = {"type": "discrete", "size": 1}

    # Load into pipeline format
    tables, relation_order, _ = load_multi_table_CUSTOM(
        dataset_meta, domain, df, verbose=False
    )

    # Clustering (no-op for single table with no parent, but required)
    tables, all_group_lengths_prob_dicts = clava_clustering(
        tables, relation_order, save_dir, configs
    )

    # Training
    models = clava_training(
        tables, relation_order, save_dir, configs
    )

    # Synthesis
    cleaned_tables, _, _ = clava_synthesizing(
        tables, relation_order, save_dir,
        all_group_lengths_prob_dicts, models, configs,
        sample_scale=sample_scale,
    )

    # Extract the single table result
    syn_df = list(cleaned_tables.values())[0]

    # Drop placeholder/id columns and reorder to match input
    drop_cols = [c for c in syn_df.columns if c in ("placeholder", f"{table_name}_id")]
    syn_df = syn_df.drop(columns=drop_cols, errors="ignore")

    # Keep only original columns, in original order
    orig_cols = [c for c in train_df.columns if c in syn_df.columns]
    syn_df = syn_df[orig_cols].reset_index(drop=True)

    # Cast types back
    for col in syn_df.columns:
        if col in train_df.columns:
            try:
                if train_df[col].dtype in (int, np.int64, np.int32):
                    syn_df[col] = pd.to_numeric(syn_df[col], errors="coerce").fillna(0).astype(int)
                elif train_df[col].dtype == float:
                    syn_df[col] = pd.to_numeric(syn_df[col], errors="coerce").fillna(0.0)
            except (ValueError, TypeError):
                pass

    return syn_df
