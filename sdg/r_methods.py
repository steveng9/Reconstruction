"""
R-based SDG methods using rpy2: Synthpop, RankSwap, CellSuppression.
"""

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


def _ensure_r_package(pkg_name):
    """Import an R package, raising a clear error if not installed."""
    try:
        return importr(pkg_name)
    except Exception as e:
        raise ImportError(
            f"R package '{pkg_name}' is not installed. "
            f"Install it with: R -e 'install.packages(\"{pkg_name}\")'"
        ) from e


def _py2rpy(df):
    """Convert pandas DataFrame to R DataFrame using localconverter."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.get_conversion().py2rpy(df)


def _rpy2py(r_obj):
    """Convert R object to pandas DataFrame using localconverter."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.get_conversion().rpy2py(r_obj)


def _to_factor(r_df, col):
    """Convert a column to factor in an R dataframe."""
    return ro.r(f'''
        (function(df) {{
            df${col} <- as.factor(df${col})
            df
        }})
    ''')(r_df)


def _to_numeric(r_df, col):
    """Convert a column to numeric in an R dataframe."""
    return ro.r(f'''
        (function(df) {{
            df${col} <- as.numeric(as.character(df${col}))
            df
        }})
    ''')(r_df)


def _restore_int_cols(result_df, original_df):
    """Convert columns back to int where the original was integer-typed."""
    for col in result_df.columns:
        if col in original_df.columns and original_df[col].dtype in (int, np.int64, np.int32):
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0).astype(int)
    return result_df


def synthpop_generate(train_df, meta, **config):
    """Generate synthetic data using R's synthpop package (CART method).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: method (str, default 'cart'),
                  maxfaclevels (int, default 1000 — max factor levels before synthpop refuses).
    Returns:
        Synthetic DataFrame.
    """
    method = config.get("method", "cart")
    maxfaclevels = config.get("maxfaclevels", 1000)
    synthpop = _ensure_r_package("synthpop")

    df = train_df.copy()
    cat_cols = [c for c in meta.get("categorical", []) + meta.get("ordinal", []) if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    r_df = _py2rpy(df)
    for col in cat_cols:
        r_df = _to_factor(r_df, col)

    syn_result = synthpop.syn(data=r_df, method=method, maxfaclevels=maxfaclevels)
    syn_df = _rpy2py(syn_result.rx2("syn"))

    return _restore_int_cols(syn_df, train_df)


def rankswap_generate(train_df, meta, **config):
    """Generate de-identified data using R's sdcMicro rankSwap.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: swap_features (list of column names to swap; defaults to continuous columns).
    Returns:
        De-identified DataFrame.
    """
    sdcMicro = _ensure_r_package("sdcMicro")

    swap_features = config.get("swap_features", meta.get("continuous", []))
    swap_features = [c for c in swap_features if c in train_df.columns]
    if not swap_features:
        raise ValueError("No swap_features specified and no continuous columns in meta")

    df = train_df.copy()
    non_swap = [c for c in df.columns if c not in swap_features]
    for col in non_swap:
        df[col] = df[col].astype(str)

    r_df = _py2rpy(df)
    for col in non_swap:
        r_df = _to_factor(r_df, col)
    for col in swap_features:
        r_df = _to_numeric(r_df, col)

    r_swap_vars = ro.StrVector(swap_features)
    result = sdcMicro.rankSwap(obj=r_df, variables=r_swap_vars)
    anon_df = _rpy2py(result)

    return _restore_int_cols(anon_df, train_df)


def cellsuppression_generate(train_df, meta, **config):
    """Generate de-identified data using R's sdcMicro local suppression.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config: key_vars (list of QI column names for k-anonymity; required),
                  k (int, default 6).
    Returns:
        De-identified DataFrame (rows with NAs removed).
    """
    sdcMicro = _ensure_r_package("sdcMicro")

    key_vars = config.get("key_vars")
    if key_vars is None:
        raise ValueError("key_vars must be specified for CellSuppression (list of QI columns)")
    key_vars = [c for c in key_vars if c in train_df.columns]
    k = config.get("k", 6)

    df = train_df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str)

    r_df = _py2rpy(df)
    for col in df.columns:
        r_df = _to_factor(r_df, col)

    r_key_vars = ro.StrVector(key_vars)
    sdc_obj = sdcMicro.createSdcObj(dat=r_df, keyVars=r_key_vars)
    sdc_obj = sdcMicro.localSuppression(sdc_obj, k=k)
    anon_r = sdcMicro.extractManipData(sdc_obj)
    anon_df = _rpy2py(anon_r)

    # Remove rows with NAs (suppressed cells)
    anon_df = anon_df.dropna().reset_index(drop=True)

    return _restore_int_cols(anon_df, train_df)
