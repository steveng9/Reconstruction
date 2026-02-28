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
            df$`{col}` <- as.factor(df$`{col}`)
            df
        }})
    ''')(r_df)


def _to_numeric(r_df, col):
    """Convert a column to numeric in an R dataframe."""
    return ro.r(f'''
        (function(df) {{
            df$`{col}` <- as.numeric(as.character(df$`{col}`))
            df
        }})
    ''')(r_df)


def _r_safe_name(name):
    """Replace R-problematic characters (hyphens, spaces, etc.) with underscores."""
    import re
    return re.sub(r'[^a-zA-Z0-9_.]', '_', name)


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
                  maxfaclevels (int, default 1000 — max factor levels before synthpop refuses),
                  high_card_threshold (int, default 20 — columns with more unique values than
                    this are moved to the end of the synthesis order and use polyreg instead of
                    cart, avoiding slow CART split-search over many factor levels as predictors).
    Returns:
        Synthetic DataFrame.
    """
    method = config.get("method", "cart")
    maxfaclevels = config.get("maxfaclevels", 1000)
    high_card_threshold = config.get("high_card_threshold", 20)
    synthpop = _ensure_r_package("synthpop")

    df = train_df.copy()
    cat_cols = [c for c in meta.get("categorical", []) + meta.get("ordinal", []) if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # High-cardinality categorical columns (e.g. IND=110 levels, BPL=89 levels) slow down
    # every CART model where they appear as predictors. Move them to the end so the default
    # lower-triangular predictor matrix excludes them as predictors for all other columns.
    # Use polyreg (multinomial logistic regression) for their own synthesis — faster than
    # CART for many-class responses and avoids exponential factor-level subset search.
    high_card_cols = [c for c in cat_cols if df[c].nunique() > high_card_threshold]
    if high_card_cols:
        other_cols = [c for c in df.columns if c not in high_card_cols]
        df = df[other_cols + high_card_cols]

    r_df = _py2rpy(df)
    for col in cat_cols:
        r_df = _to_factor(r_df, col)

    if high_card_cols:
        # Use "sample" (draw from observed marginal) rather than "polyreg" or "cart":
        # polyreg one-hot-encodes all factor predictors, producing ~6-18k parameters for
        # 89-110 class columns — nnet::multinom never converges. "sample" is instant and
        # preserves the marginal distribution; joint conditioning is sacrificed, but these
        # columns are already excluded as predictors (end of visit sequence), so other
        # variables are unaffected.
        method_vec = ro.StrVector([
            "sample" if c in set(high_card_cols) else method
            for c in df.columns
        ])
        syn_result = synthpop.syn(data=r_df, method=method_vec, maxfaclevels=maxfaclevels)
    else:
        syn_result = synthpop.syn(data=r_df, method=method, maxfaclevels=maxfaclevels)

    syn_df = _rpy2py(syn_result.rx2("syn"))
    # Restore original column order
    syn_df = syn_df[train_df.columns]

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
    if not key_vars:
        raise ValueError(
            "key_vars must be a non-empty list of categorical QI columns for CellSuppression. "
            "For purely continuous datasets, omit CellSuppression from SDG_JOBS entirely."
        )
    key_vars = [c for c in key_vars if c in train_df.columns]
    if not key_vars:
        raise ValueError(f"None of the specified key_vars exist in the training data columns.")
    k = config.get("k", 6)

    df = train_df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str)

    # sdcMicro evaluates column names as R expressions — hyphens become subtraction operators.
    # Rename to R-safe names before passing to R, then restore afterward.
    col_map = {col: _r_safe_name(col) for col in df.columns}
    reverse_map = {v: orig for orig, v in col_map.items()}
    df = df.rename(columns=col_map)
    safe_key_vars = [col_map.get(kv, kv) for kv in key_vars]

    r_df = _py2rpy(df)
    for col in df.columns:
        r_df = _to_factor(r_df, col)

    r_key_vars = ro.StrVector(safe_key_vars)
    sdc_obj = sdcMicro.createSdcObj(dat=r_df, keyVars=r_key_vars)
    sdc_obj = sdcMicro.localSuppression(sdc_obj, k=k)
    anon_r = sdcMicro.extractManipData(sdc_obj)
    anon_df = _rpy2py(anon_r)
    anon_df = anon_df.rename(columns=reverse_map)

    # Remove rows with NAs (suppressed cells)
    anon_df = anon_df.dropna().reset_index(drop=True)

    return _restore_int_cols(anon_df, train_df)
