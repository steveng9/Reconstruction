"""
SDG (Synthetic Data Generation) registry.

Provides a unified interface to all synthetic data generation methods.
Each method has the signature:

    generate(train_df, meta, **config) -> pd.DataFrame

where:
    train_df: pandas DataFrame of training data (no ID column)
    meta: dict with keys 'categorical', 'continuous', 'ordinal' -> lists of column names
    **config: method-specific parameters (epsilon, k, key_vars, etc.)
"""

from .smartnoise_methods import mst_generate, aim_generate
from .tvae_method import tvae_generate
from .ctgan_method import ctgan_generate
from .arf_method import arf_generate
from .tabddpm_method import tabddpm_generate
from .r_methods import synthpop_generate, rankswap_generate, cellsuppression_generate


SDG_REGISTRY = {
    # Differentially private methods (epsilon parameter)
    "MST": mst_generate,
    "AIM": aim_generate,

    # Deep generative models
    "TVAE": tvae_generate,
    "CTGAN": ctgan_generate,
    "ARF": arf_generate,
    "TabDDPM": tabddpm_generate,

    # R-based methods
    "Synthpop": synthpop_generate,

    # De-identification techniques
    "RankSwap": rankswap_generate,
    "CellSuppression": cellsuppression_generate,
}


def get_sdg(name):
    """Get SDG function by name from registry.

    Args:
        name: Name of the SDG method (e.g., "MST", "TVAE", "Synthpop").

    Returns:
        Generate function with signature (train_df, meta, **config) -> pd.DataFrame.

    Raises:
        KeyError: If name is not found in registry.
    """
    if name not in SDG_REGISTRY:
        available = sorted(SDG_REGISTRY.keys())
        raise KeyError(
            f"SDG method '{name}' not found. Available: {', '.join(available)}"
        )
    return SDG_REGISTRY[name]


def list_sdg():
    """Return sorted list of available SDG method names."""
    return sorted(SDG_REGISTRY.keys())


def sdg_dirname(method, params=None):
    """Derive the canonical directory name for an SDG method + params.

    Examples:
        sdg_dirname("MST", {"epsilon": 1.0})   -> "MST_eps1"
        sdg_dirname("MST", {"epsilon": 0.1})   -> "MST_eps0.1"
        sdg_dirname("TVAE")                     -> "TVAE"
        sdg_dirname("TVAE", {})                 -> "TVAE"
    """
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method


__all__ = ["SDG_REGISTRY", "get_sdg", "list_sdg", "sdg_dirname"]
