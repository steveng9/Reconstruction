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


def _unavailable(name, err):
    """Placeholder for an SDG method whose optional dependency is missing.

    `except ImportError as _e` unbinds _e when the block ends, so a fallback
    that closes over _e directly raises NameError -- hiding the real reason the
    method is missing. Passing the exception in as an argument keeps it alive.
    """
    def _fn(*a, **kw):
        raise ImportError(f"{name} unavailable: {err}") from err
    return _fn


try:
    from .privbayes_method import privbayes_generate
except ImportError as _e:
    privbayes_generate = _unavailable("PrivBayes", _e)

try:
    from .mwem_pgm_method import mwem_pgm_generate
except ImportError as _e:
    mwem_pgm_generate = _unavailable("MWEM+PGM", _e)

try:
    from .private_gsd_method import private_gsd_generate
except ImportError as _e:
    private_gsd_generate = _unavailable("Private-GSD", _e)

try:
    from .privsyn_method import privsyn_generate
except ImportError as _e:
    privsyn_generate = _unavailable("PrivSyn", _e)

try:
    from .tvae_method import tvae_generate
except ImportError as _e:
    tvae_generate = _unavailable("TVAE (SDV API mismatch)", _e)

try:
    from .ctgan_method import ctgan_generate
except ImportError as _e:
    ctgan_generate = _unavailable("CTGAN", _e)

try:
    from .arf_method import arf_generate
except ImportError as _e:
    arf_generate = _unavailable("ARF (needs synthcity)", _e)

try:
    from .tabddpm_method import tabddpm_generate
except ImportError as _e:
    tabddpm_generate = _unavailable("TabDDPM", _e)

try:
    from .r_methods import synthpop_generate, rankswap_generate, cellsuppression_generate
except ImportError as _e:
    synthpop_generate = _unavailable("Synthpop (needs rpy2 and R)", _e)
    rankswap_generate = _unavailable("RankSwap (needs rpy2 and R)", _e)
    cellsuppression_generate = _unavailable("CellSuppression (needs rpy2 and R)", _e)


SDG_REGISTRY = {
    # Differentially private methods (epsilon parameter)
    "MST": mst_generate,
    "AIM": aim_generate,
    "PrivBayes": privbayes_generate,
    "MWEMPGM": mwem_pgm_generate,
    "PrivateGSD": private_gsd_generate,
    "PrivSyn": privsyn_generate,

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
