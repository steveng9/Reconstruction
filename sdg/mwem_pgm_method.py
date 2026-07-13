"""
MWEM+PGM synthetic data generation (McKenna et al., adaptive-measurement variant of
MWEM described in the original Private-PGM paper, sec 3.3: https://arxiv.org/pdf/1012.4763.pdf).

Adapted from the reference implementation in the private-pgm repo
(https://github.com/ryan112358/private-pgm, mechanisms/mwem+pgm.py) to the
measurement-tuple API of the `mbi.FactoredInference` version pinned in this repo
(the same `mbi` package used by attacks/partialMST.py), rather than the newer
`mbi.estimation`/`LinearMeasurement` API used in more recent private-pgm checkouts.

Each round: privately select (via the exponential mechanism) the workload clique
whose current model estimate is worst, privately measure its true marginal with
Gaussian noise, and re-fit a FactoredInference graphical model on all measurements
so far. After all rounds, sample synthetic rows from the final model.
"""

import itertools
import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax

from mbi import Dataset, Domain, FactoredInference
from mbi.junction_tree import JunctionTree


def _cdp_rho(eps, delta):
    """Smallest rho such that rho-CDP implies (eps, delta)-DP (binary search).

    Self-contained port of cdp2adp.cdp_rho (Steinke, https://arxiv.org/abs/2004.00010),
    dropping the matplotlib import which that module pulls in unnecessarily.
    """
    def cdp_delta(rho, eps):
        if rho == 0:
            return 0.0
        amin, amax = 1.01, (eps + 1) / (2 * rho) + 2
        for _ in range(1000):
            alpha = (amin + amax) / 2
            derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
            if derivative < 0:
                amin = alpha
            else:
                amax = alpha
        return min(
            math.exp((alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)) / (alpha - 1.0),
            1.0,
        )

    if delta >= 1:
        return 0.0
    rhomin, rhomax = 0.0, eps + 1
    for _ in range(1000):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin


def _hypothetical_model_size(domain, cliques):
    jtree = JunctionTree(domain, cliques)
    maximal_cliques = jtree.maximal_cliques()
    cells = sum(domain.size(cl) for cl in maximal_cliques)
    return cells * 8 / 2**20  # MB, assuming float64 potentials


def _worst_approximated(true_answers, model, candidates, exp_eps):
    errors = np.array([
        np.abs(true_answers[cl] - model.project(cl).datavector()).sum() - model.domain.size(cl)
        for cl in candidates
    ])
    prob = softmax(0.5 * exp_eps / 1.0 * (errors - errors.max()))
    return candidates[np.random.choice(len(candidates), p=prob)]


def mwem_pgm_generate(train_df, meta, **config):
    """Generate synthetic data using MWEM+PGM.

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config:
            epsilon (float, default 1.0) — DP privacy budget.
            delta (float, default 1e-9).
            degree (int, default 2) — workload = all `degree`-way marginals.
            max_cells (int, default 10000) — drop workload cliques with a bigger domain.
            rounds (int, default None) — MWEM rounds; None -> one per column.
            maxsize_mb (float, default 25) — cap on graphical-model size (MB).
            pgm_iters (int, default 1000) — FactoredInference optimization iterations.
            n_bins (int, default 20) — bins for pre-discretizing continuous columns
                (mirrors sdg/smartnoise_methods.py's bin_continuous_as_ordinal).
            num_rows (int, default len(train_df)).
            seed (int, default 0).
    Returns:
        Synthetic DataFrame with the same columns/dtypes as train_df.
    """
    epsilon = config.get("epsilon", 1.0)
    delta = config.get("delta", 1e-9)
    degree = config.get("degree", 2)
    max_cells = config.get("max_cells", 10000)
    rounds = config.get("rounds", None)
    maxsize_mb = config.get("maxsize_mb", 25)
    pgm_iters = config.get("pgm_iters", 1000)
    n_bins = config.get("n_bins", 20)
    num_rows = config.get("num_rows", len(train_df))
    seed = config.get("seed", 0)

    np.random.seed(seed)

    categorical = [c for c in meta.get("categorical", []) + meta.get("ordinal", []) if c in train_df.columns]
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]

    # ── Encode: categoricals -> integer codes 0..k-1, continuous -> binned ordinal ──
    df = train_df.copy()
    cat_maps = {}      # col -> {code: original_value}
    bin_edges = {}      # col -> bin edge array (for continuous columns)

    for col in categorical:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        cat_maps[col] = dict(enumerate(uniques))

    for col in continuous:
        lo, hi = df[col].min(), df[col].max()
        if lo == hi:
            bin_edges[col] = np.array([lo, hi])
            df[col] = 0
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
            bin_edges[col] = edges
            df[col] = np.searchsorted(edges[1:-1], df[col].values)

    domain_cols = list(df.columns)
    shapes = [int(df[c].nunique()) for c in domain_cols]
    domain = Domain(domain_cols, shapes)
    data = Dataset(df, domain)

    # ── Workload: all `degree`-way marginals under the cell-count cap ──
    workload = [
        cl for cl in itertools.combinations(domain_cols, degree)
        if domain.size(cl) <= max_cells
    ]
    if not workload:
        workload = [(c,) for c in domain_cols]  # fallback: 1-way marginals always fit
    if rounds is None:
        rounds = len(domain_cols)

    rho = _cdp_rho(epsilon, delta)
    rho_per_round = rho / rounds
    alpha = 0.9
    sigma = math.sqrt(0.5 / (alpha * rho_per_round))
    exp_eps = math.sqrt(8 * (1 - alpha) * rho_per_round)

    true_answers = {cl: data.project(cl).datavector() for cl in workload}

    engine = FactoredInference(domain, iters=pgm_iters, warm_start=True)
    measurements = []
    model = None
    cliques = []

    for i in range(1, rounds + 1):
        candidates = [
            cl for cl in workload
            if _hypothetical_model_size(domain, cliques + [cl]) <= maxsize_mb * i / rounds
        ]
        if not candidates:
            break
        if model is None:
            # No model yet: pick uniformly at random for the first round.
            chosen = candidates[np.random.choice(len(candidates))]
        else:
            chosen = _worst_approximated(true_answers, model, candidates, exp_eps)

        x = true_answers[chosen]
        y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma, chosen))
        cliques.append(chosen)
        model = engine.estimate(measurements)

    if model is None:
        # Degenerate case (e.g. maxsize_mb too small for even one clique): fall back
        # to independent 1-way marginals only.
        measurements = []
        for cl in [(c,) for c in domain_cols]:
            x = data.project(cl).datavector()
            y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
            measurements.append((sparse.eye(x.size), y, sigma, cl))
        model = engine.estimate(measurements)

    synth_data = model.synthetic_data(rows=num_rows)
    synth_df = synth_data.df.copy()

    # ── Decode back to original values/dtypes ──
    for col, mapping in cat_maps.items():
        synth_df[col] = synth_df[col].map(mapping)
    for col, edges in bin_edges.items():
        if len(edges) == 2:
            synth_df[col] = float(edges[0])
        else:
            midpoints = (edges[:-1] + edges[1:]) / 2
            idx = synth_df[col].clip(0, n_bins - 1).astype(int)
            synth_df[col] = midpoints[idx]

    synth_df = synth_df[[c for c in train_df.columns if c in synth_df.columns]]
    for col in synth_df.columns:
        dtype = train_df[col].dtype
        if dtype in (np.int64, np.int32, int):
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").round().fillna(0).astype(np.int64)
        elif dtype == float:
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").astype(float)

    return synth_df.reset_index(drop=True)
