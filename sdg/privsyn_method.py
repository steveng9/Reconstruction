"""
PrivSyn synthetic data generation (Zhang, Wang, Li, Honorio, Backes, He, Chen, Zhang.
"PrivSyn: Differentially Private Data Synthesis", USENIX Security 2021,
https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf).

Unlike MST/AIM/MWEM+PGM (all graphical-model / Private-PGM-family methods already in
this repo, which fit a junction-tree-based probabilistic graphical model over noisy
marginals and then sample from it), PrivSyn uses a completely different synthesis
mechanism: it directly privately measures a workload of low-order (here: all 1-way and
2-way) marginals, makes them mutually consistent, and then iteratively "gradually
updates" (GUM = Gradually Update Method) a synthetic record set — swapping individual
synthetic records' attribute values — so that its own empirical marginals converge to
the noisy target marginals. There is no graphical model, no junction tree, and no
adversarial/generator-discriminator training.

The GUM record-synthesis algorithm (View/Consistenter/RecordSynthesizer) is vendored
from the reference implementation bundled in the SynMeter benchmark repo
(https://github.com/zealscott/SynMeter, Apache License 2.0) — see
sdg/_privsyn_vendor.py for details/attribution. That vendored code only implements the
*mechanics* of consistency-enforcement and record-swapping; it is not itself
DP-aware. The privacy-relevant pieces — which marginals to measure, and how much
Gaussian noise to add to each one — are implemented fresh below, following the same
zCDP-composition convention already used in sdg/mwem_pgm_method.py (adding
Gaussian noise calibrated so that all measured marginals together satisfy
(epsilon, delta)-DP under zCDP composition, rather than SynMeter's own ad hoc
per-marginal-group budgeting).
"""

import itertools
import math

import numpy as np
import pandas as pd

from ._privsyn_vendor import Consistenter, RecordSynthesizer, View


def _cdp_rho(eps, delta):
    """Smallest rho such that rho-CDP implies (eps, delta)-DP (binary search).

    Self-contained port of cdp2adp.cdp_rho (Steinke, https://arxiv.org/abs/2004.00010).
    Copied verbatim from sdg/mwem_pgm_method.py to keep this file independently
    importable.
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


def _one_hot(marginal_attrs, attr_index_map, num_attrs):
    onehot = [0] * num_attrs
    for attr in marginal_attrs:
        onehot[attr_index_map[attr]] = 1
    return onehot


def privsyn_generate(train_df, meta, **config):
    """Generate synthetic data using PrivSyn (GUM record synthesis over consistent,
    noisy 1-way + 2-way marginals).

    Args:
        train_df: Training DataFrame (no ID column).
        meta: Dict with 'categorical', 'continuous', 'ordinal' column lists.
        **config:
            epsilon (float, default 1.0) — DP privacy budget.
            delta (float, default 1e-9).
            degree (int, default 2) — workload = all `degree`-way marginals (PrivSyn's
                own paper also restricts to 1-way + 2-way marginals for the general
                synthesis workload; higher degree is not supported here).
            max_cells (int, default 10000) — drop 2-way marginals whose joint domain
                exceeds this (mirrors sdg/mwem_pgm_method.py); always keeps all 1-way
                marginals regardless of cardinality.
            update_iterations (int, default 30) — GUM record-update rounds (PrivSyn
                paper default).
            n_bins (int, default 20) — bins for pre-discretizing continuous columns.
                Continuous columns are pre-binned into ordinal bins using the true
                (non-private) column range, and decoded back after synthesis using
                each bin's true (non-private) empirical MEAN of the training values
                that fell in it — NOT the geometric midpoint of the bin's numeric
                range, and NOT a uniformly-random point within the bin — since
                PrivSyn/GUM only ever produces discrete bin-index records.
                Geometric-midpoint decoding (the convention used in
                sdg/privbayes_method.py / sdg/mwem_pgm_method.py) turned out to be
                insufficient here: for extreme zero-inflated columns like
                capital-gain/capital-loss (>90% exact zeros, max in the tens of
                thousands), an equal-width bin 0 spans roughly [0, range/n_bins), so
                its midpoint (~half the bin width) is far from the true concentration
                at 0 — this reproduced a >3x mean inflation even at epsilon=1000
                (near-noiseless), i.e. a genuine decode-scheme bug, not DP noise.
                Decoding to the bin's true empirical mean fixes this exactly (verified
                to reproduce the real mean to machine precision at epsilon=1000).
                This still only uses non-private statistics for the (fixed, public)
                encode/decode transform, exactly like the bin-edge computation already
                does — no additional privacy loss beyond the noisy marginals below.
            num_rows (int, default len(train_df)).
            seed (int, default 0).
    Returns:
        Synthetic DataFrame with the same columns/dtypes as train_df.
    """
    epsilon = config.get("epsilon", 1.0)
    delta = config.get("delta", 1e-9)
    degree = config.get("degree", 2)
    max_cells = config.get("max_cells", 10000)
    update_iterations = config.get("update_iterations", 30)
    n_bins = config.get("n_bins", 20)
    num_rows = config.get("num_rows", len(train_df))
    seed = config.get("seed", 0)

    np.random.seed(seed)

    categorical = [c for c in meta.get("categorical", []) + meta.get("ordinal", []) if c in train_df.columns]
    continuous = [c for c in meta.get("continuous", []) if c in train_df.columns]

    # ── Encode: categoricals -> integer codes 0..k-1, continuous -> binned ordinal ──
    df = train_df.copy()
    cat_maps = {}   # col -> {code: original_value}
    bin_edges = {}  # col -> bin edge array (for continuous columns)
    domain_sizes = {}

    for col in categorical:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        cat_maps[col] = dict(enumerate(uniques))
        domain_sizes[col] = len(uniques)

    bin_decode_values = {}  # col -> array of length n_bins: true empirical per-bin mean

    for col in continuous:
        lo, hi = df[col].min(), df[col].max()
        if lo == hi:
            bin_edges[col] = np.array([lo, hi])
            df[col] = 0
            domain_sizes[col] = 1
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
            bin_edges[col] = edges
            raw_values = df[col].values
            bin_idx = np.searchsorted(edges[1:-1], raw_values)
            df[col] = bin_idx

            # Decode value per bin = true (non-private) empirical mean of the
            # training values that landed in that bin, NOT the geometric midpoint
            # (see docstring: midpoint decoding badly under/over-shoots for
            # zero-inflated skewed columns like capital-gain/capital-loss). Falls
            # back to the geometric midpoint for any bin with zero training members
            # (can happen at low n_bins-to-cardinality ratio; harmless since GUM
            # will rarely if ever assign synthetic records to an empty bin anyway).
            midpoints = (edges[:-1] + edges[1:]) / 2
            decode_vals = midpoints.copy()
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.any():
                    decode_vals[b] = raw_values[mask].mean()
            bin_decode_values[col] = decode_vals
            domain_sizes[col] = n_bins

    attr_list = list(df.columns)
    attr_index_map = {a: i for i, a in enumerate(attr_list)}
    domain_list = np.array([domain_sizes[c] for c in attr_list], dtype=np.int64)
    records = df[attr_list].to_numpy(dtype=np.uint32)
    n = len(attr_list)

    # ── Workload: all 1-way marginals (always) + all `degree`-way marginals under the
    #    cell-count cap ──
    workload = [(c,) for c in attr_list]
    if degree >= 2:
        workload += [
            cl
            for cl in itertools.combinations(attr_list, degree)
            if np.prod([domain_sizes[c] for c in cl]) <= max_cells
        ]

    # ── Privately measure every marginal in the workload, splitting the total zCDP
    #    budget evenly across all of them (Gaussian mechanism, L2 sensitivity 1 per
    #    marginal count query — removing one record changes exactly one cell by 1) ──
    rho = _cdp_rho(epsilon, delta)
    rho_per_marginal = rho / len(workload)
    sigma = math.sqrt(1.0 / (2.0 * rho_per_marginal)) if rho_per_marginal > 0 else 0.0

    attrs_view_dict = {}
    onehot_view_dict = {}
    for marginal_attrs in workload:
        onehot = _one_hot(marginal_attrs, attr_index_map, n)
        view = View(np.array(onehot), domain_list)
        view.count_records(records)  # true (non-private) counts
        if sigma > 0:
            view.count = view.count + np.random.normal(scale=sigma, size=view.count.shape)
        attrs_view_dict[marginal_attrs] = view
        onehot_view_dict[tuple(onehot)] = view

    # ── Enforce marginal consistency (same total, same lower-order projections) ──
    consistenter = Consistenter(onehot_view_dict, domain_list)
    consistenter.consist_views()

    for view in attrs_view_dict.values():
        total = np.sum(view.count)
        view.count = view.count / total if total > 0 else np.full_like(view.count, 1.0 / view.count.size)

    # ── GUM record synthesis ──
    singleton_views = {attrs[0]: view for attrs, view in attrs_view_dict.items() if len(attrs) == 1}

    synthesizer = RecordSynthesizer(attr_list, domain_list, num_rows)
    marginal_keys = list(attrs_view_dict.keys())
    synthesizer.initialize_records(marginal_keys, method="singleton", singleton_views=singleton_views)

    for update_iteration in range(update_iterations):
        synthesizer.update_alpha(update_iteration)
        sorted_error_attrs = synthesizer.update_order(update_iteration, attrs_view_dict, marginal_keys)
        for attrs in sorted_error_attrs:
            synthesizer.update_records(attrs_view_dict[attrs], update_iteration, attrs)

    synth_df = synthesizer.df.copy()

    # ── Decode back to original values/dtypes ──
    for col, mapping in cat_maps.items():
        synth_df[col] = synth_df[col].map(mapping)
    for col, edges in bin_edges.items():
        if len(edges) == 2:
            synth_df[col] = float(edges[0])
        else:
            idx = synth_df[col].clip(0, n_bins - 1).astype(int)
            synth_df[col] = bin_decode_values[col][idx]

    synth_df = synth_df[[c for c in train_df.columns if c in synth_df.columns]]
    for col in synth_df.columns:
        dtype = train_df[col].dtype
        if dtype in (np.int64, np.int32, int):
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").round().fillna(0).astype(np.int64)
        elif dtype == float:
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").astype(float)

    return synth_df.reset_index(drop=True)
