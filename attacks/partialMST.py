"""Partial MST reconstruction attack.

Trains an MST (Maximum Spanning Tree) model on the synthetic data, then
conditionally samples hidden features given known QI values by walking the
MST's spanning tree structure and conditioning on QI values at each step.

How it works
------------
MST learns a graphical model (a factor graph whose structure is a maximum
spanning tree over features) from differentially-private pairwise marginals.
The `GraphicalModel.synthetic_data()` method samples all columns in
elimination-order by conditioning each column on the already-sampled columns
that share a clique with it.

For the reconstruction attack we re-use that same conditional-sampling
algorithm, but instead of sampling QI columns we *fix* them to the actual
target values.  Hidden features are then sampled from their conditional
distribution given the fixed QI values.

Three attack variants
---------------------
standard (PartialMST):
    Classic MST — selects pairwise (2-way) cliques by mutual-information
    weight to form a spanning tree.  Each hidden feature conditions on at
    most one other column (its tree-parent), which may or may not be a QI.

bounded (PartialMSTBounded):
    For each hidden feature h, selects the highest-scoring subset of
    (max_clique_size - 1) QI columns and forms a clique
    (QI_i, ..., QI_j, h).  During sampling, h conditions on up to
    (max_clique_size - 1) QI values *simultaneously*.  Pairwise edges are
    added afterwards to ensure the factor graph is connected.

    max_clique_size=3 (default): each hidden col conditions on 2 QI cols.
    max_clique_size=2: pairwise QI→hidden edges (QI-forcing variant).
    max_clique_size=|QI|+1: equivalent to the "star" variant where every
    hidden col directly conditions on all QI cols.

hub (PartialMSTHub):
    One single clique over all QI columns captures the joint QI
    distribution.  Each hidden feature is then connected to its highest-MI
    QI column via a pairwise edge.  The hub is shared across all hidden
    features, so FactoredInference learns a model whose joint P respects QI
    correlations.  During sampling each hidden col still conditions on one
    QI col, but within a better-calibrated model.

    If the joint QI domain exceeds MAX_HUB_DOMAIN (1 000 000), the hub
    clique is skipped and a warning is printed.

Encoding pipeline (original → model):
    1. Pre-bin continuous columns to integer bin indices (if
       bin_continuous_as_ordinal=True, same as SDG generation).
    2. Apply LabelTransformer.forward (snsynth): maps category values to
       sorted integer codes 0 … cardinality-1.
    3. Apply compress_domain forward map: MST compresses rare categories to
       a single catch-all bin.  We store the `supports` dict during fitting
       so we can reproduce this mapping.

Decoding pipeline (model → original):
    1. Apply undo_compress_fn (GraphicalModel → label codes).
    2. Apply LabelTransformer.inverse.
    3. If the column was pre-binned, map bin index back to bin midpoint.

Checkpointing
-------------
The fitted model (a _PartialMSTSynthesizer pickle) is saved to
    {dataset_dir}/{sdg_dirname}/partial_mst_artifacts_{qi_key}[_{variant}_k{k}]/mst_model.pkl
and reused on subsequent runs unless `retrain=True` is set in attack_params.
The artifact dir includes SDG subdirectory, QI key, and variant tag so each
(sample, SDG method, QI, variant) combination has its own checkpoint.
"""

import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from snsynth.mst.mst import MSTSynthesizer
from mbi import Dataset, FactoredInference


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HUB_DOMAIN = 1_000_000  # skip hub clique if joint QI domain exceeds this


# ---------------------------------------------------------------------------
# Helpers for high-order clique selection
# ---------------------------------------------------------------------------

class _UnionFind:
    """Simple union-find (disjoint-set) for graph connectivity tracking."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def connected(self, x, y):
        return self.find(x) == self.find(y)


def _pairwise_mi_from_data(data, col_a, col_b):
    """Compute mutual information I(col_a; col_b) directly from a Dataset."""
    joint = data.project([col_a, col_b]).datavector(flatten=False)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint = joint / total
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    outer = pa * pb
    mask = (joint > 0) & (outer > 0)
    return max(float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))), 0.0)


def _clique_score(pairwise_mi, clique):
    """Score a clique by the sum of pairwise mutual informations of its members."""
    return sum(
        pairwise_mi.get((a, b), 0.0)
        for i, a in enumerate(clique)
        for b in clique[i + 1:]
    )


def _sample_from_proba(proba, n, sample_mode, top_pct):
    """Draw n samples from a (normalised) 1-D probability array.

    sample_mode="sample":
        Standard multinomial sampling — current default behaviour.

    sample_mode="argmax":
        Always return the index with the highest probability.  Deterministic.

    sample_mode="top_pct":
        Keep only the top ceil(top_pct/100 * n_values) values by probability,
        re-normalise among them, then sample.  When that ceiling equals 1 the
        result is deterministic (equivalent to argmax).  top_pct=100 reduces
        to ordinary sampling.

    Args:
        proba      : 1-D float array, already normalised (sums to ~1).
        n          : number of draws required.
        sample_mode: "sample" | "argmax" | "top_pct"
        top_pct    : float in (0, 100], used only for "top_pct" mode.

    Returns:
        np.int64 array of length n with the chosen indices.
    """
    if sample_mode == "argmax":
        return np.full(n, int(np.argmax(proba)), dtype=np.int64)

    if sample_mode == "top_pct":
        n_vals = proba.size
        k = max(1, int(np.ceil(top_pct / 100.0 * n_vals)))
        if k < n_vals:
            # argsort ascending → last k are the largest
            top_idx = np.argsort(proba)[-k:]
            proba_top = proba[top_idx].copy()
            total = proba_top.sum()
            proba_top = proba_top / total if total > 0 else np.ones(k) / k
            if k == 1:
                return np.full(n, int(top_idx[0]), dtype=np.int64)
            chosen = np.random.choice(k, n, p=proba_top)
            return top_idx[chosen].astype(np.int64)
        # k >= n_vals → fall through to ordinary sampling

    # Default: sample from the full distribution
    return np.random.choice(proba.size, n, p=proba).astype(np.int64)


# ---------------------------------------------------------------------------
# Subclass that stores the compress_domain `supports` dict
# ---------------------------------------------------------------------------

class _PartialMSTSynthesizer(MSTSynthesizer):
    """MSTSynthesizer with noiseless training and high-order clique selection.

    Set these *instance* attributes before calling fit() to activate the
    bounded or hub variants:

        qi_orig_cols   : list[str]  — original QI column names
        clique_variant : str        — "standard" | "bounded" | "hub"
        max_clique_size: int        — max nodes per clique for "bounded"
    """

    qi_orig_cols    = None
    clique_variant  = "standard"
    max_clique_size = 3
    iters           = 10000

    # ---- core MST override -------------------------------------------------

    def MST(self, data, epsilon, delta):
        """Noiseless MST — bypasses DP since we train on synthetic, not private, data.

        The standard MSTSynthesizer adds Gaussian noise scaled to the privacy
        budget (sigma ≈ 10 at epsilon=1.0, sigma ≈ 1.2 at epsilon=10.0).
        That noise is irrelevant here: the synthetic data is already public, and
        adding noise only degrades the model's fidelity to the true distribution.

        Instead we use:
          sigma = 1e-6  →  effectively noiseless measurements.
                           Non-zero counts (≥1) satisfy count >= 3*sigma, so
                           all values present in synth are "supported" in
                           compress_domain.  Zero-count values (absent from
                           synth) are correctly mapped to the catch-all bin.
          rho   = 1e6   →  makes the exponential mechanism in select() a
                           deterministic argmax, giving the true maximum
                           spanning tree on mutual-information weights.
        cdp_rho() is not called at all (it overflows above epsilon ≈ 700).
        """
        sigma = 1e-6
        rho   = 1e6

        cliques = [(col,) for col in data.domain]
        log1 = self.measure(data, cliques, sigma)
        data, log1, undo_compress_fn = self.compress_domain(data, log1)
        self.undo_compress_fn = undo_compress_fn

        if self.clique_variant == "bounded":
            selected = self._select_bounded(data)
        elif self.clique_variant == "hub":
            selected = self._select_hub(data)
        else:
            selected = self.select(data, rho / 3.0, log1)

        log2 = self.measure(data, selected, sigma)
        engine = FactoredInference(data.domain, iters=self.iters)
        self.synthesizer = engine.estimate(log1 + log2)

    # ---- high-order clique selection helpers --------------------------------

    def _qi_and_hidden_model_cols(self, data):
        """Map QI/hidden original names → model col names ('colX') in the compressed domain."""
        col_names     = list(self._transformer._columns)
        all_model_cols = list(data.domain.attrs)

        qi_model = set()
        for orig_col in (self.qi_orig_cols or []):
            if orig_col in col_names:
                idx = col_names.index(orig_col)
                mc  = f"col{idx}"
                if mc in data.domain.attrs:
                    qi_model.add(mc)

        qi_list     = sorted(qi_model)
        hidden_list = [c for c in all_model_cols if c not in qi_model]
        return qi_list, hidden_list

    def _compute_all_pairwise_mi(self, data):
        """Return dict {(colA, colB): MI} for all column pairs (symmetric)."""
        all_cols = list(data.domain.attrs)
        pmi = {}
        for i, a in enumerate(all_cols):
            for b in all_cols[i + 1:]:
                mi = _pairwise_mi_from_data(data, a, b)
                pmi[(a, b)] = pmi[(b, a)] = mi
        return pmi

    def _fill_connectivity(self, all_cols, selected, uf, pairwise_mi):
        """Add pairwise edges (greedy MST) until the factor graph is connected."""
        pairs = sorted(
            ((pairwise_mi.get((a, b), 0.0), a, b)
             for i, a in enumerate(all_cols) for b in all_cols[i + 1:]),
            reverse=True,
        )
        for _, a, b in pairs:
            if not uf.connected(a, b):
                uf.union(a, b)
                selected.append((a, b))

    def _select_bounded(self, data):
        """For each hidden col, select the best k-1 QI cols and form a k-way clique.

        Clique scoring: sum of pairwise mutual informations among all members.
        Connectivity: pairwise edges are added greedily (MST on remaining
        components) to ensure the factor graph is connected.

        With max_clique_size=3 each hidden col conditions on 2 QI cols at
        sampling time.  With max_clique_size=|QI|+1 this is the star variant
        (every hidden col directly conditions on all QI cols).
        """
        all_cols          = list(data.domain.attrs)
        qi_cols, hidden_cols = self._qi_and_hidden_model_cols(data)

        if not qi_cols or not hidden_cols:
            return self.select(data, 1e6 / 3.0, [])

        pmi              = self._compute_all_pairwise_mi(data)
        n_qi_in_clique   = min(self.max_clique_size - 1, len(qi_cols))
        selected         = []
        uf               = _UnionFind(all_cols)

        for h in hidden_cols:
            if n_qi_in_clique == len(qi_cols):
                # Use all QI cols — no need to enumerate subsets
                best_clique = tuple(sorted(qi_cols + [h]))
            else:
                best_score  = -1.0
                best_clique = None
                for qi_subset in itertools.combinations(qi_cols, n_qi_in_clique):
                    # Score by MI(h, qi_i) only — QI-QI MI is constant across
                    # subsets and would bias selection away from features
                    # that are actually informative about h.
                    score = sum(pmi.get((q, h), 0.0) for q in qi_subset)
                    if score > best_score:
                        best_score  = score
                        best_clique = tuple(sorted(qi_subset + (h,)))

            selected.append(best_clique)
            for c in best_clique:
                uf.union(best_clique[0], c)

        self._fill_connectivity(all_cols, selected, uf, pmi)
        return selected

    def _select_hub(self, data):
        """Add one hub clique over all QI cols; connect each hidden col to best QI col.

        The hub clique (QI_1, ..., QI_k) is measured once and captures the
        full joint QI distribution.  Each hidden column is then connected via a
        pairwise edge to the QI column with the highest pairwise MI.  Pairwise
        edges are added to ensure full connectivity.

        During _conditional_sample, each hidden col still conditions directly on
        only one QI col (its pairwise neighbour), but the FactoredInference
        model has been calibrated with the QI joint structure via the hub
        clique, so the learned conditional P(hidden | QI_j) reflects QI-QI
        correlations.

        If the joint QI domain size exceeds MAX_HUB_DOMAIN, the hub clique is
        skipped and a warning is printed; hidden features still connect to
        their best QI col via pairwise edges.
        """
        all_cols          = list(data.domain.attrs)
        qi_cols, hidden_cols = self._qi_and_hidden_model_cols(data)

        if not qi_cols or not hidden_cols:
            return self.select(data, 1e6 / 3.0, [])

        pmi      = self._compute_all_pairwise_mi(data)
        selected = []
        uf       = _UnionFind(all_cols)

        # Hub clique — one joint measurement over all QI columns
        hub_domain = 1
        for c in qi_cols:
            hub_domain *= int(data.domain[c])

        if hub_domain <= MAX_HUB_DOMAIN:
            hub_clique = tuple(sorted(qi_cols))
            selected.append(hub_clique)
            for c in hub_clique:
                uf.union(hub_clique[0], c)
        else:
            print(
                f"[PartialMSTHub] QI hub domain size {hub_domain:,} > "
                f"{MAX_HUB_DOMAIN:,}; skipping hub clique — "
                "falling back to pairwise QI connections."
            )

        # Each hidden col → its best QI col (pairwise edge)
        for h in hidden_cols:
            best_qi = max(qi_cols, key=lambda q: pmi.get((q, h), 0.0))
            pair    = tuple(sorted([best_qi, h]))
            selected.append(pair)
            uf.union(h, best_qi)

        self._fill_connectivity(all_cols, selected, uf, pmi)
        return selected

    # ---- picklable compress_domain -----------------------------------------

    def _undo_compress(self, data):
        """Picklable decompression function (replaces the lambda in the parent)."""
        return self.reverse_data(data, self._supports)

    def compress_domain(self, data, measurements):
        # Same logic as the parent class, but we:
        #   (a) save `supports` on self for forward encoding of QI values, and
        #   (b) return a picklable bound method instead of a lambda so the
        #       fitted model can be checkpointed with pickle.
        supports = {}
        new_measurements = []
        for Q, y, sigma, proj in measurements:
            col = proj[0]
            sup = y >= 3 * sigma
            supports[col] = sup
            if supports[col].sum() == y.size:
                new_measurements.append((Q, y, sigma, proj))
            else:
                y2 = np.append(y[sup], y[~sup].sum())
                I2 = np.ones(y2.size)
                I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
                I2 = sparse.diags(I2)
                new_measurements.append((I2, y2, sigma, proj))

        self._supports = supports
        return self.transform_data(data, supports), new_measurements, self._undo_compress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_meta(cfg):
    """Load meta.json from two levels above the dataset sample dir."""
    data_dir = Path(cfg["dataset"]["dir"])
    meta_path = data_dir.parent.parent / "meta.json"
    if not meta_path.exists():
        return {"categorical": [], "continuous": [], "ordinal": []}
    with open(meta_path) as f:
        return json.load(f)


def _sdg_dirname(cfg):
    """Derive the SDG sub-directory name from cfg (mirrors get_data._sdg_dirname)."""
    method = cfg.get("sdg_method", "unknown")
    params = cfg.get("sdg_params") or {}
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method


def _fit_mst_on_synth(synth, meta, bin_continuous_as_ordinal, n_bins,
                      qi=None, clique_variant="standard", max_clique_size=3,
                      iters=10000):
    """Fit a _PartialMSTSynthesizer on synth (noiseless — no DP).

    Returns (model, bin_edges) where bin_edges is {col -> np.array of edges}
    for columns that were pre-binned (empty dict if bin_continuous_as_ordinal
    is False or there are no continuous columns).

    Args:
        qi              : list of original QI column names (for bounded/hub)
        clique_variant  : "standard" | "bounded" | "hub"
        max_clique_size : max clique size for "bounded" variant
    """
    categorical = [c for c in meta.get("categorical", []) if c in synth.columns]
    continuous  = [c for c in meta.get("continuous",  []) if c in synth.columns]
    ordinal     = [c for c in meta.get("ordinal",     []) if c in synth.columns]

    df = synth.copy()
    bin_edges = {}

    if bin_continuous_as_ordinal and continuous:
        for col in continuous:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                bin_edges[col] = np.array([lo, hi])
                df[col] = 0
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                bin_edges[col] = edges
                df[col] = np.searchsorted(edges[1:-1], df[col].values)
        ordinal = ordinal + continuous
        continuous = []

    preprocessor_eps = 0.0 if not continuous else 0.3  # only spent if continuous cols remain

    model = _PartialMSTSynthesizer(epsilon=1.0)  # epsilon is ignored (MST() override is noiseless)
    model.qi_orig_cols    = list(qi) if qi else []
    model.clique_variant  = clique_variant
    model.max_clique_size = max_clique_size
    model.iters           = iters

    model.fit(
        df,
        preprocessor_eps=preprocessor_eps,
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=ordinal,
    )
    return model, bin_edges


def _forward_compress(supports, model_col, label_codes):
    """Map label-encoded integer codes to the compressed domain.

    Supported values get a sequential index; rare/unsupported values all map
    to the catch-all index (= number of supported values).

    Args:
        supports: dict {model_col -> bool_array}  (from _PartialMSTSynthesizer._supports)
        model_col: e.g. 'col3'
        label_codes: np.int array of LabelTransformer output codes

    Returns:
        np.int array of compressed domain codes
    """
    sup = supports[model_col]
    catch_all = int(sup.sum())
    forward_map = np.full(sup.size, catch_all, dtype=np.int64)
    forward_map[sup] = np.arange(catch_all, dtype=np.int64)
    clipped = np.clip(label_codes, 0, sup.size - 1).astype(int)
    return forward_map[clipped]


def _encode_qi(model, targets_qi, qi, bin_edges):
    """Encode QI columns from original space to compressed domain codes.

    Args:
        model: fitted _PartialMSTSynthesizer
        targets_qi: DataFrame containing at least the QI columns
        qi: list of original QI column names
        bin_edges: dict {col -> bin_edge_array} for continuous columns

    Returns:
        dict {model_col_name -> np.int array of compressed codes}
    """
    transformer = model._transformer
    col_names = list(transformer._columns)  # original column names in order

    encoded = {}
    for orig_col in qi:
        if orig_col not in col_names:
            continue
        col_idx = col_names.index(orig_col)
        t = transformer.transformers[col_idx]
        model_col = f"col{col_idx}"
        values = targets_qi[orig_col].values

        # Pre-bin continuous columns (same logic as _fit_mst_on_synth)
        if orig_col in bin_edges:
            edges = bin_edges[orig_col]
            if len(edges) == 2:
                values = np.zeros(len(values), dtype=int)
            else:
                values = np.searchsorted(edges[1:-1], values.astype(float))

        # LabelTransformer forward pass (with fallback for unseen values)
        fallback = len(t.categories) - 1
        label_codes = []
        for v in values:
            try:
                label_codes.append(t._transform(v))
            except (KeyError, ValueError):
                label_codes.append(fallback)
        label_codes = np.array(label_codes, dtype=np.int64)

        encoded[model_col] = _forward_compress(model._supports, model_col, label_codes)

    return encoded


def _conditional_sample(model, encoded_qi, n_targets,
                        sample_mode="sample", top_pct=20.0, return_probas=False):
    """Sample all columns from the GraphicalModel, fixing QI columns.

    Follows the same elimination-order traversal as GraphicalModel.synthetic_data,
    but QI columns are pre-fixed to their encoded values instead of being sampled.
    Hidden columns are drawn from their conditional distribution given all
    already-determined columns (including the fixed QI values), using the
    strategy specified by sample_mode / top_pct.

    Args:
        model        : fitted _PartialMSTSynthesizer
        encoded_qi   : dict {model_col -> np.int array(n_targets)} for QI columns
        n_targets    : number of target rows
        sample_mode  : "sample" | "argmax" | "top_pct"  (see _sample_from_proba)
        top_pct      : float in (0, 100] — top-% cutoff for "top_pct" mode
        return_probas: if True, also return per-row probability distributions
                       for each hidden column in the compressed domain

    Returns:
        Dataset in the compressed domain with all columns filled in.
        If return_probas=True, returns (Dataset, col_probas) where
        col_probas is a dict {model_col -> np.array(n_targets, n_compressed_classes)}.
    """
    est = model.synthesizer  # GraphicalModel
    order = est.elimination_order[::-1]
    cols = est.domain.attrs

    # Initialise DataFrame: QI cols pre-filled, hidden cols = 0
    data = np.zeros((n_targets, len(cols)), dtype=np.int64)
    df = pd.DataFrame(data, columns=cols)
    for model_col, codes in encoded_qi.items():
        if model_col in df.columns:
            df[model_col] = codes

    qi_model_cols = set(encoded_qi.keys())
    cliques = [set(cl) for cl in est.cliques]
    # Pre-populate with all QI columns: their values are known upfront and must
    # be available for conditioning regardless of where they appear in the
    # traversal order (QI columns that are leaf nodes in the spanning tree are
    # eliminated first → appear last in the reversed order → without this
    # initialisation, hidden features sampled before them would condition on
    # nothing and degenerate to unconditional/random sampling).
    used = set(encoded_qi.keys())

    col_probas = {} if return_probas else None

    for col in order:
        if col in qi_model_cols:
            continue  # already filled and pre-marked in used

        # Find already-used columns that share a clique with `col`
        relevant_cliques = [cl for cl in cliques if col in cl]
        if relevant_cliques:
            relevant = tuple(used.intersection(set.union(*relevant_cliques)))
        else:
            relevant = ()
        used.add(col)

        # Joint marginal over (relevant..., col); shape = (card_rel0, ..., card_col)
        marg = est.project(relevant + (col,)).datavector(flatten=False)
        col_card = int(est.domain[col])

        if len(relevant) == 0:
            proba = marg.astype(float)
            total = proba.sum()
            proba = proba / total if total > 0 else np.ones_like(proba) / proba.size
            df[col] = _sample_from_proba(proba, n_targets, sample_mode, top_pct)
            if return_probas:
                col_probas[col] = np.tile(proba, (n_targets, 1))
        else:
            # Vectorised conditional sampling: group rows by their values in
            # `relevant`, sample `col` from the conditional distribution for
            # each unique combination.  Avoids pandas groupby so row order is
            # guaranteed to be preserved.
            rel_vals = df[list(relevant)].values          # shape (n_targets, |relevant|)
            new_col = np.zeros(n_targets, dtype=np.int64)
            if return_probas:
                row_probas = np.zeros((n_targets, col_card))

            # np.unique on 2D arrays works row-wise; for single relevant col
            # the array is still 2D so we always get a consistent shape.
            unique_combos, inverse = np.unique(rel_vals, axis=0, return_inverse=True)
            for i, combo in enumerate(unique_combos):
                idx = tuple(combo) if len(combo) > 1 else int(combo[0])
                counts = marg[idx].astype(float)
                total = counts.sum()
                proba = counts / total if total > 0 else np.ones(counts.size) / counts.size
                mask = inverse == i
                new_col[mask] = _sample_from_proba(proba, int(mask.sum()), sample_mode, top_pct)
                if return_probas:
                    row_probas[mask] = proba

            df[col] = new_col
            if return_probas:
                col_probas[col] = row_probas

    dataset = Dataset(df, est.domain)
    if return_probas:
        return dataset, col_probas
    return dataset


def _decode_hidden(model, sampled_dataset, hidden_cols, bin_edges):
    """Decode hidden columns from compressed domain back to original values.

    Args:
        model: fitted _PartialMSTSynthesizer
        sampled_dataset: Dataset in compressed domain (output of _conditional_sample)
        hidden_cols: list of original column names to decode
        bin_edges: dict {col -> bin_edge_array} for pre-binned continuous columns

    Returns:
        dict {orig_col_name -> list of decoded values}
    """
    # Undo compress_domain: compressed codes → label-encoded integer codes
    decompressed = model.undo_compress_fn(sampled_dataset)
    df = decompressed.df

    transformer = model._transformer
    col_names = list(transformer._columns)

    result = {}
    for orig_col in hidden_cols:
        if orig_col not in col_names:
            continue
        col_idx = col_names.index(orig_col)
        model_col = f"col{col_idx}"
        t = transformer.transformers[col_idx]
        codes = df[model_col].values

        # Inverse LabelTransformer: integer code → original category value
        decoded = [t._inverse_transform(int(c)) for c in codes]

        # Inverse bin transform for continuous columns
        if orig_col in bin_edges:
            edges = bin_edges[orig_col]
            if len(edges) == 2:
                decoded = [float(edges[0])] * len(codes)
            else:
                n_bins = len(edges) - 1
                midpoints = (edges[:-1] + edges[1:]) / 2
                bin_indices = np.clip([int(v) for v in decoded], 0, n_bins - 1)
                decoded = list(midpoints[bin_indices])

        result[orig_col] = decoded

    return result


def _decode_probas(model, col_probas, hidden_cols, bin_edges):
    """Decode per-row compressed-domain probability distributions to original label space.

    For each hidden column, maps the compressed-domain proba array
    (shape n_targets × n_compressed_classes) to the original value space:
      - Supported compressed classes map 1-to-1 to original label codes.
      - The catch-all compressed class (if present) distributes its probability
        uniformly across all unsupported original values.

    Args:
        model     : fitted _PartialMSTSynthesizer
        col_probas: dict {model_col -> np.array(n_targets, n_compressed_classes)}
        hidden_cols: list of original column names
        bin_edges : dict {col -> bin_edge_array} for pre-binned continuous columns

    Returns:
        (probas_list, classes_list) where each element corresponds to a hidden column:
            probas_list[i]  : np.array(n_targets, n_orig_classes)
            classes_list[i] : np.array of original class label values
    """
    transformer = model._transformer
    col_names   = list(transformer._columns)
    supports    = model._supports

    probas_list  = []
    classes_list = []

    for orig_col in hidden_cols:
        if orig_col not in col_names:
            probas_list.append(None)
            classes_list.append(None)
            continue

        col_idx   = col_names.index(orig_col)
        model_col = f"col{col_idx}"
        t         = transformer.transformers[col_idx]

        if model_col not in col_probas:
            probas_list.append(None)
            classes_list.append(None)
            continue

        compressed_proba = col_probas[model_col]   # (n_targets, n_compressed)
        sup              = supports[model_col]       # bool array, length = n_orig_label_codes
        n_orig           = sup.size
        n_supported      = int(sup.sum())
        has_catchall     = (compressed_proba.shape[1] == n_supported + 1)

        supported_codes   = np.where(sup)[0]
        unsupported_codes = np.where(~sup)[0]

        # Build (n_targets, n_orig) probability array in original label-code space
        n_targets   = compressed_proba.shape[0]
        orig_proba  = np.zeros((n_targets, n_orig))

        # Each supported compressed index k → original label code supported_codes[k]
        for k, orig_code in enumerate(supported_codes):
            orig_proba[:, orig_code] += compressed_proba[:, k]

        # Catch-all bin → distribute evenly among unsupported original codes
        if has_catchall and len(unsupported_codes) > 0:
            catch_all_p = compressed_proba[:, n_supported]          # (n_targets,)
            per_unsupported = catch_all_p / len(unsupported_codes)  # (n_targets,)
            orig_proba[:, unsupported_codes] += per_unsupported[:, np.newaxis]

        # Decode original label codes → original values
        all_orig_values = np.array([t._inverse_transform(int(c)) for c in range(n_orig)])

        # For pre-binned continuous columns, map bin-index values to bin midpoints
        if orig_col in bin_edges:
            edges = bin_edges[orig_col]
            if len(edges) == 2:
                all_orig_values = np.array([float(edges[0])] * n_orig)
            else:
                n_bins    = len(edges) - 1
                midpoints = (edges[:-1] + edges[1:]) / 2
                bin_idxs  = np.clip([int(v) for v in all_orig_values], 0, n_bins - 1)
                all_orig_values = midpoints[bin_idxs]

        probas_list.append(orig_proba)
        classes_list.append(all_orig_values)

    return probas_list, classes_list


def partial_mst_independent_reconstruction(cfg, synth, targets, qi, hidden_features):
    """Partial MST reconstruction, predicting each hidden feature independently.

    Calls partial_mst_reconstruction once per hidden feature, passing only
    synth[qi + [feature]] each time.  Each MST model therefore only ever sees
    two kinds of columns: the QI columns (conditioned on) and one hidden feature
    (predicted).  There is no hidden-to-hidden chaining.

    This mirrors how ML classifiers (RandomForest, LightGBM, etc.) work: one
    model per target feature, trained only on the QI.  The resulting artifact
    directories are the same as partial_mst_reconstruction but scoped to a
    single-feature synth, so the two attacks never share a checkpoint.
    """
    reconstructed = pd.DataFrame(index=targets.index)
    all_probas  = []
    all_classes = []
    for feat in hidden_features:
        synth_sub = synth[qi + [feat]].copy()
        # Give each feature its own artifact dir by tagging the QI key with
        # the feature name — otherwise all per-feature runs share one checkpoint.
        cfg_sub = {**cfg, "QI": f"{cfg.get('QI', 'QI1')}_{feat}"}
        recon_single, feat_probas, feat_classes = partial_mst_reconstruction(
            cfg_sub, synth_sub, targets, qi, [feat]
        )
        reconstructed[feat] = recon_single[feat]
        # feat_probas/feat_classes are single-element lists (one hidden feature)
        all_probas.append(feat_probas[0] if feat_probas else None)
        all_classes.append(feat_classes[0] if feat_classes else None)
    return reconstructed, all_probas, all_classes


def _needs_training(artifact_dir, cfg):
    checkpoint = os.path.join(artifact_dir, "mst_model.pkl")
    force = cfg.get("attack_params", {}).get("retrain", False)
    return force or not os.path.exists(checkpoint)


def _artifact_dir(cfg, clique_variant, max_clique_size):
    """Build artifact directory path, including variant tag when non-standard."""
    dataset_dir = cfg["dataset"]["dir"]
    sdg_dir     = _sdg_dirname(cfg)
    qi_key      = cfg.get("QI", "QI1")
    if clique_variant == "standard":
        suffix = ""
    elif clique_variant == "hub":
        suffix = "_hub"
    else:
        suffix = f"_{clique_variant}_k{max_clique_size}"
    return os.path.join(dataset_dir, sdg_dir, f"partial_mst_artifacts_{qi_key}{suffix}")


# ---------------------------------------------------------------------------
# Main attack function
# ---------------------------------------------------------------------------

def partial_mst_reconstruction(cfg, synth, targets, qi, hidden_features):
    """Partial MST reconstruction attack.

    Trains MST on the provided synthetic data, then for each target row
    samples hidden features conditioned on the known QI values using the
    learned graphical model's joint distribution.

    attack_params
    -------------
    bin_continuous_as_ordinal (bool, default True):
        Pre-bin continuous columns to `n_bins` discrete bins before fitting.
        Should match the setting used when generating the synthetic data.
    n_bins (int, default 20):
        Number of bins for continuous columns (only used when
        bin_continuous_as_ordinal=True).
    retrain (bool, default False):
        Force retraining even if a checkpoint already exists.
    clique_variant (str, default "standard"):
        "standard" | "bounded" | "hub"  — see module docstring.
    max_clique_size (int, default 3):
        Maximum clique size for the "bounded" variant (ignored otherwise).
    """
    params = cfg.get("attack_params", {})
    bin_continuous_as_ordinal = bool(params.get("bin_continuous_as_ordinal", True))
    n_bins          = int(params.get("n_bins", 20))
    iters           = int(params.get("iters", 10000))
    clique_variant  = params.get("clique_variant", "standard")
    max_clique_size = int(params.get("max_clique_size", 3))
    sample_mode     = params.get("sample_mode", "sample")
    top_pct         = float(params.get("top_pct", 20.0))

    artifact_dir = _artifact_dir(cfg, clique_variant, max_clique_size)
    os.makedirs(artifact_dir, exist_ok=True)

    checkpoint_path = os.path.join(artifact_dir, "mst_model.pkl")

    if _needs_training(artifact_dir, cfg):
        meta = _load_meta(cfg)
        # Restrict meta to columns actually present in synth
        for key in ("categorical", "continuous", "ordinal"):
            meta[key] = [c for c in meta.get(key, []) if c in synth.columns]

        model, bin_edges = _fit_mst_on_synth(
            synth, meta, bin_continuous_as_ordinal, n_bins,
            qi=qi, clique_variant=clique_variant, max_clique_size=max_clique_size,
            iters=iters,
        )
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"model": model, "bin_edges": bin_edges}, f)
    else:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        model     = ckpt["model"]
        bin_edges = ckpt["bin_edges"]

    # Encode QI columns to compressed domain
    encoded_qi = _encode_qi(model, targets[qi], qi, bin_edges)

    # Conditional sampling of ALL columns (QI fixed, hidden sampled)
    n_targets = len(targets)
    sampled, col_probas = _conditional_sample(model, encoded_qi, n_targets,
                                              sample_mode=sample_mode, top_pct=top_pct,
                                              return_probas=True)

    # Decode hidden columns back to original values
    decoded = _decode_hidden(model, sampled, hidden_features, bin_edges)

    # Decode compressed-domain probability distributions to original label space
    probas_list, classes_list = _decode_probas(model, col_probas, hidden_features, bin_edges)

    # Build output DataFrame with correct dtypes
    reconstructed = pd.DataFrame(index=range(n_targets))
    for col in hidden_features:
        if col in decoded:
            reconstructed[col] = decoded[col]
            # Restore integer dtype for integer-coded columns
            if col in targets.columns and targets[col].dtype.kind in ("i", "u"):
                reconstructed[col] = reconstructed[col].round().astype(targets[col].dtype)
        else:
            # Fallback: use the mode from synth (shouldn't happen in practice)
            reconstructed[col] = synth[col].mode().iloc[0] if col in synth.columns else 0

    return reconstructed, probas_list, classes_list


# ---------------------------------------------------------------------------
# High-order variant entry points
# ---------------------------------------------------------------------------

def partial_mst_bounded_reconstruction(cfg, synth, targets, qi, hidden_features):
    """PartialMST with bounded high-order QI cliques (bounded variant).

    Each hidden feature is placed in a direct clique with up to
    (max_clique_size - 1) QI features, chosen by maximum sum-of-pairwise-MI
    score.  During _conditional_sample, each hidden column conditions on
    multiple QI values simultaneously, giving stronger QI signal than the
    standard pairwise MST.

    attack_params (in addition to standard PartialMST params)
    ----------------------------------------------------------
    max_clique_size (int, default 3):
        Maximum nodes per high-order clique (must be ≥ 2).
        3  → each hidden col conditions on 2 QI cols.
        2  → pairwise QI-hidden edges (QI-forced variant).
        |QI|+1 → equivalent to the "star" variant (all QI cols per hidden).
    """
    params  = cfg.get("attack_params", {})
    cfg_sub = {**cfg, "attack_params": {**params, "clique_variant": "bounded"}}
    return partial_mst_reconstruction(cfg_sub, synth, targets, qi, hidden_features)


def partial_mst_hub_reconstruction(cfg, synth, targets, qi, hidden_features):
    """PartialMST with a single QI hub clique + pairwise hidden-QI edges (hub variant).

    One clique (QI_1, ..., QI_k) captures the full joint QI distribution.
    Each hidden feature connects to its highest-MI QI column via a pairwise
    edge.  FactoredInference calibrates the joint model with QI correlations
    via the hub, so the conditional P(hidden | QI_j) is better informed than
    in the standard MST even though each hidden col directly conditions on
    only one QI value at sampling time.

    If the joint QI domain exceeds MAX_HUB_DOMAIN (1 000 000), the hub clique
    is skipped with a printed warning.

    attack_params: same as standard PartialMST; max_clique_size is ignored.
    """
    params  = cfg.get("attack_params", {})
    cfg_sub = {**cfg, "attack_params": {**params, "clique_variant": "hub"}}
    return partial_mst_reconstruction(cfg_sub, synth, targets, qi, hidden_features)
