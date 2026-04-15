"""MarginalRF reconstruction attack.

Combines a Random Forest's per-feature posterior probabilities with pairwise
marginal constraints learned from the synthetic data, using belief propagation
on a graphical model built from synth pairwise mutual information.

Algorithm
---------
1. **RF phase**: Train one RandomForest per hidden feature on synth (QI →
   hidden).  Collect per-target log-posterior probabilities
   ``log p_k(v | qi_target)`` for each feature k and each of its classes v.

2. **Structure learning**: Build a graph over the hidden features weighted by
   pairwise mutual information from synth.  Only features with cardinality ≤
   ``max_pair_cardinality`` participate; the rest use RF posteriors unchanged.
   Three graph types are supported (``graph_type`` param):

   - ``"mst"`` *(default)*: maximum spanning tree — guarantees a cycle-free
     graph and enables exact sum-product BP in a single forward/backward pass.
   - ``"complete"``: fully-connected graph — captures all pairwise interactions
     at the cost of approximate (loopy) BP.
   - ``"topk"``: keep only the ``top_k_edges`` highest-MI edges — a middle
     ground that may preserve important non-tree edges while keeping the graph
     sparse.  When ``top_k_edges`` is ``None`` it defaults to 2 × |features|.

3. **Pairwise log-PMI tables**: For each graph edge (k, l) and each target t,
   compute a *local* log-PMI table from the K nearest synth records to t (by
   QI distance):
       log_pmi_t[v, w] = log P_local_t(k=v, l=w)
                       − log P_local_t(k=v) − log P_local_t(l=w)
   This approximates the *conditional* PMI given QI ≈ qi_t, removing the
   QI-mediated correlation that the RF already captures.

   When ``knn_k = None`` (global mode), the global synth marginal is used
   instead.

4. **Belief propagation**:
   - MST → exact sum-product BP (one forward + one backward pass).
   - Complete / topk → loopy BP with damped iterative message passing
     (``lbp_max_iter`` iterations, ``lbp_damping`` step-size ∈ (0, 1]).

5. **Decode**: argmax of normalised log-beliefs → final predicted class.

Key insight
-----------
Even when RF assigns 60 % to value A and 40 % to value B for some feature,
the pairwise synth structure may push the refined belief toward B because (B,
C) is much more common among synth records with similar QI to this target —
a signal the independent RF cannot see.

When to expect improvement
--------------------------
The BP correction helps when hidden features have residual correlation BEYOND
what the QI already explains.  It is most beneficial with:
  - Weak or small QI sets (QI_tiny)
  - Hidden features that have strong joint patterns not driven by QI
  - Local marginal mode (``knn_k > 0``) to avoid double-counting QI signal
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

from .ML_classifiers import _encode_qi


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_RF_N_ESTIMATORS_DEFAULT = 25
_RF_MAX_DEPTH_DEFAULT    = 25
_MAX_PAIR_CARD_DEFAULT   = 50   # features with more classes skip pairwise correction
_LAPLACE_ALPHA_DEFAULT   = 1e-6 # smoothing for empirical joint tables
_KNN_K_DEFAULT           = 100  # synth neighbours per target for local marginals
_GRAPH_TYPE_DEFAULT      = "mst"  # "mst" | "complete" | "topk"
_TOP_K_EDGES_DEFAULT     = None   # for graph_type="topk"; None → 2 × |features|
_LBP_MAX_ITER_DEFAULT    = 20
_LBP_DAMPING_DEFAULT     = 0.5   # step-size for loopy BP: 1.0=full update, 0.0=no update


# ---------------------------------------------------------------------------
# Mutual information helpers (for MST structure learning)
# ---------------------------------------------------------------------------

def _mutual_info_from_counts(col_a: pd.Series, col_b: pd.Series) -> float:
    """Estimate MI(col_a, col_b) from empirical counts."""
    joint = pd.crosstab(col_a.astype(str), col_b.astype(str)).to_numpy().astype(float)
    total = joint.sum()
    if total == 0:
        return 0.0
    M     = joint / total
    pa    = M.sum(axis=1, keepdims=True)
    pb    = M.sum(axis=0, keepdims=True)
    outer = pa * pb
    mask  = (M > 0) & (outer > 0)
    return max(float(np.sum(M[mask] * np.log(M[mask] / outer[mask]))), 0.0)


# ---------------------------------------------------------------------------
# MST structure learning
# ---------------------------------------------------------------------------

def _build_mst(synth: pd.DataFrame, features: list) -> nx.Graph:
    """Maximum spanning tree over ``features`` weighted by pairwise MI.

    Guarantees a cycle-free graph → exact BP in one forward/backward pass.
    """
    G = nx.Graph()
    G.add_nodes_from(features)
    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            mi = _mutual_info_from_counts(synth[fa], synth[fb])
            G.add_edge(fa, fb, weight=mi)
    return nx.maximum_spanning_tree(G, weight="weight")


def _build_complete_graph(synth: pd.DataFrame, features: list) -> nx.Graph:
    """Complete graph over ``features`` weighted by pairwise MI.

    Every pair of features is connected, capturing all pairwise interactions.
    Requires loopy BP (approximate) rather than exact BP.
    """
    G = nx.Graph()
    G.add_nodes_from(features)
    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            mi = _mutual_info_from_counts(synth[fa], synth[fb])
            G.add_edge(fa, fb, weight=mi)
    return G


def _build_topk_graph(synth: pd.DataFrame, features: list, k: int) -> nx.Graph:
    """Sparse graph keeping only the ``k`` highest-MI edges.

    May have cycles (if k > |features| − 1), in which case loopy BP is needed.
    ``k`` defaults to 2 × |features| when called with the sentinel ``None``.
    """
    if k is None:
        k = 2 * len(features)
    G = nx.Graph()
    G.add_nodes_from(features)
    edges = []
    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            mi = _mutual_info_from_counts(synth[fa], synth[fb])
            edges.append((mi, fa, fb))
    edges.sort(reverse=True)
    for mi, fa, fb in edges[:k]:
        G.add_edge(fa, fb, weight=mi)
    return G


# ---------------------------------------------------------------------------
# Pairwise log-PMI computation
# ---------------------------------------------------------------------------

def _log_pmi_from_counts(
    count_table: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Given a raw count table (C_a, C_b), return log-PMI array.

    log_pmi[i, j] = log P(a=i, b=j) − log P(a=i) − log P(b=j)

    Rows/columns must already be aligned to the RF class ordering for the
    two features.  Laplace smoothing of ``alpha`` avoids log(0).
    """
    M  = count_table.astype(float) + alpha
    M  = M / M.sum()
    pa = M.sum(axis=1, keepdims=True)   # (C_a, 1)
    pb = M.sum(axis=0, keepdims=True)   # (1, C_b)
    return np.log(M) - np.log(pa) - np.log(pb)  # (C_a, C_b)


def _global_log_pmi_table(
    synth: pd.DataFrame,
    feat_a: str,
    classes_a: np.ndarray,
    feat_b: str,
    classes_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Global (unconditional) log-PMI table from all of synth.

    Shape: (|classes_a|, |classes_b|).
    """
    joint = pd.crosstab(synth[feat_a].astype(str), synth[feat_b].astype(str))
    counts = (
        joint.reindex(index=classes_a, columns=classes_b, fill_value=0)
        .to_numpy()
    )
    return _log_pmi_from_counts(counts, alpha)


def _local_log_pmi_tables(
    synth: pd.DataFrame,
    feat_a: str,
    classes_a: np.ndarray,
    feat_b: str,
    classes_b: np.ndarray,
    knn_indices: np.ndarray,   # (n_targets, K)
    alpha: float,
) -> np.ndarray:
    """Per-target local log-PMI tables using K nearest synth neighbours.

    Returns shape: (n_targets, |classes_a|, |classes_b|).
    """
    n_targets = knn_indices.shape[0]
    Ca, Cb    = len(classes_a), len(classes_b)

    # Build integer codes aligned to RF class orderings
    ca_map = {str(v): i for i, v in enumerate(classes_a)}
    cb_map = {str(v): i for i, v in enumerate(classes_b)}
    va_all = np.array([ca_map.get(str(x), -1) for x in synth[feat_a]], dtype=np.int32)
    vb_all = np.array([cb_map.get(str(x), -1) for x in synth[feat_b]], dtype=np.int32)

    result = np.empty((n_targets, Ca, Cb))
    for t in range(n_targets):
        nbr = knn_indices[t]
        va  = va_all[nbr]
        vb  = vb_all[nbr]
        # Count table for these K neighbours
        counts = np.zeros((Ca, Cb), dtype=float)
        for i, j in zip(va, vb):
            if 0 <= i < Ca and 0 <= j < Cb:
                counts[i, j] += 1.0
        result[t] = _log_pmi_from_counts(counts, alpha)

    return result


# ---------------------------------------------------------------------------
# Belief propagation on a tree
# ---------------------------------------------------------------------------

def _bp_tree(
    tree: nx.Graph,
    log_unaries: dict,          # feat → (n_targets, C_feat)
    log_pmi: dict,              # (fa, fb) → (n_targets, C_a, C_b)  OR  (C_a, C_b)
) -> dict:
    """Exact sum-product belief propagation on a tree.

    ``log_pmi`` values may be either 2-D (global, shared across targets) or
    3-D (local, one table per target).  Both orientations are handled
    transparently.

    Returns log_beliefs: feat → (n_targets, C_feat) normalised.
    """
    if len(tree.nodes) == 0:
        return {}
    if len(tree.nodes) == 1:
        feat  = next(iter(tree.nodes))
        u     = log_unaries[feat]
        log_z = logsumexp(u, axis=1, keepdims=True)
        return {feat: u - log_z}

    root      = next(iter(tree.nodes))
    bfs_order = list(nx.bfs_tree(tree, root).nodes())
    parent    = {root: None}
    children  = {n: [] for n in tree.nodes}
    for node in bfs_order:
        for nbr in tree.neighbors(node):
            if nbr not in parent:
                parent[nbr] = node
                children[node].append(nbr)

    beliefs  = {f: log_unaries[f].copy() for f in tree.nodes}
    messages = {}   # (sender, receiver) → (n_targets, C_receiver)

    def _get_pmi(fa: str, fb: str) -> np.ndarray:
        """Return log_pmi so that result[..., va, vb] = PMI(fa=va, fb=vb)."""
        if (fa, fb) in log_pmi:
            return log_pmi[(fa, fb)]
        arr = log_pmi[(fb, fa)]
        # Transpose last two axes (works for both 2-D and 3-D)
        return arr.swapaxes(-2, -1)

    def _send_message(b_sender: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Compute m_{sender→recv}(vr) = logsumexp_{vs}[b_sender(vs) + J(vs,vr)].

        b_sender : (n_targets, C_sender)
        J        : (C_sender, C_recv)  OR  (n_targets, C_sender, C_recv)
        returns  : (n_targets, C_recv)
        """
        if J.ndim == 2:
            # Global J: broadcast over target dimension
            # b_sender[:, :, None] + J[None, :, :] → (T, C_s, C_r)
            return logsumexp(b_sender[:, :, None] + J[None, :, :], axis=1)
        else:
            # Local J: (T, C_s, C_r), b_sender: (T, C_s)
            return logsumexp(b_sender[:, :, None] + J, axis=1)

    def _recv_message(b_recv: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Compute m_{recv→sender}(vs) = logsumexp_{vr}[b_recv(vr) + J(vs,vr)].

        This is the backward message from receiver node to sender node.
        b_recv : (n_targets, C_recv)
        J      : (C_sender, C_recv)  OR  (n_targets, C_sender, C_recv)
        returns: (n_targets, C_sender)
        """
        if J.ndim == 2:
            # b_recv[:, None, :] + J[None, :, :] → (T, C_s, C_r); sum over C_r
            return logsumexp(b_recv[:, None, :] + J[None, :, :], axis=2)
        else:
            return logsumexp(b_recv[:, None, :] + J, axis=2)

    # Forward pass: leaves → root
    for node in reversed(bfs_order):
        par = parent[node]
        if par is None:
            continue
        J       = _get_pmi(node, par)          # (C_node, C_par) or (T, C_node, C_par)
        log_msg = _send_message(beliefs[node], J)
        messages[(node, par)] = log_msg
        beliefs[par] = beliefs[par] + log_msg

    # Backward pass: root → leaves
    for node in bfs_order:
        for child in children[node]:
            b_excl  = beliefs[node] - messages[(child, node)]
            J       = _get_pmi(child, node)    # (C_child, C_node) or (T, C_child, C_node)
            log_msg = _recv_message(b_excl, J)
            messages[(node, child)] = log_msg
            beliefs[child] = beliefs[child] + log_msg

    # Normalise
    log_beliefs = {}
    for feat in tree.nodes:
        b     = beliefs[feat]
        log_z = logsumexp(b, axis=1, keepdims=True)
        log_beliefs[feat] = b - log_z

    return log_beliefs


# ---------------------------------------------------------------------------
# Loopy belief propagation (for graphs with cycles)
# ---------------------------------------------------------------------------

def _lbp_loopy(
    graph: nx.Graph,
    log_unaries: dict,      # feat → (n_targets, C_feat)
    log_pmi: dict,          # (fa, fb) → (n_targets, C_a, C_b)  OR  (C_a, C_b)
    max_iter: int,
    damping: float,         # step-size: 1.0=full update each iter, 0.0=no update
    tol: float = 1e-4,
) -> dict:
    """Loopy sum-product belief propagation via damped iterative message passing.

    Runs ``max_iter`` rounds of parallel message updates with damping.  On a
    tree this converges in one pass to the same result as ``_bp_tree``; on
    graphs with cycles the result is approximate.

    ``damping`` ∈ (0, 1]:
        new_msg = damping * computed + (1 − damping) * old_msg
    A value of 0.5 is a safe default; use lower values if divergence is observed.

    Returns log_beliefs: feat → (n_targets, C_feat) normalised.
    """
    if len(graph.nodes) == 0:
        return {}
    if len(graph.nodes) == 1:
        feat  = next(iter(graph.nodes))
        u     = log_unaries[feat]
        log_z = logsumexp(u, axis=1, keepdims=True)
        return {feat: u - log_z}

    def _get_pmi(fa: str, fb: str) -> np.ndarray:
        if (fa, fb) in log_pmi:
            return log_pmi[(fa, fb)]
        arr = log_pmi[(fb, fa)]
        return arr.swapaxes(-2, -1)

    nodes = list(graph.nodes)

    # Initialise all directed messages to zero (log domain → uniform)
    messages: dict = {}
    for u in nodes:
        for v in graph.neighbors(u):
            C_v = log_unaries[v].shape[1]
            n_t = log_unaries[v].shape[0]
            messages[(u, v)] = np.zeros((n_t, C_v))

    for _ in range(max_iter):
        new_messages: dict = {}
        max_delta = 0.0

        for u in nodes:
            nbrs = list(graph.neighbors(u))
            # Accumulate incoming messages into the belief at u
            belief_u = log_unaries[u].copy()
            for w in nbrs:
                belief_u = belief_u + messages[(w, u)]

            for v in nbrs:
                # Outgoing message u→v: exclude v's contribution to belief_u
                b_excl = belief_u - messages[(v, u)]
                J = _get_pmi(u, v)   # (C_u, C_v) or (T, C_u, C_v)
                if J.ndim == 2:
                    raw = logsumexp(b_excl[:, :, None] + J[None, :, :], axis=1)
                else:
                    raw = logsumexp(b_excl[:, :, None] + J, axis=1)
                # Log-normalise for numerical stability
                raw = raw - logsumexp(raw, axis=1, keepdims=True)
                new_messages[(u, v)] = raw

        # Apply damping and track convergence
        for key, new_msg in new_messages.items():
            delta = float(np.abs(new_msg - messages[key]).max())
            if delta > max_delta:
                max_delta = delta
            messages[key] = damping * new_msg + (1.0 - damping) * messages[key]

        if max_delta < tol:
            break

    # Compute final beliefs
    log_beliefs: dict = {}
    for feat in nodes:
        b = log_unaries[feat].copy()
        for nbr in graph.neighbors(feat):
            b = b + messages[(nbr, feat)]
        log_z = logsumexp(b, axis=1, keepdims=True)
        log_beliefs[feat] = b - log_z

    return log_beliefs


# ---------------------------------------------------------------------------
# Main attack function
# ---------------------------------------------------------------------------

def marginal_rf_reconstruction(cfg, deid, targets, qi, hidden_features):
    """Reconstruction attack combining RF posteriors with synth pairwise marginals.

    Attack params (set via cfg["attack_params"]):
        num_estimators (int, default 25): RF trees.
        max_depth (int, default 25): RF max depth.
        max_pair_cardinality (int, default 50):
            Hidden features with more unique values than this threshold are
            excluded from the graph and use their RF posteriors unchanged.
        knn_k (int | None, default 100):
            Number of nearest synth neighbours for local (conditional) log-PMI
            tables per target.  None → global (unconditional) PMI.
        alpha (float, default 1e-6):
            Laplace smoothing for empirical joint probability tables.
        graph_type (str, default "mst"):
            Pairwise graph structure over eligible hidden features.
            "mst"      → maximum spanning tree (exact BP, one pass).
            "complete" → fully-connected graph (loopy BP).
            "topk"     → keep top_k_edges highest-MI edges (loopy BP).
        top_k_edges (int | None, default None):
            Edge budget for graph_type="topk".  None → 2 × |features|.
        lbp_max_iter (int, default 20):
            Maximum loopy BP iterations (ignored for "mst").
        lbp_damping (float, default 0.5):
            Step-size for loopy BP: 1.0 = full update, 0.0 = no update.
            Lower values improve stability on dense graphs.

    Returns
    -------
    reconstructed_df : pd.DataFrame
    probas : list[np.ndarray | None]
    classes_ : list[np.ndarray | None]
    """
    params       = cfg["attack_params"]
    n_estimators = params.get("num_estimators",       _RF_N_ESTIMATORS_DEFAULT)
    max_depth    = params.get("max_depth",            _RF_MAX_DEPTH_DEFAULT)
    max_card     = params.get("max_pair_cardinality", _MAX_PAIR_CARD_DEFAULT)
    knn_k        = params.get("knn_k",                _KNN_K_DEFAULT)
    alpha        = params.get("alpha",                _LAPLACE_ALPHA_DEFAULT)
    graph_type   = params.get("graph_type",           _GRAPH_TYPE_DEFAULT)
    top_k_edges  = params.get("top_k_edges",          _TOP_K_EDGES_DEFAULT)
    lbp_max_iter = params.get("lbp_max_iter",         _LBP_MAX_ITER_DEFAULT)
    lbp_damping  = params.get("lbp_damping",          _LBP_DAMPING_DEFAULT)

    # ── Step 1: RF posteriors ──────────────────────────────────────────────
    enc     = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = enc.fit_transform(deid[qi].astype(str))
    X_test  = enc.transform(targets[qi].astype(str))

    rf_probas:  dict = {}   # feat → (n_targets, C)
    rf_classes: dict = {}   # feat → string class labels

    for feat in hidden_features:
        y = deid[feat].astype(str)
        if y.nunique() < 2:
            rf_probas[feat]  = None
            rf_classes[feat] = None
            continue
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y)
        rf_probas[feat]  = model.predict_proba(X_test)
        rf_classes[feat] = model.classes_

    # ── Step 2: Identify tree-eligible features ────────────────────────────
    tree_features = [
        f for f in hidden_features
        if rf_probas.get(f) is not None and len(rf_classes[f]) <= max_card
    ]

    # ── Step 3: Build graph over eligible features ────────────────────────
    log_pmi_tables: dict = {}
    graph = nx.Graph()

    if len(tree_features) >= 2:
        if graph_type == "mst":
            graph = _build_mst(deid, tree_features)
        elif graph_type == "complete":
            graph = _build_complete_graph(deid, tree_features)
        elif graph_type == "topk":
            graph = _build_topk_graph(deid, tree_features, top_k_edges)
        else:
            raise ValueError(
                f"Unknown graph_type={graph_type!r}. Choose 'mst', 'complete', or 'topk'."
            )

        # ── Step 3a: KNN in QI space for local marginals ───────────────────
        if knn_k is not None:
            effective_k = min(knn_k, len(deid))
            knn         = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
            knn.fit(X_train)
            _, knn_indices = knn.kneighbors(X_test)  # (n_targets, effective_k)
        else:
            knn_indices = None

        # ── Step 3b: Compute log-PMI tables for each graph edge ────────────
        for fa, fb in graph.edges():
            ca, cb = rf_classes[fa], rf_classes[fb]
            n_cells = len(ca) * len(cb)
            # Local PMI requires on average ≥ 5 counts per cell; otherwise
            # the empirical joint is too noisy and we fall back to global.
            use_local = (
                knn_indices is not None
                and effective_k >= 5 * n_cells
            )
            if use_local:
                log_pmi_tables[(fa, fb)] = _local_log_pmi_tables(
                    deid, fa, ca, fb, cb, knn_indices, alpha
                )  # (n_targets, Ca, Cb)
            else:
                log_pmi_tables[(fa, fb)] = _global_log_pmi_table(
                    deid, fa, ca, fb, cb, alpha
                )  # (Ca, Cb)

    # ── Step 4: Belief propagation ─────────────────────────────────────────
    log_unaries: dict = {}
    for feat in tree_features:
        p = np.clip(rf_probas[feat], 1e-12, 1.0)
        log_unaries[feat] = np.log(p)

    if graph_type == "mst":
        log_beliefs = _bp_tree(graph, log_unaries, log_pmi_tables)
    else:
        log_beliefs = _lbp_loopy(graph, log_unaries, log_pmi_tables,
                                  max_iter=lbp_max_iter, damping=lbp_damping)

    # ── Step 5: Decode and assemble output ────────────────────────────────
    recon    = targets.copy()
    probas   = []
    classes_ = []

    for feat in hidden_features:
        p_rf = rf_probas.get(feat)

        if p_rf is None:
            recon[feat] = deid[feat].mode().iloc[0]
            probas.append(None)
            classes_.append(None)
            continue

        feat_classes = rf_classes[feat]

        if feat in log_beliefs:
            lb        = log_beliefs[feat]
            idx       = np.argmax(lb, axis=1)
            proba_out = np.exp(lb)
        else:
            idx       = np.argmax(p_rf, axis=1)
            proba_out = p_rf

        pred = feat_classes[idx]
        try:
            recon[feat] = pred.astype(deid[feat].dtype)
        except (ValueError, TypeError):
            recon[feat] = pred

        probas.append(proba_out)
        classes_.append(feat_classes)

    return recon, probas, classes_
