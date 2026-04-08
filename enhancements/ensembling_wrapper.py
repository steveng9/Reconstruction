"""
Generic ensembling wrapper for combining multiple reconstruction attacks.
Wraps attacks without modifying their code.

Ensembling combines predictions from multiple attack methods using various
aggregation strategies (voting, averaging, weighted, stacking).
"""

import wandb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def apply_ensembling(attack_fn, cfg):
    """
    Wrapper that creates an ensemble of multiple attack methods.

    Args:
        attack_fn: Primary attack function (can be ignored if include_primary=False)
        cfg: Configuration dictionary

    Returns:
        attack_function: Either the original attack_fn or a wrapped ensemble function
    """
    ensembling_cfg = cfg["attack_params"].get("ensembling", {})

    if not ensembling_cfg.get("enabled", False):
        # No ensembling, return original attack
        return attack_fn

    # Import here to avoid circular imports
    from attacks import get_attack

    # Get configuration
    method_names = ensembling_cfg.get("methods", [])
    aggregation = ensembling_cfg.get("aggregation", "voting")
    weights = ensembling_cfg.get("weights", None)
    include_primary = ensembling_cfg.get("include_primary", False)

    if not method_names:
        print("WARNING: Ensembling enabled but no methods specified. Using primary attack only.")
        return attack_fn

    # Get data type from config
    data_type = cfg.get("data_type", "agnostic")

    # Get attack functions for all methods
    ensemble_methods = []
    ensemble_names = []

    if include_primary:
        primary_name = cfg.get("attack_method", "Unknown")
        ensemble_methods.append(attack_fn)
        ensemble_names.append(primary_name)

    for method_name in method_names:
        # Skip if it's the same as primary and we already included it
        if include_primary and method_name == cfg.get("attack_method"):
            continue

        try:
            method_fn = get_attack(method_name, data_type)
            ensemble_methods.append(method_fn)
            ensemble_names.append(method_name)
        except KeyError as e:
            print(f"WARNING: Could not load attack method '{method_name}': {e}")
            continue

    if len(ensemble_methods) == 0:
        print("WARNING: No valid methods for ensemble. Using primary attack only.")
        return attack_fn

    # Log ensembling configuration to WandB
    wandb.config.update({
        "ensembling_enabled": True,
        "ensembling_methods": ensemble_names,
        "ensembling_aggregation": aggregation,
        "ensembling_num_methods": len(ensemble_methods)
    })

    print(f"\n{'='*60}")
    print(f"ENSEMBLING ENABLED")
    print(f"  Methods: {', '.join(ensemble_names)}")
    print(f"  Aggregation: {aggregation}")
    print(f"  Number of models: {len(ensemble_methods)}")
    print(f"{'='*60}\n")

    # Create and return ensemble wrapper function
    def ensemble_attack(cfg_inner, synth, targets, qi, hidden_features):
        """Ensemble attack function that combines multiple methods."""
        return _run_ensemble(
            ensemble_methods,
            ensemble_names,
            cfg_inner,
            cfg,  # Original config with all method params
            synth,
            targets,
            qi,
            hidden_features,
            aggregation,
            weights,
            data_type
        )

    return ensemble_attack


def _run_ensemble(methods, method_names, cfg_inner, cfg_full, synth, targets, qi, hidden_features,
                  aggregation, weights, data_type):
    """
    Run ensemble of multiple attack methods and aggregate predictions.

    Args:
        methods: List of attack functions
        method_names: List of method names (for logging)
        cfg_inner: Config passed to ensemble (may have merged params)
        cfg_full: Original full config with all method-specific params
        synth: Synthetic/de-identified data
        targets: Training data with known features
        qi: List of quasi-identifier features
        hidden_features: List of features to reconstruct
        aggregation: Aggregation strategy
        weights: Optional weights for weighted aggregation
        data_type: "categorical", "continuous", or "agnostic"

    Returns:
        reconstructed, probas, classes (aggregated predictions)
    """
    from master_experiment_script import _prepare_config

    # ── Stacking takes a separate code path ──────────────────────────────────
    if aggregation == "stacking":
        ensembling_cfg = cfg_full.get("attack_params", {}).get("ensembling", {})
        return _run_stacking_ensemble(
            methods, method_names, cfg_full, synth, targets, qi, hidden_features,
            n_folds=ensembling_cfg.get("n_folds", 5),
            use_probas=ensembling_cfg.get("use_probas", True),
        )

    # Collect predictions from all methods
    all_predictions = []
    all_probas = []
    all_classes = []

    print(f"\nRunning ensemble of {len(methods)} methods...")

    for idx, (method_fn, method_name) in enumerate(zip(methods, method_names)):
        print(f"  [{idx+1}/{len(methods)}] Running {method_name}...")

        # Prepare config with method-specific params
        method_cfg = cfg_full.copy()
        method_cfg["attack_method"] = method_name
        method_cfg = _prepare_config(method_cfg)

        # Run attack
        try:
            recon, probas, classes = method_fn(method_cfg, synth, targets, qi, hidden_features)
            all_predictions.append(recon)
            all_probas.append(probas)
            all_classes.append(classes)
        except Exception as e:
            print(f"    WARNING: {method_name} failed with error: {e}")
            continue

    if len(all_predictions) == 0:
        raise RuntimeError("All ensemble methods failed!")

    print(f"  ✓ Successfully ran {len(all_predictions)} methods\n")

    # Aggregate predictions
    reconstructed = _aggregate_predictions(
        all_predictions,
        all_probas,
        all_classes,
        hidden_features,
        aggregation,
        weights,
        data_type
    )

    # For now, return None for probas/classes (could implement aggregated probas later)
    return reconstructed, None, None


def _aggregate_predictions(predictions, all_probas, all_classes, hidden_features,
                           aggregation, weights, data_type):
    """
    Aggregate predictions from multiple models.

    Args:
        predictions: List of DataFrames with predictions
        all_probas: List of probability arrays (may be None)
        all_classes: List of class arrays (may be None)
        hidden_features: List of features being predicted
        aggregation: Aggregation strategy
        weights: Optional weights for each model
        data_type: Type of data being predicted

    Returns:
        DataFrame with aggregated predictions
    """
    n_models = len(predictions)

    # Set default weights if not provided
    if weights is None:
        weights = [1.0] * n_models
    else:
        if len(weights) != n_models:
            print(f"WARNING: {len(weights)} weights provided but {n_models} models. Using equal weights.")
            weights = [1.0] * n_models

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Initialize result with first prediction
    result = predictions[0].copy()

    # Aggregate each feature
    for feat_idx, feature in enumerate(hidden_features):
        feature_predictions = [pred[feature] for pred in predictions]

        if aggregation == "voting" or aggregation == "hard_voting":
            # Hard voting: most common prediction
            result[feature] = _hard_voting(feature_predictions)

        elif aggregation == "soft_voting" or aggregation == "weighted_voting":
            result[feature] = _soft_voting(
                feature_predictions, all_probas, all_classes, feat_idx, weights
            )

        elif aggregation == "averaging" or aggregation == "mean":
            # Averaging: for continuous values
            result[feature] = _averaging(feature_predictions, weights)

        elif aggregation == "median":
            # Median: robust to outliers
            result[feature] = _median(feature_predictions)

        elif aggregation == "weighted":
            # Weighted average
            result[feature] = _averaging(feature_predictions, weights)

        elif aggregation == "confidence_routing":
            # Per-row: pick the attack with the highest max predicted probability.
            # Attacks without real probas get confidence=0.0 and only win if no
            # other attack in the combo has probas.
            result[feature] = _confidence_routing(
                feature_predictions, all_probas, all_classes, feat_idx
            )

        else:
            print(f"WARNING: Unknown aggregation strategy '{aggregation}'. Using voting.")
            result[feature] = _hard_voting(feature_predictions)

    return result


def _hard_voting(predictions):
    """
    Hard voting: return most common prediction for each row.

    Args:
        predictions: List of Series with predictions

    Returns:
        Series with majority vote
    """
    # Stack predictions and get mode (most common value)
    stacked = pd.DataFrame(predictions).T
    return stacked.mode(axis=1)[0]


def _soft_voting(predictions, all_probas, all_classes, feat_idx, weights):
    """
    Soft voting: aggregate predicted probability distributions, take argmax.

    For models that return probas + classes (RF, LGB, NaiveBayes, SVM): uses their
    probability distribution directly, aligned to a common class space.
    For models without probas (KNN, Mode, Random, etc.): converts the hard prediction
    to a one-hot distribution (probability 1.0 on predicted class).
    Models with probas but no classes (NaiveBayes classes=None): treated as no-proba
    and one-hotted from hard predictions.

    Args:
        predictions: List of Series with hard predictions (one per model)
        all_probas:  List of proba-lists (all_probas[i][feat_idx] = (n_targets, n_classes))
        all_classes: List of classes-lists (all_classes[i][feat_idx] = class label array)
        feat_idx:    Index of this feature in hidden_features
        weights:     Per-model weights (already normalised to sum=1)

    Returns:
        Series with soft-voted predictions
    """
    n_targets = len(predictions[0])

    # Gather per-model (proba_array, class_labels) or one-hot from hard pred
    model_probas = []
    model_classes = []

    for i, hard_pred in enumerate(predictions):
        feat_probas  = all_probas[i][feat_idx]  if (all_probas[i]  is not None) else None
        feat_classes = all_classes[i][feat_idx] if (all_classes[i] is not None) else None

        if feat_probas is not None and feat_classes is not None:
            model_probas.append(feat_probas)
            model_classes.append(np.asarray(feat_classes))
        else:
            # One-hot fallback: probability 1.0 on the hard-predicted class
            unique_vals = np.array(sorted(hard_pred.unique()))
            val_to_idx  = {v: j for j, v in enumerate(unique_vals)}
            ohe = np.zeros((n_targets, len(unique_vals)))
            for row, val in enumerate(hard_pred):
                ohe[row, val_to_idx[val]] = 1.0
            model_probas.append(ohe)
            model_classes.append(unique_vals)

    # Build union of all class labels seen across models (preserve original dtype)
    all_unique = sorted(set(c for cls in model_classes for c in cls),
                        key=lambda x: (str(type(x)), x))
    n_classes  = len(all_unique)
    cls_to_idx = {c: j for j, c in enumerate(all_unique)}

    # Align each model's proba to the common class space
    aligned = []
    for p, cls in zip(model_probas, model_classes):
        a = np.zeros((n_targets, n_classes))
        for j, c in enumerate(cls):
            if c in cls_to_idx:
                a[:, cls_to_idx[c]] = p[:, j]
        aligned.append(a)

    # Weighted sum over models → argmax → class label
    agg = sum(w * a for w, a in zip(weights, aligned))
    pred_indices = np.argmax(agg, axis=1)
    predicted    = np.array([all_unique[k] for k in pred_indices])

    return pd.Series(predicted, index=predictions[0].index)


def _confidence_routing(predictions, all_probas, all_classes, feat_idx):
    """
    Per-row routing: for each target record, use the prediction from whichever
    attack has the highest max predicted probability on this feature.

    Attacks without real probas (KNN, Mode, etc.) are assigned confidence=0.0
    so they only win if every attack in the combo lacks probas, in which case
    we fall back to hard voting.

    Args:
        predictions: List of Series with hard predictions (one per model)
        all_probas:  List of proba-lists (all_probas[i][feat_idx] = (n_targets, n_classes))
        all_classes: List of classes-lists (all_classes[i][feat_idx] = class label array)
        feat_idx:    Index of this feature in hidden_features

    Returns:
        Series with confidence-routed predictions
    """
    n_targets = len(predictions[0])
    n_models  = len(predictions)

    # conf[i, row] = max predicted probability for model i on this row
    conf = np.zeros((n_models, n_targets))
    any_real_probas = False

    for i in range(n_models):
        feat_probas  = all_probas[i][feat_idx]  if (all_probas[i]  is not None) else None
        feat_classes = all_classes[i][feat_idx] if (all_classes[i] is not None) else None

        if feat_probas is not None and feat_classes is not None:
            conf[i] = np.max(feat_probas, axis=1)
            any_real_probas = True
        # else: conf[i] stays 0.0 — lowest priority

    if not any_real_probas:
        return _hard_voting(predictions)

    # For each row, pick the attack with the highest confidence
    chosen = np.argmax(conf, axis=0)               # (n_targets,)
    stacked = np.stack([p.values for p in predictions])  # (n_models, n_targets)
    result_vals = stacked[chosen, np.arange(n_targets)]

    return pd.Series(result_vals, index=predictions[0].index)


def _averaging(predictions, weights):
    """
    Weighted averaging for continuous values.

    Args:
        predictions: List of Series with predictions
        weights: Model weights

    Returns:
        Series with weighted average
    """
    weighted_sum = sum(w * pred for w, pred in zip(weights, predictions))
    return weighted_sum


def _median(predictions):
    """
    Median aggregation: robust to outliers.

    Args:
        predictions: List of Series with predictions

    Returns:
        Series with median values
    """
    stacked = pd.DataFrame(predictions).T
    return stacked.median(axis=1)


def _run_stacking_ensemble(methods, method_names, cfg_full, synth, targets, qi,
                            hidden_features, n_folds=5, use_probas=True,
                            proba_max_classes=20):
    """
    Stacking ensemble: trains a per-feature meta-model on out-of-fold (OOF)
    base-model predictions from synth, then applies it to aggregate base-model
    predictions on actual targets.

    How it works
    ------------
    OOF phase (meta-model training):
      Synth is split into K folds. For each fold k, every base model trains on
      synth minus fold k and predicts on fold k records (using only their QI,
      treating the fold k records as if they were targets). Ground truth for the
      meta-model is fold k's actual hidden feature values. This gives N labeled
      examples (one per synth row) without leaking any label directly.

    Meta-model training:
      For each hidden feature, a LogisticRegression is trained on:
        X = [ordinal-encoded base predictions, confidence (or full probas for
             low-cardinality features), ordinal-encoded QI values]
        y = true hidden feature value
      For features with n_classes <= proba_max_classes, the full probability
      vector is included. For high-cardinality features, only the max probability
      (confidence) is included to avoid an intractably large feature space.
      Including QI lets the meta-model learn which base model is reliable for
      which sub-population.

    Inference:
      All base models run on actual targets (full synth as training data).
      Their predictions + target QI values are fed to the meta-models.

    Parameters
    ----------
    methods           : list of attack functions
    method_names      : list of method name strings
    cfg_full          : original full config dict (for _prepare_config per method)
    synth             : synthetic dataset (pd.DataFrame)
    targets           : real target records, QI columns visible (pd.DataFrame)
    qi                : list of quasi-identifier column names
    hidden_features   : list of hidden feature column names
    n_folds           : number of OOF folds (default 5)
    use_probas        : if True, include base-model probability info in meta-features
    proba_max_classes : features with more unique values than this use confidence
                        (max proba) instead of the full probability vector, to
                        avoid a combinatorial explosion in meta-feature dims (default 20)

    Returns
    -------
    (reconstructed, None, None)
    """
    import gc
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OrdinalEncoder
    from master_experiment_script import _prepare_config

    n_synth   = len(synth)
    n_targets = len(targets)
    n_models  = len(methods)

    print(f"\nStacking ensemble: {n_models} base models, {n_folds} OOF folds", flush=True)
    print(f"  Base models: {', '.join(method_names)}", flush=True)

    # ── Encoders fit on full synth (stable class space at OOF + inference) ───
    print(f"  Fitting encoders...", flush=True)
    qi_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    qi_enc.fit(synth[qi].astype(str))

    feat_val_enc  = {}   # OrdinalEncoder per hidden feature, for encoding predictions
    feat_n_classes = {}  # cardinality of each hidden feature in synth
    for feat in hidden_features:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(synth[[feat]].astype(str))
        feat_val_enc[feat]   = enc
        feat_n_classes[feat] = len(enc.categories_[0])

    print(f"  Feature cardinalities: { {f: feat_n_classes[f] for f in hidden_features} }",
          flush=True)
    print(f"  Proba strategy (full if n_classes <= {proba_max_classes}, else confidence only):",
          flush=True)
    for feat in hidden_features:
        strategy = "full proba" if feat_n_classes[feat] <= proba_max_classes else "confidence only"
        print(f"    {feat}: {feat_n_classes[feat]} classes → {strategy}", flush=True)

    # ── OOF prediction storage ────────────────────────────────────────────────
    # oof_hard[feat][midx] : (n_synth,) string array of hard predictions
    # oof_prob[feat][midx] : (n_synth, n_classes) float array, or None
    # oof_cls[feat][midx]  : class label array aligned to oof_prob columns, or None
    oof_hard = {feat: {m: np.full(n_synth, '', dtype=object) for m in range(n_models)}
                for feat in hidden_features}
    oof_prob = {feat: {m: None for m in range(n_models)} for feat in hidden_features}
    oof_cls  = {feat: {m: None for m in range(n_models)} for feat in hidden_features}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(synth)):
        print(f"  OOF fold {fold_idx + 1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)})...", flush=True)
        synth_tr  = synth.iloc[train_idx].reset_index(drop=True)
        synth_val = synth.iloc[val_idx].reset_index(drop=True)

        for midx, (method_fn, method_name) in enumerate(zip(methods, method_names)):
            print(f"    fold {fold_idx+1} model [{midx+1}/{n_models}] {method_name}...",
                  flush=True)
            method_cfg = cfg_full.copy()
            method_cfg["attack_method"] = method_name
            method_cfg = _prepare_config(method_cfg)

            try:
                recon, probas, classes = method_fn(
                    method_cfg, synth_tr, synth_val, qi, hidden_features)

                for feat_idx, feat in enumerate(hidden_features):
                    oof_hard[feat][midx][val_idx] = recon[feat].astype(str).values

                    if use_probas and probas is not None and classes is not None:
                        p = probas[feat_idx]    # (n_val, n_classes_for_feat)
                        c = classes[feat_idx]   # class label array
                        if p is not None and c is not None:
                            # Use full-synth class space as canonical column order
                            full_cls = feat_val_enc[feat].categories_[0]
                            if oof_prob[feat][midx] is None:
                                oof_prob[feat][midx] = np.zeros((n_synth, len(full_cls)))
                                oof_cls[feat][midx]  = np.asarray(full_cls)
                            # Project fold probas onto canonical class columns
                            cls_to_full = {str(cls): j for j, cls in enumerate(full_cls)}
                            p_aligned = np.zeros((len(val_idx), len(full_cls)))
                            for j_fold, cls_label in enumerate(c):
                                j_full = cls_to_full.get(str(cls_label))
                                if j_full is not None:
                                    p_aligned[:, j_full] = p[:, j_fold]
                            oof_prob[feat][midx][val_idx] = p_aligned

                print(f"    fold {fold_idx+1} model [{midx+1}/{n_models}] {method_name} done",
                      flush=True)
            except Exception as e:
                print(f"    [WARN] {method_name} failed on fold {fold_idx}: {e}", flush=True)

        gc.collect()

    # ── Build meta-features for synth (OOF) and train meta-models ───────────
    print(f"  Training meta-models...", flush=True)
    X_qi_synth = qi_enc.transform(synth[qi].astype(str))

    meta_models   = {}   # {feat: fitted LogisticRegression}
    meta_feat_dim = {}   # {feat: int} — number of meta-features per model

    def _proba_block_oof(feat, midx, n_rows):
        """Return proba meta-feature block for OOF phase."""
        if not use_probas or oof_prob[feat][midx] is None:
            return None
        n_cls = feat_n_classes[feat]
        if n_cls <= proba_max_classes:
            return oof_prob[feat][midx]           # (n_synth, n_cls) full distribution
        else:
            return oof_prob[feat][midx].max(axis=1, keepdims=True)  # (n_synth, 1) confidence

    for feat_i, feat in enumerate(hidden_features):
        X_blocks = []

        for midx in range(n_models):
            hard_col = oof_hard[feat][midx].reshape(-1, 1)
            X_blocks.append(feat_val_enc[feat].transform(hard_col))
            blk = _proba_block_oof(feat, midx, n_synth)
            if blk is not None:
                X_blocks.append(blk)

        X_blocks.append(X_qi_synth)
        X_meta = np.hstack(X_blocks)
        y_meta = synth[feat].astype(str).values

        n_cls = feat_n_classes[feat]
        print(f"  Meta-model [{feat_i+1}/{len(hidden_features)}] {feat}: "
              f"X={X_meta.shape}, n_classes={n_cls} ...", flush=True)

        meta_clf = LogisticRegression(max_iter=200, C=1.0, solver='saga',
                                      multi_class='auto', n_jobs=-1)
        meta_clf.fit(X_meta, y_meta)
        meta_models[feat]   = meta_clf
        meta_feat_dim[feat] = X_meta.shape[1]
        print(f"  Meta-model [{feat_i+1}/{len(hidden_features)}] {feat}: done", flush=True)

    print(f"  Meta-model input dims: "
          f"{ {f: meta_feat_dim[f] for f in hidden_features} }", flush=True)

    # ── Run base models on actual targets (full synth as training data) ──────
    print(f"  Running base models on targets...", flush=True)
    inf_hard  = {feat: [] for feat in hidden_features}
    inf_prob  = {feat: [] for feat in hidden_features}
    inf_cls   = {feat: [] for feat in hidden_features}

    for midx, (method_fn, method_name) in enumerate(zip(methods, method_names)):
        print(f"    [{midx+1}/{n_models}] {method_name}...", flush=True)
        method_cfg = cfg_full.copy()
        method_cfg["attack_method"] = method_name
        method_cfg = _prepare_config(method_cfg)

        try:
            recon, probas, classes = method_fn(
                method_cfg, synth, targets, qi, hidden_features)

            for feat_idx, feat in enumerate(hidden_features):
                inf_hard[feat].append(recon[feat].astype(str).values)

                p = (probas[feat_idx]  if (use_probas and probas  is not None) else None)
                c = (classes[feat_idx] if (use_probas and classes is not None) else None)
                inf_prob[feat].append(p if (p is not None and c is not None) else None)
                inf_cls[feat].append(c if (p is not None and c is not None) else None)

            print(f"    [{midx+1}/{n_models}] {method_name} done", flush=True)
        except Exception as e:
            print(f"    [WARN] {method_name} failed at inference: {e}", flush=True)
            for feat in hidden_features:
                inf_hard[feat].append(np.array([str(synth[feat].mode().iloc[0])] * n_targets))
                inf_prob[feat].append(None)
                inf_cls[feat].append(None)

    # ── Apply meta-models ─────────────────────────────────────────────────────
    print(f"  Applying meta-models to targets...", flush=True)
    X_qi_targets = qi_enc.transform(targets[qi].astype(str))
    result        = targets.copy()

    for feat in hidden_features:
        X_blocks = []

        for midx in range(n_models):
            hard_col = inf_hard[feat][midx].reshape(-1, 1)
            X_blocks.append(feat_val_enc[feat].transform(hard_col))

            # Proba block — must use same strategy (full vs confidence) as OOF
            n_cls = feat_n_classes[feat]
            has_oof_prob = use_probas and oof_prob[feat][midx] is not None
            has_inf_prob = inf_prob[feat][midx] is not None and inf_cls[feat][midx] is not None

            if has_oof_prob and has_inf_prob:
                oof_classes = oof_cls[feat][midx]
                p_inf       = inf_prob[feat][midx]
                inf_classes_arr = np.asarray(inf_cls[feat][midx])

                if n_cls <= proba_max_classes:
                    # Full distribution — re-align inf probas to OOF column order
                    n_oof_cls = len(oof_classes)
                    aligned   = np.zeros((n_targets, n_oof_cls))
                    cls_to_oof = {str(c): j for j, c in enumerate(oof_classes)}
                    for j_inf, cls_label in enumerate(inf_classes_arr):
                        j_oof = cls_to_oof.get(str(cls_label))
                        if j_oof is not None:
                            aligned[:, j_oof] = p_inf[:, j_inf]
                    X_blocks.append(aligned)
                else:
                    # Confidence only (must match OOF strategy)
                    X_blocks.append(p_inf.max(axis=1, keepdims=True))

            elif has_oof_prob:
                # OOF had probas but inference didn't — fill with zeros
                if n_cls <= proba_max_classes:
                    X_blocks.append(np.zeros((n_targets, len(oof_cls[feat][midx]))))
                else:
                    X_blocks.append(np.zeros((n_targets, 1)))

        X_blocks.append(X_qi_targets)
        X_meta_test = np.hstack(X_blocks)

        meta_pred = meta_models[feat].predict(X_meta_test)
        try:
            result[feat] = meta_pred.astype(synth[feat].dtype)
        except (ValueError, TypeError):
            result[feat] = meta_pred

    print(f"  Stacking inference complete.", flush=True)
    return result, None, None
