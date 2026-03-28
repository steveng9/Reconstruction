"""
Generic chaining wrapper for any reconstruction attack.
Wraps existing attacks without modifying their code.

Chaining predicts hidden features sequentially, adding each predicted
feature to the known features for subsequent predictions.
"""

import wandb
import numpy as np
import pandas as pd
from scoring import calculate_reconstruction_score, calculate_continuous_vals_reconstruction_score
from get_data import get_meta_data_for_diffusion


def apply_chaining(attack_fn, cfg, synth, targets, qi, hidden_features):
    """
    Wrapper that applies chaining to any attack function.

    Args:
        attack_fn: Original attack function with signature (cfg, synth, targets, qi, hidden_features)
        cfg: Configuration dictionary
        synth: Synthetic/de-identified data
        targets: Training data with known features (QI)
        qi: List of quasi-identifier (known) features
        hidden_features: List of features to reconstruct

    Returns:
        reconstructed, probas, classes (same as attack_fn)
    """
    chaining_cfg = cfg["attack_params"].get("chaining", {})

    if not chaining_cfg.get("enabled", False):
        # No chaining, call attack directly
        return attack_fn(cfg, synth, targets, qi, hidden_features)

    # Determine prediction order
    order_strategy = chaining_cfg.get("order_strategy", "default")
    data_type = cfg.get("data_type", "categorical")
    order = _get_chaining_order(chaining_cfg, hidden_features, synth, targets, qi,
                                data_type=data_type)

    # Log chaining configuration to WandB
    wandb.config.update({
        "chaining_enabled": True,
        "chaining_order_strategy": order_strategy,
        "chaining_order": order
    })

    print(f"\n{'='*60}")
    print(f"CHAINING ENABLED")
    print(f"  Strategy: {order_strategy}")
    print(f"  Order: {order}")
    print(f"{'='*60}\n")

    # Soft chaining: one-hot training / probability-vector inference
    mode = chaining_cfg.get("mode", "hard")
    if mode == "soft":
        if data_type != "categorical":
            print(f"[WARN] Soft chaining is only defined for categorical data; "
                  f"falling back to hard chaining")
        else:
            attack_method = cfg.get("attack_method", "")
            clf = _make_soft_chain_clf(attack_method, cfg.get("attack_params", {}))
            if clf is not None:
                return _run_soft_chained_sklearn(
                    attack_method, cfg.get("attack_params", {}),
                    synth, targets, qi, order,
                ), None, None
            else:
                print(f"[WARN] Soft chaining not supported for {attack_method!r}; "
                      f"falling back to hard chaining")

    # Get feature type information (continuous vs discrete) — only needed for
    # intermediate logging, so skip the call when log_intermediate is False.
    log_intermediate = chaining_cfg.get("log_intermediate", True)
    if log_intermediate:
        _, domain = get_meta_data_for_diffusion(cfg)
    else:
        domain = {}

    # Initialize reconstruction
    reconstructed = targets.copy()
    known_features = qi.copy()
    all_probas = []
    all_classes = []

    # Sequentially predict each feature
    for idx, feature in enumerate(order):
        print(f"\n  Chaining step {idx+1}/{len(order)}: predicting {feature}")

        # Predict single feature using current known features
        recon_step, probas, classes = attack_fn(
            cfg, synth, reconstructed[known_features], known_features, [feature]
        )

        # Update reconstruction
        reconstructed[feature] = recon_step[feature]

        # Log intermediate accuracy if enabled
        if chaining_cfg.get("log_intermediate", True):
            _log_intermediate_score(cfg, targets, reconstructed, feature, idx, domain)

        # Add predicted feature to known features for next iteration
        known_features.append(feature)

        # Store probabilities/classes if provided
        if probas is not None:
            all_probas.extend(probas if isinstance(probas, list) else [probas])
        if classes is not None:
            all_classes.extend(classes if isinstance(classes, list) else [classes])

    return reconstructed, all_probas if all_probas else None, all_classes if all_classes else None


def _log_intermediate_score(cfg, targets, reconstructed, feature, step_idx, domain):
    """
    Log intermediate reconstruction score for a single feature.
    Handles both continuous and categorical features.
    """
    # Determine if feature is continuous or discrete
    feature_type = domain.get(feature, {}).get("type", "discrete")

    if feature_type == "continuous":
        # Use continuous scoring
        scores_df = calculate_continuous_vals_reconstruction_score(
            targets, reconstructed, [feature]
        )

        # Log multiple continuous metrics
        metrics = scores_df.loc[feature].to_dict()
        log_dict = {
            f"chain_step_{step_idx+1:02d}_{feature}_mae": metrics['mean_abs_error'],
            f"chain_step_{step_idx+1:02d}_{feature}_normalized_mae": metrics['normalized_mae'],
            f"chain_step_{step_idx+1:02d}_{feature}_rmse": metrics['rmse'],
            f"chain_step_{step_idx+1:02d}_{feature}_normalized_rmse": metrics['normalized_rmse'],
        }
        wandb.log(log_dict)

        print(f"    → Normalized MAE: {metrics['normalized_mae']:.4f}, "
              f"Normalized RMSE: {metrics['normalized_rmse']:.4f}")

    else:
        # Use categorical scoring (rarity-weighted accuracy)
        score = calculate_reconstruction_score(targets, reconstructed, [feature])[0]
        wandb.log({f"chain_step_{step_idx+1:02d}_{feature}_accuracy": score})

        print(f"    → Rarity-weighted accuracy: {score:.1f}%")


def _get_chaining_order(chaining_cfg, hidden_features, synth, targets, qi,
                        data_type="categorical"):
    """Determine the order to predict features."""
    strategy = chaining_cfg.get("order_strategy", "default")

    if strategy == "manual":
        order = chaining_cfg.get("order", hidden_features)
        if set(order) != set(hidden_features):
            raise ValueError(
                f"Manual order must contain all hidden features.\n"
                f"Expected: {set(hidden_features)}\nGot: {set(order)}"
            )
        return order

    elif strategy == "default":
        return hidden_features

    elif strategy == "random":
        import random
        order = hidden_features.copy()
        random.seed(chaining_cfg.get("random_seed", 42))
        random.shuffle(order)
        return order

    elif strategy == "correlation":
        return _order_by_correlation(synth, qi, hidden_features, ascending=False)

    elif strategy == "reverse_correlation":
        return _order_by_correlation(synth, qi, hidden_features, ascending=True)

    elif strategy == "mutual_info":
        return _order_by_mutual_info(synth, qi, hidden_features, data_type=data_type)

    elif strategy == "dynamic_mutual_info":
        return _dynamic_order(synth, qi, hidden_features, data_type=data_type)

    else:
        raise ValueError(f"Unknown chaining order strategy: {strategy}")


def _order_by_correlation(synth, qi, hidden_features, ascending=False):
    """
    Order features by average absolute Pearson correlation with QI features.
    Handles string-valued columns via OrdinalEncoding before computing correlation.
    ascending=False: most-correlated first. ascending=True: least-correlated first.
    """
    from sklearn.preprocessing import OrdinalEncoder
    data = synth[qi + hidden_features].copy()
    str_cols = [c for c in data.columns if data[c].dtype == object]
    if str_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        data[str_cols] = enc.fit_transform(data[str_cols].astype(str))
    data = data.apply(pd.to_numeric, errors='coerce')
    corr = data.corr().abs()
    scores = {hf: corr.loc[hf, qi].mean() for hf in hidden_features}
    return sorted(hidden_features, key=lambda f: scores[f], reverse=(not ascending))


def _order_by_mutual_info(synth, qi, hidden_features, data_type="categorical"):
    """
    Order features by mean mutual information with QI features.
    Handles string-valued and mixed-type columns.
    Uses mutual_info_classif for categorical targets, mutual_info_regression for continuous.
    Highest-MI features first.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    enc_data = synth.copy()
    for col in enc_data.columns:
        if enc_data[col].dtype == object:
            le = LabelEncoder()
            enc_data[col] = le.fit_transform(enc_data[col].astype(str))
    mi_fn = mutual_info_regression if data_type == "continuous" else mutual_info_classif
    kwargs = {"random_state": 42}
    if data_type != "continuous":
        kwargs["discrete_features"] = True
    scores = {hf: float(mi_fn(enc_data[qi], enc_data[hf], **kwargs).mean())
              for hf in hidden_features}
    return sorted(hidden_features, key=lambda f: scores[f], reverse=True)


def _dynamic_order(synth, qi, hidden_features, data_type="categorical"):
    """
    Greedy dynamic ordering: at each step select the remaining hidden feature
    with the highest mutual information with the CURRENT known set
    (QI + all features already placed in the chain).

    Unlike static mutual_info ordering (which scores only against the original QI),
    this updates the conditioning set at each step, capturing how dependencies
    change as the chain grows. O(N^2) MI computations where N = len(hidden_features).
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    enc_data = synth.copy()
    for col in enc_data.columns:
        if enc_data[col].dtype == object:
            le = LabelEncoder()
            enc_data[col] = le.fit_transform(enc_data[col].astype(str))
    mi_fn = mutual_info_regression if data_type == "continuous" else mutual_info_classif
    kwargs = {"random_state": 42}
    if data_type != "continuous":
        kwargs["discrete_features"] = True

    remaining, order, current_known = list(hidden_features), [], list(qi)
    while remaining:
        best_feat, best_score = None, -1.0
        for feat in remaining:
            score = float(mi_fn(enc_data[current_known], enc_data[feat], **kwargs).mean())
            if score > best_score:
                best_score, best_feat = score, feat
        order.append(best_feat)
        remaining.remove(best_feat)
        current_known.append(best_feat)
    return order


def _make_soft_chain_clf(attack_name, attack_params):
    """
    Instantiate a fresh sklearn classifier for one step of soft chaining.
    Returns None if the attack is not supported (caller falls back to hard chaining).
    Supported: RandomForest, MLP, LightGBM, LogisticRegression, NaiveBayes, KNN.
    """
    p = attack_params
    if attack_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=p.get("num_estimators", 25),
            max_depth=p.get("max_depth", 25),
            random_state=42,
        )
    elif attack_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden_dims", [300])),
            max_iter=p.get("epochs", 250),
            learning_rate_init=p.get("learning_rate", 0.001),
            early_stopping=False,
            random_state=42,
        )
    elif attack_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=p.get("lgb_num_estimators", 100),
            verbosity=-1,
            random_state=42,
        )
    elif attack_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=p.get("max_iter", 100),
            random_state=42,
        )
    elif attack_name == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif attack_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        weights = "distance" if p.get("use_weights", True) else "uniform"
        return KNeighborsClassifier(n_neighbors=p.get("k", 1), weights=weights)
    else:
        return None


def _run_soft_chained_sklearn(attack_name, attack_params, synth, targets, qi, chain_order,
                              confidence_threshold=None):
    """
    Generic soft/probabilistic chaining for any sklearn classifier with predict_proba.

    Works for: RandomForest, MLP, LightGBM, LogisticRegression, NaiveBayes, KNN.

    Training (on synth), step k:
      X_train = [OrdinalEncode(QI)] | [OneHot(synth[feat_1])] | ... | [OneHot(synth[feat_{k-1}])]
      True synth values become one-hot vectors — a degenerate (certain) probability distribution.

    Inference (on targets), step k:
      X_test  = [OrdinalEncode(targets_QI)] | [probas_1] | ... | [probas_{k-1}]
      probas_j is the K_j-length predicted probability distribution from step j.
      When confident, probas ≈ one-hot (matches training). When uncertain, mass spreads,
      propagating uncertainty to downstream steps rather than a single wrong hard label.

    confidence_threshold: if set, rows whose max predicted probability falls below this
      value have their conditioning block replaced with a uniform distribution for the
      next step, signalling "unknown" rather than propagating a low-confidence guess.
      The reconstruction of the current feature still uses the model's best-guess label.
    """
    from sklearn.preprocessing import OrdinalEncoder

    qi_enc     = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_synth_qi = qi_enc.fit_transform(synth[qi].astype(str))
    X_test_qi  = qi_enc.transform(targets[qi].astype(str))

    X_synth_blocks = [X_synth_qi]
    X_test_blocks  = [X_test_qi]
    reconstructed  = targets.copy()

    for feat in chain_order:
        X_synth_step = np.hstack(X_synth_blocks)
        X_test_step  = np.hstack(X_test_blocks)

        y_synth = synth[feat].astype(str)
        clf = _make_soft_chain_clf(attack_name, attack_params)
        if attack_name == "MLP":
            classes = np.unique(y_synth)
            n_epochs = clf.max_iter
            for ep in range(n_epochs):
                clf.partial_fit(X_synth_step, y_synth, classes=classes)
                if (ep + 1) % 20 == 0:
                    print(f'    Epoch {ep+1}/{n_epochs}, loss={clf.loss_:.6f}')
        else:
            clf.fit(X_synth_step, y_synth)

        classes      = clf.classes_
        n_classes    = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        probas_test = clf.predict_proba(X_test_step)
        pred_labels = classes[np.argmax(probas_test, axis=1)]
        try:
            reconstructed[feat] = pred_labels.astype(synth[feat].dtype)
        except (ValueError, TypeError):
            reconstructed[feat] = pred_labels

        # Confidence gating: for rows below threshold, replace the conditioning block
        # with a uniform distribution so the next step treats this feature as unknown.
        probas_for_next = probas_test
        if confidence_threshold is not None:
            max_probs = probas_test.max(axis=1)
            low_conf  = max_probs < confidence_threshold
            if low_conf.any():
                probas_for_next = probas_test.copy()
                probas_for_next[low_conf] = 1.0 / n_classes

        # Synth block: one-hot of true synth values
        synth_codes  = synth[feat].astype(str).map(class_to_idx).fillna(0).astype(int)
        onehot_synth = np.zeros((len(synth), n_classes))
        onehot_synth[np.arange(len(synth)), synth_codes.values] = 1.0
        X_synth_blocks.append(onehot_synth)
        X_test_blocks.append(probas_for_next)

    return reconstructed


def _run_masked_hard_chained_sklearn(attack_name, attack_params, synth, targets, qi,
                                     chain_order, confidence_threshold=0.7):
    """
    Confidence-gated hard chaining for any sklearn classifier with predict_proba.

    Works for: RandomForest, MLP, LightGBM, LogisticRegression, NaiveBayes, KNN.

    The key difference from soft chaining: the conditioning block sent to the next
    step is always a one-hot vector (or uniform), never a raw probability distribution.
    This avoids the collapse seen in soft chaining for high-cardinality features,
    where spreading probability mass over hundreds of classes creates a noisy,
    high-dimensional input that overwhelms the QI signal.

    Training (on synth), step k:
      X_train = [OrdinalEncode(QI)] | [OneHot(synth[feat_1])] | ... | [OneHot(synth[feat_{k-1}])]
      Synth always uses one-hot (true value is certain).

    Inference (on targets), step k:
      High-confidence rows (max_prob >= threshold): one-hot at predicted class
        → matches training distribution, propagates a confident hard prediction
      Low-confidence rows (max_prob < threshold): uniform (1/n_classes per class)
        → signals "unknown" without injecting a likely-wrong hard label

    Returns
    -------
    reconstructed : pd.DataFrame
        Full reconstruction of targets (same shape, all hidden features predicted).
    conf_stats : dict[str, float]
        {feature: fraction_of_rows_above_threshold} — diagnostic for how often the
        model is confident enough to propagate a prediction for each feature.
    """
    from sklearn.preprocessing import OrdinalEncoder

    qi_enc     = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_synth_qi = qi_enc.fit_transform(synth[qi].astype(str))
    X_test_qi  = qi_enc.transform(targets[qi].astype(str))

    X_synth_blocks = [X_synth_qi]
    X_test_blocks  = [X_test_qi]
    reconstructed  = targets.copy()
    conf_stats     = {}  # feature → fraction of high-confidence rows

    for feat in chain_order:
        X_synth_step = np.hstack(X_synth_blocks)
        X_test_step  = np.hstack(X_test_blocks)

        y_synth = synth[feat].astype(str)
        clf = _make_soft_chain_clf(attack_name, attack_params)
        if attack_name == "MLP":
            classes  = np.unique(y_synth)
            n_epochs = clf.max_iter
            for ep in range(n_epochs):
                clf.partial_fit(X_synth_step, y_synth, classes=classes)
                if (ep + 1) % 20 == 0:
                    print(f'    Epoch {ep+1}/{n_epochs}, loss={clf.loss_:.6f}')
        else:
            clf.fit(X_synth_step, y_synth)

        classes      = clf.classes_
        n_classes    = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        probas_test  = clf.predict_proba(X_test_step)
        pred_indices = np.argmax(probas_test, axis=1)
        pred_labels  = classes[pred_indices]
        try:
            reconstructed[feat] = pred_labels.astype(synth[feat].dtype)
        except (ValueError, TypeError):
            reconstructed[feat] = pred_labels

        # Confidence gating: build a one-hot-or-uniform conditioning block.
        # Uniform (1/K) for all rows by default (low-confidence / unknown signal).
        max_probs        = probas_test.max(axis=1)
        high_conf        = max_probs >= confidence_threshold
        conf_stats[feat] = float(high_conf.mean())

        cond_block = np.full((len(targets), n_classes), 1.0 / n_classes)
        if high_conf.any():
            hi_idx             = np.where(high_conf)[0]
            cond_block[hi_idx] = 0.0
            cond_block[hi_idx, pred_indices[hi_idx]] = 1.0

        # Synth block: one-hot of true synth values (always certain)
        synth_codes  = synth[feat].astype(str).map(class_to_idx).fillna(0).astype(int)
        onehot_synth = np.zeros((len(synth), n_classes))
        onehot_synth[np.arange(len(synth)), synth_codes.values] = 1.0
        X_synth_blocks.append(onehot_synth)
        X_test_blocks.append(cond_block)

    return reconstructed, conf_stats
