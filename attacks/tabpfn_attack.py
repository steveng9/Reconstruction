"""TabPFN reconstruction attack.

TabPFN (Prior-Data Fitted Networks, Hollmann et al. 2022) is a transformer
pre-trained on synthetic classification tasks.  At inference time it takes
the entire synth dataset as an "in-context" training set and predicts each
hidden feature given the quasi-identifiers — no gradient updates required.

Key properties
--------------
- Extremely fast at small n (≤ 1024 synth rows) — no training loop.
- Hard limit: ≤ 10 distinct classes per target feature (TabPFN v1).
- Features exceeding the class limit fall back to a RandomForest classifier.
- For synth datasets larger than `max_train_samples`, a random subsample
  is drawn (seeded for reproducibility).

Install (once, in the recon_ environment):
    conda run -n recon_ pip install tabpfn==0.1.11
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .ML_classifiers import _encode_qi


_MAX_CLASSES_TABPFN  = 10    # TabPFN v1 hard class-count limit
_MAX_TRAIN_DEFAULT   = 1024  # TabPFN v1 training-size limit
_N_ENSEMBLE_DEFAULT  = 32    # number of ensemble configurations
_TEST_BATCH_DEFAULT  = 1000  # max test rows per forward pass (prevents OOM on large datasets)


def tabpfn_reconstruction(cfg, deid, targets, qi, hidden_features):
    """Reconstruction attack using TabPFN in-context classification.

    For each hidden feature, trains TabPFN (or a RF fallback) on the
    synthetic data with QI columns as inputs, then predicts the hidden
    feature values for every target record.

    Attack params (set via cfg["attack_params"]):
        max_train_samples (int, default 1024):
            Maximum synth rows fed to TabPFN.  If synth has more rows a
            random subsample (seed 42) is used.  Does not affect the RF
            fallback, which always uses the full synth.
        n_ensemble_configurations (int, default 32):
            Number of TabPFN ensemble members.  Lower values are faster
            but less accurate.  32 is the TabPFN v1 default.
        device (str, default "cpu"):
            PyTorch device for TabPFN ("cpu" or "cuda").
        rf_fallback_n_estimators (int, default 25):
            Trees in the RF fallback used for high-cardinality features.
        rf_fallback_max_depth (int, default 25):
            Max depth for the RF fallback.

    Returns
    -------
    reconstructed_df : pd.DataFrame
        Copy of `targets` with hidden feature columns filled in.
    probas : list[np.ndarray | None]
        Per-feature (n_targets × n_classes) probability arrays.
        None for constant-valued features.
    classes_ : list[np.ndarray | None]
        Per-feature class label arrays (strings).
        None for constant-valued features.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError(
            "TabPFN is not installed.  Install it with:\n"
            "  conda run -n recon_ pip install tabpfn==0.1.11"
        ) from e

    params          = cfg["attack_params"]
    max_train       = params.get("max_train_samples",         _MAX_TRAIN_DEFAULT)
    n_ensemble      = params.get("n_ensemble_configurations", _N_ENSEMBLE_DEFAULT)
    device          = params.get("device",                    "cpu")
    test_batch_size = params.get("test_batch_size",           _TEST_BATCH_DEFAULT)
    rf_n_estimators = params.get("rf_fallback_n_estimators",  25)
    rf_max_depth    = params.get("rf_fallback_max_depth",      25)

    X_train_full, X_test = _encode_qi(deid, targets, qi)

    # Subsample synth for TabPFN if needed (RF fallback uses full synth)
    if len(deid) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(deid), max_train, replace=False)
        X_train_pfn = X_train_full[idx]
        deid_pfn    = deid.iloc[idx].reset_index(drop=True)
    else:
        X_train_pfn = X_train_full
        deid_pfn    = deid

    clf = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)

    # RF fallback — instantiated lazily so we avoid creating it when not needed
    rf_clf = None

    recon    = targets.copy()
    probas   = []
    classes_ = []

    for feat in hidden_features:
        y_pfn = deid_pfn[feat].astype(str)
        n_classes = y_pfn.nunique()

        # Constant feature — no information to predict from
        if n_classes < 2:
            recon[feat] = deid[feat].mode().iloc[0]
            probas.append(None)
            classes_.append(None)
            continue

        if n_classes <= _MAX_CLASSES_TABPFN:
            clf.fit(X_train_pfn, y_pfn.values)
            # Chunk predictions to avoid OOM on large test sets (e.g. 100k targets).
            # Use only predict_proba — predict() just calls it again internally.
            if test_batch_size and len(X_test) > test_batch_size:
                proba_chunks = []
                for start in range(0, len(X_test), test_batch_size):
                    proba_chunks.append(clf.predict_proba(X_test[start:start + test_batch_size]))
                proba = np.concatenate(proba_chunks, axis=0)
            else:
                proba = clf.predict_proba(X_test)
            pred         = clf.classes_[np.argmax(proba, axis=1)]
            feat_classes = clf.classes_
        else:
            # High-cardinality: fall back to RF on the full synth
            if rf_clf is None:
                rf_clf = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                )
            y_full = deid[feat].astype(str)
            rf_clf.fit(X_train_full, y_full.values)
            pred         = rf_clf.predict(X_test)
            proba        = rf_clf.predict_proba(X_test)
            feat_classes = rf_clf.classes_

        try:
            recon[feat] = pred.astype(deid[feat].dtype)
        except (ValueError, TypeError):
            recon[feat] = pred

        probas.append(proba)
        classes_.append(feat_classes)

    return recon, probas, classes_
