"""
Default parameter values for all reconstruction attacks.

These mirror the *_default constants defined in each attack's implementation
module (attacks/ML_classifiers.py, attacks/NN_classifier.py, etc.) and in
the external repos (MIA_on_diffusion/, recon-synth/).

Used by experiment scripts to log the full effective parameters to WandB,
even when no overrides are passed:

    from attacks.defaults import ATTACK_PARAM_DEFAULTS
    effective = {**ATTACK_PARAM_DEFAULTS.get(method, {}), **explicit_overrides}

If a default changes in the underlying implementation, update it here too.
"""

ATTACK_PARAM_DEFAULTS: dict[str, dict] = {

    # ── Baselines ────────────────────────────────────────────────────────────
    # attacks/baselines_classifiers.py / baselines_continuous.py
    "Mode":        {},
    "Random":      {},
    "Mean":        {},
    "Median":      {},
    "MeasureDeid": {},
    "NearestNeighbor": {},
    "RandomNormal":    {},

    # ── ML classifiers (attacks/ML_classifiers.py) ───────────────────────────
    "KNN": {
        "k":           1,
        "use_weights": True,
    },
    "NaiveBayes": {},
    "LogisticRegression": {
        "max_iter": 100,
    },
    "SVM": {
        "kernel":      "rbf",
        "C":           1.0,
        "gamma":       "scale",
    },
    "RandomForest": {
        "num_estimators": 25,
        "max_depth":      25,
    },
    "LightGBM": {
        "lgb_num_estimators": 100,
        "lgb_objective":      "multiclass",
        "lgb_metric":         "multi_logloss",
        "lgb_verbosity":      -1,
    },

    # ── TabPFN (attacks/tabpfn_attack.py) ───────────────────────────────────
    # In-context classification: synth → training set, QI → features.
    # Falls back to RF for features with > 10 distinct values.
    "TabPFN": {
        "max_train_samples":         1024,  # subsample synth if larger
        "n_ensemble_configurations": 32,    # TabPFN ensemble size
        "device":                    "cpu",
        "rf_fallback_n_estimators":  25,
        "rf_fallback_max_depth":     25,
    },

    # ── MarginalRF (attacks/marginal_rf.py) ─────────────────────────────────
    # RF posteriors + sum-product BP on an MST of pairwise synth marginals.
    # Reduces to plain RF when features are independent in synth (PMI ≈ 0).
    "MarginalRF": {
        "num_estimators":       25,
        "max_depth":            25,
        "max_pair_cardinality": 50,    # features above this skip pairwise correction
        "knn_k":                100,   # synth neighbours per target for local (conditional) PMI;
                                       # set to None for global (unconditional) PMI
        "alpha":                1e-6,  # Laplace smoothing for joint tables
        "graph_type":           "mst", # "mst" (exact BP) | "complete" | "topk" (both loopy BP)
        "top_k_edges":          None,  # edge budget for graph_type="topk"; None → 2 × |features|
        "lbp_max_iter":         20,    # max loopy BP iterations (ignored for "mst")
        "lbp_damping":          0.5,   # loopy BP step-size: 1.0=full update, 0.0=no update
        # Column marginal correction — nudges aggregate predictions toward the synth marginal
        "col_correction_alpha": 0.5,   # strength: 0.0=off, 1.0=full; TODO: ablation sweep over [0,0.25,0.5,0.75,1.0]
        "col_correction_mode":  "global", # "global" (cross-row normalisation) | "knn" (per-row local marginal)
        "col_correction_iters": 1,     # outer (row-BP + col-correction) iterations; >1 = alternating refinement
    },

    # ── ML regressors (attacks/ML_regression.py) ─────────────────────────────
    "LinearRegression":     {},
    "Ridge":                {"alpha": 1.0},
    "Lasso":                {"alpha": 1.0},
    "ElasticNet":           {"alpha": 1.0, "l1_ratio": 0.5},
    "PolynomialRegression": {"degree": 2},
    "BayesianRidge":        {},
    "HuberRegressor":       {"epsilon": 1.35},
    "RANSACRegressor":      {},
    "SGDRegressor":         {"max_iter_sdg": 1000, "penalty": "l2", "alpha_sdg": 0.0001},

    # ── Neural networks (attacks/NN_classifier.py) ───────────────────────────
    "MLP": {
        "test_size":     0.2,
        "hidden_dims":   [300],
        "batch_size":    264,
        "learning_rate": 0.0003,
        "epochs":        500,
        "patience":      60,
        "dropout_rate":  0.2,
    },

    # ── Attention (attacks/attention_classifier.py) ──────────────────────────
    "Attention": {
        "num_heads":       4,
        "embedding_dim":   64,   # must be divisible by num_heads
        "num_layers":      2,
        "feedforward_dim": 128,
        "dropout_rate":    0.2,
        "test_size":       0.2,
        "batch_size":      128,
        "learning_rate":   0.001,
        "epochs":          100,
        "patience":        30,
    },
    "AttentionAutoregressive": {
        "num_heads_AR":       4,
        "embedding_dim_AR":   64,   # must be divisible by num_heads_AR
        "num_layers_AR":      2,
        "feedforward_dim_AR": 128,
        "dropout_rate_AR":    0.15,
        "test_size_AR":       0.2,
        "batch_size_AR":      64,
        "learning_rate_AR":   0.001,
        "epochs_AR":          200,
        "patience_AR":        30,
        "feature_order":      None,  # None → use hidden_features order
    },

    # ── Partial diffusion — MIA_on_diffusion/midst_models/single_table_TabDDPM/
    #    tabddpm_reconstruction_attack.py lines 37-44, pipeline_utils.py line 576
    #    All three attacks train the same model (TabDDPM & ConditionedRePaint
    #    share the artifact dir); they differ only in sampling strategy.
    "TabDDPM": {
        "hidden_dims":       [512, 1024, 1024, 1024, 1024, 512],  # d_layers of the diffusion MLP; required (no fallback in tabddpm_reconstruction_attack.py)
        "dropout":           0.1,
        "batch_size":        4096,
        "lr":                0.0006,
        "weight_decay":      1e-5,
        "num_epochs":        200_000,
        "num_timesteps":     2000,
        "resamples":         10,
        "jump_fn":           "jump_max10",   # serialised as string for WandB
        "sample_batch_size": 8192,
    },
    "RePaint": {
        "hidden_dims":       [512, 1024, 1024, 1024, 1024, 512],
        "dropout":           0.1,
        "batch_size":        4096,
        "lr":                0.0006,
        "weight_decay":      1e-5,
        "num_epochs":        200_000,
        "num_timesteps":     2000,
        "resamples":         10,
        "jump_fn":           "jump_max10",
        "sample_batch_size": 8192,
    },
    "ConditionedRePaint": {
        "hidden_dims":       [512, 1024, 1024, 1024, 1024, 512],
        "dropout":           0.1,
        "batch_size":        4096,
        "lr":                0.0006,
        "weight_decay":      1e-5,
        "num_epochs":        200_000,
        "num_timesteps":     2000,
        "resamples":         10,
        "jump_fn":           "jump_max10",
        "sample_batch_size": 8192,
    },
    "TabDDPMEnsemble": {
        "hidden_dims":         [512, 1024, 1024, 1024, 1024, 512],
        "dropout":             0.1,
        "batch_size":          4096,
        "lr":                  0.0006,
        "weight_decay":        1e-5,
        "num_epochs":          200_000,
        "num_timesteps":       2000,
        "resamples":           10,
        "jump_fn":             "jump_max10",
        "n_diffusion_samples": 5,
        "ensemble_agg":        "majority",
    },
    "TabDDPMWithMLP": {
        "hidden_dims":     [512, 1024, 1024, 1024, 1024, 512],
        "dropout":         0.1,
        "batch_size":      4096,
        "lr":              0.0006,
        "weight_decay":    1e-5,
        "num_epochs":      200_000,
        "num_timesteps":   2000,
        "resamples":       10,
        "jump_fn":         "jump_max10",
        "stacking_frac":   0.5,
        "mlp_hidden_dims": [1024],
        "mlp_epochs":      1000,
        "mlp_lr":          0.001,
    },

    # ── Partial MST (attacks/partialMST.py) ──────────────────────────────────
    # PartialMSTIndependent reuses the same defaults; each call trains on a
    # single-feature synth so checkpoints are automatically distinct.
    "PartialMST": {
        "bin_continuous_as_ordinal": True,
        "n_bins":                   20,
        "iters":                    10000,
        "retrain":                  False,
        "sample_mode":              "sample",  # "sample" | "argmax" | "top_pct"
        "top_pct":                  20.0,      # used only when sample_mode="top_pct"
    },
    "PartialMSTIndependent": {
        "bin_continuous_as_ordinal": True,
        "n_bins":                   20,
        "iters":                    10000,
        "retrain":                  False,
        "sample_mode":              "sample",
        "top_pct":                  20.0,
    },
    "PartialMSTBounded": {
        "bin_continuous_as_ordinal": True,
        "n_bins":                   20,
        "iters":                    10000,
        "retrain":                  False,
        "clique_variant":           "bounded",
        "max_clique_size":          3,
        "sample_mode":              "top_pct",
        "top_pct":                  10.0,
    },
    "PartialMSTHub": {
        "bin_continuous_as_ordinal": True,
        "n_bins":                   20,
        "iters":                    10000,
        "retrain":                  False,
        "clique_variant":           "hub",
        "sample_mode":              "sample",
        "top_pct":                  20.0,
    },

    # ── SOTA — recon-synth / Reconstruction/SOTA_attacks/linear_reconstruction.py
    "LinearReconstruction": {
        "k":       3,   # k-way marginal queries
        "n_procs": 4,   # Gurobi threads
    },
    "LinearReconstructionCategorical": {
        "k":       3,
        "n_procs": 4,
    },
    "LinearReconstructionJoint": {
        "k":       3,
        "n_procs": 4,
    },
}
