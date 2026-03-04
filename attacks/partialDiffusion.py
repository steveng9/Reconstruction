
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

from get_data import get_meta_data_for_diffusion

from tabddpm_reconstruction_attack import (
    train_diffusion_for_reconstruction,
    reconstruct_data_categorical,
    dump_artifact,
    load_artifact,
)


def _artifact_dir(cfg: dict, name: str) -> str:
    """Return a per-sample, per-SDG-method, per-QI artifact directory path.

    Pattern: {sample_dir}/{sdg_label}/{name}_{qi}
    Mirrors the Attention attack convention so parallel jobs across different
    SDG methods or QI variants never share the same checkpoint.
    """
    method = cfg.get("sdg_method", "unknown")
    params = cfg.get("sdg_params") or {}
    eps = params.get("epsilon") or params.get("eps")
    sdg_label = f"{method}_eps{eps:g}" if eps is not None else method
    qi = cfg.get("QI", "QI")
    return cfg["dataset"]["dir"] + f"/{sdg_label}/{name}_{qi}"


def _needs_training(artifact_dir: str, cfg: dict) -> bool:
    """Return True if the diffusion model should be trained."""
    checkpoint = os.path.join(artifact_dir, "model_ckpt.pkl")
    force = cfg.get("attack_params", {}).get("retrain", False)
    return force or not os.path.exists(checkpoint)


def _needs_mlp_training(mlp_artifact_dir: str, cfg: dict) -> bool:
    """Return True if the stacked MLP should be trained.

    Triggered by retrain=True (retrain everything) OR retrain_mlp=True
    (retrain only the MLP stacker, keeping the diffusion checkpoint).
    """
    mlp_file = os.path.join(mlp_artifact_dir, "stacked_mlp.pkl")
    params = cfg.get("attack_params", {})
    force = params.get("retrain", False) or params.get("retrain_mlp", False)
    return force or not os.path.exists(mlp_file)


def _write_mask_for_n_rows(n_rows, qi, hidden_features, artifact_dir):
    """Write known_features_mask.pkl sized for n_rows.

    The external reconstruct_data_categorical loads this file and passes it
    directly to the diffusion model, which requires it to match the number of
    rows being reconstructed.  The mask saved during training uses len(synth)
    rows, which differs from len(targets) when the SDG changes the dataset
    size (e.g. CellSuppression drops records).  We regenerate it on the fly —
    the pattern is always the same: 1 for QI columns, 0 for hidden.
    """
    column_order = qi + hidden_features
    mask = np.zeros((n_rows, len(column_order)))
    mask[:, :len(qi)] = 1
    dump_artifact(mask, os.path.join(artifact_dir, "known_features_mask.pkl"))


def _round_to_support(pred, support):
    """Snap each value in pred to the nearest value in the sorted support array.

    Used to post-process MLPRegressor float outputs so they match the exact
    integer values present in synth, enabling non-zero categorical accuracy.
    E.g. pred=39.7 → support contains 40 → snapped to 40 → exact match.
    """
    support = np.asarray(support, dtype=float)
    pred = np.asarray(pred, dtype=float)
    idx = np.searchsorted(support, pred)
    idx = np.clip(idx, 1, len(support) - 1)
    left = support[idx - 1]
    right = support[idx]
    snapped = np.where(np.abs(pred - left) <= np.abs(pred - right), left, right)
    return snapped


def _fix_feature_dtypes(reconstruction, hidden_features, domain):
    """Cast hidden features to the correct dtype based on domain type."""
    for feat in hidden_features:
        if feat not in reconstruction.columns:
            continue
        feat_type = domain.get(feat, {}).get("type", "discrete")
        try:
            if feat_type == "continuous":
                reconstruction[feat] = reconstruction[feat].astype(float)
            else:
                reconstruction[feat] = reconstruction[feat].astype(int)
        except (ValueError, TypeError):
            pass
    return reconstruction


def partial_tabddpm_reconstruction(cfg, synth, targets, qi, hidden_features):
    artifact_dir = _artifact_dir(cfg, "partial_tabddpm_artifacts")
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features)
        dump_artifact(domain, os.path.join(artifact_dir, "domain.pkl"))
    domain_path = os.path.join(artifact_dir, "domain.pkl")
    if os.path.exists(domain_path):
        domain = load_artifact(domain_path)
    _write_mask_for_n_rows(len(targets), qi, hidden_features, artifact_dir)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features)
    reconstruction = _fix_feature_dtypes(reconstruction, hidden_features, domain)
    return reconstruction, None, None


def repaint_reconstruction(cfg, synth, targets, qi, hidden_features):
    artifact_dir = _artifact_dir(cfg, "repaint_artifacts")
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=True)
        dump_artifact(domain, os.path.join(artifact_dir, "domain.pkl"))
    domain_path = os.path.join(artifact_dir, "domain.pkl")
    if os.path.exists(domain_path):
        domain = load_artifact(domain_path)
    _write_mask_for_n_rows(len(targets), qi, hidden_features, artifact_dir)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=True)
    reconstruction = _fix_feature_dtypes(reconstruction, hidden_features, domain)
    return reconstruction, None, None


def conditioned_repaint_reconstruction(cfg, synth, targets, qi, hidden_features):
    """Hybrid: QI-conditioned training (same as TabDDPM) + RePaint sampling.

    Shares artifact directory with partial_tabddpm_reconstruction — both use
    identical QI-conditioned training, so the checkpoint can be reused.
    If TabDDPM has already run for this (SDG method, QI), training is skipped automatically.
    """
    artifact_dir = _artifact_dir(cfg, "partial_tabddpm_artifacts")
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=False)
        dump_artifact(domain, os.path.join(artifact_dir, "domain.pkl"))
    domain_path = os.path.join(artifact_dir, "domain.pkl")
    if os.path.exists(domain_path):
        domain = load_artifact(domain_path)
    _write_mask_for_n_rows(len(targets), qi, hidden_features, artifact_dir)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=True)
    reconstruction = _fix_feature_dtypes(reconstruction, hidden_features, domain)
    return reconstruction, None, None


def tabddpm_ensemble_reconstruction(cfg, synth, targets, qi, hidden_features):
    """Run TabDDPM N times and aggregate: mode for discrete, mean for continuous."""
    artifact_dir = _artifact_dir(cfg, "partial_tabddpm_artifacts")
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features)
        dump_artifact(domain, os.path.join(artifact_dir, "domain.pkl"))
    domain_path = os.path.join(artifact_dir, "domain.pkl")
    if os.path.exists(domain_path):
        domain = load_artifact(domain_path)

    n_samples = cfg.get("attack_params", {}).get("n_diffusion_samples", 5)

    _write_mask_for_n_rows(len(targets), qi, hidden_features, artifact_dir)
    all_recs = []
    for _ in range(n_samples):
        rec = reconstruct_data_categorical(cfg, targets.copy(), qi, hidden_features)
        rec = _fix_feature_dtypes(rec, hidden_features, domain)
        all_recs.append(rec)

    # Aggregate per feature: mode for discrete, mean for continuous
    aggregated = all_recs[0].copy()
    for feat in hidden_features:
        feat_type = domain.get(feat, {}).get("type", "discrete")
        stacked = np.stack([r[feat].values for r in all_recs], axis=1)  # (n_targets, n_samples)
        if feat_type == "continuous":
            aggregated[feat] = stacked.mean(axis=1)
        else:
            # Majority vote
            def row_mode(row):
                vals, counts = np.unique(row, return_counts=True)
                return vals[np.argmax(counts)]
            aggregated[feat] = np.apply_along_axis(row_mode, 1, stacked)

    return aggregated, None, None


_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchMLP(nn.Module):
    """Lightweight GPU-capable MLP with a sklearn-compatible .predict() interface.

    task="classify": CrossEntropyLoss, .predict() returns integer class indices.
    task="regress":  MSELoss,           .predict() returns float values.
    """

    def __init__(self, in_dim, hidden_dims, out_dim, task="classify"):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.task = task
        self.loss_curve_ = []

    def forward(self, x):
        return self.net(x)

    def fit(self, X_np, y_np, lr=0.001, epochs=200, batch_size=512):
        device = _TORCH_DEVICE
        self.to(device)
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        if self.task == "classify":
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            criterion = nn.CrossEntropyLoss()
        else:
            y = torch.tensor(y_np, dtype=torch.float32, device=device)
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_curve_ = []
        self.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self(xb)
                if self.task == "regress":
                    out = out.squeeze(-1)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            self.loss_curve_.append(epoch_loss / len(X))
        self.to("cpu")
        return self

    def predict(self, X_np):
        device = _TORCH_DEVICE
        self.to(device)
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X_np, dtype=torch.float32, device=device)
            out = self(X)
            if self.task == "classify":
                pred = out.argmax(dim=1).cpu().numpy()
            else:
                pred = out.squeeze(-1).cpu().numpy()
        self.to("cpu")
        return pred


def _train_tabddpm_mlp_stacker(cfg, synth, qi, hidden_features, mlp_artifact_dir, domain):
    """Train a stacked MLP that corrects TabDDPM imputations using synth as training data."""
    attack_params = cfg.get("attack_params", {})
    stacking_frac = attack_params.get("stacking_frac", 0.2)
    mlp_hidden_dims = attack_params.get("mlp_hidden_dims", [128, 128])
    mlp_epochs = attack_params.get("mlp_epochs", 200)
    mlp_lr = attack_params.get("mlp_lr", 0.001)

    # Use a slice of synth as stacking training data (treating synth rows as pseudo-targets)
    n_stack = min(max(1, int(len(synth) * stacking_frac)), len(synth))
    synth_stack = synth.sample(n=n_stack, random_state=42).reset_index(drop=True)

    # Get diffusion imputations for the stacking set (QI known, hidden imputed)
    _write_mask_for_n_rows(len(synth_stack), qi, hidden_features, cfg["dataset"]["artifacts"])
    rec = reconstruct_data_categorical(cfg, synth_stack.copy(), qi, hidden_features)
    rec = rec.reset_index(drop=True)
    rec = _fix_feature_dtypes(rec, hidden_features, domain)
    # Align row count in case clava_reconstructing returns a different length
    n_rows = min(len(synth_stack), len(rec))
    synth_stack = synth_stack.iloc[:n_rows].reset_index(drop=True)
    rec = rec.iloc[:n_rows].reset_index(drop=True)

    # Encode QI features (handles string and numeric values uniformly)
    qi_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_qi = qi_encoder.fit_transform(synth_stack[qi].astype(str)).astype(np.float64)

    # Encode diffusion hint features; continuous hints used as-is
    hint_encoders = {}
    X_hints = np.zeros((n_rows, len(hidden_features)), dtype=np.float64)
    for i, feat in enumerate(hidden_features):
        feat_type = domain.get(feat, {}).get("type", "discrete")
        hint_vals = rec[feat].values.reshape(-1, 1)
        if feat_type == "discrete":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_hints[:, i] = enc.fit_transform(hint_vals.astype(str)).ravel()
            hint_encoders[feat] = enc
        else:
            X_hints[:, i] = hint_vals.ravel().astype(np.float64)
            hint_encoders[feat] = None

    X_train = np.hstack([X_qi, X_hints]).astype(np.float64)
    # Replace any NaN/inf (can appear from suppressed cells in CellSuppression synth)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features — critical for MLP convergence: OrdinalEncoded QI codes (0–15)
    # and raw continuous hint values (fnlwgt up to 1M) are on wildly different scales.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train one _TorchMLP per hidden feature (classifier for discrete, regressor for continuous).
    # LabelEncoder converts string labels → integer class indices for CrossEntropyLoss.
    n_feats = len(hidden_features)
    print(f"\n  [TabDDPMWithMLP] Training stacker MLPs on {n_rows} synth rows "
          f"({len(qi)} QI + {n_feats} hint features)  →  {n_feats} targets  "
          f"[device: {_TORCH_DEVICE}]", flush=True)

    mlps = {}
    label_encoders = {}
    for feat in hidden_features:
        feat_type = domain.get(feat, {}).get("type", "discrete")
        y = synth_stack[feat]
        if feat_type == "continuous":
            mlp = _TorchMLP(X_train.shape[1], mlp_hidden_dims, 1, task="regress")
            mlp.fit(X_train, y.astype(float).values, lr=mlp_lr, epochs=mlp_epochs)
            label_encoders[feat] = None
            n_classes = None
        else:
            le = LabelEncoder()
            y_int = le.fit_transform(y.astype(str))
            n_classes = len(le.classes_)
            mlp = _TorchMLP(X_train.shape[1], mlp_hidden_dims, n_classes, task="classify")
            mlp.fit(X_train, y_int, lr=mlp_lr, epochs=mlp_epochs)
            label_encoders[feat] = le

        losses = mlp.loss_curve_
        n = len(losses)
        # Print loss at 0%, 25%, 50%, 75%, 100% of training
        ckpts = [losses[int(i * (n - 1) / 4)] for i in range(5)]
        extra = f"  {n_classes} classes" if feat_type == "discrete" else "  regressor"
        print(f"    {feat:<22} {n:>4} iters  "
              f"loss: {ckpts[0]:.4f} → {ckpts[2]:.4f} → {ckpts[4]:.4f}{extra}", flush=True)

        mlps[feat] = mlp

    stacker = {
        "mlps": mlps,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "qi_encoder": qi_encoder,
        "hint_encoders": hint_encoders,
        "domain": domain,
    }
    os.makedirs(mlp_artifact_dir, exist_ok=True)
    dump_artifact(stacker, os.path.join(mlp_artifact_dir, "stacked_mlp.pkl"))
    return stacker


def tabddpm_mlp_reconstruction(cfg, synth, targets, qi, hidden_features):
    """TabDDPM imputations used as features for a discriminative MLP stacker."""
    artifact_dir = _artifact_dir(cfg, "partial_tabddpm_artifacts")
    mlp_artifact_dir = _artifact_dir(cfg, "partial_tabddpm_mlp_artifacts")
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features)
        dump_artifact(domain, os.path.join(artifact_dir, "domain.pkl"))
    domain_path = os.path.join(artifact_dir, "domain.pkl")
    if os.path.exists(domain_path):
        domain = load_artifact(domain_path)

    if _needs_mlp_training(mlp_artifact_dir, cfg):
        stacker = _train_tabddpm_mlp_stacker(cfg, synth, qi, hidden_features, mlp_artifact_dir, domain)
    else:
        stacker = load_artifact(os.path.join(mlp_artifact_dir, "stacked_mlp.pkl"))

    mlps = stacker["mlps"]
    label_encoders = stacker.get("label_encoders", {})
    scaler = stacker.get("scaler")
    qi_encoder = stacker["qi_encoder"]
    hint_encoders = stacker["hint_encoders"]
    domain = stacker["domain"]

    # Get diffusion imputations for targets
    _write_mask_for_n_rows(len(targets), qi, hidden_features, artifact_dir)
    rec = reconstruct_data_categorical(cfg, targets.copy(), qi, hidden_features)
    rec = _fix_feature_dtypes(rec, hidden_features, domain)

    # Build X for inference
    X_qi = qi_encoder.transform(targets[qi].astype(str)).astype(np.float64)
    X_hints = np.zeros((len(targets), len(hidden_features)), dtype=np.float64)
    for i, feat in enumerate(hidden_features):
        feat_type = domain.get(feat, {}).get("type", "discrete")
        hint_vals = rec[feat].values.reshape(-1, 1)
        if feat_type == "discrete" and hint_encoders.get(feat) is not None:
            X_hints[:, i] = hint_encoders[feat].transform(hint_vals.astype(str)).ravel()
        else:
            X_hints[:, i] = hint_vals.ravel().astype(np.float64)
    X_test = np.hstack([X_qi, X_hints]).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Build support sets from synth for nearest-value snapping of continuous predictions
    supports = {}
    for feat in hidden_features:
        if domain.get(feat, {}).get("type") == "continuous" and feat in synth.columns:
            supports[feat] = np.sort(synth[feat].dropna().unique().astype(float))

    # Predict with per-feature MLPs
    reconstruction = rec.copy()
    for feat in hidden_features:
        feat_type = domain.get(feat, {}).get("type", "discrete")
        pred = mlps[feat].predict(X_test)
        if feat_type == "continuous":
            support = supports.get(feat)
            if support is not None and len(support) > 0:
                pred = _round_to_support(pred, support)
            reconstruction[feat] = pred.astype(float)
        else:
            le = label_encoders.get(feat)
            if le is not None:
                pred = le.inverse_transform(pred.astype(int))
            try:
                reconstruction[feat] = pred.astype(int)
            except (ValueError, TypeError):
                reconstruction[feat] = pred

    return reconstruction, None, None
