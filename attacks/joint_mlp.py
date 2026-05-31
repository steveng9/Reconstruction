"""
JointMLP: single neural network that predicts all hidden features simultaneously.

Trains a shared-trunk multi-output network with one softmax head per hidden feature.
Loss = sum of per-feature cross-entropies; predictions are the argmax of each head.

Contrast with MLP (attacks/NN_classifier.py), which trains one model per feature.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

_TEST_SIZE     = 0.2
_HIDDEN_DIMS   = [2000]
_BATCH_SIZE    = 256
_LEARNING_RATE = 0.0005
_EPOCHS        = 300
_PATIENCE      = 50
_DROPOUT_RATE  = 0.0


def joint_mlp_reconstruction(cfg, synth, targets, qi, hidden_features):
    p = cfg["attack_params"]
    return _reconstruct(
        synth, targets, qi, hidden_features,
        test_size     = p.get("test_size",     _TEST_SIZE),
        hidden_dims   = p.get("hidden_dims",   _HIDDEN_DIMS),
        batch_size    = p.get("batch_size",    _BATCH_SIZE),
        learning_rate = p.get("learning_rate", _LEARNING_RATE),
        epochs        = p.get("epochs",        _EPOCHS),
        patience      = p.get("patience",      _PATIENCE),
        dropout_rate  = p.get("dropout_rate",  _DROPOUT_RATE),
    )


def _reconstruct(synth, targets, qi, hidden_features,
                 test_size, hidden_dims, batch_size,
                 learning_rate, epochs, patience, dropout_rate):

    X_enc, y_enc, X_tgt_enc, feat_enc, y_enc_obj = _encode(
        synth, qi, hidden_features, targets
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=42
    )

    zeros_y = np.zeros((len(X_tgt_enc), y_enc.shape[1]))
    train_loader   = DataLoader(_DS(X_tr,      y_tr),    batch_size=batch_size, shuffle=True)
    val_loader     = DataLoader(_DS(X_val,     y_val),   batch_size=batch_size)
    targets_loader = DataLoader(_DS(X_tgt_enc, zeros_y), batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = _JointMLP(X_enc.shape[1], hidden_dims, y_enc.shape[1], dropout_rate).to(device)
    opt    = optim.Adam(model.parameters(), lr=learning_rate)

    model = _train(model, train_loader, val_loader, opt, device, y_enc_obj, epochs, patience)

    probas_list, classes_list, ohe_preds = _predict(model, targets_loader, device, y_enc_obj)

    original_predictions = y_enc_obj.inverse_transform(ohe_preds)

    reconstructed_targets = targets.copy()
    reconstructed_targets[hidden_features] = original_predictions

    return reconstructed_targets, probas_list, classes_list


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode(synth, qi, hidden_features, targets):
    feat_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_enc     = feat_enc.fit_transform(synth[qi].values)
    X_tgt_enc = feat_enc.transform(targets[qi].values)

    y_enc_obj = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    y_enc     = y_enc_obj.fit_transform(synth[hidden_features].values)

    return X_enc, y_enc, X_tgt_enc, feat_enc, y_enc_obj


class _DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _JointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _joint_loss(outputs, targets_ohe, y_encoder):
    """Sum of per-feature cross-entropies and mean per-feature accuracy."""
    loss  = outputs.new_zeros(())
    acc   = outputs.new_zeros(())
    n_feat = len(y_encoder.categories_)
    offset = 0
    for cats in y_encoder.categories_:
        n       = len(cats)
        logits  = outputs[:, offset:offset + n]
        tgt_idx = targets_ohe[:, offset:offset + n].long().argmax(dim=1)
        loss   += nn.functional.cross_entropy(logits, tgt_idx)
        acc    += (logits.argmax(dim=1) == tgt_idx).float().mean()
        offset += n
    return loss, acc / n_feat


def _train(model, train_loader, val_loader, opt, device, y_encoder, epochs, patience):
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss, _ = _joint_loss(model(X_b), y_b, y_encoder)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(X_b)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                loss, acc = _joint_loss(model(X_b), y_b, y_encoder)
                val_loss += loss.item() * len(X_b)
                val_acc  += acc.item() * len(X_b)
        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1:>4}/{epochs}  train={tr_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, targets_loader, device, y_encoder):
    model.eval()
    chunks = []
    with torch.no_grad():
        for X_b, _ in targets_loader:
            chunks.append(model(X_b.to(device)).cpu().numpy())
    all_out = np.concatenate(chunks, axis=0)

    n = len(all_out)
    ohe_preds    = np.zeros_like(all_out)
    probas_list  = []
    classes_list = []
    offset = 0
    for cats in y_encoder.categories_:
        k      = len(cats)
        logits = all_out[:, offset:offset + k]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs   = np.exp(shifted)
        probs  /= probs.sum(axis=1, keepdims=True)
        idx     = probs.argmax(axis=1)
        ohe_preds[np.arange(n), offset + idx] = 1.0
        probas_list.append(probs)
        classes_list.append(cats)
        offset += k

    return probas_list, classes_list, ohe_preds
