"""
Attention-based reconstruction for categorical tabular data.
Inspired by transformer architecture but adapted for tabular reconstruction.

This model uses multi-head self-attention to learn complex relationships between
features and performs autoregressive reconstruction (like LLM next-token prediction).
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from get_data import load_data
from scoring import calculate_reconstruction_score

# ============================================================================
# DEFAULTS - ATTENTION MODEL
# ============================================================================

# Architecture - one feature at a time prediction model
num_heads_default = 4
embedding_dim_default = 64  # Must be divisible by num_heads
num_layers_default = 2
feedforward_dim_default = 128
dropout_rate_default = 0.2

# Training
test_size_default = 0.2
batch_size_default = 128
learning_rate_default = 0.001
epochs_default = 100
patience_default = 30


# Architecture - True Autoregressive (causal transformer, LLM-style)
num_heads_AR_default = 4
embedding_dim_AR_default = 64  # Must be divisible by num_heads
num_layers_AR_default = 2
feedforward_dim_AR_default = 128
dropout_rate_AR_default = 0.15

# Training - True Autoregressive
test_size_AR_default = 0.2
batch_size_AR_default = 64
learning_rate_AR_default = 0.001
epochs_AR_default = 200
patience_AR_default = 30


# Feature ordering (for autoregressive prediction)
# Features will be predicted in this order, each building on previous predictions
feature_order_default = ['F23', 'F13', 'F11', 'F43', 'F36', 'F25', 'F18', 'F30', 'F5', 'F15', 'F33', 'F10', 'F12', 'F50', 'F1', 'F3', 'F9', 'F21']

PLOT = False

CONFIG_PATH_default = "/Users/stevengolob/Documents/school/PhD/reconstruction_project/configs/dev_config.yaml"
# model_save_dir = "/Users/stevengolob/PycharmProjects/Reconstruction/models/"


def _development():
    config = load_config(CONFIG_PATH_default)

    print(f"\n\n{'=' * 50}")

    train, synth, qi, hidden_features, _ = load_data(config)
    hidden_features = hidden_features[:2]
    print("hidden_features = ", hidden_features)

    reconstructed, _, _ = attention_reconstruction(config, synth, train[qi], qi, hidden_features)
    scores = calculate_reconstruction_score(train, reconstructed, hidden_features)

    results = {f"RA_{k}": v for k, v in zip(hidden_features, scores)}
    results["RA_mean"] = np.mean(scores)

    scores = list(results.values())
    print(f"\n{np.array(scores[:-1])}")
    print(f"ave: {scores[-1]}\n{'=' * 50}")



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



# ============================================================================
# DATASET
# ============================================================================

class TabularDataset(Dataset):
    """Dataset for tabular data with categorical features."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AutoregressiveDataset(Dataset):
    """Dataset for autoregressive training: X is full sequence, Y is all hidden targets."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================================
# ATTENTION-BASED MODEL ARCHITECTURE
# ============================================================================

class FeatureEmbedding(nn.Module):
    """
    Embeds categorical features into continuous space.
    Each feature gets its own embedding table.
    """

    def __init__(self, feature_vocab_sizes, embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in feature_vocab_sizes
        ])
        self.num_features = len(feature_vocab_sizes)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features) - categorical indices
        Returns:
            (batch_size, num_features, embedding_dim)
        """
        embeddings = []
        for i in range(self.num_features):
            emb = self.embeddings[i](x[:, i])
            embeddings.append(emb)
        return torch.stack(embeddings, dim=1)


class PositionalEncoding(nn.Module):
    """Adds positional information to distinguish different feature positions."""

    def __init__(self, num_features, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features, embedding_dim))

    def forward(self, x):
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    """Single transformer block: Multi-head attention + FFN."""

    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, embedding_dim)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TabularAttentionModel(nn.Module):
    """Attention-based model for categorical tabular data."""

    def __init__(self, feature_vocab_sizes, num_classes, embedding_dim,
                 num_heads, num_layers, feedforward_dim, dropout_rate):
        if torch.cuda.is_available():
            print("Using CUDA device :)")
        else:
            print("NOT Using CUDA!")
        super(TabularAttentionModel, self).__init__()

        self.num_features = len(feature_vocab_sizes)

        self.feature_embedding = FeatureEmbedding(feature_vocab_sizes, embedding_dim)
        self.pos_encoding = PositionalEncoding(self.num_features, embedding_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * self.num_features, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature_embedding(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = x.flatten(start_dim=1)
        x = self.output_projection(x)

        return x


class AutoregressiveTabularModel(nn.Module):
    """
    True autoregressive transformer for tabular data with causal attention masking.
    Like an LLM: each position can only attend to itself and prior positions.

    Input sequence: [QI_0, ..., QI_{n-1}, H_0, ..., H_{m-1}]
    To predict H_k, uses the representation at position (num_qi - 1 + k),
    which has only attended to QI features + H_0..H_{k-1} (via causal mask).
    """

    def __init__(self, feature_vocab_sizes, num_qi, num_classes_per_hidden,
                 embedding_dim, num_heads, num_layers, feedforward_dim, dropout_rate):
        super().__init__()
        if torch.cuda.is_available():
            print("Using CUDA device :)")
        else:
            print("NOT Using CUDA!")

        self.num_features = len(feature_vocab_sizes)
        self.num_qi = num_qi
        self.num_hidden = len(num_classes_per_hidden)

        # +1 per feature for PAD token (index = original vocab_size)
        self.pad_indices = list(feature_vocab_sizes)
        padded_vocab_sizes = [vs + 1 for vs in feature_vocab_sizes]

        self.feature_embedding = FeatureEmbedding(padded_vocab_sizes, embedding_dim)
        self.pos_encoding = PositionalEncoding(self.num_features, embedding_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output head k: takes representation at position (num_qi - 1 + k) to predict H_k
        self.output_heads = nn.ModuleList([
            nn.Linear(embedding_dim, n_classes)
            for n_classes in num_classes_per_hidden
        ])

    def _causal_mask(self, seq_len, device):
        """Causal mask: position i can attend to positions 0..i only."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features) — full sequence (QI + hidden/PAD values)
        Returns:
            List of (batch_size, num_classes_k) logits, one per hidden feature
        """
        emb = self.feature_embedding(x)
        emb = self.pos_encoding(emb)

        mask = self._causal_mask(self.num_features, emb.device)

        h = emb
        for block in self.transformer_blocks:
            h = block(h, mask=mask)

        # Output at position (num_qi - 1 + k) predicts hidden feature k
        outputs = []
        for k in range(self.num_hidden):
            pos = self.num_qi - 1 + k
            outputs.append(self.output_heads[k](h[:, pos, :]))

        return outputs


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(cfg, model, train_loader, val_loader, criterion, optimizer, device,
                epochs, early_stopping_patience):
    """Train the attention model."""
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg["dataset"]["artifacts"], 'best_attention_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    if PLOT:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    model.load_state_dict(torch.load(os.path.join(cfg["dataset"]["artifacts"], 'best_attention_model.pth')))
    return model


def train_autoregressive_model(cfg, model, train_loader, val_loader, criterion, optimizer,
                               device, epochs, early_stopping_patience, num_hidden):
    """Train the autoregressive model on full sequences with causal masking."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        total_samples = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # list of (batch, num_classes_k)

            loss = sum(criterion(outputs[k], Y_batch[:, k]) for k in range(num_hidden))

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        train_loss /= total_samples

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)

                loss = sum(criterion(outputs[k], Y_batch[:, k]) for k in range(num_hidden))
                val_loss += loss.item() * X_batch.size(0)

                for k in range(num_hidden):
                    _, preds = torch.max(outputs[k], 1)
                    total += Y_batch.size(0)
                    correct += (preds == Y_batch[:, k]).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg["dataset"]["artifacts"], 'best_autoregressive_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    model.load_state_dict(torch.load(os.path.join(cfg["dataset"]["artifacts"], 'best_autoregressive_model.pth')))
    return model


def predict(model, data_loader, device):
    """Make predictions with the trained model."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    return np.array(predictions)


# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def encode_features(df, feature_columns):
    """Encode categorical features to integer indices."""
    encoded_df = pd.DataFrame()
    encoders = {}
    vocab_sizes = []

    for col in feature_columns:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        vocab_sizes.append(len(encoder.classes_))

    return encoded_df, encoders, vocab_sizes


def handle_unseen_categories(targets, synth, features, verbose=True):
    targets_cleaned = targets.copy()
    total_issues = 0

    if verbose:
        print("\n" + "=" * 70)
        print("CHECKING FOR UNSEEN CATEGORIES IN TARGET DATA")
        print("=" * 70)

    for col in features:
        target_values = targets_cleaned[col].astype(str)

        # Get categories from both datasets
        seen_categories = set(synth[col].astype(str).unique())
        target_categories = set(target_values.unique())
        unseen_categories = target_categories - seen_categories
        unseen_mask = ~target_values.isin(seen_categories)

        if unseen_mask.any():
            # Replace unseen categories with the most common category from training
            most_common = synth[col].astype(str).mode()[0]
            num_unseen_rows = unseen_mask.sum()
            num_unseen_cats = len(unseen_categories)
            num_seen_cats = len(seen_categories)
            total_issues += 1

            if verbose:
                print(f"\n'{col}':")
                print(f"  ├─ {num_unseen_cats} unseen categories (out of {num_seen_cats} in synth)")
                print(f"  ├─ {num_unseen_rows} rows affected ({100 * num_unseen_rows / len(target_values):.1f}%)")
                print(f"  └─ Mapping to most common: '{most_common}'")

            targets_cleaned[col] = target_values.where(~unseen_mask, most_common)

    if verbose:
        if total_issues == 0:
            print("\n✓ No unseen categories found")
        else:
            print(f"\n⚠ Total features with unseen categories: {total_issues}/{len(features)}")
        print("=" * 70 + "\n")

    return targets_cleaned


# ============================================================================
# RECONSTRUCTION METHODS
# ============================================================================

def attention_reconstruction_single(cfg, synth, targets, known_features, hidden_features,
                                    embedding_dim, num_heads, num_layers,
                                    feedforward_dim, dropout_rate, test_size,
                                    batch_size, learning_rate, epochs, patience):
    """
    Single-feature prediction: predict each hidden feature independently.
    """
    # Handle unseen categories ONCE at the beginning
    targets_cleaned = handle_unseen_categories(targets, synth, known_features, verbose=True)
    targets_copy = targets_cleaned.copy()

    for hidden_feature in hidden_features:
        print(f'\n\nPredicting hidden feature: {hidden_feature}')

        # Encode all features
        all_features = known_features.copy()
        train_df = synth[all_features + [hidden_feature]].copy()

        encoded_train, encoders, vocab_sizes = encode_features(train_df, all_features)
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(train_df[hidden_feature].astype(str))

        # Encode targets - already cleaned, no need to check again
        encoded_targets = pd.DataFrame()
        for col in all_features:
            encoded_targets[col] = encoders[col].transform(targets_cleaned[col].astype(str))

        # Split data
        X_train, X_val, y_train_split, y_val = train_test_split(
            encoded_train.values, y_train, test_size=test_size, random_state=42
        )

        # Create datasets
        train_dataset = TabularDataset(X_train, y_train_split)
        val_dataset = TabularDataset(X_val, y_val)
        target_dataset = TabularDataset(encoded_targets.values, np.zeros(len(encoded_targets)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        target_loader = DataLoader(target_dataset, batch_size=batch_size)

        # Create model
        num_classes = len(target_encoder.classes_)
        model = TabularAttentionModel(
            vocab_sizes, num_classes, embedding_dim,
            num_heads, num_layers, feedforward_dim, dropout_rate
        )

        # Train
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trained_model = train_model(cfg,
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs, patience
        )

        # Predict
        predictions = predict(trained_model, target_loader, device)
        original_predictions = target_encoder.inverse_transform(predictions)
        targets_copy[hidden_feature] = original_predictions

    return targets_copy.astype(int)

#
# def attention_reconstruction_autoregressive_retrain(cfg, synth, targets, known_features,
#                                                     hidden_features, embedding_dim, num_heads,
#                                                     num_layers, feedforward_dim, dropout_rate,
#                                                     test_size, batch_size, learning_rate,
#                                                     epochs, patience, feature_order=None):
#     """
#     Autoregressive with retraining: trains separate model for each feature position.
#     Simpler but less efficient.
#     """
#     # Handle unseen categories ONCE for initial known features
#     targets_cleaned = handle_unseen_categories(targets, synth, known_features, verbose=True)
#     targets_copy = targets_cleaned.copy()
#     known_features_copy = known_features.copy()
#
#     # Determine order
#     if feature_order is None:
#         feature_order = hidden_features
#     else:
#         assert set(feature_order) == set(hidden_features)
#
#     print(f'\nAutoregressive prediction order: {feature_order}')
#     print(f'Training separate model for each position...\n')
#
#     for idx, hidden_feature in enumerate(feature_order):
#         print(f'\n=== Position {idx + 1}/{len(feature_order)}: Predicting {hidden_feature} ===')
#
#         # Encode features
#         all_features = known_features_copy.copy()
#         train_df = synth[all_features + [hidden_feature]].copy()
#
#         encoded_train, encoders, vocab_sizes = encode_features(train_df, all_features)
#         target_encoder = LabelEncoder()
#         y_train = target_encoder.fit_transform(train_df[hidden_feature].astype(str))
#
#         # Encode targets - check for new unseen categories in predicted features
#         encoded_targets = pd.DataFrame()
#         for col in all_features:
#             col_values = targets_copy[col].astype(str)
#             seen_cats = set(encoders[col].classes_)
#             unseen_mask = ~col_values.isin(seen_cats)
#
#             if unseen_mask.any():
#                 most_common = train_df[col].astype(str).mode()[0]
#                 col_values = col_values.where(~unseen_mask, most_common)
#
#             encoded_targets[col] = encoders[col].transform(col_values)
#
#         # Split, create datasets, train (same as single)
#         X_train, X_val, y_train_split, y_val = train_test_split(
#             encoded_train.values, y_train, test_size=test_size, random_state=42
#         )
#
#         train_dataset = TabularDataset(X_train, y_train_split)
#         val_dataset = TabularDataset(X_val, y_val)
#         target_dataset = TabularDataset(encoded_targets.values, np.zeros(len(encoded_targets)))
#
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)
#         target_loader = DataLoader(target_dataset, batch_size=batch_size)
#
#         num_classes = len(target_encoder.classes_)
#         model = TabularAttentionModel(
#             vocab_sizes, num_classes, embedding_dim,
#             num_heads, num_layers, feedforward_dim, dropout_rate
#         )
#
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         trained_model = train_model(
#             model, train_loader, val_loader, criterion, optimizer,
#             device, epochs, patience
#         )
#
#         predictions = predict(trained_model, target_loader, device)
#         original_predictions = target_encoder.inverse_transform(predictions)
#         targets_copy[hidden_feature] = original_predictions
#
#         # Add to known features for next iteration
#         known_features_copy.append(hidden_feature)
#
#     return targets_copy.astype(int)


def attention_reconstruction_autoregressive_true(cfg, synth, targets, known_features,
                                                 hidden_features, embedding_dim, num_heads,
                                                 num_layers, feedforward_dim, dropout_rate,
                                                 test_size, batch_size, learning_rate,
                                                 epochs, patience, feature_order=None):
    """
    TRUE autoregressive reconstruction with causal attention masking.
    Like an LLM: trains on full sequences with teacher forcing and causal mask,
    then predicts hidden features one at a time at inference.
    """
    # Handle unseen categories ONCE
    targets_cleaned = handle_unseen_categories(targets, synth, known_features, verbose=True)
    targets_copy = targets_cleaned.copy()

    # Determine prediction order
    if feature_order is None:
        feature_order = list(hidden_features)
    else:
        assert set(feature_order) == set(hidden_features)

    print(f'\nAutoregressive prediction order: {feature_order}')
    print(f'Training causal transformer on full sequences...\n')

    # Encode ALL features (QI + hidden in prediction order)
    all_features = known_features + feature_order
    train_df = synth[all_features].copy()
    encoded_all, encoders, vocab_sizes = encode_features(train_df, all_features)

    num_qi = len(known_features)
    num_hidden = len(feature_order)
    num_classes_per_hidden = [len(encoders[f].classes_) for f in feature_order]

    # Training data: full sequences with teacher forcing
    X_full = encoded_all.values  # (n_synth, num_qi + num_hidden)
    Y_hidden = encoded_all[feature_order].values  # (n_synth, num_hidden)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_full, Y_hidden, test_size=test_size, random_state=42
    )

    train_dataset = AutoregressiveDataset(X_train, Y_train)
    val_dataset = AutoregressiveDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = AutoregressiveTabularModel(
        vocab_sizes, num_qi, num_classes_per_hidden,
        embedding_dim, num_heads, num_layers, feedforward_dim, dropout_rate
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_model = train_autoregressive_model(
        cfg, model, train_loader, val_loader, criterion, optimizer,
        device, epochs, patience, num_hidden
    )

    # Autoregressive inference on targets
    print('\nPerforming autoregressive inference on targets...')

    # Encode QI features of targets
    encoded_target_qi = pd.DataFrame()
    for col in known_features:
        col_values = targets_copy[col].astype(str)
        seen_cats = set(encoders[col].classes_)
        unseen_mask = ~col_values.isin(seen_cats)
        if unseen_mask.any():
            most_common = train_df[col].astype(str).mode()[0]
            col_values = col_values.where(~unseen_mask, most_common)
        encoded_target_qi[col] = encoders[col].transform(col_values)

    # Initialize sequence: QI values + PAD tokens for hidden features
    n_targets = len(targets_copy)
    X_inference = np.zeros((n_targets, len(all_features)), dtype=np.int64)
    X_inference[:, :num_qi] = encoded_target_qi.values
    for k in range(num_hidden):
        X_inference[:, num_qi + k] = vocab_sizes[num_qi + k]  # PAD index

    # Predict hidden features one at a time, filling in each before the next
    trained_model.eval()
    with torch.no_grad():
        for k, hidden_feature in enumerate(feature_order):
            print(f'  Predicting: {hidden_feature} ({k + 1}/{num_hidden})')

            predictions = []
            for start in range(0, n_targets, batch_size):
                end = min(start + batch_size, n_targets)
                X_batch = torch.tensor(X_inference[start:end], dtype=torch.long).to(device)
                outputs = trained_model(X_batch)
                _, preds = torch.max(outputs[k], 1)
                predictions.extend(preds.cpu().numpy())

            predictions = np.array(predictions)
            # Fill in prediction for next autoregressive step
            X_inference[:, num_qi + k] = predictions

            # Decode back to original values
            targets_copy[hidden_feature] = encoders[hidden_feature].inverse_transform(predictions)

    return targets_copy.astype(int)


# ============================================================================
# ADAPTER FUNCTIONS
# ============================================================================

def attention_reconstruction(cfg, synth, targets, known_features, hidden_features):
    """Single-feature attention reconstruction."""
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/attention_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    return attention_reconstruction_single(
        cfg, synth, targets, known_features, hidden_features,
        cfg["attack_params"].get("embedding_dim", embedding_dim_default),
        cfg["attack_params"].get("num_heads", num_heads_default),
        cfg["attack_params"].get("num_layers", num_layers_default),
        cfg["attack_params"].get("feedforward_dim", feedforward_dim_default),
        cfg["attack_params"].get("dropout_rate", dropout_rate_default),
        cfg["attack_params"].get("test_size", test_size_default),
        cfg["attack_params"].get("batch_size", batch_size_default),
        cfg["attack_params"].get("learning_rate", learning_rate_default),
        cfg["attack_params"].get("epochs", epochs_default),
        cfg["attack_params"].get("patience", patience_default)
    ), None, None


def attention_autoregressive_reconstruction(cfg, synth, targets, known_features, hidden_features):
    """TRUE autoregressive: single model, multiple heads (more efficient, LLM-like)."""
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/attention_AR_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    return attention_reconstruction_autoregressive_true(
        cfg, synth, targets, known_features, hidden_features,
        cfg["attack_params"].get("embedding_dim_AR", embedding_dim_AR_default),
        cfg["attack_params"].get("num_heads_AR", num_heads_AR_default),
        cfg["attack_params"].get("num_layers_AR", num_layers_AR_default),
        cfg["attack_params"].get("feedforward_dim_AR", feedforward_dim_AR_default),
        cfg["attack_params"].get("dropout_rate_AR", dropout_rate_AR_default),
        cfg["attack_params"].get("test_size_AR", test_size_AR_default),
        cfg["attack_params"].get("batch_size_AR", batch_size_AR_default),
        cfg["attack_params"].get("learning_rate_AR", learning_rate_AR_default),
        cfg["attack_params"].get("epochs_AR", epochs_AR_default),
        cfg["attack_params"].get("patience_AR", patience_AR_default),
        cfg["attack_params"].get("feature_order", feature_order_default)
    ), None, None


if __name__ == "__main__":
    _development()
