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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from get_data import load_data
from scoring import calculate_reconstruction_score

# ============================================================================
# DEFAULTS - ATTENTION MODEL
# ============================================================================

# Architecture
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

# Feature ordering (for autoregressive prediction)
# Features will be predicted in this order, each building on previous predictions
feature_order_default = None  # Will use natural order if None

PLOT = False

CONFIG_PATH_default = "/Users/stevengolob/Documents/school/PhD/reconstruction_project/configs/dev_config.yaml"
# model_save_dir = "/Users/stevengolob/PycharmProjects/Reconstruction/models/"


def _development():
    config = load_config(CONFIG_PATH_default)

    print(f"\n\n{'=' * 50}")

    train, synth, qi, hidden_features = load_data(config)
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


# ============================================================================
# ATTENTION-BASED MODEL
# ============================================================================

class FeatureEmbedding(nn.Module):
    """
    Embeds categorical features into continuous space.
    Each feature gets its own embedding table.
    """

    def __init__(self, feature_vocab_sizes, embedding_dim):
        """
        Args:
            feature_vocab_sizes: List of vocabulary sizes for each feature
            embedding_dim: Dimension of embeddings
        """
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
    """
    Adds positional information to distinguish different feature positions.
    """

    def __init__(self, num_features, embedding_dim):
        super(PositionalEncoding, self).__init__()
        # Learnable positional embeddings (simpler than sinusoidal for small number of features)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features, embedding_dim))

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features, embedding_dim)
        Returns:
            (batch_size, num_features, embedding_dim)
        """
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    """
    Single transformer block: Multi-head attention + FFN.
    """

    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_rate):
        super(TransformerBlock, self).__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, embedding_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, num_features, embedding_dim)
            mask: Optional attention mask
        Returns:
            (batch_size, num_features, embedding_dim)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TabularAttentionModel(nn.Module):
    """
    Attention-based model for categorical tabular data.
    Uses transformer architecture to predict target feature from input features.
    """

    def __init__(self, feature_vocab_sizes, num_classes, embedding_dim,
                 num_heads, num_layers, feedforward_dim, dropout_rate):
        super(TabularAttentionModel, self).__init__()

        self.num_features = len(feature_vocab_sizes)

        # Embedding layer
        self.feature_embedding = FeatureEmbedding(feature_vocab_sizes, embedding_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.num_features, embedding_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feedforward_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * self.num_features, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features) - categorical indices
        Returns:
            (batch_size, num_classes) - logits
        """
        # Embed features
        x = self.feature_embedding(x)  # (batch_size, num_features, embedding_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Flatten and project to output
        x = x.flatten(start_dim=1)  # (batch_size, num_features * embedding_dim)
        x = self.output_projection(x)

        return x


# ============================================================================
# TRAINING
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

    # Visualize training progress
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

    # Load best model
    model.load_state_dict(torch.load(os.path.join(cfg["dataset"]["artifacts"], 'best_attention_model.pth'))
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
# DATA PROCESSING
# ============================================================================

def encode_features(df, feature_columns):
    """
    Encode categorical features to integer indices.
    Returns encoded data and list of encoders.
    """
    encoded_df = pd.DataFrame()
    encoders = {}
    vocab_sizes = []

    for col in feature_columns:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        vocab_sizes.append(len(encoder.classes_))

    return encoded_df, encoders, vocab_sizes


# ============================================================================
# RECONSTRUCTION METHODS
# ============================================================================

def attention_reconstruction_single(cfg, synth, targets, known_features, hidden_features,
                                    embedding_dim, num_heads, num_layers,
                                    feedforward_dim, dropout_rate, test_size,
                                    batch_size, learning_rate, epochs, patience):
    """
    Single-feature prediction: predict each hidden feature independently.

    This is the simpler approach - each hidden feature is predicted from known features
    without considering predictions of other hidden features.
    """
    targets_copy = targets.copy()

    for hidden_feature in hidden_features:
        print(f'\n\nPredicting hidden feature: {hidden_feature}')

        # Encode all features (known + current hidden)
        all_features = known_features.copy()
        train_df = synth[all_features + [hidden_feature]].copy()

        encoded_train, encoders, vocab_sizes = encode_features(train_df, all_features)
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(train_df[hidden_feature].astype(str))

        # Encode targets (only known features)
        # Handle unseen categories by mapping them to a special index or the most common class
        encoded_targets = pd.DataFrame()
        for col in all_features:
            target_values = targets[col].astype(str)

            # Check for unseen categories
            seen_categories = set(encoders[col].classes_)
            unseen_mask = ~target_values.isin(seen_categories)

            if unseen_mask.any():
                # Replace unseen categories with the most common category from training
                most_common = train_df[col].astype(str).mode()[0]
                target_values = target_values.copy()
                target_values[unseen_mask] = most_common
                print(f"Warning: {unseen_mask.sum()} / {len(seen_categories)} categories unseen in '{col}' mapped to '{most_common}'")

            encoded_targets[col] = encoders[col].transform(target_values)

        # Split data
        X_train, X_val, y_train_split, y_val = train_test_split(
            encoded_train.values, y_train, test_size=test_size, random_state=42
        )

        # Create datasets
        train_dataset = TabularDataset(X_train, y_train_split)
        val_dataset = TabularDataset(X_val, y_val)
        target_dataset = TabularDataset(
            encoded_targets.values,
            np.zeros(len(encoded_targets))
        )

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


def attention_reconstruction_autoregressive(cfg, synth, targets, known_features,
                                            hidden_features, embedding_dim, num_heads,
                                            num_layers, feedforward_dim, dropout_rate,
                                            test_size, batch_size, learning_rate,
                                            epochs, patience, feature_order=None):
    """
    Autoregressive prediction: predict hidden features sequentially, using previous
    predictions as input for subsequent predictions (like LLM next-token prediction).

    This approach can capture dependencies between hidden features and potentially
    achieve better reconstruction by learning the joint distribution.
    """
    targets_copy = targets.copy()
    known_features_copy = known_features.copy()

    # Determine order of feature prediction
    if feature_order is None:
        feature_order = hidden_features
    else:
        # Validate that feature_order contains all hidden_features
        assert set(feature_order) == set(hidden_features), \
            "feature_order must contain exactly the hidden_features"

    print(f'\nAutoregressive prediction order: {feature_order}')

    for hidden_feature in feature_order:
        print(f'\n\nPredicting hidden feature: {hidden_feature}')
        print(f'Using features: {known_features_copy}')

        # Encode all features (known + current hidden)
        all_features = known_features_copy.copy()
        train_df = synth[all_features + [hidden_feature]].copy()

        encoded_train, encoders, vocab_sizes = encode_features(train_df, all_features)
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(train_df[hidden_feature].astype(str))

        # Encode targets (known features + previously predicted hidden features)
        # Handle unseen categories by mapping them to a special index or the most common class
        encoded_targets = pd.DataFrame()
        for col in all_features:
            target_values = targets_copy[col].astype(str)

            # Check for unseen categories
            seen_categories = set(encoders[col].classes_)
            unseen_mask = ~target_values.isin(seen_categories)

            if unseen_mask.any():
                # Replace unseen categories with the most common category from training
                most_common = train_df[col].astype(str).mode()[0]
                target_values = target_values.copy()
                target_values[unseen_mask] = most_common
                print(f"Warning: {unseen_mask.sum()} / {len(seen_categories)} categories unseen in '{col}' mapped to '{most_common}'")

            encoded_targets[col] = encoders[col].transform(target_values)

        # Split data
        X_train, X_val, y_train_split, y_val = train_test_split(
            encoded_train.values, y_train, test_size=test_size, random_state=42
        )

        # Create datasets
        train_dataset = TabularDataset(X_train, y_train_split)
        val_dataset = TabularDataset(X_val, y_val)
        target_dataset = TabularDataset(
            encoded_targets.values,
            np.zeros(len(encoded_targets))
        )

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

        # Add this feature to known features for next iteration
        known_features_copy.append(hidden_feature)

    return targets_copy.astype(int)


# ============================================================================
# ADAPTER FUNCTIONS (INTERFACE FOR CONFIG-BASED CALLS)
# ============================================================================

def attention_reconstruction(cfg, synth, targets, known_features, hidden_features):
    """
    Main adapter for single-feature attention-based reconstruction.
    Predicts each hidden feature independently.
    """
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
    """
    Main adapter for autoregressive attention-based reconstruction.
    Predicts hidden features sequentially, using previous predictions.
    """

    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/attention_AR_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    return attention_reconstruction_autoregressive(
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
        cfg["attack_params"].get("patience", patience_default),
        cfg["attack_params"].get("feature_order", feature_order_default)
    ), None, None


if __name__ == "__main__":
    _development()
