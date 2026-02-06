import sys
import os
import numpy as np
import pandas as pd
from os.path import isfile, join
import matplotlib.pyplot as plt

from util import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# defaults
test_size_default = 0.2
hidden_dims_default = [300] # or [128, 96, 64]
batch_size_default = 264
learning_rate_default = 0.0003
epochs_default = 150 # or 400
patience_default = 30
dropout_rate_default = 0.2
PLOT = False


def mlp_repression_reconstruction(cfg, synth, targets, known_features, hidden_features):
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/mlp_continuous_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    return nn_regression_reconstruction(
        synth, targets, known_features, hidden_features,
        cfg["attack_params"].get("test_size", test_size_default),
        cfg["attack_params"].get("hidden_dims", hidden_dims_default),
        cfg["attack_params"].get("batch_size", batch_size_default),
        cfg["attack_params"].get("learning_rate", learning_rate_default),
        cfg["attack_params"].get("epochs", epochs_default),
        cfg["attack_params"].get("patience", patience_default),
        cfg["attack_params"].get("dropout_rate", dropout_rate_default)
    ), None, None


def chained_mlp_regression_reconstruction(cfg, synth, targets, known_features, hidden_features):
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/mlp_continuous_chained_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    """
    Chained reconstruction: predict each hidden feature sequentially,
    adding each predicted feature to the known features for the next prediction.
    """
    adequate_epochs = {
        'F23': 10, 'F13': 75, 'F11': 70, 'F43': 15, 'F36': 60,
        'F15': 90, 'F33': 70, 'F25': 30, 'F18': 20, 'F5': 45,
        'F30': 20, 'F10': 25, 'F12': 15, 'F50': 15, 'F3': 15,
        'F1': 25, 'F9': 20, 'F21': 25
    }

    reconstructed_targets = targets.copy()
    known_features_copy = known_features.copy()

    for hidden_feature in hidden_features:
        # Adjust hidden dimension based on number of known features
        dim_size = 300 + (len(known_features_copy) - 7) * 20

        # Get epochs for this feature if specified, otherwise use default
        feature_epochs = adequate_epochs.get(hidden_feature, 50)

        recon = nn_regression_reconstruction(
            synth, reconstructed_targets, known_features_copy, [hidden_feature],
            test_size=.2, hidden_dims=[dim_size], batch_size=264,
            learning_rate=.0002, epochs=feature_epochs
        )

        reconstructed_targets[hidden_feature] = recon[hidden_feature]
        known_features_copy.append(hidden_feature)

    return reconstructed_targets, None, None



# Custom dataset class for continuous data
class ContinuousDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural Network model for regression
class ContinuousNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        if torch.cuda.is_available():
            print("Using CUDA device :)")
        else:
            print("NOT Using CUDA!")
        super(ContinuousNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(cfg, model, train_loader, val_loader, criterion, optimizer, device, epochs,
                early_stopping_patience):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_maes = []
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                # Calculate MAE (Mean Absolute Error)
                mae = torch.abs(outputs - labels).sum().item()
                val_mae_sum += mae
                total += labels.numel()

        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae_sum / total
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)

        if epoch % 20 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val MAE: {val_mae:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(cfg["dataset"]["artifacts"], 'best_model_continuous.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    # Part 6: Visualize training progress
    # ------------------------------------
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
        plt.plot(val_maes, label='val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Load best model
    model.load_state_dict(torch.load(os.path.join(cfg["dataset"]["artifacts"], 'best_model_continuous.pth')))
    return model, training_history


# Function to make predictions
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.numpy())

    return np.array(predictions), np.array(true_labels)


# Process continuous features
def process_continuous_data(df, target_column, targets):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[[target_column]].values  # Keep as 2D array

    # Standardize features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    targets_scaled = feature_scaler.transform(targets)

    # Standardize target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, targets_scaled, feature_scaler, target_scaler



def nn_regression_reconstruction(cfg, synth, targets, known_features, hidden_features,
                                 test_size, hidden_dims, batch_size, learning_rate, epochs, patience, dropout_rate):
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        print(f'\n\nHidden feature: {hidden_feature}')

        # Process continuous data
        X_scaled, y_scaled, targets_scaled, feature_scaler, target_scaler = process_continuous_data(
            synth[known_features + [hidden_feature]], hidden_feature, targets[known_features]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42)

        # Create datasets and dataloaders
        train_dataset = ContinuousDataset(X_train, y_train)
        test_dataset = ContinuousDataset(X_test, y_test)
        targets_dataset = ContinuousDataset(targets_scaled, np.zeros((len(targets_scaled), 1)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        targets_loader = DataLoader(targets_dataset, batch_size=batch_size)

        # Determine dimensions
        input_dim = X_train.shape[1]
        output_dim = 1  # Single continuous target

        # Create model
        model = ContinuousNN(input_dim, hidden_dims, output_dim, dropout_rate)

        # Loss function and optimizer - MSE for regression
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Train model
        trained_model, history = train_model(cfg, model, train_loader, test_loader, criterion, optimizer, device, epochs, patience)

        # Make predictions
        predictions, _ = predict(trained_model, targets_loader, device)

        # Convert scaled predictions back to original scale
        original_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        reconstructed_targets[hidden_feature] = original_predictions

    return reconstructed_targets
