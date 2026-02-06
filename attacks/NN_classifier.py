
import sys

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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# defaults
test_size_default = 0.2
hidden_dims_default = [300]  # or [128, 96, 64]
batch_size_default = 264
learning_rate_default = 0.0003
epochs_default = 250
patience_default = 200
dropout_rate_default = 0.2
PLOT = False


def _development():
    qis = [
        "QI1",
        # "QI2",
    ]
    sdg_practice_problems = [
        "25_Demo_AIM_e1_25f_Deid.csv",
        "25_Demo_TVAE_25f_Deid.csv",
        # "25_Demo_CellSupression_25f_Deid.csv",
        # "25_Demo_Synthpop_25f_Deid.csv",
        # "25_Demo_ARF_25f_Deid.csv",
        # "25_Demo_RANKSWAP_25f_Deid.csv",
        "25_Demo_MST_e10_25f_Deid.csv",
    ]

    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    target_filename = "25_Demo_25f_OriginalData.csv"
    targets_original = pd.read_csv(join(mypath, target_filename))

    for qi_name in qis:
        qi = QIs[qi_name]
        hidden_features = list(set(features_25).difference(set(qi)))

        reconstruction_scores = pd.DataFrame(index=features_25)
        reconstruction_scores["nunique"] = targets_original.nunique()
        reconstruction_scores[f"{qi_name}_random_guess"] = np.NAN
        reconstruction_scores.loc[hidden_features, f"{qi_name}_random_guess"] = pd.Series(round(1 / targets_original.nunique() * 100, 1), index=features_25)
        targets = targets_original[qi]

        for deid_filename in sdg_practice_problems:

            sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
            print("\n\n", qi_name, sdg_method_name)
            deid = pd.read_csv(join(mypath, deid_filename))

            recon_method_name = f"{qi_name}_{sdg_method_name}"
            # recon = logistic_regression_reconstruction(deid, targets, qi, hidden_features)
            # recon = nn_reconstruction(deid, targets, qi, hidden_features)
            recon = chained_mlp_reconstruction(deid, targets, qi, hidden_features)
            reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

            print(qi_name, recon_method_name)
            # for x in reconstruction_scores.loc[sorted(hidden_features), recon_method_name].T.to_numpy():
            #     print(x, end=",")
            print(reconstruction_scores[recon_method_name].mean())

        # Set display width very large to avoid wrapping and truncating
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)


        for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy()[2:]:
            for x in l:
                print(x, end=",")
            print()
            # print(round(l.mean(), 2))
        print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.iloc[2:].mean().mean(), 2))



def mlp_classification_reconstruction(cfg, synth, targets, known_features, hidden_features):
    """Main adapter that reads from config or uses defaults."""
    return nn_classification_reconstruction(
        synth, targets, known_features, hidden_features,
        cfg["attack_params"].get("test_size", test_size_default),
        cfg["attack_params"].get("hidden_dims", hidden_dims_default),
        cfg["attack_params"].get("batch_size", batch_size_default),
        cfg["attack_params"].get("learning_rate", learning_rate_default),
        cfg["attack_params"].get("epochs", epochs_default),
        cfg["attack_params"].get("patience", patience_default),
        cfg["attack_params"].get("dropout_rate", dropout_rate_default)
    ), None, None


def chained_mlp_classification_reconstruction(cfg, synth, targets, known_features, hidden_features):
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

        recon = nn_classification_reconstruction(
            synth, reconstructed_targets, known_features_copy, [hidden_feature],
            test_size=0.2,
            hidden_dims=[dim_size],
            batch_size=264,
            learning_rate=0.0002,
            epochs=feature_epochs,
            patience=patience_default,
            dropout_rate=dropout_rate_default
        )

        reconstructed_targets[hidden_feature] = recon[hidden_feature]
        known_features_copy.append(hidden_feature)

    return reconstructed_targets, None, None


# Custom dataset class for categorical data
class CategoricalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural Network model
class CategoricalNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        if torch.cuda.is_available():
            print("Using CUDA device :)")
        else:
            print("NOT Using CUDA!")
        super(CategoricalNN, self).__init__()
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


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping_patience):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

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

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_accuracy)

        if epoch % 20 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '../models/best_model_categorical.pth')
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
    model.load_state_dict(torch.load('best_model_categorical.pth'))
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
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    return np.array(predictions), np.array(true_labels)


# Process categorical features
def process_categorical_data(df, target_column, targets):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # Set up one-hot encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    targets_encoded = encoder.transform(targets)

    # Encode target if it's categorical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded, targets_encoded, encoder, label_encoder


def nn_classification_reconstruction(synth, targets, known_features, hidden_features,
                                     test_size, hidden_dims, batch_size, learning_rate,
                                     epochs, patience, dropout_rate):
    reconstructed_targets = targets.copy()

    for hidden_feature in hidden_features:
        print(f'\n\nHidden feature: {hidden_feature}')

        X_encoded, y_encoded, targets_encoded, feature_encoder, label_encoder = process_categorical_data(
            synth[known_features + [hidden_feature]], hidden_feature, targets[known_features]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=42
        )

        train_dataset = CategoricalDataset(X_train, y_train)
        test_dataset = CategoricalDataset(X_test, y_test)
        targets_dataset = CategoricalDataset(targets_encoded, np.zeros(len(targets_encoded)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        targets_loader = DataLoader(targets_dataset, batch_size=batch_size)

        # Determine dimensions
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_encoded))

        model = CategoricalNN(input_dim, hidden_dims, num_classes, dropout_rate)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trained_model, history = train_model(
            model, train_loader, test_loader, criterion, optimizer, device, epochs, patience
        )

        predictions, _ = predict(trained_model, targets_loader, device)

        original_predictions = label_encoder.inverse_transform(predictions)
        reconstructed_targets[hidden_feature] = original_predictions

    return reconstructed_targets.astype(int)