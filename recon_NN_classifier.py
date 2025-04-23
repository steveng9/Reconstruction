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


test_size_ = 0.2
hidden_dims_ = [128, 96, 64]
batch_size_ = 264
learning_rate_ = 0.003
epochs_ = 400
patience = 7



def main():
    qis = [
        "QI1",
        "QI2",
    ]
    sdg_practice_problems = [
        "25_Demo_AIM_e1_25f_Deid.csv",
        "25_Demo_TVAE_25f_Deid.csv",
        "25_Demo_CellSupression_25f_Deid.csv",
        "25_Demo_Synthpop_25f_Deid.csv",
        "25_Demo_ARF_25f_Deid.csv",
        "25_Demo_RANKSWAP_25f_Deid.csv",
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
            recon = nn_reconstruction(deid, targets, qi, hidden_features)
            reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

        # Set display width very large to avoid wrapping and truncating
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        print(reconstruction_scores)
        sums = reconstruction_scores.iloc[:, 1:].sum()
        print(sums)








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
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
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


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, early_stopping_patience=patience):
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

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break


    # Part 6: Visualize training progress
    # ------------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
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


def nn_reconstruction(deid, targets, qi, hidden_features, test_size=test_size_, hidden_dims=hidden_dims_, batch_size=batch_size_, learning_rate=learning_rate_, epochs=epochs_):

    targets_copy = targets.copy()
    for hidden_feature in hidden_features:
        print(f'\n\nHidden feature: {hidden_feature}')

        # Process categorical data
        X_encoded, y_encoded, targets_encoded, feature_encoder, label_encoder = process_categorical_data(deid[qi+[hidden_feature]], hidden_feature, targets)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=test_size, random_state=42)

        # Create datasets and dataloaders
        train_dataset = CategoricalDataset(X_train, y_train)
        test_dataset = CategoricalDataset(X_test, y_test)
        targets_dataset = CategoricalDataset(targets_encoded, np.zeros(len(targets_encoded)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        targets_loader = DataLoader(targets_dataset, batch_size=batch_size)

        # Determine dimensions
        input_dim = X_train.shape[1]  # This will be the total number of one-hot encoded features
        num_classes = len(np.unique(y_encoded))

        # Create model
        model = CategoricalNN(input_dim, hidden_dims, num_classes)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Train model
        trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)

        # Make predictions
        predictions, _ = predict(trained_model, targets_loader, device)

        # Convert encoded predictions back to original labels
        original_predictions = label_encoder.inverse_transform(predictions)
        targets_copy[hidden_feature] = original_predictions

    # return trained_model, history, original_predictions, original_true_labels, feature_encoder, label_encoder
    return targets_copy.astype(int)




# # Alternative approach: Using embeddings for categorical features
# class EmbeddingNN(nn.Module):
#     def __init__(self, categorical_features, embedding_dims, hidden_dims, output_dim, dropout_rate=0.2):
#         """
#         categorical_features: List of (num_categories, embedding_dim) tuples
#         embedding_dims: List of embedding dimensions for each categorical feature
#         """
#         super(EmbeddingNN, self).__init__()
#
#         # Create embeddings for each categorical feature
#         self.embeddings = nn.ModuleList()
#         total_embedding_dim = 0
#
#         for idx, (num_categories, embedding_dim) in enumerate(categorical_features):
#             self.embeddings.append(nn.Embedding(num_categories, embedding_dim))
#             total_embedding_dim += embedding_dim
#
#         # Create sequential layers
#         layers = []
#         prev_dim = total_embedding_dim
#
#         # Hidden layers
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.Dropout(dropout_rate))
#             prev_dim = hidden_dim
#
#         # Output layer
#         layers.append(nn.Linear(prev_dim, output_dim))
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x_categorical):
#         """
#         x_categorical: List of tensors, each tensor contains indices for one categorical feature
#         """
#         # Get embeddings for each feature
#         embeddings = []
#         for i, embedding_layer in enumerate(self.embeddings):
#             embeddings.append(embedding_layer(x_categorical[:, i].long()))
#
#         # Concatenate all embeddings
#         x = torch.cat(embeddings, dim=1)
#
#         # Pass through the rest of the network
#         return self.model(x)
#
#
# # Function to process data for embedding approach
# def process_data_for_embeddings(df, target_column):
#     # Process target
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(df[target_column].values)
#
#     # Process features
#     X = df.drop(columns=[target_column])
#     categorical_encoders = {}
#     encoded_features = []
#     categorical_features_info = []  # Store (num_categories, embedding_dim) for each feature
#
#     for column in X.columns:
#         # Create a label encoder for each feature
#         cat_encoder = LabelEncoder()
#         encoded_col = cat_encoder.fit_transform(X[column].values)
#         encoded_features.append(encoded_col.reshape(-1, 1))
#
#         # Store encoder
#         categorical_encoders[column] = cat_encoder
#
#         # Calculate embedding dimension (rule of thumb: min(50, num_categories/2))
#         num_categories = len(cat_encoder.classes_)
#         embedding_dim = min(50, max(1, num_categories // 2))
#         categorical_features_info.append((num_categories, embedding_dim))
#
#     # Combine all encoded features
#     X_encoded = np.hstack(encoded_features)
#
#     return X_encoded, y_encoded, categorical_encoders, label_encoder, categorical_features_info


# Example usage with embeddings approach
"""
def run_embedding_classification(df, target_column, test_size=0.2, hidden_dims=[128, 64],
                               batch_size=64, learning_rate=0.001, epochs=50):
    # Process data for embeddings
    X_encoded, y_encoded, cat_encoders, label_encoder, cat_features_info = process_data_for_embeddings(df, target_column)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=test_size, random_state=42)

    # Create datasets
    train_dataset = CategoricalDataset(X_train, y_train)  # Reuse our dataset class
    test_dataset = CategoricalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model with embeddings
    num_classes = len(np.unique(y_encoded))
    model = EmbeddingNN(cat_features_info, [emb_dim for _, emb_dim in cat_features_info], 
                       hidden_dims, num_classes)

    # Rest of training and evaluation code is the same...
"""

# Example usage
"""
import pandas as pd

# Example dataset with categorical features (adjust with your own data)
data = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'High School'],
    'occupation': ['Engineer', 'Teacher', 'Doctor', 'Engineer', 'Artist', 'Teacher'],
    'marital_status': ['Single', 'Married', 'Divorced', 'Single', 'Married', 'Divorced'],
    'income_class': ['Low', 'Medium', 'High', 'Medium', 'Low', 'Medium']
}

df = pd.DataFrame(data)

# Run classification
model, history, preds, true_labels, feature_encoder, label_encoder = run_categorical_classification(
    df, 'income_class', hidden_dims=[64, 32], epochs=20
)
"""







if __name__ == "__main__":
    main()