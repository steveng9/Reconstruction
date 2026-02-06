# Node Classification with PyTorch Geometric for Tabular Data
import sys
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from util import features_25, calculate_reconstruction_score, QIs

lr_ = 0.01
epochs_ = 700
hidden_dim_ = 64
num_hidden_layers_ = 2
early_stopping_patience = 13

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
        reconstruction_scores = pd.DataFrame(index=features_25)
        reconstruction_scores["nunique"] = targets_original.nunique()

        qi = QIs[qi_name]
        hidden_features = list(set(features_25).difference(set(qi)))
        reconstruction_scores[f"{qi_name}_random_guess"] = np.NAN
        reconstruction_scores.loc[hidden_features, f"{qi_name}_random_guess"] = pd.Series(round(1 / targets_original.nunique() * 100, 1), index=features_25)
        targets = targets_original[qi].copy()

        for deid_filename in sdg_practice_problems:

            sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
            print("\n\n", qi_name, sdg_method_name)
            deid = pd.read_csv(join(mypath, deid_filename))

            recon_method_name = f"{qi_name}_{sdg_method_name}"
            recon = gnn_node_classification_reconstruction(deid, targets, qi, hidden_features)
            reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

        # Set display width very large to avoid wrapping and truncating
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        np.set_printoptions(linewidth=np.inf)

        print(qi_name)
        for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy()[2:]:
            for x in l:
                print(x, end=",")
            print()



def gnn_node_classification_reconstruction(deid, targets, qi, hidden_features):
    recon = targets.copy()
    X = deid[qi]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    targets_scaled = scaler.transform(targets[qi])
    graph_params = {'k': 5, 'threshold': 0.5, 'method': 'knn'}
    train_graph = create_graph_from_tabular(X_scaled, graph_params)
    targets_graph = create_graph_from_tabular(targets_scaled, graph_params)

    for hidden_feature in hidden_features:
        print("\n\n", hidden_feature)

        y = deid[hidden_feature]
        train_graph.y = torch.tensor(y, dtype=torch.long)

        # Initialize model
        input_dim = X.shape[1]  # Number of features
        hidden_dim = hidden_dim_
        output_dim = len(np.unique(y))  # Number of classes
        num_hidden_layers = num_hidden_layers_

        model = GNN(input_dim, hidden_dim, num_hidden_layers, output_dim, model_type='GraphSAGE')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=5e-4)

        # Train model
        train_losses, train_accs, idx_to_class = train_gnn(train_graph, model, optimizer, epochs=epochs_)

        # Evaluate model on test data
        targets_preds = evaluate_gnn(model, targets_graph, idx_to_class)
        recon[hidden_feature] = targets_preds

        # # Part 6: Visualize training progress
        # # ------------------------------------
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(train_losses)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        # plt.subplot(1, 2, 2)
        # plt.plot(train_accs)
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training Accuracy')
        #
        # plt.tight_layout()

    return recon


def create_graph_from_tabular(features, graph_params, labels=None):
    """
    Create a graph from tabular data:
    - Each row becomes a node
    - Node features are the row values
    - Edges are created based on similarity (either KNN or threshold-based)

    Args:
        features: Feature matrix (rows are samples, columns are features)
        labels: Target labels
        k: Number of nearest neighbors (for KNN method)
        threshold: Similarity threshold (for threshold method)
        method: 'knn' or 'threshold'

    Returns:
        PyG Data object representing the graph
    """
    x = torch.FloatTensor(features)

    # Calculate similarity matrix (using cosine similarity)
    sim_matrix = cosine_similarity(features)

    if graph_params['method'] == 'knn':
        edge_list = []
        for i in range(len(features)):
            # Get indices of k most similar nodes (excluding self)
            similarities = sim_matrix[i]
            similarities[i] = -np.inf  # Exclude self-connection
            top_k_indices = np.argsort(similarities)[-graph_params['k']:]

            # Add edges (in both directions for undirected graph)
            for j in top_k_indices:
                edge_list.append((i, j))
                # edge_list.append((j, i))

    elif graph_params['method'] == 'threshold':
        # Connect nodes with similarity above threshold
        edge_list = []
        for i in range(len(features)):
            for j in range(len(features)):
                if i != j and sim_matrix[i, j] > graph_params['threshold']:
                    edge_list.append((i, j))

    # Create edge index tensor [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create labels tensor if provided
    y = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data



class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim, model_type='GCN'):
        super(GNN, self).__init__()
        self.convs = []

        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            for _ in range(num_hidden_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        else:  # 'GraphSAGE'
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            for _ in range(num_hidden_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        for i, conv in enumerate(self.convs):
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            if i < len(self.convs)-1:
                x = conv(x, edge_index)
            else:
                x = conv(x)
        return x

#
# def one_hot_encode_sorted(arr):
#     """One-hot encodes a NumPy array based on its sorted unique values.
#
#     Args:
#         arr (np.ndarray): The input NumPy array.
#
#     Returns:
#          np.ndarray: The one-hot encoded array.
#     """
#     unique_sorted = np.unique(arr)
#     mapping = {val: i for i, val in enumerate(unique_sorted)}
#
#     one_hot_encoded = np.zeros((arr.size, len(unique_sorted)), dtype=int)
#     one_hot_encoded[np.arange(arr.size), [mapping[x] for x in arr.flatten()]] = 1
#
#     return one_hot_encoded


def train_gnn(data, model, optimizer, epochs=400):
    """Train the GNN model on the given data."""
    model.train()

    train_losses = []
    train_accs = []
    best_loss = float('inf')

    unique_classes = torch.unique(data.y)
    class_to_idx = {class_val.item(): idx for idx, class_val in enumerate(unique_classes)}
    idx_to_class = {idx: class_val for class_val, idx in class_to_idx.items()}
    label_indices = torch.tensor([class_to_idx[label.item()] for label in data.y])

    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        # loss = F.cross_entropy(out, data.y.squeeze())
        loss = F.cross_entropy(out, label_indices)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        pred_idx = out.argmax(dim=1)
        predicted_classes = torch.tensor([idx_to_class[idx.item()] for idx in pred_idx])

        acc = predicted_classes.eq(data.y).sum().item() / len(data.y)

        # Save metrics
        train_losses.append(loss.item())
        train_accs.append(acc)

        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')


        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return train_losses, train_accs, idx_to_class


def evaluate_gnn(model, targets_graph, model_output_idx_to_classes):
    model.eval()
    with torch.no_grad():
        out = model(targets_graph.x, targets_graph.edge_index)
        pred_idx = out.argmax(dim=1).numpy()
        pred_classes = torch.tensor([model_output_idx_to_classes[idx.item()] for idx in pred_idx])

    return pred_classes

#
# # Part 7: Function to make predictions on new data
# # ------------------------------------
# def predict_new_samples(model, train_features, new_features, graph_params):
#     """
#     Make predictions for new data samples:
#     1. Create a graph connecting new samples to training data
#     2. Run inference on this combined graph
#     3. Extract predictions for new samples
#     """
#     model.eval()
#
#     # Combine training and new features
#     combined_features = np.vstack([train_features, new_features])
#
#     # Create a graph from combined features (without labels)
#     combined_graph = create_graph_from_tabular(
#         combined_features, labels=None,
#         k=graph_params['k'],
#         threshold=graph_params['threshold'],
#         method=graph_params['method']
#     )
#
#     with torch.no_grad():
#         # Get predictions for all nodes
#         out = model(combined_graph.x, combined_graph.edge_index)
#         preds = out.argmax(dim=1).numpy()
#
#         # Extract predictions for new samples only
#         new_sample_preds = preds[len(train_features):]
#
#     return new_sample_preds


# Example usage:
# Assuming we have some new samples to predict
# new_samples = ...
# predictions = predict_new_samples(model, X_train, new_samples, graph_params)
# print("Predictions for new samples:", predictions)

# Example of how you would use this with your own data:
"""
# 1. Load your tabular data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# 2. Preprocess (scaling, encoding, etc.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# 4. Create graph and train model as shown above

# 5. Make predictions on new data
new_data = ... # Your new tabular data
new_data_scaled = scaler.transform(new_data)
predictions = predict_new_samples(model, X_train, new_data_scaled, graph_params)
"""

if __name__ == '__main__':
    main()