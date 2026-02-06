import sys

import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from util import calculate_reconstruction_score

# from pytorch_geometric.examples.signed_gcn import pos_edge_index, neg_edge_index


testing = True
update = 10

# For reproducibility
# torch.manual_seed(42)
node_embedding_size = 20
testing_data_size = 100
num_nonhidden_features = 7
num_hidden_features = 3

num_epochs = 200
learning_rate = .05
hidden_channels=128
out_channels=64
num_hidden_layers=1

# Simple GNN model for link prediction
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hidden_layers, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = [GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers)]
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        # Get node embeddings
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = conv(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Simple dot product decoder
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)

    def decode_all(self, z):
        # Compute pairwise dot products for all possible node pairs
        prob_adj = z @ z.t()
        return prob_adj

    def forward(self, x, pos_edge_index, neg_edge_index=None):
        z = self.encode(x, pos_edge_index)

        pos_score = self.decode(z, pos_edge_index)

        if neg_edge_index is not None:
            neg_score = self.decode(z, neg_edge_index)
            return pos_score, neg_score

        return pos_score

def convert_to_graph(synth, targets, nonhidden_features, hidden_features):

    # NODE NUMBERING SCHEME:
    # node i is from row (i//num_cols) and column (i%num_cols)
    # and row i, col j makes node (i*num_cols+j)

    # assert synth.shape[1] == targets.shape[1]
    num_features = synth.shape[1]
    num_synth_rows = synth.shape[0]
    num_target_rows = targets.shape[0]
    num_synth_nodes = num_synth_rows * num_features
    num_target_nodes = num_target_rows * num_features

    synth_target_known_edge_index_i, synth_target_known_edge_index_j = [], []
    synth_pos_inference_edge_index_i, synth_pos_inference_edge_index_j = [], []
    synth_neg_inference_edge_index_i, synth_neg_inference_edge_index_j = [], []
    target_inference_edge_index_i, target_inference_edge_index_j = [], []

    # fully connect all nodes belonging to same row
    for row in range(num_synth_rows + num_target_rows):
        for node_i, node_j in combinations(range(row*num_features, (row+1)*num_features), 2):
            synth_target_known_edge_index_i.append(node_i)
            synth_target_known_edge_index_j.append(node_j)

    # fully connect all NONHIDDEN nodes representing corresponding features
    # (i.e. node # (k * num_features + feature_num) for any integer k) IF they share the same value.

    # KNOWN FEATURES
    # Don't need negative links for these
    for feature_num, feature in enumerate(nonhidden_features):

        value_to_nodes = {value: [] for value in list(synth[feature].unique())+list(targets[feature].unique())}
        synth.apply(lambda row_: value_to_nodes[row_[feature]].append(row_.name), axis=1)
        targets.apply(lambda row_: value_to_nodes[row_[feature]].append(row_.name+num_synth_rows), axis=1)

        for _, rows in value_to_nodes.items():
            for row_a, row_b in combinations(rows, 2):
                node_i = row_a * num_features + feature_num
                node_j = row_b * num_features + feature_num
                synth_target_known_edge_index_i.append(node_i)
                synth_target_known_edge_index_j.append(node_j)

    # TRAINING FEATURES
    # need to separate edges from "negative" edges
    for x, feature in enumerate(hidden_features):
        feature_num = x + num_nonhidden_features
        value_to_nodes = {value: [] for value in list(synth[feature].unique())}
        synth.apply(lambda row_: value_to_nodes[row_[feature]].append(row_.name), axis=1)
        edges_for_this_feature = set()

        for _, rows in value_to_nodes.items():
            for row_a, row_b in combinations(rows, 2):
                node_i = row_a * num_features + feature_num
                node_j = row_b * num_features + feature_num
                synth_pos_inference_edge_index_i.append(node_i)
                synth_pos_inference_edge_index_j.append(node_j)
                edges_for_this_feature.add((node_i, node_j))

        for node_i, node_j in combinations(range(feature_num, num_synth_nodes, num_features), 2):
            if (node_i, node_j) not in edges_for_this_feature and (node_j, node_i) not in edges_for_this_feature:
                synth_neg_inference_edge_index_i.append(node_i)
                synth_neg_inference_edge_index_j.append(node_j)

    # HIDDEN / INFERENCE FEATURES
    # look at all potential links between target nodes and synth nodes for hidden features
    for x, feature in enumerate(hidden_features):
        feature_num = x + num_nonhidden_features

        # create edge from each target to EVERY synth node in same column for link prediction
        for row_k in range(num_synth_rows, num_synth_rows+num_target_rows):
            target_inference_edge_index_i += [row_k*num_features + feature_num] * num_synth_rows
            target_inference_edge_index_j += list(range(feature_num, num_synth_rows*num_features, num_features))

    synth_target_known_edge_index = to_undirected(torch.tensor([synth_target_known_edge_index_i, synth_target_known_edge_index_j], dtype=torch.long))
    synth_pos_inference_edge_index = torch.tensor([synth_pos_inference_edge_index_i, synth_pos_inference_edge_index_j], dtype=torch.long)
    synth_neg_inference_edge_index = torch.tensor([synth_neg_inference_edge_index_i, synth_neg_inference_edge_index_j], dtype=torch.long)
    # todo: should I make this one for inference also undirected, or just one direction?
    target_inference_edge_index = torch.tensor([target_inference_edge_index_i, target_inference_edge_index_j], dtype=torch.long)

    # data = Data(x=torch.from_numpy(np.zeros((num_synth_nodes+num_target_nodes, node_embedding_size))).float(), edge_index=synth_target_known_edge_index)
    data = Data(x=torch.from_numpy(np.ones((num_synth_nodes+num_target_nodes, node_embedding_size))).float(), edge_index=synth_target_known_edge_index)
    # data = Data(x=torch.from_numpy(np.random.random((num_synth_nodes+num_target_nodes, node_embedding_size))).float(), edge_index=synth_target_known_edge_index)
    # todo: try putting the node values in the embeddings to begin

    return data, synth_target_known_edge_index, synth_pos_inference_edge_index, synth_neg_inference_edge_index, target_inference_edge_index




def main():

    synth_full = pd.read_csv("/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/10_MST_e10_50f_QID2_Deid.csv")
    targets = pd.read_csv('/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/10_MST_e10_50f_QID2_AttackTargets.csv')
    all_nonhidden_features = targets.drop('TargetID', axis=1, inplace=False).columns.tolist()
    all_hidden_features = list(set(synth_full.columns.tolist()).difference(set(all_nonhidden_features)))
    nonhidden_features = all_nonhidden_features[:num_nonhidden_features]
    hidden_features = all_hidden_features[:num_hidden_features]
    synth = synth_full[nonhidden_features + hidden_features].iloc[:testing_data_size]
    if testing:
        targets_with_solution = synth_full[nonhidden_features + hidden_features].iloc[testing_data_size:].sample(n=testing_data_size)
        targets_with_solution.index = list(range(testing_data_size))
        targets = targets_with_solution[nonhidden_features]
        targets_solution = targets_with_solution[hidden_features]
    else:
        targets = targets[nonhidden_features].iloc[:testing_data_size]

    num_nodes = synth.shape[0] * synth.shape[1]
    assert num_nodes == testing_data_size * (num_nonhidden_features + num_hidden_features)

    (data,
     synth_target_known_edge_index,
     synth_pos_inference_edge_index,
     synth_neg_inference_edge_index,
     target_inference_edge_index) = convert_to_graph(synth, targets, nonhidden_features, hidden_features)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNLinkPredictor(
        # in_channels=num_features,
        in_channels=node_embedding_size, # have no features (i.e. one column of zeros), and entirely use edge connectivity
        hidden_channels=hidden_channels,
        num_hidden_layers=num_hidden_layers,
        out_channels=out_channels
    ).to(device)

    # Move data to device
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    def train(pos_edge_index, neg_edge_index):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, synth_target_known_edge_index)

        pos_score = model.decode(z, pos_edge_index)
        neg_score = model.decode(z, neg_edge_index)

        # Compute BCE loss
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(pos_edge_index, neg_edge_index):
        model.eval()
        z = model.encode(data.x, synth_target_known_edge_index)

        pos_score = model.decode(z, pos_edge_index)
        neg_score = model.decode(z, neg_edge_index)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

        return roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())

    losses = []
    auc_scores = []
    for epoch in range(num_epochs):
        (train_pos_edge,
         train_neg_edge,
         test_pos_edge,
         test_neg_edge
         ) = custom_train_test_split_edges(
            synth_pos_inference_edge_index,
            synth_neg_inference_edge_index,
            train_proportion=0.8
        )

        loss = train(train_pos_edge, train_neg_edge)
        auc = test(test_pos_edge, test_neg_edge)
        losses.append(loss)
        auc_scores.append(auc)

        if epoch % update == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')


    plot_losses(losses, auc_scores)


    @torch.no_grad()
    def visualize_embeddings():
        model.eval()
        z = model.encode(data.x, pos_edge_index)
        z = z.cpu().numpy()

        # Use t-SNE for dimensionality reduction
        from sklearn.manifold import TSNE
        z_2d = TSNE(n_components=2).fit_transform(z)

        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=20, c=data.y.cpu().numpy(), cmap='tab10')
        plt.colorbar()
        plt.title('Node Embeddings')
        plt.savefig('node_embeddings.png')
        plt.show()

    # visualize_embeddings()

    @torch.no_grad()
    def predict_links():
        model.eval()
        z = model.encode(data.x, synth_target_known_edge_index)

        return model.decode(z, target_inference_edge_index)
        # for i, score in enumerate(pos_scores):
        #     print(f"Edge {pos_edges[0, i].item()} → {pos_edges[1, i].item()}: {torch.sigmoid(score).item():.4f}")

    scores = predict_links()
    print()

    def convert_predicted_links_to_table_attributes():
        num_features = synth.shape[1]
        num_synth_rows = synth.shape[0]
        num_target_rows = targets.shape[0]
        reconstruction = pd.DataFrame(index=range(num_target_rows), columns=hidden_features)

        for x, feature in enumerate(hidden_features):
            feature_num = x + num_nonhidden_features
            i = x * num_synth_rows * num_target_rows
            for row_k in range(num_target_rows):
                predicted_value = get_value_from_highest_link_score(feature_num, scores[i:i+num_synth_rows])
                reconstruction.iloc[row_k, x] = predicted_value
                i += num_synth_rows
        return reconstruction


    def get_value_from_highest_link_score(feature_num, scores_for_this_target):
        index_of_predicted_value = torch.argmax(scores_for_this_target)
        return synth.iloc[index_of_predicted_value.item(), feature_num]

    reconstruction = convert_predicted_links_to_table_attributes()

    calculate_reconstruction_score(targets_solution, reconstruction)




def custom_train_test_split_edges(synth_pos_inference_edge_index, synth_neg_inference_edge_index, train_proportion=0.8):
    num_pos_edges = synth_pos_inference_edge_index.shape[1]
    indices = np.random.permutation(num_pos_edges)
    train_cols = int(num_pos_edges * train_proportion)
    train_indices = indices[:train_cols]
    test_indices = indices[train_cols:]

    train_pos_edge = synth_pos_inference_edge_index[:, train_indices]
    test_pos_edge = synth_pos_inference_edge_index[:, test_indices]

    num_neg_edges = synth_neg_inference_edge_index.shape[1]
    num_neg_edges_used = min(num_pos_edges, num_neg_edges)
    indices = np.random.permutation(num_neg_edges)
    train_cols = int(num_neg_edges_used * train_proportion)
    train_indices = indices[:train_cols]
    test_indices = indices[train_cols:num_neg_edges_used]

    train_neg_edge = synth_neg_inference_edge_index[:, train_indices]
    test_neg_edge = synth_neg_inference_edge_index[:, test_indices]

    # return train_pos_edge, train_neg_edge, test_pos_edge, test_neg_edge
    return to_undirected(train_pos_edge), to_undirected(train_neg_edge), to_undirected(test_pos_edge), to_undirected(test_neg_edge)


def plot_losses(losses, auc_scores):
    # Plot training metrics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(auc_scores)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('ROC-AUC Score')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()




main()