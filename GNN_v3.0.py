#import relevant libraries 
import networkx as nx
import matplotlib.pyplot as plt
import os 
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.manifold import TSNE

# Importing the Plantoid Cora Dataset.
def load_dataset():
    valid_datasets = ['Cora', 'CiteSeer', 'PubMed']
    dataset_name = ''
    while dataset_name not in valid_datasets:
        dataset_name = input("Please enter a dataset name (Cora, CiteSeer, or PubMed): ")

    dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    return dataset

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(dataset.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the selected dataset
dataset = load_dataset()

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print(data)

def select_gnn_type():
    gnn_types = ['Spectral GCN', 'Spatial GCN', 'GAT']
    selected_gnn = ''

    while selected_gnn not in gnn_types:
        selected_gnn = input("Please choose a GNN type (Spectral GCN, Spatial GCN, or GAT): ")

    return selected_gnn

# User selects the GNN type
gnn_type = select_gnn_type()

if gnn_type == 'Spectral GCN':
    model = GCN(hidden_channels=16)
elif gnn_type == 'Spatial GCN':
    model = GraphSAGE(hidden_channels=16)
else:  # gnn_type == 'GAT'
    model = GAT(hidden_channels=8, heads=8)

print(model)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# Visualizing the Untrained GAT network
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# Step 6: Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Step 7: Test the model
test_acc = test(model, data)
print(f'{gnn_type} Accuracy: {test_acc:.4f}')

# Step 8: Visualize the trained model
model.eval()
out = model(data.x, data.edge_index) 
visualize(out, color=data.y)



