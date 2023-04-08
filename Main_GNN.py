import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Importing the Plantoid Cora Dataset.
def load_dataset():
    valid_datasets = ['Cora', 'CiteSeer', 'PubMed']
    dataset_name = ''
    while dataset_name not in valid_datasets:
        dataset_name = input("Please enter a dataset name (Cora, CiteSeer, or PubMed): ")

    dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    return dataset


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, activation, dropout_rate):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.activation = activation
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, activation, dropout_rate):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(dataset.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes)
        self.activation = activation
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
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

# Define the optimized hyperparameters
optimized_hyperparams = {
    'Spectral GCN': {
        'Cora': {'hidden_size': 16, 'activation': torch.tanh, 'dropout_rate': 0.5, 'learning_rate': 0.01, 'weight_decay': 0.0005},
        'CiteSeer': {'hidden_size': 64, 'activation': torch.tanh, 'dropout_rate': 0.3, 'learning_rate': 0.01, 'weight_decay': 0.0005},
        'PubMed': {'hidden_size': 64, 'activation': F.relu, 'dropout_rate': 0.3, 'learning_rate': 0.01, 'weight_decay': 0.0005}
    },
    'Spatial GCN': {
        'Cora': {'hidden_size': 16, 'activation': F.relu, 'dropout_rate': 0.5, 'learning_rate': 0.1, 'weight_decay': 0.0005},
        'CiteSeer': {'hidden_size': 16, 'activation': torch.tanh, 'dropout_rate': 0.7, 'learning_rate': 0.1, 'weight_decay': 0.005},
        'PubMed': {'hidden_size': 16, 'activation': torch.tanh, 'dropout_rate': 0.5, 'learning_rate': 0.01, 'weight_decay': 0.005}
    },
    'GAT': {
        'Cora': {'hidden_channels': 16, 'heads': 8, 'learning_rate': 0.01, 'weight_decay': 0.0005, 'dropout': 0.6},
        'CiteSeer': {'hidden_channels': 8, 'heads': 8, 'learning_rate': 0.005, 'weight_decay': 0.0005, 'dropout': 0.6},
        'PubMed': {'hidden_channels': 8, 'heads': 8, 'learning_rate': 0.005, 'weight_decay': 0.0005, 'dropout': 0.6}
    }
}

# Get the optimal hyperparameters for the selected GNN type and dataset
hyperparams = optimized_hyperparams[gnn_type][dataset.name]

if gnn_type == 'Spectral GCN':
    model = GCN(hidden_channels=hyperparams['hidden_size'], activation=hyperparams['activation'], dropout_rate=hyperparams['dropout_rate'])
elif gnn_type == 'Spatial GCN':
    model = GraphSAGE(hidden_channels=hyperparams['hidden_size'], activation=hyperparams['activation'], dropout_rate=hyperparams['dropout_rate'])
else:  # gnn_type == 'GAT'
    model = GAT(hidden_channels=hyperparams['hidden_channels'], heads=hyperparams['heads'], dropout=hyperparams['dropout'])

print(model)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc

# Visualizing the Untrained GAT network
def visualize(h, color, title):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.title(title)
    plt.show()

model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y, title=f'{gnn_type} on {dataset.name}')

# Step 6: Train the model
allvalidation_accuracy = []
alltest_acc = []

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])

for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)
    validation_accuracy = test(model, data, data.val_mask)
    test_accuracy = test(model, data, data.test_mask)
    allvalidation_accuracy.append(validation_accuracy)
    alltest_acc.append(test_accuracy)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Step 7: Test the model (final test accuracy)
test_accuracy = test(model, data, data.test_mask)
print(f'{gnn_type} Final Accuracy: {test_accuracy:.4f}')

# Step 8: Visualize the trained model
model.eval()
out = model(data.x, data.edge_index) 
visualize(out, color=data.y, title=f'{gnn_type} on {dataset.name}')

# Plotting the val and test accuracy
plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(allvalidation_accuracy) + 1), allvalidation_accuracy, label='Validation accuracy', c='blue')
plt.plot(np.arange(1, len(alltest_acc) + 1), alltest_acc, label='Testing accuracy', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'{gnn_type} on {dataset.name}')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig(f'{gnn_type}_accuracy.png')
plt.show()







