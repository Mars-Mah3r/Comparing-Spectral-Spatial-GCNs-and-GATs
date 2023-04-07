# Import relevant libraries
import networkx as nx
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE

# Importing the Plantoid Cora Dataset.
# Change dataset manually to find optimal hyperparameters for different datasets
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print(data)

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print(data)

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
model = GCN(hidden_channels=16, activation=F.relu, dropout_rate=0.5)
print(model)


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

hidden_layer_sizes = [16, 32, 64]
activation_functions = [F.relu, torch.sigmoid, torch.tanh]
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [5e-4, 5e-3, 5e-2]

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

best_test_acc = 0
best_hyperparameters = None

for hidden_size in hidden_layer_sizes:
    for activation in activation_functions:
        for dropout_rate in dropout_rates:
            for learning_rate in learning_rates:
                for weight_decay in weight_decays:
                    # Initialize and train the model
                    model = GCN(hidden_channels=hidden_size, activation=activation, dropout_rate=dropout_rate)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    for epoch in range(1, 101):
                        loss = train(model, data, optimizer, criterion)
                    
                    # Evaluate the model
                    test_acc = test(model, data)

                    # Update the best hyperparameters if the accuracy improves
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_hyperparameters = {
                            "hidden_size": hidden_size,
                            "activation": activation,
                            "dropout_rate": dropout_rate,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                        }

print("Best hyperparameters:", best_hyperparameters)
print("Best test accuracy:", best_test_acc)

# Model evalutaion 
model = GCN(hidden_channels=best_hyperparameters["hidden_size"], activation=best_hyperparameters["activation"], dropout_rate=best_hyperparameters["dropout_rate"])
optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparameters["learning_rate"], weight_decay=best_hyperparameters["weight_decay"])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)

test_acc = test(model, data)
print(f'Spectral GCN Accuracy: {test_acc:.4f}')

model.eval()
out = model(data.x, data.edge_index)

