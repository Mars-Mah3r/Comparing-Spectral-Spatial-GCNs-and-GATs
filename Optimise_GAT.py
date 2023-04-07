import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
from sklearn.manifold import TSNE

dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())

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

# Creating a GAT model structure containing two GATConv layers
# with a dropout rate of 0.6. Model contains 8 hidden channels.
# Learning rate: 0.005

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads, dropout_rate, activation_function):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1)
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

hidden_layer_sizes = [8, 16, 32]
heads = [4, 8, 16]
activation_functions = [F.relu, torch.sigmoid, torch.tanh]
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [5e-4, 5e-3, 5e-2]
activation_functions = [F.relu, F.elu, F.leaky_relu]


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
    for head in heads:
        for dropout_rate in dropout_rates:
            for learning_rate in learning_rates:
                for weight_decay in weight_decays:
                    for activation_function in activation_functions:
                        # Initialize and train the model
                        model = GAT(hidden_channels=hidden_size, heads=head, dropout_rate=dropout_rate, activation_function=activation_function)
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
                                "heads": head,
                                "dropout_rate": dropout_rate,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "activation_function": activation_function,
                            }

print("Best hyperparameters:", best_hyperparameters)
print("Best test accuracy:", best_test_acc)

# Model evaluation
model = GAT(hidden_channels=best_hyperparameters["hidden_size"], heads=best_hyperparameters["heads"], dropout_rate=best_hyperparameters["dropout_rate"], activation_function=best_hyperparameters["activation_function"])
optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparameters["learning_rate"], weight_decay=best_hyperparameters["weight_decay"])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)

test_acc = test(model, data)
print(f'Optimized GAT Accuracy: {test_acc:.4f}')

model.eval()
out = model(data.x, data.edge_index)
