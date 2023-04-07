import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
from sklearn.manifold import TSNE

# Function to load the Planetoid dataset (Cora, CiteSeer, or PubMed)
def load_dataset():
    valid_datasets = ['Cora', 'CiteSeer', 'PubMed']
    dataset_name = ''
    while dataset_name not in valid_datasets:
        dataset_name = input("Please enter a dataset name (Cora, CiteSeer, or PubMed): ")

    dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    return dataset

# Load the selected dataset
dataset = load_dataset()

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Get the first graph object from the dataset
data = dataset[0] 
print(data)

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

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Defining the testing function 
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# Define the training and evaluation process
# Define the training function
# Define the training and evaluation process
def train_and_evaluate(hidden_channels, heads, learning_rate, weight_decay, dropout):
    model = GAT(hidden_channels, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc

    for epoch in range(1, 101):
        loss = train()
    
    test_acc = test()
    return test_acc


# Define the hyperparameter search space
hidden_channels_list = [8, 16]
heads_list = [8, 16]
learning_rate_list = [0.005, 0.01]
weight_decay_list = [5e-4, 1e-4]
dropout_list = [0.6, 0.4]

best_test_acc = 0
best_params = None

for hidden_channels in hidden_channels_list:
    for heads in heads_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                for dropout in dropout_list:
                    test_acc = train_and_evaluate(hidden_channels, heads, learning_rate, weight_decay, dropout)
                    
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_params = {
                            'hidden_channels': hidden_channels,
                            'heads': heads,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'dropout': dropout
                        }

print(f"Best test accuracy: {best_test_acc}")
print(f"Best hyperparameters: {best_params}")

# Train and evaluate the model with the best hyperparameters
model = GAT(best_params['hidden_channels'], best_params['heads'])
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train()

test_acc = test()
print(f'GAT Accuracy: {test_acc:.4f}')

model.eval()

out = model(data.x, data.edge_index)
