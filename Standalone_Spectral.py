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


# Define the GCN model structure containing two GCNConv layers
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model with 16 hidden channels
model = GCN(hidden_channels=16)
print(model)

# Visulazing the Untrained GCN network 
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

# Training the Spectral GNN
model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

# Define the testing function
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

# Train the model for 100 epochs and print the loss for each epoch
for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# Model evalutaion 
test_acc = test()
print(f'Spectral GCN Accuracy: {test_acc:.4f}')

model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

best_test_acc = 0
best_epoch = 0


                            #### UNcomment to obtain best epoxh ####
# for epoch in range(1, 101):
#     loss = train()
#     current_test_acc = test()
#     if current_test_acc > best_test_acc:
#         best_test_acc = current_test_acc
#         best_epoch = epoch
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {current_test_acc:.4f}')

# print(f'Best Test Accuracy: {best_test_acc:.4f} at Epoch: {best_epoch}')
