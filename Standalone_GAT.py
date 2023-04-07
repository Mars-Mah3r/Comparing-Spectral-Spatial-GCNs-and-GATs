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

# Creating a GAT model structure containing two GATConv layers
# with a dropout rate of 0.6. Model contains 8 hidden channels.
# Learning rate: 0.005

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

model = GAT(hidden_channels=8, heads=8)
print(model)

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

# Training the GAT
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

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

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Model evaluation
test_acc = test()
print(f'GAT Accuracy: {test_acc:.4f}')

model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# best_test_acc = 0
# best_epoch = 0

# for epoch in range(1, 101):
#     loss = train()
#     current_test_acc = test()
#     if current_test_acc > best_test_acc:
#         best_test_acc = current_test_acc
#         best_epoch = epoch
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {current_test_acc:.4f}')

# print(f'Best Test Accuracy: {best_test_acc:.4f} at Epoch: {best_epoch}')