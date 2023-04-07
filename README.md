# Comparing-Spectral-Spatial-GCNs-and-GATs

## Abstract

This repository will include all files that were used in my 2023 6CCE3EEP Individual Project. 

In order to create GNN, the following article by Awan, A. A named [“A Comprehensive Introduction to Graph Neural Networks (GNNs).”](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial) was adapted. From this article, an understanding of the core techniques for implementing GNNs was obtained and hence a the code architecture was adapted to better suit the specific problem this paper aims to tackle.

This is the full breakdown of how the [source-code](https://app.datacamp.com/workspace/w/4bb43f79-421f-4a6e-ba51-f972ea996db8) was adapted:

-	The original data set used was “Cora” from Planetoid ( more information on this in section 3.14 ), the code was changed to allow the user to select one of three Datasets from Planetoid, consisting of Cora, CiteSeer, and PubMed
-	The original article only provided a rough template on creating Spectral GCN. This was adapted and expanded to include a method of implementing a Spatial GCN and a Graph Attention Network. 
-	The article had provided pre-determined hyperparameters, with little to no information on how they were obtained. In turn the code was adapted to have the hyperparameters tuned depending on the GNN being used 
-	Through the above mentioned change, the training process was adjusted.
-	The original code was refactored to become more modular, efficient , readable and to be better integrated in accordance with this papers objectives. 

## Requirements 
_Note: This repository requires a minimum of Python 3.7 for PyTorch 1.13.x, and Python 3.8 for Pytorch 2.0._

Recommended PiP installations:
```
pip install networkx
pip install matplotlib
pip install torch
pip install torch-geometric
pip install scikit-learn
```

This repository the Python 3.10.5 and the following Packages:
- torch             ==       1.13.1
- torch-cluster      ==      1.6.0
- torch-geometric      ==    2.3.0
- torch-scatter       ==     2.1.0
- torch-sparse        ==     0.6.16
- torch-spline-conv    ==    1.2.1
- torchvision        ==      0.14.1
- matplotlib       ==        3.7.1
- matplotlib-inline      ==  0.1.6
- networkx         ==        3.0
- numpy              ==      1.24.2 

## Running the Model
To run the model, simply run the following command:
``` 
$ python Main_GNN.py
```
You will then be prompted to provide two inputs, the first asking for a dataset, input one of:
```
Cora, CiteSeer, PubMed
```
Then select which GNN method: 
```
Spectral_GCN, Spatial_GCN, GAT
```

_Note: after the dispalyed visualisation of the untrained network, please close the window to conitnue executing the code_

### Other files
This repository also includes standalone GNN versions, that were used for testing and improving the code

Also 3, optimal_x files are included which can be executed to calculate the optimal hyperparameters in a given search space. Do note that grid method is employed and is computaionally taxing.
