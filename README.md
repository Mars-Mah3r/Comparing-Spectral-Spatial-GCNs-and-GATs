# Comparing-Spectral-Spatial-GCNs-and-GATs

## Abstract

This repository will include all files that were used in my 2023 6CCE3EEP Individual Project. 

In order to create GNN, the following article by Awan, A. A named “A Comprehensive Introduction to Graph Neural Networks (GNNs).” [1] was adapted. From this article, an understanding of the core techniques for implementing GNNs was obtained and hence a the code architecture was adapted to better suit the specific problem this paper aims to tackle.

This is the full breakdown of how the source-code[2] was adapted:

-	The original data set used was “Cora” from Planetoid ( more information on this in section 3.14 ), the code was changed to allow the user to select one of three Datasets from Planetoid, consisting of Cora, CiteSeer, and PubMed
-	The original article only provided a rough template on creating Spectral GCN. This was adapted and expanded to include a method of implementing a Spatial GCN and a Graph Attention Network. 
-	The article had provided pre-determined hyperparameters, with little to no information on how they were obtained. In turn the code was adapted to have the hyperparameters tuned depending on the GNN being used 
-	Through the above mentioned change, the training process was adjusted.
-	The original code was refactored to become more modular, efficient , readable and to be better integrated in accordance with this papers objectives. 

## Environment Setup
