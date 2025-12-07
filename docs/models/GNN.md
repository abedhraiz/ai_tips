# GNN - Graph Neural Networks

## Overview

**Graph Neural Networks** are designed to work with graph-structured data, where entities (nodes) are connected by relationships (edges). Unlike CNNs (grids) or Transformers (sequences), GNNs can handle irregular, non-Euclidean data structures.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Input** | Graph (nodes + edges) |
| **Architecture** | Message passing layers |
| **Output** | Node/edge/graph embeddings |
| **Key Operation** | Neighborhood aggregation |
| **Domains** | Social networks, molecules, knowledge graphs |

## How It Works

```
Graph Neural Network: Message Passing

Step 1: Aggregate neighbor features
Step 2: Update node representation
Step 3: Repeat for K layers

       ┌───┐         ┌───┐
       │ A │─────────│ B │
       └─┬─┘         └─┬─┘
         │             │
         │   ┌───┐     │
         └───│ C │─────┘
             └─┬─┘
               │
             ┌─┴─┐
             │ D │
             └───┘

For node C:
  h_C^(k+1) = UPDATE(h_C^(k), AGGREGATE({h_A^(k), h_B^(k), h_D^(k)}))

After K layers, each node's representation contains information 
from its K-hop neighborhood.
```

## Types of GNNs

### By Architecture

| Type | Aggregation | Formula |
|------|-------------|---------|
| **GCN** | Mean | h = σ(D⁻¹AXW) |
| **GraphSAGE** | Sample + aggregate | h = σ(W·[h ∥ AGG(neighbors)]) |
| **GAT** | Attention-weighted | h = σ(Σ α_ij W h_j) |
| **GIN** | Sum (injective) | h = MLP((1+ε)h + Σh_neighbors) |
| **MPNN** | General message passing | m = Σ M(h_i, h_j, e_ij) |

### By Task

| Task | Level | Example |
|------|-------|---------|
| **Node Classification** | Node | User categorization |
| **Link Prediction** | Edge | Friend recommendation |
| **Graph Classification** | Graph | Molecule property |
| **Node Clustering** | Node | Community detection |
| **Graph Generation** | Graph | Drug design |

## Popular Models

| Model | Key Innovation | Use Case |
|-------|---------------|----------|
| **GCN** | Spectral convolution | General graphs |
| **GraphSAGE** | Inductive learning | Large graphs |
| **GAT** | Attention mechanism | Weighted aggregation |
| **GIN** | Maximally expressive | Graph classification |
| **PNA** | Multiple aggregators | Molecular properties |
| **SchNet** | Continuous filters | 3D molecules |
| **DimeNet** | Directional messages | Quantum chemistry |
| **Graphormer** | Transformer for graphs | General graphs |

## Examples with Code

### Example 1: Node Classification with GCN

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load Cora dataset (citation network)
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train
model = GCN(dataset.num_features, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Test accuracy
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = correct / data.test_mask.sum()
print(f'Accuracy: {accuracy:.4f}')  # ~81%
```

### Example 2: Graph Attention Network (GAT)

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GAT(dataset.num_features, dataset.num_classes)
# GAT typically achieves ~83% on Cora
```

### Example 3: GraphSAGE for Large Graphs

```python
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Mini-batch training for large graphs
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Sample 25 neighbors, then 10
    batch_size=512,
    input_nodes=data.train_mask
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

### Example 4: Graph Classification (Molecules)

```python
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Load molecular dataset
dataset = TUDataset(root='./data', name='PROTEINS')

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        
        # GIN layers
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)
        
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Pool to graph-level representation
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)

# Training loop
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = GIN(dataset.num_features, 64, dataset.num_classes)

for batch in loader:
    out = model(batch.x, batch.edge_index, batch.batch)
    loss = F.cross_entropy(out, batch.y)
```

### Example 5: Link Prediction

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # Inner product decoder
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edge_index)
        neg_score = self.decode(z, neg_edge_index)
        return pos_score, neg_score

# Training
model = LinkPredictor(num_features, 64)

# Generate negative samples
neg_edge_index = negative_sampling(
    edge_index=data.edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=data.edge_index.size(1)
)

pos_score, neg_score = model(data.x, data.edge_index, 
                             data.edge_index, neg_edge_index)

# Binary cross entropy loss
pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
loss = pos_loss + neg_loss
```

### Example 6: Molecular Property Prediction

```python
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import AttentiveFP

# Load ESOL dataset (solubility prediction)
dataset = MoleculeNet(root='./data', name='ESOL')

# AttentiveFP is designed for molecules
model = AttentiveFP(
    in_channels=dataset.num_features,
    hidden_channels=64,
    out_channels=1,  # Regression
    edge_dim=dataset.num_edge_features,
    num_layers=2,
    num_timesteps=2,
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    loss = F.mse_loss(out.squeeze(), batch.y)
```

## Use Cases

| Domain | Application | Example |
|--------|-------------|---------|
| **Social Networks** | Recommendation | Friend suggestion |
| **Drug Discovery** | Property prediction | Toxicity, solubility |
| **Fraud Detection** | Anomaly detection | Transaction networks |
| **Knowledge Graphs** | Link prediction | Entity relationships |
| **Computer Vision** | Scene graphs | Object relationships |
| **Chemistry** | Molecule generation | Drug design |
| **Physics** | Particle simulations | N-body problems |
| **Chip Design** | Circuit optimization | EDA tools |

## GNN vs Other Architectures

| Aspect | GNN | CNN | Transformer |
|--------|-----|-----|-------------|
| **Data Structure** | ✅ Graphs | Grids | Sequences |
| **Irregular Data** | ✅ Yes | No | Limited |
| **Permutation Invariant** | ✅ Yes | No | No |
| **Scalability** | Varies | Good | O(n²) |
| **Locality** | ✅ Built-in | Built-in | Learned |
| **Long-range** | K-hop limited | Limited | ✅ Global |

## Common Pooling Operations

```python
from torch_geometric.nn import (
    global_mean_pool,   # Average all nodes
    global_max_pool,    # Max over nodes
    global_add_pool,    # Sum all nodes
    TopKPooling,        # Select top-k nodes
    SAGPooling,         # Self-attention pooling
)

# For graph-level tasks, aggregate nodes to graph
graph_embedding = global_mean_pool(node_embeddings, batch)
```

## Benchmarks

### Node Classification Accuracy

| Dataset | GCN | GAT | GraphSAGE | Best |
|---------|-----|-----|-----------|------|
| Cora | 81.5% | 83.0% | 82.5% | 84.2% |
| CiteSeer | 70.3% | 72.5% | 71.0% | 73.8% |
| PubMed | 79.0% | 79.0% | 78.5% | 80.1% |

### Graph Classification Accuracy

| Dataset | GCN | GIN | GAT | Best |
|---------|-----|-----|-----|------|
| PROTEINS | 73.5% | 75.1% | 74.0% | 77.2% |
| MUTAG | 85.0% | 89.4% | 86.5% | 91.2% |
| PTC | 55.0% | 64.6% | 60.0% | 68.5% |

## Memory Requirements

| Model | Nodes | Edges | VRAM |
|-------|-------|-------|------|
| GCN-2L | 10K | 50K | 0.5 GB |
| GCN-2L | 100K | 1M | 2 GB |
| GCN-2L | 1M | 10M | 16 GB |
| GAT-2L | 10K | 50K | 1 GB |
| GAT-2L | 100K | 1M | 8 GB |

## Scaling GNNs

### Mini-batching Strategies

```python
# 1. Neighbor Sampling (GraphSAGE)
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=512)

# 2. Cluster-GCN
from torch_geometric.loader import ClusterData, ClusterLoader
cluster_data = ClusterData(data, num_parts=100)
loader = ClusterLoader(cluster_data, batch_size=10)

# 3. GraphSAINT
from torch_geometric.loader import GraphSAINTRandomWalkSampler
loader = GraphSAINTRandomWalkSampler(data, batch_size=1000, walk_length=2)
```

## Choosing the Right GNN

```
What's your task?
├── Node Classification
│   ├── Small graphs → GCN, GAT
│   └── Large graphs → GraphSAGE, ClusterGCN
│
├── Graph Classification
│   ├── General → GIN (most expressive)
│   ├── Molecules → AttentiveFP, SchNet
│   └── Social → Hierarchical pooling
│
├── Link Prediction
│   ├── Simple → GCN + inner product
│   └── Complex → R-GCN for typed edges
│
└── Knowledge Graphs
    ├── TransE, RotatE (embeddings)
    └── R-GCN (message passing)
```

## Related Models

- **[LLM](./LLM.md)** - Can be enhanced with graph reasoning
- **[RAG](./RAG.md)** - Knowledge graphs for retrieval
- **[World Models](./WORLD_MODELS.md)** - Graphs for relational reasoning

## Resources

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [DGL (Deep Graph Library)](https://www.dgl.ai/)
- [GCN Paper](https://arxiv.org/abs/1609.02907)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [GAT Paper](https://arxiv.org/abs/1710.10903)
- [Open Graph Benchmark](https://ogb.stanford.edu/)
