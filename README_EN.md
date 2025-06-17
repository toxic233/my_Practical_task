# Counterfactual Reasoning-based Heterogeneous Graph Neural Network Representation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

> A novel representation learning framework that combines counterfactual reasoning with heterogeneous graph neural networks to enhance model interpretability and generalization.

## 🚀 Project Overview

Heterogeneous Graph Neural Networks (HGNNs) excel at processing complex network data with multiple types of nodes and edges, but suffer from black-box issues and lack of interpretability. This project introduces counterfactual reasoning into HGNNs to improve model interpretability and robustness through causal inference.

### Core Problems
- Existing HGNNs lack interpretability
- Models tend to learn spurious correlations, leading to overfitting
- Insufficient exploration of complex causal relationships in heterogeneous graphs

### Our Solution
- Counterfactual reasoning-based heterogeneous graph representation learning framework
- Reinforcement learning-optimized counterfactual decision mechanism
- Multi-dimensional perturbation strategies for complex heterogeneous graph structures

## ✨ Key Features

### 🎯 Enhanced Interpretability
- **Causal Relationship Mining**: Discover causal relationships in heterogeneous graphs through counterfactual reasoning
- **Explanatory Subgraph Generation**: Generate interpretable key subgraph structures
- **Spurious Correlation Elimination**: Reduce model dependency on spurious correlations

### 🚄 Efficient Decision Mechanism
- **Reinforcement Learning Optimization**: Use RL to optimize counterfactual decision processes
- **Search Space Reduction**: Effectively handle exponential search spaces in heterogeneous graphs
- **Multi-dimensional Perturbation**: Support various operation types for nodes and edges

### 🔧 Robustness Enhancement
- **Out-of-Distribution Generalization**: Address OOD issues in explanatory subgraphs
- **Structure Preservation**: Maintain original structural features of heterogeneous graphs
- **Random Perturbation Enhancement**: Improve model robustness against interference

## 🏗️ Technical Architecture

### System Architecture
[Figure 4 Research Process Model](image/图片6.png)

### Core Modules

#### 1. Counterfactual Reasoning Module
```
Objective: Generate subgraphs with opposite predictions using minimal perturbations
Method: Multi-dimensional perturbation strategies + Neighbor selection optimization
Output: Explanatory subgraph G_s and counterfactual subgraph G'
```

#### 2. Reinforcement Learning Decision Module
```
State: Probability of target node label change
Action: {E_add, E_del, E_mod, V_add, V_del, V_mod}
Reward: Minimal perturbations + Maximum prediction change probability
```

#### 3. Structure Preservation Module
```
Explanatory Subgraph Optimization: Random edge and node supplementation
Counterfactual Subgraph Enhancement: Circular perturbation strategies
Goal: Address size and distribution differences
```

## 📊 Datasets & Experiments

### Datasets
| Dataset Type | Dataset Names | Purpose |
|--------------|---------------|---------|
| Synthetic | Dataset1-3 | Comparison with other counterfactual reasoning methods |
| Real-world | DBLP, ACM, IMDB | Heterogeneous graph processing capability evaluation |

### Experimental Setup
- **Bibliographic Data Heterogeneous Network Example**:
  
  [Figure 1 Bibliographic Data Heterogeneous Network](image/图片1.png)[Figure 1 Bibliographic Data Heterogeneous Network](image/图片2.png)

- **Counterfactual Reasoning Application Scenarios**:
  
  [Figure 2 Main Research Areas of Counterfactual Reasoning](image/图片3.png)

- **Perturbation Operation Examples**:
  
  [(a) Node Perturbation Example](image/图片4.png)[(b) Edge Perturbation Example](image/图片5.png)

### Evaluation Metrics
- **Fidelity**: Importance degree of explanatory subgraphs for target node predictions
- **Robustness**: Anti-interference capability of counterfactual subgraphs
- **Minimality**: Graph edit distance, explanation size, etc.
- **Prediction Accuracy**: Basic classification performance
- **Runtime**: Algorithm efficiency evaluation

## 📈 Experimental Results

> **Note**: This section will be filled with actual experimental results

### Performance Comparison
```
TODO: Add comparison results with baseline methods
- Traditional HGNNs (HAN, MAGNN, SeHGNN)
- Existing counterfactual reasoning methods (RCExplainer, CF2, GOAt)
```

### Ablation Studies
```
TODO: Add contribution analysis of each module
- Counterfactual reasoning module effectiveness
- Reinforcement learning decision mechanism contribution
- Structure preservation strategy impact
```

### Visualization Analysis
```
TODO: Add explanatory subgraph visualization results
- Key node and edge identification
- Causal relationship mining effectiveness
- Performance across different datasets
```

## 🛠️ Quick Start

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.12.0
DGL >= 0.6.0
NumPy >= 1.19.0
FastAPI >= 0.104.1
```

### Installation
```bash
git clone https://github.com/your-username/counterfactual-heterogeneous-gnn.git
cd counterfactual-heterogeneous-gnn
pip install -r requirements.txt
```

### Start API Service
```bash
# Development mode
python main.py

# Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production environment
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Usage Examples

#### 1. Python API
```python
from src.model import CounterfactualHGNN
from src.data import load_heterogeneous_graph

# Load heterogeneous graph data
graph = load_heterogeneous_graph('DBLP')

# Initialize model
model = CounterfactualHGNN(
    node_types=graph.node_types,
    edge_types=graph.edge_types,
    hidden_dim=128
)

# Train model
model.fit(graph, epochs=100)

# Generate counterfactual explanations
explanations = model.explain(target_nodes=[0, 1, 2])
```

#### 2. REST API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Single node explanation
curl -X POST "http://localhost:8000/api/v1/explain/single" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "dataset=Cora&model=graphsage&task=node&node_id=0&timeout=30"

# Available models
curl http://localhost:8000/api/v1/models
```

## 🔌 API Service Guide

### Supported Datasets & Models

#### Dataset Categories
| Type | Dataset | Description | Nodes |
|------|---------|-------------|-------|
| Homogeneous | Cora | Academic paper network, 7 classes | 2708 |
| Homogeneous | CiteSeer | Academic paper network, 6 classes | 3327 |
| Homogeneous | PubMed | Biomedical papers, 3 classes | 19717 |
| Heterogeneous | ACM | Conference, paper, author network | - |
| Heterogeneous | IMDB | Movie, actor, director network | 19061 |
| Heterogeneous | DBLP | Author, paper, journal, conference | 18448 |
| Synthetic | syn1-syn4 | Various complexity synthetic graphs | - |

#### Model Types
- **GraphSAGE**: Inductive learning for large-scale graphs, supports node classification and link prediction
- **DGI**: Unsupervised node representation learning based on mutual information maximization
- **MAGNN**: Heterogeneous graph analysis using meta-path aggregation

### Main API Endpoints

#### 1. Single Node Explanation `POST /api/v1/explain/single`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset | string | ✅ | - | Dataset name (case-insensitive) |
| model | string | ✅ | - | Model type |
| task | string | ✅ | - | Task type (node/link) |
| node_id | integer | ✅ | - | Target node ID |
| neighbors_cnt | integer | ❌ | 5 | Number of nearest neighbors |
| maxiter | integer | ❌ | 1000 | MCTS maximum iterations |
| timeout | integer | ❌ | 60 | Timeout (seconds) |

**Response Example:**
```json
{
  "node_id": 0,
  "importance": 1.2345,
  "subgraph_size": 15,
  "subgraph_nodes": [0, 1, 2, 3, 5, 8, 13, 21],
  "subgraph_edges": [[0, 1], [1, 2], [2, 3]],
  "processing_time": 5.67,
  "model_info": {
    "dataset": "Cora",
    "model_name": "graphsage",
    "task": "node",
    "device": "cuda:0"
  },
  "status": "success"
}
```

#### 2. Batch Node Explanation `POST /api/v1/explain/batch`

Supports Server-Sent Events (SSE) streaming response with real-time progress.

**Request Body Example:**
```json
{
  "dataset": "Cora",
  "model": "graphsage", 
  "task": "node",
  "node_ids": [0, 1, 2, 3, 4],
  "timeout": 300
}
```

**Streaming Event Types:**
- `status`: Status updates
- `progress`: Individual node results
- `timeout`: Timeout notifications
- `completed`: Final result statistics

#### 3. System Monitoring Endpoints

```bash
# Health check
GET /api/v1/health

# Available model information
GET /api/v1/models

# API basic information
GET /
```

### Error Handling

| HTTP Status | Description | Solution |
|-------------|-------------|----------|
| 400 | Bad Request | Check parameter format and range |
| 408 | Request Timeout | Increase timeout or reduce complexity |
| 422 | Validation Error | Verify parameter types and required fields |
| 500 | Internal Server Error | Check model files and system resources |

### Special Dataset Notes

#### DBLP Dataset
- **Node Types**: Authors(0-4056), Papers(4057-18405), Journals(18406-18425), Conferences(18426-18447)
- **Explanation Scope**: Only supports author node explanation (node_id < 4057)

#### IMDB Dataset  
- **Node Types**: Movies(0-4277), Actors(4278-16777), Directors(16778-19061)
- **Explanation Scope**: Only supports movie node explanation

### Performance Optimization

- **GPU Acceleration**: Automatic detection and use of available GPU
- **Model Caching**: Automatic caching after first load for faster response
- **Batch Processing**: Recommended for multiple node explanations
- **Parameter Tuning**: Adjust maxiter and timeout based on graph scale

### 📋 Detailed API Documentation

For comprehensive API usage instructions, please refer to:
- 📖 [Complete API Usage Guide](API_GUIDE.md) - Detailed endpoint descriptions, parameter configurations, error handling
- 🔧 Best practices and performance optimization tips
- ❓ FAQ and troubleshooting

## 🎯 Significance & Applications

### Academic Value
- **Theoretical Contribution**: First systematic application of counterfactual reasoning to HGNNs
- **Methodological Innovation**: Novel multi-dimensional perturbation strategies and RL optimization framework
- **Performance Improvement**: Significant enhancement in model interpretability and robustness

### Practical Applications
- **Recommendation Systems**: Provide interpretable recommendation results and reasoning
- **Social Network Analysis**: Identify key factors influencing user behavior
- **Bioinformatics**: Discover causal mechanisms in protein interactions
- **Knowledge Graphs**: Enhance trustworthiness and transparency in knowledge reasoning

### Technical Impact
- Advance HGNNs toward explainable AI
- Provide new tools for causal analysis of complex network data
- Promote application of counterfactual reasoning in graph learning

## 🤝 Contributing

We welcome community contributions! Here's how you can participate:

1. **Issue Reporting**: Report bugs or request features in [Issues](https://github.com/your-username/counterfactual-heterogeneous-gnn/issues)
2. **Feature Development**: Fork the project and submit Pull Requests
3. **Documentation**: Help improve project documentation and examples
4. **Test Cases**: Add more test datasets and scenarios

## 📝 Related Work

### Heterogeneous Graph Neural Networks
- **HAN**: Hierarchical attention mechanism for meta-paths
- **MAGNN**: Meta-path instance aggregation method
- **SeHGNN**: Simplified semantic fusion module

### Counterfactual Reasoning
- **RCExplainer**: Reinforcement learning-based edge selection
- **CF2**: Balanced factual and counterfactual reasoning
- **GOAt**: Gradient-based importance computation

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

- **Email**: your-email@example.com
- **Project Homepage**: https://github.com/your-username/counterfactual-heterogeneous-gnn
- **Paper**: [Preprint Link]

---

If this project helps you, please give us a ⭐ Star! 