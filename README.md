# SentinelGNN

This repository contains the code and data for the paper "SentinelGNN: A Neural Network Architecture for Detecting Anomalies in Attributed Multi-graphs". SentinelGNN is a novel graph neural network designed to detect various types of anomalies in complex blockchain transaction networks.

## Repository Structure

- `figures/`: Contains visualizations and plots from the experiments
- `README.md`: This file, providing an overview of the project
- `data_processing.py`: Functions for processing and transforming the input graph data
- `datainfos.txt`: Statistics and insights about the dataset  
- `main.py`: Main script for training and evaluating the SentinelGNN model
- `model.py`: Implements the core SentinelGNN model architecture
- `pyg_gnn_wrapper.py`: Wrapper functions for key PyTorch Geometric operations used in the model
- `utils.py`: Utility functions for data loading, metrics, etc.

## Installation

1. Clone the repo: `git clone https://github.com/yourusername/SentinelGNN.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset and place it in the appropriate directory (see `datainfos.txt` for details)

## Usage

To preprocess the data:
```
python data_processing.py --data_dir /path/to/data 
```

To train SentinelGNN:
```
python main.py --data_dir /path/to/data --model SentinelGNN --epochs 50
```

To evaluate a trained model:
```
python main.py --data_dir /path/to/data --model_path /path/to/model --evaluate
```

Refer to the code in each script for additional command-line arguments and options.

## Dataset

See `datainfos.txt` for a detailed description of the dataset, including statistics on the graph size, node/edge features, and anomaly labels.

## Model Architecture

The SentinelGNN architecture is implemented across several files:

- `model.py` contains the core GNN layers, attention mechanisms, and anomaly detection head
- `pyg_gnn_wrapper.py` provides wrapper functions for key PyTorch Geometric operations
- `utils.py` includes utility layers like a multi-layer perceptron 

Refer to the code comments and docstrings for details on each component.

## Results

Key results and visualizations are provided in the `figures/` directory. This includes:

- ROC and Precision-Recall curves for anomaly detection performance
- t-SNE plots of learned embeddings showing separation of anomalous nodes
- Attention weight visualizations that highlight important graph structures

The `README.md` in `figures/` provides additional context for each result.
