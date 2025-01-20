# SentinelGNN

This repository contains the code and data for the paper "SentinelGNN: A Neural Network Architecture for Detecting Anomalies in Attributed Multi-graphs". SentinelGNN is a novel graph neural network designed to detect various types of anomalies in complex blockchain transaction networks.

## Paper Abstract

The rapid growth of blockchain networks has introduced unprecedented challenges in detecting anomalous activities within complex transaction graphs. While existing approaches struggle with multi-edge scenarios and temporal dependencies, we present SentinelGNN, a novel graph neural network architecture that achieves state-of-the-art performance in detecting five types of blockchain anomalies: point, contextual, collective, temporal, and structural. 

Through extensive experimentation on a large-scale Ethereum dataset containing 6.08M nodes and 38.90M edges, SentinelGNN achieves ROC-AUC scores of 0.980 ± 0.01 and PR-AUC scores of 0.975 ± 0.01, outperforming traditional graph neural networks by significant margins.

## Repository Structure

- `data/`: Contains the Ethereum transaction dataset and preprocessing scripts
- `models/`: Implements the SentinelGNN architecture and baseline models 
- `utils/`: Utility functions for data loading, evaluation metrics, etc.
- `experiments/`: Scripts to reproduce the experiments from the paper
- `docs/`: Additional documentation and tutorials

## Installation

1. Clone the repo: `git clone https://github.com/SentinelGNN/implementation.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python data/download_data.py`
4. Preprocess the data: `python data/preprocess.py`

## Usage

To train SentinelGNN on the Ethereum dataset:
```
python train.py --data_dir data/ethereum/ --model SentinelGNN --epochs 50
```

To evaluate a trained model:
```
python evaluate.py --data_dir data/ethereum/ --model_path checkpoints/sentinelgnn.pt
```

Refer to the `experiments/` directory for scripts to reproduce specific results from the paper.

## Dataset

The primary dataset is a large-scale Ethereum transaction graph from Oct 2018 to May 2020:

- 6,083,422 nodes (unique addresses) 
- 38,901,039 edges (transactions)
- Node features: Account properties, balance history, contract flags
- Edge features: Transaction amount, gas price, timestamp
- Temporal resolution: Block-level timestamps

Ground truth anomaly labels are provided by Etherscan for 296 addresses corresponding to illicit activities. 

## Extending SentinelGNN

To apply SentinelGNN to a new dataset or blockchain system:

1. Format your data into a suitable graph representation with node/edge features
2. Modify the data loading and preprocessing scripts in `data/` 
3. Adjust model hyperparameters like feature dimensions in `models/sentinelgnn.py`
4. Define new anomaly types or labels in `utils/anomalies.py`
5. Train, evaluate, and analyse the model performance

For more advanced modifications to the GNN architecture itself, refer to the documentation in `models/` and `docs/architecture.md`.
