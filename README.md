# Evolve-DGN
Code for paper titled "Evolve-DGN: An Evolving Dynamic Graph Network for Adaptive and Equitable Resource Allocation in Disaster Response"

## Overview
Evolve-DGN is a novel deep learning approach that combines Graph Neural Networks (GNNs) with Reinforcement Learning (RL) to optimize resource allocation during disaster response scenarios. The model is designed to adapt to dynamic changes in the environment while ensuring equitable distribution of resources.

## Key Features
- Dynamic Graph Neural Network architecture for processing evolving disaster scenarios
- Reinforcement Learning integration for adaptive decision-making
- Fairness-aware resource allocation using Jain's Fairness Index
- Comparative analysis with baseline models (Static GNN, T-GCN, EvolveGCN)
- Ablation studies to validate model components

## Repository Structure
```
Evolve-DGN/
├── model_weights/           # Trained model weights
│   ├── ablation1_weq_0/    # Ablation study without equity weight
│   ├── ablation2_no_attention/ # Ablation study without attention mechanism
│   └── full_model/         # Complete Evolve-DGN model weights
├── paper/                  # Research paper PDF
└── training_script/        # Training implementation
    ├── ablation1_weq_0/
    ├── ablation2_no_attention/
    └── full_model/
```

## Environment
The simulation environment (`DisasterEnv`) models a disaster response scenario with:
- Multiple demand nodes (affected areas)
- Supply nodes (resource centers)
- Hospital nodes
- Dynamic graph structure representing the transportation network
- Time-varying demand and supply levels

## Model Architecture
Evolve-DGN consists of:
1. Dynamic Graph Neural Network for processing evolving network states
2. Attention mechanism for prioritizing critical areas
3. Equity-aware reward function incorporating Jain's Fairness Index
4. PPO (Proximal Policy Optimization) for reinforcement learning

## Training Scripts
The repository includes three training configurations:
1. `full_model/model_trainer.ipynb`: Complete Evolve-DGN implementation
2. `ablation1_weq_0/ablation1_model_trainer.ipynb`: Model without equity weight
3. `ablation2_no_attention/ablation2_model_trainer.ipynb`: Model without attention mechanism

## Usage
1. Install dependencies:
```bash
pip install gymnasium networkx stable-baselines3
```

2. Train the model:
```python
from training_script.full_model.model_trainer import train_evolve_dgn_model
model_path, model_type = train_evolve_dgn_model(FlatDisasterEnv)
```

3. Evaluate performance:
```python
results = evaluate_model(model_path, model_type)
print(results)  # Displays metrics like delivery time, fill rate, and fairness index
```

## Citation
If you use this code in your research, please cite our paper:
```
[Citation information to be added after publication]
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Sachin Kumar
