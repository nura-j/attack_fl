# Genetic Algorithm-Based Dynamic Backdoor Attack on Federated Learning-Based Network Traffic Classification

## Overview

This repository contains the implementation of a novel genetic algorithm-based dynamic backdoor attack against federated learning systems for network traffic classification. The research explores the vulnerability of federated learning models to sophisticated backdoor attacks that evolve over time using genetic algorithms.

## Research Focus

- Implementation of dynamic backdoor attacks on federated learning systems
- Application of genetic algorithms to create evolving triggers
- Testing attack resilience across different RNN model architectures
- Network traffic classification in a federated learning context

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.16.1
- Keras 3.3.3
- NumPy 1.26.4
- Pandas 2.2.2
- Scikit-learn 1.5.0
- Matplotlib 3.9.0

## Project Structure

- `main.py`: Main experiment runner for testing all RNN models
- `models/`: Implementation of different RNN model architectures
- `utils/`: Helper functions for data processing and attack implementation
- `results/`: Output directory for experimental results
- `data/`: Network traffic datasets

## Models

The project implements and compares three RNN model architectures:
1. `RNNModel1`: Simple RNN architecture
2. `RNNModel2`: LSTM-based architecture
3. `RNNModel3`: Complex GRU-based architecture

## Usage

### Training Models

```bash
python lstm_train.py --dataset data/moore_clean_cols_all.csv --num_classes 6 --epochs 10 --batch_size 100
```

### Running Federated Learning Experiments with Backdoor Attacks

```bash
python main.py --num_clients 50 --num_rounds 50 --batch_size 32 --epochs 1
```

### Command Line Arguments

- `--num_clients`: Number of clients in federated learning (default: 50)
- `--num_rounds`: Number of federated learning rounds (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Epochs per round (default: 1)
- `--output_dir`: Directory for output files (default: 'results')
- `--models_dir`: Directory for saved models (default: 'models')

## Attack Implementation

The genetic algorithm-based backdoor attack consists of:

1. Trigger generation using genetic algorithms
2. Backdoor insertion into client data
3. Model poisoning in federated learning rounds
4. Evaluation of attack success rate

## Experiment Setup

- Number of clients: 50
- Participation rate: 20% (10 clients per round)
- Attack rate: 10% (1 attacker)
- Non-IID data distribution
- Multiple trials to ensure consistent results

## Results

Results are saved in the `results/` directory with the following metrics:
- Global model accuracy on clean data (GAC)
- Global model accuracy on poisoned data (GAP)
- Malicious model accuracy on clean data (MAC)
- Malicious model accuracy on poisoned data (MAP)

## Citation

If you use this code in your research, please cite:

```
@inproceedings{nazzal2023genetic,
  title={Genetic Algorithm-Based Dynamic Backdoor Attack on Federated Learning-Based Network Traffic Classification},
  author={Nazzal, Mahmoud and Aljaafari, Nura and Sawalmeh, Ahmed and Khreishah, Abdallah and Anan, Muhammad and Algosaibi, Abdulelah and Alnaeem, Mohammed and Aldalbahi, Adel and Alhumam, Abdulaziz and Vizcarra, Conrado P and others},
  booktitle={2023 Eighth International Conference on Fog and Mobile Edge Computing (FMEC)},
  pages={204--209},
  year={2023},
  organization={IEEE}
}
```

## License

[MIT License](LICENSE)
```

This README provides a comprehensive overview of the project, including installation instructions, usage examples, project structure, and experiment setup. It's designed to help users understand and use your codebase effectively.
