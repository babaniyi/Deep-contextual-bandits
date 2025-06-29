# Deep Contextual Bandits (PyTorch Edition)

This library is a PyTorch-based benchmark for testing decision-making algorithms for contextual bandits. It is inspired by the "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling" paper (ICLR 2018) by Carlos Riquelme, George Tucker, and Jasper Snoek.

## Overview

This repository provides a comprehensive benchmark for contextual bandit algorithms, implementing various approaches based on approximate Bayesian Neural Networks and Thompson sampling. The original TensorFlow implementation can be found at [TensorFlow Models Archive](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits).

## Features

- **Multiple Algorithms**: Implements various contextual bandit algorithms including:
  - Neural Bandit (point estimate)
  - Neural Linear (Bayesian linear regression on neural features)
  - Bootstrapped Neural Networks
  - Variational Neural Networks
  - Parameter Noise Sampling
  - And more...

- **Diverse Datasets**: Supports both synthetic and real-world datasets:
  - **Synthetic**: Linear, sparse linear, wheel bandit
  - **Real**: Mushroom, Financial, Jester, Statlog, Adult, Covertype, Census

- **PyTorch Implementation**: Modern, clean PyTorch codebase with type hints and comprehensive testing

- **Modular Design**: Easy to extend with new algorithms and datasets

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Deep-contextual-bandits

# Install dependencies
pip install -r bandits/requirements.txt
```

## Quick Start

### Basic Usage

```python
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.core.contextual_bandit import run_contextual_bandit

# Sample synthetic data
dataset, _, (opt_rewards, opt_actions) = sample_linear_data(
    num_contexts=1000, 
    dim_context=10, 
    num_actions=5
)

# Define hyperparameters
hparams = {
    "context_dim": 10,
    "num_actions": 5,
    "layer_sizes": [100, 100],
    "initial_lr": 0.001,
    "batch_size": 512,
    "training_epochs": 100
}

# Create algorithm
algorithm = NeuralBanditModel(hparams, name="neural_bandit")

# Run experiment
actions, rewards = run_contextual_bandit(
    context_dim=10,
    num_actions=5,
    dataset=dataset,
    algos=[algorithm]
)
```

### Running Experiments

```bash
# Run with synthetic linear data
python3 example_main.py --data_type=linear --num_contexts=2000

# Run with wheel bandit data
python3 example_main.py --data_type=wheel --num_contexts=1000

# Run with multiple algorithms
python3 example_main.py --data_type=linear --num_contexts=1000 --algorithms neural_bandit neural_linear

# Run with custom seed
python3 example_main.py --data_type=linear --num_contexts=1000 --seed 123

# Run with real dataset (requires dataset files in datasets/ folder)
python3 example_main.py --data_type=mushroom --num_contexts=2000 --data_dir datasets
```

### Command Line Options

- `--data_type`: Type of dataset (`linear`, `sparse_linear`, `wheel`, `mushroom`, `financial`, `jester`)
- `--num_contexts`: Number of contexts to sample (default: 2000)
- `--algorithms`: List of algorithms to run (`neural_bandit`, `neural_linear`)
- `--data_dir`: Directory containing dataset files (default: `datasets`)
- `--seed`: Random seed for reproducibility (default: 42)

## Dataset Setup

### Real Datasets

To use real datasets, download them to a `datasets/` folder:

```bash
mkdir datasets
cd datasets

# Download datasets (you'll need to obtain these files)
# - mushroom.data (UCI Mushroom dataset)
# - raw_stock_contexts (Financial data)
# - jester_data_40jokes_19181users.npy (Jester dataset)
# - shuttle.trn (Statlog dataset)
# - adult.full (UCI Adult dataset)
# - covtype.data (Covertype dataset)
# - USCensus1990.data.txt (Census dataset)
```

### Dataset Sources

- **Mushroom**: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
- **Adult**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Covertype**: [UCI Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)
- **Census**: [UCI Census Dataset](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29)
- **Statlog**: [UCI Statlog Dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Shuttle%29)

## Project Structure

```
Deep-contextual-bandits/
├── bandits/
│   ├── algorithms/          # Bandit algorithm implementations
│   │   ├── neural_bandit_model.py
│   │   ├── neural_linear_sampling.py
│   │   └── ...
│   ├── core/               # Core abstractions
│   │   ├── bayesian_nn.py
│   │   ├── contextual_bandit.py
│   │   └── contextual_dataset.py
│   ├── data/               # Data samplers
│   │   ├── data_sampler.py
│   │   └── synthetic_data_sampler.py
│   └── requirements.txt
├── datasets/               # Real dataset files (not included)
├── example_main.py         # Main experiment script
├── test_neural_bandit.py   # Unit tests
└── README.md
```

## Algorithms

### Neural Bandit
- **Description**: Point estimate neural network for contextual bandits
- **Use Case**: Baseline comparison, when computational efficiency is important

### Neural Linear
- **Description**: Bayesian linear regression on neural network features
- **Use Case**: Good balance between expressiveness and uncertainty quantification

### Bootstrapped Neural Networks
- **Description**: Ensemble of neural networks with bootstrap sampling
- **Use Case**: Simple uncertainty estimation through model averaging

### Variational Neural Networks
- **Description**: Bayesian neural networks using variational inference
- **Use Case**: Full uncertainty quantification with neural networks

## Testing

Run the test suite:

```bash
python test_neural_bandit.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{riquelme2018deep,
  title={Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling},
  author={Riquelme, Carlos and Tucker, George and Snoek, Jasper},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the original TensorFlow code by Carlos Riquelme, George Tucker, and Jasper Snoek. The PyTorch port maintains the same experimental setup and evaluation methodology.
