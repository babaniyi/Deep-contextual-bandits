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

## Experimental Results

We conducted comprehensive experiments comparing different algorithms across multiple datasets. The results below show the performance of our PyTorch implementations on various benchmark datasets.

### Performance Summary

#### Synthetic Datasets

| Dataset | Neural Bandit | Neural Linear | Linear Full Posterior | Uniform |
|---------|---------------|---------------|----------------------|---------|
| **Linear** | 524.3 | 518.9 | **170.0** | 701.5 |
| **Sparse Linear** | 381.1 | 389.3 | **134.2** | 598.7 |
| **Wheel** | **972.6** | 1351.9 | 1673.7 | 1834.9 |

*Table: Cumulative Regret on Synthetic Datasets (lower is better)*

| Dataset | Neural Bandit | Neural Linear | Linear Full Posterior | Uniform |
|---------|---------------|---------------|----------------------|---------|
| **Linear** | 0.339 | 0.350 | **1.047** | -0.016 |
| **Sparse Linear** | 0.419 | 0.402 | **0.913** | -0.017 |
| **Wheel** | **3.550** | 2.791 | 2.147 | 1.825 |

*Table: Average Reward on Synthetic Datasets (higher is better)*

#### Real-World Datasets

| Dataset | Neural Bandit | Neural Linear | Linear Full Posterior | Uniform |
|---------|---------------|---------------|----------------------|---------|
| **Mushroom** | 1415.0 | 1325.0 | **455.0** | 2650.0 |
| **Statlog** | **194.0** | 352.0 | 203.0 | 431.0 |
| **Adult** | 282.0 | 95.0 | **68.0** | 251.0 |

*Table: Cumulative Regret on Real-World Datasets (lower is better)*

| Dataset | Neural Bandit | Neural Linear | Linear Full Posterior | Uniform |
|---------|---------------|---------------|----------------------|---------|
| **Mushroom** | -0.400 | -0.220 | **1.520** | -2.870 |
| **Statlog** | **0.612** | 0.296 | 0.594 | 0.138 |
| **Adult** | 0.436 | 0.810 | **0.864** | 0.498 |

*Table: Average Reward on Real-World Datasets (higher is better)*

### Key Findings

1. **Linear Full Posterior Dominance**: The Linear Full Posterior algorithm consistently outperformed other methods across most datasets, achieving the lowest cumulative regret and highest average rewards. This demonstrates the effectiveness of proper Bayesian inference for linear problems.

2. **Dataset-Specific Performance**:
   - **Synthetic Linear/Sparse Linear**: Linear Full Posterior excelled, showing the advantage of proper Bayesian linear regression
   - **Wheel Bandit**: Neural Bandit performed best (3.55 average reward), likely due to the non-linear nature of the wheel bandit problem
   - **Mushroom**: Linear Full Posterior achieved the best performance (1.52 average reward vs -0.22 for Neural Linear)
   - **Statlog**: Neural Bandit achieved the best performance (61.2% accuracy vs 29.6% for Neural Linear)
   - **Adult**: Linear Full Posterior excelled with 86.4% accuracy, significantly outperforming other methods

3. **Algorithm Characteristics**:
   - **Linear Full Posterior**: Best overall performance, especially on linear problems and real-world datasets
   - **Neural Bandit**: Good performance on non-linear problems (Wheel, Statlog)
   - **Neural Linear**: Competitive performance, good balance between expressiveness and uncertainty quantification
   - **Uniform**: Serves as a reliable baseline, computationally efficient

4. **Computational Efficiency**: Uniform sampling was the fastest algorithm, while Neural Linear and Linear Full Posterior required more computational time due to their Bayesian inference components.

5. **Robustness**: Linear Full Posterior showed excellent performance across different dataset types, making it the most reliable choice for general applications.

### Experimental Setup

- **Number of contexts**: 500 per dataset
- **Random seed**: 42 (for reproducibility)
- **Hardware**: CPU-based computation
- **Framework**: PyTorch 2.0+
- **Algorithms tested**: Neural Bandit, Neural Linear, Linear Full Posterior, Uniform

### Running Your Own Experiments

To reproduce these results or run your own experiments:

```bash
# Run all experiments
python3 run_all_experiments.py --num_contexts=500 --datasets linear sparse_linear wheel mushroom statlog adult --algorithms neural_bandit neural_linear uniform linear_full_posterior

# Run specific dataset/algorithm combinations
python3 run_all_experiments.py --datasets linear --algorithms neural_bandit neural_linear

# Custom experiment parameters
python3 run_all_experiments.py --num_contexts=1000 --seed=123
```

Results will be saved in the `results/` directory with detailed CSV files and summary tables.

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
├── results/                # Experiment results and summaries
├── example_main.py         # Main experiment script
├── run_all_experiments.py  # Batch experiment runner
├── test_neural_bandit.py   # Unit tests
└── README.md
```

## Algorithms

### Neural Bandit
- **Description**: Point estimate neural network for contextual bandits
- **Use Case**: Baseline comparison, when computational efficiency is important
- **Performance**: Good general performance, especially on non-linear problems

### Neural Linear
- **Description**: Bayesian linear regression on neural network features
- **Use Case**: Good balance between expressiveness and uncertainty quantification
- **Performance**: Competitive performance across most datasets

### Linear Full Posterior
- **Description**: Thompson Sampling with independent linear models and unknown noise variance
- **Use Case**: Best choice for linear problems and real-world datasets
- **Performance**: Best overall performance, especially on linear problems and real-world datasets

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

If you use this code in your research, please cite both the original paper and this PyTorch implementation:

```bibtex
@inproceedings{riquelme2018deep,
  title={Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling},
  author={Riquelme, Carlos and Tucker, George and Snoek, Jasper},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@software{olaniyi2024deep,
  title={Deep Contextual Bandits: PyTorch Implementation},
  author={Olaniyi, Babaniyi},
  year={2024},
  url={https://github.com/babaniyi/Deep-contextual-bandits},
  note={PyTorch port of the original TensorFlow implementation with comprehensive benchmarking}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the original TensorFlow code by Carlos Riquelme, George Tucker, and Jasper Snoek. The PyTorch port maintains the same experimental setup and evaluation methodology.
