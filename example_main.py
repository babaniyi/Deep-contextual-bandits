#!/usr/bin/env python3
"""
Simple example of contextual bandits simulation using PyTorch.

This script demonstrates how to run experiments with different algorithms
and datasets, inspired by the original TensorFlow implementation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

import argparse
import time
import numpy as np
import os
from typing import Dict, Any, List, Tuple

# Import our PyTorch implementations
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.contextual_bandit import run_contextual_bandit, ContextualBandit

# Import data samplers
from bandits.data.synthetic_data_sampler import (
    sample_linear_data, 
    sample_sparse_linear_data, 
    sample_wheel_bandit_data
)
from bandits.data.data_sampler import (
    sample_mushroom_data, sample_stock_data, sample_jester_data,
    sample_statlog_data, sample_adult_data, sample_census_data,
    sample_covertype_data, sample_statlog_shuttle_data
)


def sample_data(data_type: str, num_contexts: int, data_dir: str = "datasets") -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
  """Sample data from given 'data_type'.
    
  Args:
        data_type: Dataset from which to sample
        num_contexts: Number of contexts to sample
        data_dir: Directory containing dataset files
        
  Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act)
        opt_rewards: Vector of expected optimal reward for each context
        opt_actions: Vector of optimal action for each context
        num_actions: Number of available actions
        context_dim: Dimension of each context
  """

  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim, num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
        
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
        
  elif data_type == 'wheel':
        # Create wheel bandit dataset
    delta = 0.95
    num_actions = 5
    context_dim = 2
    mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01
        dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel
        
    elif data_type == 'mushroom':
        num_actions = 2
        num_contexts = min(8124, num_contexts)
        sampled_vals = sample_mushroom_data(num_contexts, r_noeat=0, r_eat_safe=5, r_eat_poison_bad=-35, r_eat_poison_good=5, prob_poison_bad=0.5)
        dataset, opt_vals = sampled_vals
        opt_rewards, opt_actions = opt_vals
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'financial':
        num_actions = 8
        context_dim = 21
        num_contexts = min(3713, num_contexts)
        file_name = os.path.join(data_dir, 'raw_stock_contexts')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Financial dataset not found at {file_name}.")
        dataset, opt_vals = sample_stock_data(file_name, context_dim, num_actions, num_contexts, sigma=0.01, shuffle_rows=True)
        opt_rewards, opt_actions = opt_vals
        
    elif data_type == 'jester':
        num_actions = 8
        context_dim = 32
        num_contexts = min(19181, num_contexts)
        file_name = os.path.join(data_dir, 'jester_data_40jokes_19181users.npy')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Jester dataset not found at {file_name}.")
        dataset, opt_jester = sample_jester_data(file_name, context_dim, num_actions, num_contexts, shuffle_rows=True, shuffle_cols=True)
        opt_rewards, opt_actions = opt_jester
        
    elif data_type == 'statlog':
        file_name = os.path.join(data_dir, 'statlog.trn')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Statlog dataset not found at {file_name}.")
        num_actions = 7
        num_contexts = min(43500, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_statlog_data(file_name, num_contexts, shuffle_rows=True)
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'adult':
        num_contexts = min(48842, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_adult_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'covertype':
        num_contexts = min(581012, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_covertype_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'census':
        num_contexts = min(2458285, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_census_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    elif data_type == 'statlog_shuttle':
        num_contexts = min(58000, num_contexts)
        dataset, (opt_rewards, opt_actions) = sample_statlog_shuttle_data(num_contexts, shuffle_rows=True)
        num_actions = len(np.unique(opt_actions))
        context_dim = dataset.shape[1] - num_actions
        
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Supported types: linear, sparse_linear, wheel, mushroom, financial, jester, statlog, adult, covertype, census, statlog_shuttle")

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def create_algorithms(num_actions: int, context_dim: int, algorithm_names: List[str]) -> List:
    """Create algorithm instances based on names.
    
    Args:
        num_actions: Number of actions
        context_dim: Context dimension
        algorithm_names: List of algorithm names to create
        
    Returns:
        List of algorithm instances
    """
    algorithms = []
    
    for name in algorithm_names:
        if name == 'neural_bandit':
            hparams = {
                "context_dim": context_dim,
                "num_actions": num_actions,
                "layer_sizes": [100, 100],
                "activation": "relu",
                "initial_lr": 0.001,
                "batch_size": 512,
                "init_scale": 0.3,
                "use_dropout": False,
                "dropout_rate": 0.1,
                "layer_norm": False,
                "verbose": False
            }
            algorithms.append(NeuralBanditModel(hparams, name="neural_bandit"))
            
        elif name == 'neural_linear':
            hparams = {
                "context_dim": context_dim,
                "num_actions": num_actions,
                "layer_sizes": [100, 100],
                "activation": "relu",
                "initial_lr": 0.001,
                "batch_size": 512,
                "init_scale": 0.3,
                "use_dropout": False,
                "dropout_rate": 0.1,
                "layer_norm": False,
                "verbose": False,
                # Neural Linear specific parameters
                "lambda_prior": 0.25,
                "a0": 6,
                "b0": 6,
                "training_freq": 100,
                "training_freq_network": 100,
                "training_epochs": 100,
                "initial_pulls": 2
            }
            algorithms.append(NeuralLinearPosteriorSampling(hparams, name="neural_linear"))
            
        else:
            raise ValueError(f"Unknown algorithm: {name}. Supported: neural_bandit, neural_linear")
    
    return algorithms


def evaluate_performance(actions: np.ndarray, rewards: np.ndarray, opt_rewards: np.ndarray, opt_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate algorithm performance.
    
    Args:
        actions: Matrix of actions taken by algorithms
        rewards: Matrix of rewards received by algorithms
        opt_rewards: Vector of optimal rewards
        opt_actions: Vector of optimal actions
        
    Returns:
        Dictionary of performance metrics
    """
    num_algorithms = actions.shape[1]
    metrics = {}
    
    for i in range(num_algorithms):
        algo_actions = actions[:, i]
        algo_rewards = rewards[:, i]
        
        # Cumulative regret
        cumulative_regret = np.sum(opt_rewards - algo_rewards)
        
        # Average reward
        avg_reward = np.mean(algo_rewards)
        
        # Regret at each step
        regret = opt_rewards - algo_rewards
        cumulative_regret_steps = np.cumsum(regret)
        
        # Store metrics
        metrics[f'algo_{i}_cumulative_regret'] = cumulative_regret
        metrics[f'algo_{i}_avg_reward'] = avg_reward
        metrics[f'algo_{i}_final_regret'] = cumulative_regret_steps[-1]
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run contextual bandit experiments')
    parser.add_argument('--data_type', type=str, default='linear', 
                       choices=['linear', 'sparse_linear', 'wheel', 'mushroom', 'financial', 'jester', 'statlog', 'adult', 'covertype', 'census', 'statlog_shuttle'],
                       help='Type of dataset to use')
    parser.add_argument('--num_contexts', type=int, default=2000,
                       help='Number of contexts to sample')
    parser.add_argument('--algorithms', nargs='+', default=['neural_bandit', 'neural_linear'],
                       choices=['neural_bandit', 'neural_linear'],
                       help='Algorithms to run')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing dataset files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Running experiment with:")
    print(f"  Data type: {args.data_type}")
    print(f"  Number of contexts: {args.num_contexts}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Sample data
    print("Sampling data...")
    start_time = time.time()
    dataset, opt_rewards, opt_actions, num_actions, context_dim = sample_data(
        args.data_type, args.num_contexts, args.data_dir)
    print(f"Data sampled in {time.time() - start_time:.2f} seconds")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Context dimension: {context_dim}")
    print()
    
    # Create algorithms
    print("Creating algorithms...")
    algorithms = create_algorithms(num_actions, context_dim, args.algorithms)
    print(f"Created {len(algorithms)} algorithms")
    print()
    
    # Run experiment
    print("Running contextual bandit experiment...")
    start_time = time.time()
    actions, rewards = run_contextual_bandit(context_dim, num_actions, dataset, algorithms)
    experiment_time = time.time() - start_time
    print(f"Experiment completed in {experiment_time:.2f} seconds")
    print()
    
    # Evaluate performance
    print("Evaluating performance...")
    metrics = evaluate_performance(actions, rewards, opt_rewards, opt_actions)
    
    # Print results
    print("Results:")
    print("=" * 50)
    for i, algo_name in enumerate(args.algorithms):
        print(f"\n{algo_name.upper()}:")
        print(f"  Cumulative Regret: {metrics[f'algo_{i}_cumulative_regret']:.4f}")
        print(f"  Average Reward: {metrics[f'algo_{i}_avg_reward']:.4f}")
        print(f"  Final Regret: {metrics[f'algo_{i}_final_regret']:.4f}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Total time: {experiment_time:.2f} seconds")


if __name__ == "__main__":
    main()
