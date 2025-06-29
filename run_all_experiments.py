#!/usr/bin/env python3
"""
Batch experiment runner for Deep Contextual Bandits.

This script runs all available algorithms on all datasets and saves the results
to a CSV file for analysis and comparison.

Inspired by the original TensorFlow implementation:
https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits
"""

import argparse
import time
import numpy as np
import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Import our PyTorch implementations
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling

# Import core functionality
from bandits.core.contextual_bandit import run_contextual_bandit

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
    """Sample data from given 'data_type'."""
    
    if data_type == 'linear':
        num_actions = 8
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim, num_actions, sigma=noise_stds)
        opt_rewards, opt_actions = opt_linear
        
    elif data_type == 'sparse_linear':
        num_actions = 7
        context_dim = 10
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
        num_nnz_dims = int(context_dim / 3.0)
        dataset, _, opt_sparse_linear = sample_sparse_linear_data(
            num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
        opt_rewards, opt_actions = opt_sparse_linear
        
    elif data_type == 'wheel':
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
        raise ValueError(f"Unknown data_type: {data_type}")
    
    return dataset, opt_rewards, opt_actions, num_actions, context_dim


def create_algorithm(name: str, num_actions: int, context_dim: int) -> Any:
    """Create algorithm instance based on name."""
    
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
        return NeuralBanditModel(hparams, name="neural_bandit")
        
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
            "lambda_prior": 0.25,
            "a0": 6,
            "b0": 6,
            "training_freq": 100,
            "training_freq_network": 100,
            "training_epochs": 100,
            "initial_pulls": 2
        }
        return NeuralLinearPosteriorSampling(hparams, name="neural_linear")
        
    elif name == 'uniform':
        # Create a simple hparams object for uniform sampling
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
        
        hparams = HParams(num_actions)
        return UniformSampling("uniform", hparams)
        
    elif name == 'linear_full_posterior':
        # Create a simple hparams object for linear full posterior
        class HParams:
            def __init__(self, context_dim, num_actions):
                self.context_dim = context_dim
                self.num_actions = num_actions
                self.lambda_prior = 0.25
                self.a0 = 6
                self.b0 = 6
                self.initial_pulls = 2
        
        hparams = HParams(context_dim, num_actions)
        return LinearFullPosteriorSampling("linear_full_posterior", hparams)
        
    elif name == 'bootstrapped_bnn':
        # Create a simple hparams object for bootstrapped BNN
        class HParams:
            def __init__(self, context_dim, num_actions):
                self.context_dim = context_dim
                self.num_actions = num_actions
                self.training_freq = 100
                self.training_epochs = 100
                self.q = 10  # number of models
                self.p = 0.8  # probability of including each datapoint
                self.initial_pulls = 2
                self.buffer_s = 10000
        
        hparams = HParams(context_dim, num_actions)
        return BootstrappedBNNSampling("bootstrapped_bnn", hparams)
        
    elif name == 'fixed_policy':
        # Create a simple hparams object for fixed policy
        class HParams:
            def __init__(self, num_actions):
                self.num_actions = num_actions
        
        hparams = HParams(num_actions)
        # Create uniform policy
        p = np.ones(num_actions) / num_actions
        return FixedPolicySampling("fixed_policy", p, hparams)
        
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def evaluate_performance(actions: np.ndarray, rewards: np.ndarray, opt_rewards: np.ndarray, opt_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate algorithm performance."""
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


def run_single_experiment(data_type: str, algorithm_name: str, num_contexts: int, 
                         data_dir: str, seed: int) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    print(f"Running {algorithm_name} on {data_type} dataset...")
    
    # Set random seed
    np.random.seed(seed)
    
    try:
        # Sample data
        start_time = time.time()
        dataset, opt_rewards, opt_actions, num_actions, context_dim = sample_data(
            data_type, num_contexts, data_dir)
        data_time = time.time() - start_time
        
        # Create algorithm
        algorithm = create_algorithm(algorithm_name, num_actions, context_dim)
        
        # Run experiment
        start_time = time.time()
        actions, rewards = run_contextual_bandit(context_dim, num_actions, dataset, [algorithm])
        experiment_time = time.time() - start_time
        
        # Evaluate performance
        metrics = evaluate_performance(actions, rewards, opt_rewards, opt_actions)
        
        # Prepare results
        results = {
            'data_type': data_type,
            'algorithm': algorithm_name,
            'num_contexts': num_contexts,
            'context_dim': context_dim,
            'num_actions': num_actions,
            'cumulative_regret': metrics['algo_0_cumulative_regret'],
            'avg_reward': metrics['algo_0_avg_reward'],
            'final_regret': metrics['algo_0_final_regret'],
            'data_time': data_time,
            'experiment_time': experiment_time,
            'total_time': data_time + experiment_time,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        print(f"  ✓ Completed in {results['total_time']:.2f}s")
        print(f"    Cumulative Regret: {results['cumulative_regret']:.4f}")
        print(f"    Average Reward: {results['avg_reward']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return {
            'data_type': data_type,
            'algorithm': algorithm_name,
            'num_contexts': num_contexts,
            'context_dim': None,
            'num_actions': None,
            'cumulative_regret': None,
            'avg_reward': None,
            'final_regret': None,
            'data_time': None,
            'experiment_time': None,
            'total_time': None,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments for all algorithms and datasets')
    parser.add_argument('--num_contexts', type=int, default=1000,
                       help='Number of contexts to sample (default: 1000)')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing dataset files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--datasets', nargs='+', 
                       default=['linear', 'sparse_linear', 'wheel', 'mushroom', 'statlog', 'adult'],
                       help='Datasets to run (default: linear, sparse_linear, wheel, mushroom, statlog, adult)')
    parser.add_argument('--algorithms', nargs='+',
                       default=['neural_bandit', 'neural_linear', 'uniform', 'linear_full_posterior'],
                       help='Algorithms to run')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define all available datasets and algorithms
    all_datasets = ['linear', 'sparse_linear', 'wheel', 'mushroom', 'financial', 'jester', 
                   'statlog', 'adult', 'covertype', 'census', 'statlog_shuttle']
    all_algorithms = ['neural_bandit', 'neural_linear', 'uniform', 'linear_full_posterior', 
                     'bootstrapped_bnn', 'fixed_policy']
    
    # Filter datasets and algorithms based on args
    datasets_to_run = [d for d in args.datasets if d in all_datasets]
    algorithms_to_run = [a for a in args.algorithms if a in all_algorithms]
    
    print(f"Running batch experiments:")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Algorithms: {algorithms_to_run}")
    print(f"  Number of contexts: {args.num_contexts}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Run experiments
    all_results = []
    total_experiments = len(datasets_to_run) * len(algorithms_to_run)
    completed_experiments = 0
    
    start_time = time.time()
    
    for data_type in datasets_to_run:
        for algorithm_name in algorithms_to_run:
            completed_experiments += 1
            print(f"[{completed_experiments}/{total_experiments}] ", end="")
            
            results = run_single_experiment(
                data_type, algorithm_name, args.num_contexts, args.data_dir, args.seed
            )
            all_results.append(results)
            
            # Save intermediate results
            if completed_experiments % 5 == 0:
                save_results(all_results, args.output_dir)
    
    total_time = time.time() - start_time
    
    # Save final results
    save_results(all_results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len([r for r in all_results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in all_results if r['status'] == 'failed'])}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
    print(f"Results saved to: {args.output_dir}/")
    
    # Create summary table
    create_summary_table(all_results, args.output_dir)


def save_results(results: List[Dict], output_dir: str):
    """Save results to CSV and JSON files."""
    
    # Save as CSV
    csv_file = os.path.join(output_dir, 'results.csv')
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
    
    # Save as JSON
    json_file = os.path.join(output_dir, 'results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def create_summary_table(results: List[Dict], output_dir: str):
    """Create a summary table of results."""
    
    successful_results = [r for r in results if r['status'] == 'success']
    if not successful_results:
        print("No successful results to create summary table.")
        return
    
    # Create pivot table
    df = pd.DataFrame(successful_results)
    
    # Create regret table
    regret_table = df.pivot(index='data_type', columns='algorithm', values='cumulative_regret')
    regret_file = os.path.join(output_dir, 'regret_summary.csv')
    regret_table.to_csv(regret_file)
    
    # Create reward table
    reward_table = df.pivot(index='data_type', columns='algorithm', values='avg_reward')
    reward_file = os.path.join(output_dir, 'reward_summary.csv')
    reward_table.to_csv(reward_file)
    
    print(f"\nSummary tables saved:")
    print(f"  Regret summary: {regret_file}")
    print(f"  Reward summary: {reward_file}")
    
    # Print summary tables
    print("\nCUMULATIVE REGRET SUMMARY:")
    print(regret_table.round(2))
    
    print("\nAVERAGE REWARD SUMMARY:")
    print(reward_table.round(4))


if __name__ == "__main__":
    main() 