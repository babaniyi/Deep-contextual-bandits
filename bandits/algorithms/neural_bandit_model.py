# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a family of neural network architectures for bandits (PyTorch version).
The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from absl import flags
from bandits.core.bayesian_nn import BayesianNeuralNetwork

FLAGS = flags.FLAGS


class NeuralBanditModel(BayesianNeuralNetwork):
    """Implements a neural network for bandit problems using PyTorch."""

    def __init__(self, hparams: Dict[str, Any], name: str = "neural_bandit"):
        """Initialize the neural bandit model.
        
        Args:
            hparams: Dictionary containing hyperparameters
            name: Name of the model
        """
        super().__init__()
        
        self.name = name
        self.hparams = hparams
        self.verbose = hparams.get("verbose", True)
        self.times_trained = 0
        
        # Extract hyperparameters
        self.context_dim = hparams["context_dim"]
        self.num_actions = hparams["num_actions"]
        self.layer_sizes = hparams.get("layer_sizes", [100, 100])
        self.activation = hparams.get("activation", "relu")
        self.learning_rate = hparams.get("initial_lr", 0.001)
        self.batch_size = hparams.get("batch_size", 512)
        self.init_scale = hparams.get("init_scale", 0.3)
        self.use_dropout = hparams.get("use_dropout", False)
        self.dropout_rate = hparams.get("dropout_rate", 0.1)
        self.layer_norm = hparams.get("layer_norm", False)
        
        self.build_model()

    def build_model(self):
        """Build the neural network architecture."""
        layers = []
        input_dim = self.context_dim
        
        # Build hidden layers
        for layer_size in self.layer_sizes:
            if layer_size > 0:
                layers.append(nn.Linear(input_dim, layer_size))
                
                if self.layer_norm:
                    layers.append(nn.LayerNorm(layer_size))
                    
                if self.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.activation == "tanh":
                    layers.append(nn.Tanh())
                elif self.activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                    
                if self.use_dropout:
                    layers.append(nn.Dropout(self.dropout_rate))
                    
                input_dim = layer_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.verbose:
            print(f"Initialized model {self.name} with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -self.init_scale, self.init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, context_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_actions)
        """
        return self.network(x)
    
    def sample_weights(self):
        """Sample weights from the posterior (or approximate posterior).
        For this basic implementation, we return the current weights.
        Subclasses can override this for Bayesian methods.
        """
        return self.state_dict()
    
    def train_step(self, contexts: torch.Tensor, rewards: torch.Tensor, 
                   actions: torch.Tensor) -> float:
        """Perform one training step.
        
        Args:
            contexts: Context tensors of shape (batch_size, context_dim)
            rewards: Reward tensors of shape (batch_size,)
            actions: Action indices of shape (batch_size,)
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.forward(contexts)  # (batch_size, num_actions)
        
        # Create target tensor with rewards only for taken actions
        targets = torch.zeros_like(predictions)
        targets.scatter_(1, actions.unsqueeze(1), rewards.unsqueeze(1))
        
        # Compute loss (only for observed actions)
        loss = F.mse_loss(predictions, targets, reduction='none')
        mask = (targets != 0).float()
        weighted_loss = (loss * mask).sum() / mask.sum()
        
        # Backward pass
        weighted_loss.backward()
        self.optimizer.step()
        
        return weighted_loss.item()
    
    def train_model(self, data, num_steps: int):
        """Train the network for num_steps using the provided data.
        
        Args:
            data: ContextualDataset object that provides the data
            num_steps: Number of training steps
        """
        if self.verbose:
            print(f"Training {self.name} for {num_steps} steps...")
        
        self.train()  # Set to training mode
        
        for step in range(num_steps):
            # Sample batch from data
            batch_indices = np.random.choice(len(data), self.batch_size, replace=True)
            contexts_batch = []
            rewards_batch = []
            actions_batch = []
            
            for idx in batch_indices:
                context, reward = data[idx]
                action = np.random.randint(0, self.num_actions)  # Placeholder
                contexts_batch.append(context)
                # Always extract the reward for the chosen action as a float
                if isinstance(reward, torch.Tensor):
                    if reward.dim() > 0:
                        reward_val = reward[action].item()
                    else:
                        reward_val = reward.item()
                elif isinstance(reward, (np.ndarray, list)):
                    reward_val = float(reward[action]) if len(reward) > 1 else float(reward)
                else:
                    reward_val = float(reward)
                rewards_batch.append(reward_val)
                actions_batch.append(action)
            
            contexts = torch.stack(contexts_batch)
            rewards = torch.tensor(rewards_batch, dtype=torch.float32)
            actions = torch.tensor(actions_batch, dtype=torch.long)
            
            loss = self.train_step(contexts, rewards, actions)
            
            if self.verbose and step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
        
        self.times_trained += 1
    
    def predict(self, contexts: torch.Tensor) -> torch.Tensor:
        """Predict rewards for given contexts.
        
        Args:
            contexts: Context tensors of shape (batch_size, context_dim)
            
        Returns:
            Predicted rewards of shape (batch_size, num_actions)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            return self.forward(contexts)
    
    def get_action(self, context: np.ndarray) -> int:
        """Get the best action for a given context.
        
        Args:
            context: Context array of shape (context_dim,)
            
        Returns:
            Action index
        """
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        predictions = self.predict(context_tensor)
        return torch.argmax(predictions, dim=1).item()
    
    def action(self, context: np.ndarray) -> int:
        """Select action for context (interface for run_contextual_bandit).
        
        Args:
            context: Context array of shape (context_dim,)
            
        Returns:
            Action index
        """
        return self.get_action(context)
    
    def update(self, context: np.ndarray, action: int, reward: float):
        """Update the model with new data (interface for run_contextual_bandit).
        
        Args:
            context: Context array
            action: Action taken
            reward: Reward received
        """
        # For now, we'll just store the data
        # In a more sophisticated implementation, you might want to retrain periodically
        if not hasattr(self, 'data_buffer'):
            from bandits.core.contextual_dataset import ContextualDataset
            self.data_buffer = ContextualDataset(
                np.empty((0, self.context_dim)), 
                np.empty((0, self.num_actions))
            )
        
        self.data_buffer.add(context, action, reward)
        
        # Optionally retrain periodically
        if len(self.data_buffer) % 100 == 0 and len(self.data_buffer) > 0:
            self.train_model(self.data_buffer, 10)  # Retrain for 10 steps