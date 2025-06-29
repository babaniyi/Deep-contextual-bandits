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

"""Define a data buffer for contextual bandit algorithms (PyTorch version)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


class ContextualDataset(Dataset):
    """PyTorch Dataset for contextual bandit data."""
    
    def __init__(self, contexts: np.ndarray, rewards: np.ndarray):
        """Initialize the dataset.
        
        Args:
            contexts: Context data of shape (num_samples, context_dim)
            rewards: Reward data of shape (num_samples, num_actions) or (num_samples,)
        """
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        
        # Handle different reward formats
        if len(rewards.shape) == 1:
            # Single reward per sample
            self.rewards = torch.tensor(rewards, dtype=torch.float32)
        else:
            # Multiple rewards per sample (one per action)
            self.rewards = torch.tensor(rewards, dtype=torch.float32)
        
        self.actions = []  # Track actions taken
        self.buffer_size = -1  # -1 means use all data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (context, reward)
        """
        return self.contexts[idx], self.rewards[idx]

    def add(self, context: np.ndarray, action: int, reward: float):
        """Add a new triplet (context, action, reward) to the dataset.
        
        Args:
            context: Context vector
            action: Action index
            reward: Reward value
        """
        # Convert context to tensor and add to contexts
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        self.contexts = torch.cat([self.contexts, context_tensor], dim=0)
        
        # Handle reward based on format
        if len(self.rewards.shape) == 1:
            # Single reward per sample
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            self.rewards = torch.cat([self.rewards, reward_tensor], dim=0)
        else:
            # Multiple rewards per sample
            reward_vector = torch.zeros(self.rewards.shape[1], dtype=torch.float32)
            reward_vector[action] = reward
            reward_tensor = reward_vector.unsqueeze(0)
            self.rewards = torch.cat([self.rewards, reward_tensor], dim=0)
        
        self.actions.append(action)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random minibatch of (contexts, rewards).
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Tuple of (contexts, rewards) tensors
        """
        n = len(self.contexts)
        if self.buffer_size == -1:
            # Use all data
            indices = np.random.choice(range(n), batch_size, replace=True)
        else:
            # Use only buffer (last buffer_size observations)
            start_idx = max(0, n - self.buffer_size)
            indices = np.random.choice(range(start_idx, n), batch_size, replace=True)
        
        return self.contexts[indices], self.rewards[indices]

    def get_batch_with_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a random minibatch with one-hot weights for actions.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Tuple of (contexts, rewards, weights) tensors
        """
        contexts, rewards = self.get_batch(batch_size)
        
        # Create weights tensor
        if len(self.actions) > 0:
            num_actions = rewards.shape[1] if len(rewards.shape) > 1 else 1
            weights = torch.zeros(batch_size, num_actions)
            
            # For now, assume random actions (in practice, this would come from the algorithm)
            random_actions = torch.randint(0, num_actions, (batch_size,))
            weights.scatter_(1, random_actions.unsqueeze(1), 1.0)
        else:
            weights = torch.zeros(batch_size, 1)
        
        return contexts, rewards, weights

    def get_batch_for_action(self, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data for a specific action.
        
        Args:
            action: Action index to filter by
            
        Returns:
            Tuple of (contexts, rewards) for the specified action
        """
        if not self.actions:
            return np.empty((0, self.contexts.shape[1])), np.empty((0,))
        
        # Find indices where the action was taken
        action_indices = [i for i, a in enumerate(self.actions) if a == action]
        
        if not action_indices:
            return np.empty((0, self.contexts.shape[1])), np.empty((0,))
        
        contexts_for_action = self.contexts[action_indices].numpy()
        
        # Handle different reward formats
        if len(self.rewards.shape) == 1:
            rewards_for_action = self.rewards[action_indices].numpy()
        else:
            rewards_for_action = self.rewards[action_indices, action].numpy()
        
        return contexts_for_action, rewards_for_action

    def set_buffer_size(self, buffer_size: int):
        """Set the buffer size for sampling.
        
        Args:
            buffer_size: Number of recent samples to use (-1 for all)
        """
        self.buffer_size = buffer_size

    def num_points(self) -> int:
        """Return the number of points in the dataset."""
        return len(self.contexts)

    @property
    def context_dim(self):
        return self.contexts.shape[1]

    @property
    def num_actions(self):
        return self.rewards.shape[1]

    @property
    def contexts(self):
        return self._contexts

    @contexts.setter
    def contexts(self, value):
        self._contexts = value

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        self._rewards = value