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

"""Define a contextual bandit from which we can sample and compute rewards (PyTorch version).
We can feed the data, sample a context, its reward for a specific action, and
also the optimal action for a given context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import Tuple, List, Optional


def run_contextual_bandit(context_dim: int, num_actions: int, dataset: np.ndarray, 
                          algos: List) -> Tuple[np.ndarray, np.ndarray]:
    """Run a contextual bandit problem on a set of algorithms.
    
    Args:
        context_dim: Dimension of the context.
        num_actions: Number of available actions.
        dataset: Matrix where every row is a context + num_actions rewards.
        algos: List of algorithms to use in the contextual bandit instance.
        
    Returns:
        h_actions: Matrix with actions: size (num_context, num_algorithms).
        h_rewards: Matrix with rewards: size (num_context, num_algorithms).
    """
    num_contexts = dataset.shape[0]

    # Create contextual bandit
    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)

    h_actions = np.empty((0, len(algos)), float)
    h_rewards = np.empty((0, len(algos)), float)

    # Run the contextual bandit process
    for i in range(num_contexts):
        context = cmab.context(i)
        actions = [a.action(context) for a in algos]
        rewards = [cmab.reward(i, action) for action in actions]

        for j, a in enumerate(algos):
            a.update(context, actions[j], rewards[j])

        h_actions = np.vstack((h_actions, np.array(actions)))
        h_rewards = np.vstack((h_rewards, np.array(rewards)))

    return h_actions, h_rewards


class ContextualBandit:
    """Contextual Bandit environment for benchmarking algorithms."""
    
    def __init__(self, n_arms: int, context_dim: int):
        """Initialize the contextual bandit.
        
        Args:
            n_arms: Number of arms/actions available
            context_dim: Dimension of the context vectors
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self._number_contexts = 0
        self.data = None
        self.order = None
        self.reset()

    def feed_data(self, data: np.ndarray):
        """Feed the data (contexts + rewards) to the bandit object.
        
        Args:
            data: Numpy array with shape [n, d+k], where n is the number of contexts,
                d is the dimension of each context, and k the number of arms (rewards).
                
        Raises:
            ValueError: when data dimensions do not correspond to the object values.
        """
        if data.shape[1] != self.context_dim + self.n_arms:
            raise ValueError('Data dimensions do not match.')

        self._number_contexts = data.shape[0]
        self.data = data
        self.order = np.arange(self._number_contexts)

    def reset(self):
        """Reset the environment and randomly shuffle the order of contexts."""
        if self.data is not None:
            self.order = np.random.permutation(self._number_contexts)
        self.t = 0

    def context(self, number: int) -> np.ndarray:
        """Return the number-th context.
        
        Args:
            number: Index of the context to return
            
        Returns:
            Context vector of shape (context_dim,)
        """
        if self.data is None:
            raise ValueError("No data has been fed to the bandit.")
        return self.data[self.order[number]][:self.context_dim]

    def reward(self, number: int, action: int) -> float:
        """Return the reward for the number-th context and action.
        
        Args:
            number: Index of the context
            action: Action index
            
        Returns:
            Reward value
        """
        if self.data is None:
            raise ValueError("No data has been fed to the bandit.")
        return self.data[self.order[number]][self.context_dim + action]

    def optimal(self, number: int) -> int:
        """Return the optimal action (in hindsight) for the number-th context.
        
        Args:
            number: Index of the context
            
        Returns:
            Optimal action index
        """
        if self.data is None:
            raise ValueError("No data has been fed to the bandit.")
        return np.argmax(self.data[self.order[number]][self.context_dim:])

    def get_context(self) -> np.ndarray:
        """Return the current context vector (for step-based interface).
        
        Returns:
            Current context vector
        """
        if self.data is None:
            # Fallback to random context if no data
            return np.random.randn(self.context_dim)
        return self.context(self.t)

    def get_reward(self, action: int, context: np.ndarray) -> float:
        """Return the reward for the given action and context.
        
        Args:
            action: Action index
            context: Context vector
            
        Returns:
            Reward value
        """
        if self.data is None:
            # Fallback to random reward if no data
            return np.random.randn()
        return self.reward(self.t, action)

    def step(self, action: int) -> Tuple[np.ndarray, float]:
        """Take an action, return (next_context, reward).
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_context, reward)
        """
        context = self.get_context()
        reward = self.get_reward(action, context)
        self.t += 1
        return context, reward

    @property
    def num_actions(self) -> int:
        """Number of actions available."""
        return self.n_arms

    @property
    def number_contexts(self) -> int:
        """Number of contexts in the dataset."""
        return self._number_contexts