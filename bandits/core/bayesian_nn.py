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

"""Define the abstract class for Bayesian Neural Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BayesianNeuralNetwork(nn.Module, ABC):
    """
    Base class for Bayesian Neural Networks for contextual bandits.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the network.
        """
        pass

    @abstractmethod
    def sample_weights(self):
        """
        Sample weights from the posterior (or approximate posterior).
        """
        pass

    def log_likelihood(self, x, y):
        """
        Compute the log likelihood of the data under the model.
        Optional: override in subclasses if needed.
        """
        raise NotImplementedError

class BayesianNN(object):
  """A Bayesian neural network keeps a distribution over neural nets."""

  def __init__(self, optimizer):
    pass

  def build_model(self):
    pass

  def train(self, data):
    pass

  def sample(self, steps):
    pass