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

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma
import torch

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralLinearPosteriorSampling(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net (PyTorch version)."""

  def __init__(self, hparams, name="neural_linear"):
    self.name = name
    self.hparams = hparams
    self.latent_dim = self.hparams["layer_sizes"][-1]
    self._lambda_prior = self.hparams["lambda_prior"]
    self.mu = [np.zeros(self.latent_dim) for _ in range(self.hparams["num_actions"])]
    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])]
    self.precision = [self.lambda_prior * np.eye(self.latent_dim) for _ in range(self.hparams["num_actions"])]
    self._a0 = self.hparams["a0"]
    self._b0 = self.hparams["b0"]
    self.a = [self._a0 for _ in range(self.hparams["num_actions"])]
    self.b = [self._b0 for _ in range(self.hparams["num_actions"])]
    self.update_freq_lr = self.hparams["training_freq"]
    self.update_freq_nn = self.hparams["training_freq_network"]
    self.t = 0
    self.num_epochs = self.hparams["training_epochs"]
    self.data_h = ContextualDataset(np.empty((0, self.hparams["context_dim"])), np.empty((0, self.hparams["num_actions"])))
    self.latent_h = ContextualDataset(np.empty((0, self.latent_dim)), np.empty((0, self.hparams["num_actions"])))
    self.bnn = NeuralBanditModel(self.hparams, name=f"{name}-bnn")

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams["num_actions"] * self.hparams["initial_pulls"]:
      return self.t % self.hparams["num_actions"]

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.hparams["num_actions"])
    ]

    try:
      beta_s = [
          np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
          for i in range(self.hparams["num_actions"])
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite
      print(f'Exception when sampling for {self.name}. Details: {e}')
      d = self.latent_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for _ in range(self.hparams["num_actions"])
      ]

    # Compute last-layer representation for the current context
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)  # Exclude last layer

    # Apply Thompson Sampling to last-layer representation
    vals = [
        np.dot(beta_s[i], z_context) for i in range(self.hparams["num_actions"])
    ]
    return int(np.argmax(vals))

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.data_h.add(context, action, reward)
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      z_context = self.bnn.network[:-1](context_tensor).numpy().squeeze(0)
    self.latent_h.add(z_context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:
      self.bnn.train_model(self.data_h, self.num_epochs)

      # Update the latent representation of every datapoint collected so far
      all_contexts = self.data_h.contexts.numpy()
      with torch.no_grad():
        new_z = self.bnn.network[:-1](torch.tensor(all_contexts, dtype=torch.float32)).numpy()
      self.latent_h.contexts = torch.tensor(new_z, dtype=torch.float32)

    # Update the Bayesian Linear Regression
    if self.t % self.update_freq_lr == 0:
      actions_to_update = self.latent_h.actions[:-self.update_freq_lr] if self.update_freq_lr < len(self.latent_h.actions) else self.latent_h.actions
      for action_v in np.unique(actions_to_update):
        z, y = self.latent_h.get_batch_for_action(action_v)
        s = np.dot(z.T, z)
        precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(z.T, y))
        a_post = self.a0 + z.shape[0] / 2.0
        b_upd = 0.5 * np.dot(y.T, y) - 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
        b_post = self.b0 + b_upd
        self.mu[action_v] = mu_a
        self.cov[action_v] = cov_a
        self.precision[action_v] = precision_a
        self.a[action_v] = a_post
        self.b[action_v] = b_post

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior