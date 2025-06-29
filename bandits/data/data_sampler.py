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

"""Functions to create bandit problems from datasets (PyTorch/NumPy version)."""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def sample_mushroom_data(num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """Samples bandit game from Mushroom UCI Dataset using ucimlrepo.
    Args:
        num_contexts: Number of points to sample.
        r_noeat: Reward for not eating a mushroom.
        r_eat_safe: Reward for eating a non-poisonous mushroom.
        r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
        r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
        prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
    Returns:
        dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
        opt_vals: Vector of expected optimal (reward, action) for each context.
    """
    # Fetch mushroom dataset from UCI
    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.features
    targets = mushroom.data.targets
    
    # Combine features and targets
    df = pd.concat([df, targets], axis=1)
    df = one_hot(df, df.columns[:-1])  # One-hot encode all features except the target
    
    # Sample contexts
    if num_contexts > len(df):
        num_contexts = len(df)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=False)

    contexts = df.iloc[ind, 2:]  # Skip the first two columns (edible, poisonous indicators)
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    
    # Create random poison effects
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    
    # Calculate eat rewards based on edible/poisonous indicators
    eat_reward = r_eat_safe * df.iloc[ind, 0]  # edible indicator
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])  # poisonous indicator
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # Calculate optimal expected rewards
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad + r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
        opt_actions = df.iloc[ind, 0]  # indicator of edible
    else:
        opt_actions = np.ones((num_contexts, 1))

    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals


def sample_stock_data(file_name, context_dim, num_actions, num_contexts,
                      sigma, shuffle_rows=True):
    """Samples linear bandit game from stock prices dataset."""
    contexts = np.loadtxt(file_name, skiprows=1)
    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    mean_rewards = np.dot(contexts, betas)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise

    opt_actions = np.argmax(mean_rewards, axis=1)
    opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
    return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)


def sample_jester_data(file_name, context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset."""
    dataset = np.load(file_name)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])

    return dataset, (opt_rewards, opt_actions)


def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI statlog data."""
    data = np.loadtxt(file_name)
    num_actions = 7
    if shuffle_rows:
        np.random.shuffle(data)
    data = data[:num_contexts, :]
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_adult_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI adult data using ucimlrepo."""
    # Fetch adult dataset from UCI
    adult = fetch_ucirepo(id=2)
    features = adult.data.features
    targets = adult.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # One-hot encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = one_hot(data, categorical_columns)
    
    # Convert to numpy array and ensure float32 type
    data = data.astype(np.float32).values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_census_data(num_contexts, shuffle_rows=True,
                       remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI census data using ucimlrepo."""
    # Fetch US Census 1990 dataset from UCI
    census = fetch_ucirepo(id=116)
    features = census.data.features
    targets = census.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # One-hot encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = one_hot(data, categorical_columns)
    
    # Convert to numpy array and ensure float32 type
    data = data.astype(np.float32).values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_covertype_data(num_contexts, shuffle_rows=True,
                          remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI covertype data using ucimlrepo."""
    # Fetch covertype dataset from UCI
    covertype = fetch_ucirepo(id=31)
    features = covertype.data.features
    targets = covertype.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # Convert to numpy array
    data = data.values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)


def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Converts classification data to bandit problem format."""
    n = contexts.shape[0]
    if num_actions is None:
        num_actions = np.max(labels) + 1
    rewards = np.zeros((n, num_actions))
    rewards[np.arange(n), labels] = 1.0
    opt_actions = labels
    opt_rewards = np.ones(n)
    return np.hstack((contexts, rewards)), (opt_rewards, opt_actions)


def safe_std(values):
    """Compute the standard deviation, returns 1.0 if values are constant."""
    std = np.std(values)
    return std if std > 0 else 1.0


def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Remove classes with very few samples."""
    unique, counts = np.unique(labels, return_counts=True)
    mask = np.isin(labels, unique[counts > thresh * len(labels)])
    return features[mask], labels[mask]


def sample_statlog_shuttle_data(num_contexts, shuffle_rows=True,
                                remove_underrepresented=False):
    """Returns bandit problem dataset based on the Statlog Shuttle dataset using ucimlrepo."""
    # Fetch Statlog Shuttle dataset from UCI
    shuttle = fetch_ucirepo(id=148)
    features = shuttle.data.features
    targets = shuttle.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # Convert to numpy array
    data = data.values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)
