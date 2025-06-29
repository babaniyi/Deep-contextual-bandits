"""Tests for the PyTorch neural bandit model."""

import unittest
import torch
import numpy as np
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.core.contextual_dataset import ContextualDataset


class TestNeuralBanditModel(unittest.TestCase):
    """Test cases for NeuralBanditModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hparams = {
            "context_dim": 10,
            "num_actions": 5,
            "layer_sizes": [50, 25],
            "activation": "relu",
            "initial_lr": 0.001,
            "batch_size": 32,
            "init_scale": 0.3,
            "use_dropout": False,
            "dropout_rate": 0.1,
            "layer_norm": False,
            "verbose": False
        }
        
        self.model = NeuralBanditModel(self.hparams, name="test_model")
        
        # Create synthetic data for testing
        self.num_samples = 100
        self.contexts = np.random.randn(self.num_samples, self.hparams["context_dim"])
        self.rewards = np.random.randn(self.num_samples, self.hparams["num_actions"])
        
        self.dataset = ContextualDataset(self.contexts, self.rewards)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.context_dim, self.hparams["context_dim"])
        self.assertEqual(self.model.num_actions, self.hparams["num_actions"])
        self.assertEqual(self.model.layer_sizes, self.hparams["layer_sizes"])
        self.assertEqual(self.model.name, "test_model")
        self.assertEqual(self.model.times_trained, 0)
        
        # Check that the network has the expected structure
        self.assertIsNotNone(self.model.network)
        self.assertIsNotNone(self.model.optimizer)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        batch_size = 16
        x = torch.randn(batch_size, self.hparams["context_dim"])
        
        # Test forward pass
        output = self.model.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.hparams["num_actions"]))
        
        # Check that output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
    
    def test_predict(self):
        """Test prediction method."""
        batch_size = 8
        contexts = torch.randn(batch_size, self.hparams["context_dim"])
        
        predictions = self.model.predict(contexts)
        
        # Check output shape
        self.assertEqual(predictions.shape, (batch_size, self.hparams["num_actions"]))
        
        # Check that model is in eval mode after prediction
        self.assertFalse(self.model.training)
    
    def test_get_action(self):
        """Test action selection."""
        context = np.random.randn(self.hparams["context_dim"])
        
        action = self.model.get_action(context)
        
        # Check that action is valid
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.hparams["num_actions"])
    
    def test_train_step(self):
        """Test single training step."""
        batch_size = 16
        contexts = torch.randn(batch_size, self.hparams["context_dim"])
        rewards = torch.randn(batch_size)
        actions = torch.randint(0, self.hparams["num_actions"], (batch_size,))
        
        # Get initial loss
        initial_predictions = self.model.forward(contexts)
        initial_loss = torch.nn.functional.mse_loss(initial_predictions, torch.zeros_like(initial_predictions))
        
        # Perform training step
        loss = self.model.train_step(contexts, rewards, actions)
        
        # Check that loss is a float
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
    
    def test_training(self):
        """Test full training loop."""
        num_steps = 10
        
        # Train the model
        self.model.train_model(self.dataset, num_steps)
        
        # Check that training counter increased
        self.assertEqual(self.model.times_trained, 1)
    
    def test_sample_weights(self):
        """Test weight sampling."""
        weights = self.model.sample_weights()
        
        # Check that weights is a state dict
        self.assertIsInstance(weights, dict)
        
        # Check that it contains the expected keys
        expected_keys = set()
        for name, _ in self.model.named_parameters():
            expected_keys.add(name)
        
        self.assertEqual(set(weights.keys()), expected_keys)
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]
        
        for activation in activations:
            hparams = self.hparams.copy()
            hparams["activation"] = activation
            
            model = NeuralBanditModel(hparams, name=f"test_{activation}")
            
            # Test forward pass
            x = torch.randn(4, self.hparams["context_dim"])
            output = model.forward(x)
            
            self.assertEqual(output.shape, (4, self.hparams["num_actions"]))
    
    def test_dropout(self):
        """Test dropout functionality."""
        hparams = self.hparams.copy()
        hparams["use_dropout"] = True
        hparams["dropout_rate"] = 0.5
        
        model = NeuralBanditModel(hparams, name="test_dropout")
        
        # Test forward pass in training mode
        model.train()  # PyTorch's train() method
        x = torch.randn(4, self.hparams["context_dim"])
        output1 = model.forward(x)
        
        # Test forward pass in eval mode
        model.eval()  # PyTorch's eval() method
        output2 = model.forward(x)
        
        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(output1, output2))
    
    def test_layer_norm(self):
        """Test layer normalization."""
        hparams = self.hparams.copy()
        hparams["layer_norm"] = True
        
        model = NeuralBanditModel(hparams, name="test_layer_norm")
        
        # Test forward pass
        x = torch.randn(4, self.hparams["context_dim"])
        output = model.forward(x)
        
        self.assertEqual(output.shape, (4, self.hparams["num_actions"]))


if __name__ == "__main__":
    unittest.main() 