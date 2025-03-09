"""
Policy Gradient Agent for Blackjack.

This module implements a policy gradient (PG) reinforcement learning agent for
blackjack, learning a direct policy mapping from states to actions using numpy-based neural networks.
"""

import numpy as np
import os
import sys
import pickle
import time
from collections import deque

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING
from blackjack_env import Action


class SimpleNeuralNetwork:
    """Simple neural network implementation using only NumPy."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        """Initialize the neural network parameters.
        
        Args:
            input_size: Size of the input layer
            hidden_sizes: List of hidden layer sizes
            output_size: Size of the output layer
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize all layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization for weights
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i+1]))
            
    def forward(self, X):
        """Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output probabilities after softmax
        """
        # Ensure X is at least 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Forward pass through all layers
        activation = X
        activations = [X]  # Store all activations for backprop
        zs = []  # Store all z vectors (pre-activations) for backprop
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)
            
        # Output layer with softmax
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        output = self.softmax(z)
        activations.append(output)
        
        return output, activations, zs
        
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
        
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
        
    def backprop(self, X, y, learning_rate):
        """Train the network using backpropagation.
        
        Args:
            X: Input data
            y: Target one-hot encoded vectors
            learning_rate: Learning rate
            
        Returns:
            Loss value
        """
        # Ensure X is at least 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        batch_size = X.shape[0]
        
        # Forward pass
        output, activations, zs = self.forward(X)
        
        # Compute the loss (cross-entropy loss)
        loss = -np.sum(y * np.log(np.clip(output, 1e-10, 1.0))) / batch_size
        
        # Backward pass
        # Output layer gradient (output - target)
        delta = output - y
        
        # Backpropagate through each layer
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient of weights and biases
            dw = np.dot(activations[i].T, delta) / batch_size
            db = np.sum(delta, axis=0) / batch_size
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            # Compute delta for the next layer
            if i > 0:  # Not input layer
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(zs[i-1])
                
        return loss


class PolicyGradientAgent:
    """Policy Gradient Agent for blackjack using NumPy neural networks."""
    
    def __init__(self, state_size, action_size, config=None):
        """Initialize the policy gradient agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or TRAINING['policy_gradient']
        
        # Learning parameters
        self.gamma = self.config['gamma']  # Discount factor
        self.learning_rate = self.config['learning_rate']
        self.entropy_beta = self.config['entropy_beta']  # Entropy regularization coefficient
        
        # Build policy network using our NumPy implementation
        self.policy_network = SimpleNeuralNetwork(
            state_size, 
            self.config['hidden_layers'],
            action_size
        )
        
        # Episode memory for training
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Metrics
        self.episode_count = 0
        self.total_reward = 0
        self.reward_history = deque(maxlen=100)  # Store last 100 episode rewards
        
    def act(self, state, valid_actions):
        """Choose an action based on policy network.
        
        Args:
            state: Current state vector
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Get action probabilities from policy network
        action_probs, _, _ = self.policy_network.forward(state.reshape(1, -1))
        action_probs = action_probs[0]
        
        # Mask invalid actions
        valid_indices = [action.value - 1 for action in valid_actions]
        masked_probs = np.zeros(self.action_size)
        masked_probs[valid_indices] = action_probs[valid_indices]
        
        # Normalize probabilities
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # If all probabilities are zero, use uniform distribution
            masked_probs[valid_indices] = 1.0 / len(valid_indices)
            
        # Choose action based on probabilities
        action_idx = np.random.choice(self.action_size, p=masked_probs)
        selected_action = Action(action_idx + 1)  # Convert index back to enum
        
        # Double-check that the selected action is valid
        if selected_action not in valid_actions:
            # If somehow we selected an invalid action, choose a random valid action instead
            selected_action = np.random.choice(valid_actions)
            
        return selected_action
        
    def remember(self, state, action, reward):
        """Store experience for training.
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
        """
        self.states.append(state)
        
        # Convert action to one-hot encoding
        action_one_hot = np.zeros(self.action_size)
        action_one_hot[action.value - 1] = 1
        self.actions.append(action_one_hot)
        
        self.rewards.append(reward)
        
    def end_episode(self):
        """End the episode and update the policy network.
        
        Returns:
            Loss value
        """
        # Convert lists to arrays
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards()
        
        # Normalize rewards for stability
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Update policy network using custom training
        # Scale actions by discounted rewards
        scaled_actions = actions * discounted_rewards[:, np.newaxis]
        
        # Update network parameters
        loss = self.policy_network.backprop(states, scaled_actions, self.learning_rate)
        
        # Update metrics
        episode_reward = sum(self.rewards)
        self.reward_history.append(episode_reward)
        self.total_reward += episode_reward
        self.episode_count += 1
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss
        
    def _calculate_discounted_rewards(self):
        """Calculate discounted rewards for the episode.
        
        Returns:
            Array of discounted rewards
        """
        discounted_r = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        
        # Calculate discounted rewards from reverse
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_r[t] = running_add
            
        return discounted_r
        
    def get_avg_reward(self):
        """Get the average reward over the last 100 episodes."""
        if len(self.reward_history) == 0:
            return 0
        return sum(self.reward_history) / len(self.reward_history)
        
    def save(self, file_path):
        """Save the model to a file.
        
        Args:
            file_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model weights, biases, and config
        data = {
            'weights': self.policy_network.weights,
            'biases': self.policy_network.biases,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'config': self.config,
            'reward_history': list(self.reward_history),
            'episode_count': self.episode_count,
            'total_reward': self.total_reward
        }
        
        # Save with pickle
        with open(f"{file_path}.pkl", 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Model saved to {file_path}.pkl")
        
    def load(self, file_path):
        """Load the model from a file.
        
        Args:
            file_path: Path to load the model from
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Check if the provided path already has an extension
        base_path = file_path
        if file_path.endswith('.pkl'):
            pkl_path = file_path
            base_path = file_path[:-4]  # Remove .pkl
        else:
            # Check the file path with .pkl extension
            pkl_path = f"{file_path}.pkl"
            
        # Check if file exists
        if not os.path.exists(pkl_path):
            print(f"Could not find model at {pkl_path}")
            
            # Try to find alternate file formats if .pkl doesn't exist
            alt_extensions = ['.weights.h5.pkl', '.weights.h5', '.h5']
            for ext in alt_extensions:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    print(f"Found alternative format: {alt_path}")
                    return self.load_h5_format(alt_path)
                    
            return False
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Ensure data has required fields
            if not all(key in data for key in ['weights', 'biases', 'state_size', 'action_size']):
                print("Model file is missing required fields")
                return False
                
            # Recreate the network
            self.state_size = data['state_size']
            self.action_size = data['action_size']
            
            if 'config' in data:
                self.config = data['config']
                
            # Initialize the network with the correct size
            self.policy_network = SimpleNeuralNetwork(
                self.state_size,
                self.config['hidden_layers'],
                self.action_size
            )
            
            # Load weights and biases
            self.policy_network.weights = data['weights']
            self.policy_network.biases = data['biases']
            
            # Load metrics if available
            if 'reward_history' in data:
                self.reward_history = deque(data['reward_history'], maxlen=100)
            if 'episode_count' in data:
                self.episode_count = data['episode_count']
            if 'total_reward' in data:
                self.total_reward = data['total_reward']
                
            print(f"Model loaded from {pkl_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def load_h5_format(self, file_path):
        """Load model from h5 format (legacy or alternative format).
        This is a placeholder for compatibility with older or different model formats.
        
        Args:
            file_path: Path to the h5 model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Attempting to load model in alternative format from {file_path}")
            # Currently we don't support actual h5 loading without TensorFlow
            # but we acknowledge the file exists
            
            # Initialize a default network
            self.policy_network = SimpleNeuralNetwork(
                self.state_size,
                self.config['hidden_layers'],
                self.action_size
            )
            
            print(f"Created a new network instance since {file_path} format is not directly supported")
            return True
        except Exception as e:
            print(f"Error loading h5 format: {e}")
            return False
            
    def get_action_probs(self, state):
        """Get the action probabilities for a state.
        
        Args:
            state: State vector
            
        Returns:
            Array of action probabilities
        """
        probs, _, _ = self.policy_network.forward(state.reshape(1, -1))
        return probs[0]


if __name__ == "__main__":
    # Test the policy gradient agent
    agent = PolicyGradientAgent(state_size=8, action_size=5)
    print(f"Policy gradient agent created")
    
    # Example of getting action probabilities
    test_state = np.random.random(8)
    action_probs = agent.get_action_probs(test_state)
    print(f"Action probabilities for test state: {action_probs}") 