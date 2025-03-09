"""
Policy Gradient Agent for Blackjack.

This module implements a policy gradient (PG) reinforcement learning agent for
blackjack, learning a direct policy mapping from states to actions using neural networks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import sys
import pickle
import time
from collections import deque

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING
from blackjack_env import Action


class PolicyNetwork(Model):
    """Policy network model for policy gradient methods."""
    
    def __init__(self, state_size, action_size, hidden_layers):
        """Initialize the policy network.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()
        
        self.hidden_layers = []
        for units in hidden_layers:
            self.hidden_layers.append(Dense(units, activation='relu'))
            
        self.output_layer = Dense(action_size, activation='softmax')
        
    def call(self, inputs):
        """Forward pass through the network.
        
        Args:
            inputs: State vector
            
        Returns:
            Action probabilities
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.output_layer(x)


class PolicyGradientAgent:
    """Policy Gradient Agent for blackjack using neural networks."""
    
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
        
        # Build policy network
        self.policy_network = PolicyNetwork(
            state_size, 
            action_size, 
            self.config['hidden_layers']
        )
        
        # Compile policy network
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
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
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action_probs = self.policy_network(state_tensor)[0].numpy()
        
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
        return Action(action_idx + 1)  # Convert index back to enum
        
    def remember(self, state, action, reward):
        """Store experience in episode memory.
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
        """
        self.states.append(state)
        
        # Convert action to one-hot encoding
        action_idx = action.value - 1
        action_onehot = np.zeros(self.action_size)
        action_onehot[action_idx] = 1
        self.actions.append(action_onehot)
        
        self.rewards.append(reward)
        self.total_reward += reward
        
    def end_episode(self):
        """End the episode and update policy network."""
        self.episode_count += 1
        self.reward_history.append(self.total_reward)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards()
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-7)
        
        # Convert to tensors
        states = tf.convert_to_tensor(np.vstack(self.states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.vstack(self.actions), dtype=tf.float32)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        
        # Train policy network
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs = self.policy_network(states)
            
            # Calculate loss
            neg_log_prob = -tf.math.log(tf.reduce_sum(action_probs * actions, axis=1) + 1e-7)
            policy_loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
            
            # Add entropy regularization for exploration
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-7), axis=1)
            entropy_loss = -self.entropy_beta * tf.reduce_mean(entropy)
            
            # Total loss
            loss = policy_loss + entropy_loss
            
        # Calculate gradients and update network
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        # Reset episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.total_reward = 0
        
        return loss.numpy()
        
    def _calculate_discounted_rewards(self):
        """Calculate discounted rewards for the episode.
        
        Returns:
            Array of discounted rewards
        """
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        cumulative_reward = 0
        
        # Calculate discounted rewards from the end of the episode
        for i in reversed(range(len(self.rewards))):
            cumulative_reward = self.rewards[i] + self.gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward
            
        return discounted_rewards
        
    def get_avg_reward(self):
        """Get average reward over last 100 episodes.
        
        Returns:
            Average reward
        """
        if not self.reward_history:
            return 0
        return np.mean(self.reward_history)
        
    def save(self, file_path):
        """Save the policy network to file.
        
        Args:
            file_path: Path to save the agent
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model weights and metadata
        self.policy_network.save_weights(file_path + '.h5')
        
        metadata = {
            'episode_count': self.episode_count,
            'reward_history': list(self.reward_history),
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'entropy_beta': self.entropy_beta
        }
        
        with open(file_path + '.meta', 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Policy gradient agent saved to {file_path}")
        
    def load(self, file_path):
        """Load the policy network from file.
        
        Args:
            file_path: Path to load the agent from
        """
        # Build network if not already built
        if not self.policy_network.built:
            dummy_state = np.zeros((1, self.state_size))
            self.policy_network(dummy_state)
            
        # Load model weights
        if os.path.exists(file_path + '.h5'):
            self.policy_network.load_weights(file_path + '.h5')
            
            # Load metadata
            if os.path.exists(file_path + '.meta'):
                with open(file_path + '.meta', 'rb') as f:
                    metadata = pickle.load(f)
                    
                self.episode_count = metadata.get('episode_count', 0)
                self.reward_history = deque(metadata.get('reward_history', []), maxlen=100)
                self.gamma = metadata.get('gamma', self.gamma)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.entropy_beta = metadata.get('entropy_beta', self.entropy_beta)
                
                # Update optimizer learning rate
                self.optimizer.learning_rate.assign(self.learning_rate)
                
            print(f"Policy gradient agent loaded from {file_path}")
        else:
            print(f"No weights found at {file_path}.h5")
            
    def get_action_probs(self, state):
        """Get action probabilities for the current state.
        
        Args:
            state: Current state vector
            
        Returns:
            Array of action probabilities
        """
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.policy_network(state_tensor)[0].numpy()


if __name__ == "__main__":
    # Test the policy gradient agent
    agent = PolicyGradientAgent(state_size=8, action_size=5)
    print(f"Policy gradient agent created")
    
    # Example of getting action probabilities
    test_state = np.random.random(8)
    action_probs = agent.get_action_probs(test_state)
    print(f"Action probabilities for test state: {action_probs}") 