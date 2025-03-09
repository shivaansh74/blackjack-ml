"""
Deep Q-Network (DQN) Agent for Blackjack.

This module implements a reinforcement learning agent using a Deep Q-Network
to learn optimal blackjack strategy.
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import os
import sys
import pickle

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING
from blackjack_env import Action


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample random batch of experiences."""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent for blackjack."""
    
    def __init__(self, state_size, action_size, config=None):
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or TRAINING['dqn']
        
        # Learning parameters
        self.gamma = self.config['gamma']  # Discount factor
        self.epsilon = self.config['epsilon_start']  # Exploration rate
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        self.learning_rate = self.config['learning_rate']
        
        # Replay memory
        self.memory = ReplayBuffer(self.config['memory_size'])
        self.batch_size = self.config['batch_size']
        
        # Q-Networks (main and target)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Tracking variables
        self.target_update_counter = 0
        self.training_steps = 0
        
    def _build_model(self):
        """Build the neural network model for DQN."""
        model = Sequential()
        
        # Hidden layers
        hidden_layers = self.config['hidden_layers']
        model.add(Dense(hidden_layers[0], input_dim=self.state_size, activation='relu'))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            
        # Output layer (Q-values for each action)
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
        
    def update_target_model(self):
        """Update target network with weights from main network."""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        action_idx = action.value - 1  # Convert enum to index
        self.memory.add(state, action_idx, reward, next_state, done)
        
    def act(self, state, valid_actions):
        """Choose action based on epsilon-greedy policy.
        
        Args:
            state: Current state vector
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Convert valid_actions to indices
        valid_indices = [action.value - 1 for action in valid_actions]
        
        # Exploration: random action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
            
        # Exploitation: best action based on Q-values
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        
        # Mask invalid actions with very negative values
        masked_q_values = np.full(self.action_size, -np.inf)
        masked_q_values[valid_indices] = q_values[valid_indices]
        
        # Choose action with highest Q-value
        action_idx = np.argmax(masked_q_values)
        return Action(action_idx + 1)  # Convert index back to enum
        
    def replay(self):
        """Train the model with experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch of experiences
        batch = self.memory.sample(self.batch_size)
        
        # Extract data from batch
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Predict current Q-values and next Q-values
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        current_q = self.model.predict(states_tensor, verbose=0)
        next_q = self.target_model.predict(next_states_tensor, verbose=0)
        
        # Update Q-values for the actions taken
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i, actions[i]] = rewards[i]
            else:
                current_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
                
        # Train the model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target model periodically
        self.target_update_counter += 1
        if self.target_update_counter >= self.config['target_update']:
            self.update_target_model()
            self.target_update_counter = 0
            
        self.training_steps += 1
        
    def load(self, file_path):
        """Load model weights from file."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                saved_data = pickle.load(f)
                
            self.model.set_weights(saved_data['model_weights'])
            self.target_model.set_weights(saved_data['model_weights'])
            self.epsilon = saved_data.get('epsilon', self.epsilon_min)
            self.training_steps = saved_data.get('training_steps', 0)
            
            print(f"Model loaded from {file_path}")
            
    def save(self, file_path):
        """Save model weights to file."""
        data = {
            'model_weights': self.model.get_weights(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Model saved to {file_path}")
        
    def get_q_values(self, state):
        """Get Q-values for all actions in current state."""
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.model.predict(state_tensor, verbose=0)[0]


if __name__ == "__main__":
    # Test the DQN agent with a small state size and action size
    agent = DQNAgent(state_size=8, action_size=5)
    print(f"DQN agent created with {agent.model.count_params()} parameters")
    
    # Test prediction
    test_state = np.random.random(8)
    q_values = agent.get_q_values(test_state)
    print(f"Q-values for test state: {q_values}") 