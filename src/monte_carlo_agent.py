"""
Monte Carlo Agent for Blackjack.

This module implements a Monte Carlo learning agent for blackjack,
which learns optimal policy through exploring action-state pairs and
updating their values based on experienced returns.
"""

import numpy as np
import random
import os
import sys
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING
from blackjack_env import Action


class MonteCarloAgent:
    """Monte Carlo learning agent for blackjack."""
    
    def __init__(self, config=None):
        """Initialize the Monte Carlo agent.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or TRAINING['monte_carlo']
        self.learning_rate = self.config['learning_rate']
        
        # Initialize Q-values: Q(state, action) -> expected reward
        # Use a nested defaultdict for easier access
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Keep track of returns for each state-action pair
        self.returns = defaultdict(lambda: defaultdict(list))
        
        # Keep track of state-action pairs visited during an episode
        self.episode_states_actions = []
        
        # Keep track of the policy: state -> action
        self.policy = {}
        
        # Exploration probability
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.05
        
        # Learning progress
        self.episodes_trained = 0
        
    def state_to_key(self, state):
        """Convert state dict to hashable key for Q-table.
        
        Args:
            state: State dictionary from the environment
            
        Returns:
            Tuple representing the state
        """
        player_hand = state['player_hand']
        dealer_upcard = state['dealer_upcard']
        
        # Key components:
        # 1. Player hand value
        # 2. Whether player has a usable ace (soft hand)
        # 3. Dealer's upcard value
        # 4. Whether splitting is possible
        # 5. True count (for card counting, rounded to nearest integer)
        
        player_value = player_hand.value
        player_has_usable_ace = 1 if player_hand.is_soft else 0
        dealer_value = dealer_upcard.value
        can_split = 1 if state['can_split'] else 0
        true_count = round(state['true_count'])
        
        return (player_value, player_has_usable_ace, dealer_value, can_split, true_count)
        
    def choose_action(self, state, valid_actions):
        """Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        state_key = self.state_to_key(state)
        
        # Explore: random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        # Exploit: best known action
        if state_key in self.policy:
            best_action = self.policy[state_key]
            if best_action in valid_actions:
                return best_action
                
        # If no policy exists for this state or the best action is not valid,
        # choose the valid action with highest Q-value
        q_values = {action: self.Q[state_key][action.name] for action in valid_actions}
        
        # If all Q-values are the same, choose randomly
        if len(set(q_values.values())) == 1:
            return random.choice(valid_actions)
            
        return max(q_values, key=q_values.get)
        
    def start_episode(self):
        """Initialize a new episode."""
        self.episode_states_actions = []
        
    def record_step(self, state, action):
        """Record a state-action pair for the current episode.
        
        Args:
            state: Current state
            action: Action taken
        """
        state_key = self.state_to_key(state)
        self.episode_states_actions.append((state_key, action))
        
    def update_policy(self, rewards):
        """Update Q-values and policy based on the episode.
        
        Args:
            rewards: List of rewards received during the episode
        """
        # Calculate return (sum of rewards for the episode)
        G = sum(rewards)
        
        # Update Q-values for all state-action pairs visited in this episode
        for state_key, action in self.episode_states_actions:
            # Add return to returns for this state-action pair
            self.returns[state_key][action.name].append(G)
            
            # Update Q-value using incremental mean
            self.Q[state_key][action.name] = sum(self.returns[state_key][action.name]) / len(self.returns[state_key][action.name])
            
            # Update policy for this state to the action with highest Q-value
            self.policy[state_key] = max(
                Action, 
                key=lambda a: self.Q[state_key][a.name]
            )
            
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.episodes_trained += 1
        
    def get_q_values(self, state):
        """Get Q-values for all actions in current state.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary mapping action names to Q-values
        """
        state_key = self.state_to_key(state)
        return {action.name: self.Q[state_key][action.name] for action in Action}
        
    def get_policy_action(self, state, valid_actions):
        """Get the best action according to the learned policy.
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            Best action
        """
        state_key = self.state_to_key(state)
        
        if state_key in self.policy and self.policy[state_key] in valid_actions:
            return self.policy[state_key]
            
        # If no policy exists for this state or the best action is not valid,
        # choose the valid action with highest Q-value
        q_values = {action: self.Q[state_key][action.name] for action in valid_actions}
        return max(q_values, key=q_values.get)
        
    def save(self, file_path):
        """Save the learned policy and Q-values to file.
        
        Args:
            file_path: Path to save the agent
        """
        data = {
            'Q': dict(self.Q),
            'policy': self.policy,
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Monte Carlo agent saved to {file_path}")
        
    def load(self, file_path):
        """Load the policy and Q-values from file.
        
        Args:
            file_path: Path to load the agent from
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Convert the defaultdict back from regular dict
            self.Q = defaultdict(lambda: defaultdict(float))
            for state_key, action_values in data['Q'].items():
                for action_name, value in action_values.items():
                    self.Q[state_key][action_name] = value
                    
            self.policy = data['policy']
            self.epsilon = data.get('epsilon', self.epsilon_min)
            self.episodes_trained = data.get('episodes_trained', 0)
            
            print(f"Monte Carlo agent loaded from {file_path}")
            
    def plot_policy(self, output_path=None):
        """Visualize the learned policy for different player and dealer values.
        
        Args:
            output_path: Path to save the visualization
        """
        # Create a grid for player values (1-21) and dealer values (1-10)
        player_values = np.arange(4, 22)
        dealer_values = np.arange(2, 12)
        
        # Create separate plots for hard and soft hands
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Action colors
        action_colors = {
            'HIT': 'red',
            'STAND': 'green',
            'DOUBLE': 'blue',
            'SPLIT': 'purple',
            'SURRENDER': 'black'
        }
        
        for usable_ace, ax in enumerate(axes):
            policy_grid = np.zeros((len(player_values), len(dealer_values)))
            
            for i, player_value in enumerate(player_values):
                for j, dealer_value in enumerate(dealer_values):
                    # Check policy for this state (ignore card counting and splitting for visualization)
                    state_key = (player_value, usable_ace, dealer_value, 0, 0)
                    
                    if state_key in self.policy:
                        action = self.policy[state_key]
                        # Convert action to numeric value for visualization
                        action_value = {'HIT': 0, 'STAND': 1, 'DOUBLE': 2, 'SPLIT': 3, 'SURRENDER': 4}[action.name]
                        policy_grid[i, j] = action_value
                        
            # Create heatmap
            im = ax.imshow(policy_grid, cmap='viridis')
            
            # Set labels
            ax.set_xticks(np.arange(len(dealer_values)))
            ax.set_yticks(np.arange(len(player_values)))
            ax.set_xticklabels(dealer_values)
            ax.set_yticklabels(player_values)
            
            ax.set_xlabel("Dealer's Card")
            ax.set_ylabel("Player's Sum")
            ax.set_title(f"{'Soft' if usable_ace else 'Hard'} Hand")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(['HIT', 'STAND', 'DOUBLE', 'SPLIT', 'SURRENDER'])
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Policy visualization saved to {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Test the Monte Carlo agent
    agent = MonteCarloAgent()
    print(f"Monte Carlo agent created")
    
    # Example of state and Q-values
    test_state = {
        'player_hand': None,  # This would be a Hand object in real usage
        'dealer_upcard': None,  # This would be a Card object in real usage
        'can_split': False,
        'true_count': 0
    }
    
    # In real usage, we would call:
    # q_values = agent.get_q_values(test_state)
    # print(f"Q-values for test state: {q_values}") 