"""
Configuration file for Blackjack AI project hyperparameters.
"""

# Game Rules
BLACKJACK_RULES = {
    'num_decks': 6,  # Number of decks in the shoe
    'dealer_hit_soft_17': True,  # True if dealer hits on soft 17
    'double_after_split': True,  # Allow doubling after splitting
    'allow_surrender': True,  # Allow surrender
    'blackjack_payout': 1.5,  # Blackjack pays 3:2
    'max_split_hands': 4,  # Maximum number of hands after splitting
    'resplit_aces': False,  # Allow re-splitting aces
    'hit_split_aces': False,  # Allow hitting split aces
}

# Betting Parameters
BETTING = {
    'min_bet': 1,  # Minimum bet in units
    'max_bet': 500,  # Maximum bet in units
    'initial_bankroll': 1000,  # Starting bankroll
}

# Training Parameters
TRAINING = {
    # Deep Q-Network
    'dqn': {
        'learning_rate': 0.001,
        'gamma': 0.99,  # Discount factor
        'epsilon_start': 1.0,  # Exploration rate start
        'epsilon_end': 0.01,  # Exploration rate end
        'epsilon_decay': 0.995,  # Exploration rate decay
        'batch_size': 64,
        'memory_size': 10000,  # Replay buffer size
        'target_update': 500,  # Target network update frequency
        'hidden_layers': [128, 128],  # Hidden layer sizes
    },
    
    # Monte Carlo
    'monte_carlo': {
        'num_episodes': 1000000,  # Number of episodes to run
        'learning_rate': 0.01,
    },
    
    # Policy Gradient
    'policy_gradient': {
        'learning_rate': 0.001,
        'gamma': 0.99,  # Discount factor
        'hidden_layers': [128, 128],  # Hidden layer sizes
        'entropy_beta': 0.01,  # Entropy coefficient for exploration
    },
}

# Simulation Parameters
SIMULATION = {
    'num_hands': 1000000,  # Number of hands to simulate for evaluation
    'evaluation_interval': 10000,  # Evaluate model every X hands
    'checkpoint_interval': 50000,  # Save model every X hands
}

# Card Counting Parameters
CARD_COUNTING = {
    'system': 'hi_lo',  # Card counting system (hi_lo, ko, hi_opt_i, etc.)
    'count_factor': 0.5,  # Betting factor based on count (increase bets by X% per true count)
}

# Interactive Play
INTERACTIVE = {
    'default_bet': 10,  # Default bet for interactive play
    'show_ai_reasoning': True,  # Show AI's decision process
    'show_count': True,  # Show card counting information
} 