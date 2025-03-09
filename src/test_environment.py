"""
Test script for blackjack environment and models.

This script performs basic tests on the blackjack environment and models
to ensure they're working correctly.
"""

import os
import sys
import time

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_env import BlackjackEnv, Action
from q_learning_agent import DQNAgent
from monte_carlo_agent import MonteCarloAgent
from policy_network import PolicyGradientAgent
from card_counter import CardCounter
from bet_optimizer import BetOptimizer, BettingStrategy


def test_blackjack_environment():
    """Test the blackjack environment."""
    print("\n=== Testing Blackjack Environment ===")
    
    # Create environment
    env = BlackjackEnv()
    
    # Reset environment
    state = env.reset()
    print(f"Initial state created with:")
    print(f"  Player hand: {state['player_hand']}")
    print(f"  Dealer showing: {state['dealer_upcard']}")
    print(f"  Valid actions: {env.get_valid_actions()}")
    
    # Set bet
    env.set_bet(10)
    print(f"Bet set to 10")
    
    # Take a HIT action
    if Action.HIT in env.get_valid_actions():
        print(f"Taking HIT action...")
        next_state, reward, done = env.step(Action.HIT)
        print(f"  New player hand: {next_state['player_hand']}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
    
    # Take a STAND action on a new game
    env.reset()
    env.set_bet(10)
    if Action.STAND in env.get_valid_actions():
        print(f"Taking STAND action...")
        next_state, reward, done = env.step(Action.STAND)
        print(f"  Final dealer hand: {env.dealer_hand}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
    
    print("Blackjack environment test completed.\n")


def test_dqn_agent():
    """Test the DQN agent."""
    print("\n=== Testing DQN Agent ===")
    
    # Create environment
    env = BlackjackEnv()
    
    # Get state size and action size
    state = env.reset()
    state_vector = env.vectorize_state()
    state_size = len(state_vector)
    action_size = len(list(Action))
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    print(f"Created DQN agent with state size {state_size} and action size {action_size}")
    
    # Test action selection
    valid_actions = env.get_valid_actions()
    action = agent.act(state_vector, valid_actions)
    print(f"Agent chose action: {action}")
    
    # Test Q-values
    q_values = agent.get_q_values(state_vector)
    print(f"Q-values: {q_values}")
    
    print("DQN agent test completed.\n")


def test_monte_carlo_agent():
    """Test the Monte Carlo agent."""
    print("\n=== Testing Monte Carlo Agent ===")
    
    # Create environment
    env = BlackjackEnv()
    
    # Create agent
    agent = MonteCarloAgent()
    print(f"Created Monte Carlo agent")
    
    # Test action selection
    state = env.reset()
    valid_actions = env.get_valid_actions()
    action = agent.choose_action(state, valid_actions)
    print(f"Agent chose action: {action}")
    
    # Test state representation
    state_key = agent.state_to_key(state)
    print(f"State key: {state_key}")
    
    print("Monte Carlo agent test completed.\n")


def test_policy_gradient_agent():
    """Test the Policy Gradient agent."""
    print("\n=== Testing Policy Gradient Agent ===")
    
    # Create environment
    env = BlackjackEnv()
    
    # Get state size and action size
    state = env.reset()
    state_vector = env.vectorize_state()
    state_size = len(state_vector)
    action_size = len(list(Action))
    
    # Create agent
    agent = PolicyGradientAgent(state_size, action_size)
    print(f"Created Policy Gradient agent with state size {state_size} and action size {action_size}")
    
    # Test action selection
    valid_actions = env.get_valid_actions()
    action = agent.act(state_vector, valid_actions)
    print(f"Agent chose action: {action}")
    
    # Test action probabilities
    action_probs = agent.get_action_probs(state_vector)
    print(f"Action probabilities: {action_probs}")
    
    print("Policy Gradient agent test completed.\n")
    

def test_card_counter():
    """Test the card counter."""
    print("\n=== Testing Card Counter ===")
    
    # Create card counter
    counter = CardCounter()
    print(f"Created card counter with {counter.system} system")
    
    # Create environment and get some cards
    env = BlackjackEnv()
    state = env.reset()
    
    # Get dealt cards
    player_cards = state['player_hand'].cards
    dealer_upcard = state['dealer_upcard']
    
    # Update count with these cards
    print(f"Updating count with player cards: {player_cards}")
    for card in player_cards:
        count = counter.update(card)
        print(f"  After {card}: Running count = {counter.running_count}, True count = {counter.true_count:.2f}")
    
    print(f"Updating count with dealer upcard: {dealer_upcard}")
    count = counter.update(dealer_upcard)
    print(f"  After {dealer_upcard}: Running count = {counter.running_count}, True count = {counter.true_count:.2f}")
    
    # Get betting recommendation
    bet_multiplier = counter.get_bet_multiplier()
    print(f"Recommended bet multiplier: {bet_multiplier:.2f}x")
    
    print("Card counter test completed.\n")


def test_bet_optimizer():
    """Test the bet optimizer."""
    print("\n=== Testing Bet Optimizer ===")
    
    # Create bet optimizer
    optimizer = BetOptimizer(initial_bankroll=1000, strategy=BettingStrategy.KELLY)
    print(f"Created bet optimizer with {optimizer.get_strategy_name()} strategy")
    
    # Test with different true counts
    for true_count in [-3, 0, 3]:
        optimizer.card_counter.true_count = true_count
        bet = optimizer.get_optimal_bet()
        print(f"True count {true_count}: Optimal bet = ${bet}")
    
    # Test updating bankroll
    result = 50  # Win $50
    new_bankroll = optimizer.update_bankroll(result)
    print(f"After winning ${result}: New bankroll = ${new_bankroll}")
    
    # Try a different strategy
    optimizer.set_strategy(BettingStrategy.PROPORTIONAL)
    print(f"Switched to {optimizer.get_strategy_name()} strategy")
    
    # Test again with different true counts
    for true_count in [-3, 0, 3]:
        optimizer.card_counter.true_count = true_count
        bet = optimizer.get_optimal_bet()
        print(f"True count {true_count}: Optimal bet = ${bet}")
    
    print("Bet optimizer test completed.\n")


def run_all_tests():
    """Run all tests."""
    start_time = time.time()
    
    test_blackjack_environment()
    test_dqn_agent()
    test_monte_carlo_agent()
    test_policy_gradient_agent()
    test_card_counter()
    test_bet_optimizer()
    
    elapsed_time = time.time() - start_time
    print(f"All tests completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    run_all_tests() 