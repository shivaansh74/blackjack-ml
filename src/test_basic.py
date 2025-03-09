"""
Basic test script for the blackjack environment.

This script performs basic tests on the blackjack environment
without requiring TensorFlow or other dependencies.
"""

import os
import sys
import random

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blackjack_env import BlackjackEnv, Action


def test_blackjack_environment():
    """Test the blackjack environment."""
    print("\n=== Testing Blackjack Environment ===")
    
    # Create environment
    env = BlackjackEnv()
    print("Successfully created BlackjackEnv instance")
    
    # Reset environment
    state = env.reset()
    print(f"Initial state created with:")
    print(f"  Player hand: {state['player_hand']}")
    print(f"  Dealer showing: {state['dealer_upcard']}")
    
    valid_actions = env.get_valid_actions()
    print(f"  Valid actions: {valid_actions}")
    
    # Set bet
    env.set_bet(10)
    print(f"Bet set to 10")
    
    # Play a simple game with STAND action
    print("\nPlaying a simple game with STAND action:")
    next_state, reward, done = env.step(Action.STAND)
    
    print(f"Game complete!")
    print(f"  Final dealer hand: {env.dealer_hand}")
    print(f"  Final reward: {reward}")
    
    # Play a game with HIT action
    print("\nPlaying a game with HIT action:")
    state = env.reset()
    env.set_bet(10)
    print(f"New initial state:")
    print(f"  Player hand: {state['player_hand']}")
    print(f"  Dealer showing: {state['dealer_upcard']}")
    
    next_state, reward, done = env.step(Action.HIT)
    print(f"  After HIT: {next_state['player_hand']}")
    
    if not done:
        # Stand after hitting
        next_state, reward, done = env.step(Action.STAND)
        print(f"  Standing after hit")
        print(f"  Final dealer hand: {env.dealer_hand}")
    
    print(f"  Final reward: {reward}")
    
    # Test DOUBLE action
    print("\nTesting DOUBLE action:")
    state = env.reset()
    env.set_bet(10)
    print(f"New initial state:")
    print(f"  Player hand: {state['player_hand']}")
    print(f"  Dealer showing: {state['dealer_upcard']}")
    print(f"  Initial bet: {env.player_hands[0].bet}")
    
    if Action.DOUBLE in env.get_valid_actions():
        next_state, reward, done = env.step(Action.DOUBLE)
        print(f"  Doubled bet to: {env.player_hands[0].bet}")
        print(f"  Final player hand: {env.player_hands[0]}")
        print(f"  Final dealer hand: {env.dealer_hand}")
        print(f"  Final reward: {reward}")
    else:
        print("  DOUBLE action not available for this hand.")
    
    # Test SURRENDER action
    print("\nTesting SURRENDER action:")
    state = env.reset()
    env.set_bet(10)
    print(f"New initial state:")
    print(f"  Player hand: {state['player_hand']}")
    print(f"  Dealer showing: {state['dealer_upcard']}")
    
    if Action.SURRENDER in env.get_valid_actions():
        next_state, reward, done = env.step(Action.SURRENDER)
        print(f"  Surrendered the hand")
        print(f"  Final reward: {reward}")
    else:
        print("  SURRENDER action not available for this hand.")
    
    print("\nBlackjack environment test completed successfully!\n")


def test_card_counting():
    """Test the card counting feature."""
    print("\n=== Testing Card Counting ===")
    
    env = BlackjackEnv()
    state = env.reset()
    
    print(f"Initial running count: {env.running_count}")
    print(f"Initial true count: {env.true_count}")
    
    # Play a few hands to see count change
    for i in range(3):
        print(f"\nHand {i+1}:")
        state = env.reset()
        print(f"  Player hand: {state['player_hand']}")
        print(f"  Dealer showing: {state['dealer_upcard']}")
        print(f"  Running count: {env.running_count}")
        print(f"  True count: {env.true_count}")
        
        # Play hand
        env.set_bet(10)
        _, _, _ = env.step(Action.STAND)
    
    print("\nCard counting test completed successfully!\n")


if __name__ == "__main__":
    test_blackjack_environment()
    test_card_counting() 