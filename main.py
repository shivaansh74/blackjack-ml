"""
Blackjack AI - Main Module

This script combines all functionality from the blackjack-ml project.
It trains multiple AI agents, compares their performance with different
betting strategies, and displays visualizations of the results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add src directory to the path
sys.path.append('src')

# Import project modules
from src.blackjack_env import BlackjackEnv, Action
from src.q_learning_agent import DQNAgent
from src.monte_carlo_agent import MonteCarloAgent
from src.policy_network import PolicyGradientAgent
from src.card_counter import CardCounter
from src.bet_optimizer import BetOptimizer, BettingStrategy
from src.model_comparison import ModelComparison
from src.train_all_models import train_all_models
from config import BETTING, SIMULATION, BLACKJACK_RULES

def main():
    """Main function to run the entire Blackjack AI pipeline."""
    print("=" * 80)
    print("BLACKJACK AI PROJECT")
    print("=" * 80)
    
    # Create directories
    output_dir = 'models'
    results_dir = 'results'
    figures_dir = os.path.join(results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Train models or load pre-trained models
    print("\n[1/3] Training AI models...\n")
    
    # Initialize agents
    env = BlackjackEnv()
    env.reset()  # Important: Reset before accessing vectorize_state
    state_size = len(env.vectorize_state())
    action_size = len(list(Action))
    
    dqn_agent = DQNAgent(state_size, action_size)
    mc_agent = MonteCarloAgent()
    pg_agent = PolicyGradientAgent(state_size, action_size)
    
    # Try to load pre-trained models, train them if not available
    dqn_loaded = False
    try:
        dqn_agent.load(os.path.join(output_dir, "training_checkpoints", "dqn_checkpoint_ep10000_20250309_145852.pkl"))
        print(f"Model loaded from {os.path.join(output_dir, 'training_checkpoints', 'dqn_checkpoint_ep10000_20250309_145852.pkl')}")
        dqn_loaded = True
    except Exception as e:
        print(f"Could not load DQN model: {e}")
        print("Will train DQN model from scratch.")
        
    mc_loaded = False
    try:
        mc_agent.load(os.path.join(output_dir, "training_checkpoints", "monte_carlo_checkpoint_ep10000_20250309_145852.pkl"))
        print(f"Monte Carlo agent loaded from {os.path.join(output_dir, 'training_checkpoints', 'monte_carlo_checkpoint_ep10000_20250309_145852.pkl')}")
        mc_loaded = True
    except Exception as e:
        print(f"Could not load Monte Carlo agent: {e}")
        print("Will train Monte Carlo agent from scratch.")
        
    pg_loaded = False
    try:
        # Try different formats for policy gradient model
        pg_file_options = [
            "policy_gradient_checkpoint_ep10000_20250309_145852.pkl",
            "policy_gradient_checkpoint_ep10000_20250309_145852",
            "policy_gradient_checkpoint_ep10000_20250309_145852.weights.h5.pkl",
            "policy_gradient_checkpoint_ep10000_20250309_145852.weights.h5",
            "policy_gradient_checkpoint_ep10000_20250309_145852.h5"
        ]
        
        # Check if any policy gradient files exist first
        pg_files_exist = False
        for file_option in pg_file_options:
            file_path = os.path.join(output_dir, "training_checkpoints", file_option)
            if os.path.exists(file_path):
                pg_files_exist = True
                break
        
        if not pg_files_exist:
            print("No policy gradient checkpoint files found in the training_checkpoints directory.")
            print("Will train the policy gradient agent from scratch.")
            pg_loaded = False  # Ensure pg_loaded is False to trigger training
        else:
            # Try to load the files if they exist
            for file_option in pg_file_options:
                try:
                    pg_agent.load(os.path.join(output_dir, "training_checkpoints", file_option))
                    print(f"Policy gradient agent loaded from {os.path.join(output_dir, 'training_checkpoints', file_option)}")
                    pg_loaded = True
                    break
                except Exception as file_error:
                    print(f"Failed to load {file_option}: {file_error}")
                    continue
                    
            if not pg_loaded:
                print("Could not load policy gradient model in any of the expected formats.")
                print("Will train the policy gradient agent from scratch.")
    except Exception as e:
        print(f"Unexpected error checking for policy gradient agent: {e}")
        print("Will train policy gradient agent from scratch.")
    
    # Train models that couldn't be loaded
    if not (dqn_loaded and mc_loaded and pg_loaded):
        print("\nTraining models that couldn't be loaded...\n")
        training_episodes = 5000  # Reduced from 10000 for quicker execution
        
        # Only train the specific models that aren't loaded
        if not dqn_loaded or not mc_loaded or not pg_loaded:
            # Create a new environment for training if needed
            train_env = BlackjackEnv()
            training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # If all models need training, use train_all_models
            if not dqn_loaded and not mc_loaded and not pg_loaded:
                print(f"Training all models for {training_episodes} episodes...")
                dqn_agent, mc_agent, pg_agent = train_all_models(
                    num_episodes=training_episodes,
                    output_dir=output_dir,
                    checkpoint_interval=1000,
                    verbose=True
                )
                dqn_loaded = mc_loaded = pg_loaded = True
            else:
                # Train individual models that couldn't be loaded
                from src.train_all_models import train_dqn_agent, train_monte_carlo_agent, train_policy_gradient_agent
                
                if not dqn_loaded:
                    print(f"Training only DQN agent for {training_episodes} episodes...")
                    dqn_agent = DQNAgent(state_size, action_size)
                    train_dqn_agent(dqn_agent, train_env, training_episodes, 1000, output_dir, training_timestamp, True)
                    dqn_loaded = True
                    
                if not mc_loaded:
                    print(f"Training only Monte Carlo agent for {training_episodes} episodes...")
                    mc_agent = MonteCarloAgent()
                    train_monte_carlo_agent(mc_agent, train_env, training_episodes, 1000, output_dir, training_timestamp, True)
                    mc_loaded = True
                    
                if not pg_loaded:
                    print(f"Training only Policy Gradient agent for {training_episodes} episodes...")
                    try:
                        pg_agent = PolicyGradientAgent(state_size, action_size)
                        train_policy_gradient_agent(pg_agent, train_env, training_episodes, 1000, output_dir, training_timestamp, True)
                        pg_loaded = True
                        print("Policy Gradient agent training completed successfully.")
                    except Exception as train_error:
                        print(f"Error during Policy Gradient agent training: {train_error}")
                        print("Using a baseline policy gradient agent instead.")
                        # Create a fallback agent with default policy
                        pg_agent = PolicyGradientAgent(state_size, action_size)
                        pg_loaded = True
    
    # Step 2: Compare models with different betting strategies
    print("\n[2/3] Comparing models with different betting strategies...\n")
    
    # Initialize model comparison
    comparison = ModelComparison(output_dir=results_dir)
    
    # Use the agents we trained/loaded
    comparison.dqn_agent = dqn_agent if dqn_loaded else None
    comparison.monte_carlo_agent = mc_agent if mc_loaded else None
    comparison.policy_gradient_agent = pg_agent if pg_loaded else None
    
    if not dqn_loaded:
        print("Warning: DQN agent not available. Skipping DQN agent comparisons.")
    if not mc_loaded: 
        print("Warning: Monte Carlo agent not available. Skipping MC agent comparisons.")
    if not pg_loaded:
        print("Warning: Policy gradient agent not available. Skipping PG agent comparisons.")
    
    # Run simulation to compare agents and betting strategies
    results_df = comparison.run_comparison(
        num_hands=5000,  # Reduce for quicker execution
        initial_bankroll=BETTING['initial_bankroll'],
        verbose=True
    )
    
    # Step 3: Display and save results
    print("\n[3/3] Displaying results and generating visualizations...\n")
    
    # Print detailed results
    comparison.print_results()
    
    # Save results to CSV
    results_path = os.path.join(results_dir, f"comparison_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    print("\nVisualization files saved to the 'results/visualizations' directory")
    
    # Show all plots (if running in interactive mode)
    plt.show()
    
    print("\nBlackjack AI analysis complete!")

if __name__ == "__main__":
    main() 