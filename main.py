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
from src.config import BETTING, SIMULATION, BLACKJACK_RULES

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
    
    # Choose one option:
    # Option 1: Train new models (takes time)
    dqn_agent, mc_agent, pg_agent = train_all_models(
        num_episodes=10000,  # Reduce from default for quicker execution
        output_dir=output_dir,
        checkpoint_interval=1000,
        verbose=True
    )
    
    # Option 2: Load pre-trained models (if available)
    """
    # Initialize agents
    env = BlackjackEnv()
    env.reset()  # Important: Reset before accessing vectorize_state
    state_size = len(env.vectorize_state())
    action_size = len(list(Action))
    
    dqn_agent = DQNAgent(state_size, action_size)
    mc_agent = MonteCarloAgent()
    pg_agent = PolicyGradientAgent(state_size, action_size)
    
    # Load pre-trained models
    dqn_agent.load(os.path.join(output_dir, "dqn_model_latest.h5"))
    mc_agent.load(os.path.join(output_dir, "monte_carlo_model_latest.pkl"))
    pg_agent.load(os.path.join(output_dir, "policy_gradient_model_latest.h5"))
    """
    
    # Step 2: Compare models with different betting strategies
    print("\n[2/3] Comparing models with different betting strategies...\n")
    
    # Initialize model comparison
    comparison = ModelComparison(output_dir=results_dir)
    
    # Load agents (if not using the trained ones from step 1)
    # comparison.load_agents(
    #     dqn_path=os.path.join(output_dir, "dqn_model_latest.h5"),
    #     mc_path=os.path.join(output_dir, "monte_carlo_model_latest.pkl"),
    #     pg_path=os.path.join(output_dir, "policy_gradient_model_latest.h5")
    # )
    
    # Use the agents we trained/loaded in step 1
    comparison.dqn_agent = dqn_agent
    comparison.mc_agent = mc_agent
    comparison.pg_agent = pg_agent
    
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