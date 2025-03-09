"""
Model Comparison for Blackjack AI.

This module compares different reinforcement learning models and betting strategies
to identify the best performing combinations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_env import BlackjackEnv, Action
from q_learning_agent import DQNAgent
from monte_carlo_agent import MonteCarloAgent
from policy_network import PolicyGradientAgent
from card_counter import CardCounter
from bet_optimizer import BetOptimizer, BettingStrategy
from config import BETTING, SIMULATION


class ModelComparison:
    """Comparison of different AI models and betting strategies."""
    
    def __init__(self, output_dir=None):
        """Initialize the model comparison.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir or os.path.join('results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dqn_agent = None
        self.monte_carlo_agent = None
        self.policy_gradient_agent = None
        
        self.env = BlackjackEnv()
        self.state_size = len(self.env.vectorize_state())
        self.action_size = len(list(Action))
        
        self.results = []
        
    def load_agents(self, dqn_path=None, mc_path=None, pg_path=None):
        """Load pre-trained agents.
        
        Args:
            dqn_path: Path to DQN agent
            mc_path: Path to Monte Carlo agent
            pg_path: Path to Policy Gradient agent
        """
        # Load DQN agent
        self.dqn_agent = DQNAgent(self.state_size, self.action_size)
        if dqn_path and os.path.exists(dqn_path):
            self.dqn_agent.load(dqn_path)
            print(f"Loaded DQN agent from {dqn_path}")
        else:
            print("Using untrained DQN agent")
            
        # Load Monte Carlo agent
        self.monte_carlo_agent = MonteCarloAgent()
        if mc_path and os.path.exists(mc_path):
            self.monte_carlo_agent.load(mc_path)
            print(f"Loaded Monte Carlo agent from {mc_path}")
        else:
            print("Using untrained Monte Carlo agent")
            
        # Load Policy Gradient agent
        self.policy_gradient_agent = PolicyGradientAgent(self.state_size, self.action_size)
        if pg_path and os.path.exists(pg_path):
            self.policy_gradient_agent.load(pg_path)
            print(f"Loaded Policy Gradient agent from {pg_path}")
        else:
            print("Using untrained Policy Gradient agent")
            
    def run_comparison(self, num_hands=10000, initial_bankroll=1000, verbose=True):
        """Run comparison between models and betting strategies.
        
        Args:
            num_hands: Number of hands to simulate
            initial_bankroll: Starting bankroll
            verbose: Whether to print progress
            
        Returns:
            DataFrame with comparison results
        """
        # Agent types
        agents = {
            'DQN': self.dqn_agent,
            'Monte Carlo': self.monte_carlo_agent,
            'Policy Gradient': self.policy_gradient_agent,
            'Basic Strategy': None  # Will use hardcoded basic strategy
        }
        
        # Betting strategies
        betting_strategies = [
            BettingStrategy.FLAT,
            BettingStrategy.PROPORTIONAL,
            BettingStrategy.KELLY,
            BettingStrategy.OSCAR,
            BettingStrategy.MARTINGALE,
            BettingStrategy.ANTI_MARTINGALE,
            BettingStrategy.FIBONACCI
        ]
        
        results = []
        
        # Loop through each combination of agent and betting strategy
        for agent_name, agent in agents.items():
            for betting_strategy in betting_strategies:
                # Skip invalid combinations
                if agent_name == 'Basic Strategy' and betting_strategy not in [BettingStrategy.FLAT, BettingStrategy.PROPORTIONAL, BettingStrategy.KELLY]:
                    continue
                    
                # Create bet optimizer
                bet_optimizer = BetOptimizer(
                    initial_bankroll=initial_bankroll,
                    strategy=betting_strategy
                )
                
                # Run simulation
                result = self._run_simulation(
                    agent=agent,
                    agent_name=agent_name,
                    bet_optimizer=bet_optimizer,
                    num_hands=num_hands,
                    verbose=verbose
                )
                
                # Add to results
                result['agent'] = agent_name
                result['betting_strategy'] = bet_optimizer.get_strategy_name()
                results.append(result)
                
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by ROI
        results_df = results_df.sort_values('roi', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(os.path.join(self.output_dir, f"model_comparison_{timestamp}.csv"), index=False)
        
        # Plot results
        self._plot_results(results_df, timestamp)
        
        self.results = results_df
        return results_df
        
    def _run_simulation(self, agent, agent_name, bet_optimizer, num_hands, verbose=True):
        """Run simulation for a specific agent and betting strategy.
        
        Args:
            agent: AI agent
            agent_name: Name of the agent
            bet_optimizer: Bet optimizer
            num_hands: Number of hands to simulate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with simulation results
        """
        # Reset environment and bet optimizer
        env = BlackjackEnv()
        bet_optimizer.reset_session()
        
        # Initialize tracking variables
        rewards = []
        bankroll_history = [bet_optimizer.current_bankroll]
        win_count = 0
        loss_count = 0
        push_count = 0
        blackjack_count = 0
        
        # Progress bar
        progress_bar = tqdm(range(num_hands)) if verbose else range(num_hands)
        
        for _ in progress_bar:
            # Reset environment for new hand
            state = env.reset()
            
            # Get bet amount from optimizer
            bet_amount = bet_optimizer.get_optimal_bet()
            env.set_bet(bet_amount)
            
            # Handle player's turn
            done = False
            while not done:
                # Get valid actions
                valid_actions = env.get_valid_actions()
                
                # Choose action based on agent type
                if agent_name == 'Basic Strategy':
                    # Use basic strategy
                    action = self._basic_strategy_action(state, valid_actions)
                else:
                    # Use agent
                    state_vector = env.vectorize_state()
                    
                    if agent_name == 'DQN':
                        action = agent.act(state_vector, valid_actions)
                    elif agent_name == 'Monte Carlo':
                        action = agent.choose_action(state, valid_actions)
                    elif agent_name == 'Policy Gradient':
                        action = agent.act(state_vector, valid_actions)
                        
                # Take action
                state, reward, done = env.step(action)
                
            # Update bet optimizer
            bet_optimizer.update_bankroll(reward)
            
            # Update tracking variables
            rewards.append(reward)
            bankroll_history.append(bet_optimizer.current_bankroll)
            
            if reward > 0:
                win_count += 1
            elif reward < 0:
                loss_count += 1
            else:
                push_count += 1
                
            # Check for blackjack
            if env.player_hands[0].is_blackjack:
                blackjack_count += 1
                
            # Update progress bar description
            if verbose:
                avg_reward = sum(rewards) / len(rewards)
                win_rate = win_count / (win_count + loss_count + push_count)
                progress_bar.set_description(
                    f"{agent_name} + {bet_optimizer.get_strategy_name()}: "
                    f"Avg: ${avg_reward:.2f}, Win: {win_rate:.2f}, "
                    f"Bank: ${bet_optimizer.current_bankroll:.2f}"
                )
                
            # Stop if bankroll is depleted
            if bet_optimizer.current_bankroll < bet_optimizer.min_bet:
                if verbose:
                    print(f"\nBankroll depleted after {_ + 1} hands!")
                break
                
        # Calculate statistics
        final_bankroll = bet_optimizer.current_bankroll
        net_profit = final_bankroll - initial_bankroll
        roi = net_profit / initial_bankroll
        avg_reward = sum(rewards) / len(rewards)
        win_rate = win_count / (win_count + loss_count + push_count)
        
        return {
            'final_bankroll': final_bankroll,
            'net_profit': net_profit,
            'roi': roi,
            'avg_reward': avg_reward,
            'win_rate': win_rate,
            'hands_played': len(rewards),
            'blackjack_rate': blackjack_count / len(rewards),
            'bankroll_history': bankroll_history
        }
        
    def _basic_strategy_action(self, state, valid_actions):
        """Apply basic strategy for blackjack.
        
        Args:
            state: Current state
            valid_actions: Valid actions
            
        Returns:
            Action according to basic strategy
        """
        player_hand = state['player_hand']
        dealer_upcard = state['dealer_upcard']
        
        player_value = player_hand.value
        dealer_value = dealer_upcard.value
        is_soft = player_hand.is_soft
        
        # Check for surrender
        if Action.SURRENDER in valid_actions:
            # Hard 16 vs 9, 10, A
            if player_value == 16 and not is_soft and dealer_value in [9, 10, 11]:
                return Action.SURRENDER
            # Hard 15 vs 10
            elif player_value == 15 and not is_soft and dealer_value == 10:
                return Action.SURRENDER
                
        # Check for split
        if Action.SPLIT in valid_actions:
            # Always split Aces and 8s
            if player_hand.cards[0].rank == 1 or player_hand.cards[0].rank == 8:
                return Action.SPLIT
            # Never split 10s, 5s, or 4s
            elif player_hand.cards[0].rank in [10, 5, 4]:
                pass
            # Split 9s vs 2-6, 8-9
            elif player_hand.cards[0].rank == 9 and dealer_value in [2, 3, 4, 5, 6, 8, 9]:
                return Action.SPLIT
            # Split 7s vs 2-7
            elif player_hand.cards[0].rank == 7 and 2 <= dealer_value <= 7:
                return Action.SPLIT
            # Split 6s vs 2-6
            elif player_hand.cards[0].rank == 6 and 2 <= dealer_value <= 6:
                return Action.SPLIT
            # Split 3s or 2s vs 2-7
            elif player_hand.cards[0].rank in [2, 3] and 2 <= dealer_value <= 7:
                return Action.SPLIT
                
        # Check for double down
        if Action.DOUBLE in valid_actions:
            # Hard 11 always double
            if player_value == 11 and not is_soft:
                return Action.DOUBLE
            # Hard 10 vs 2-9
            elif player_value == 10 and not is_soft and dealer_value <= 9:
                return Action.DOUBLE
            # Hard 9 vs 3-6
            elif player_value == 9 and not is_soft and 3 <= dealer_value <= 6:
                return Action.DOUBLE
            # Soft 19 vs 6
            elif player_value == 19 and is_soft and dealer_value == 6:
                return Action.DOUBLE
            # Soft 18 vs 2-6
            elif player_value == 18 and is_soft and 2 <= dealer_value <= 6:
                return Action.DOUBLE
            # Soft 17 vs 3-6
            elif player_value == 17 and is_soft and 3 <= dealer_value <= 6:
                return Action.DOUBLE
            # Soft 16 vs 4-6
            elif player_value == 16 and is_soft and 4 <= dealer_value <= 6:
                return Action.DOUBLE
            # Soft 15 vs 4-6
            elif player_value == 15 and is_soft and 4 <= dealer_value <= 6:
                return Action.DOUBLE
            # Soft 14 vs 5-6
            elif player_value == 14 and is_soft and 5 <= dealer_value <= 6:
                return Action.DOUBLE
                
        # Soft totals
        if is_soft:
            # Always stand on soft 19 or higher
            if player_value >= 19:
                return Action.STAND
            # Stand on soft 18 vs 2, 7-8
            elif player_value == 18 and (dealer_value == 2 or 7 <= dealer_value <= 8):
                return Action.STAND
            # Hit on soft 18 vs 9-A
            elif player_value == 18 and dealer_value >= 9:
                return Action.HIT
            # Always hit soft 17 or lower
            else:
                return Action.HIT
                
        # Hard totals
        # Stand on 17 or higher
        if player_value >= 17:
            return Action.STAND
        # Stand on 13-16 vs 2-6
        elif 13 <= player_value <= 16 and 2 <= dealer_value <= 6:
            return Action.STAND
        # Stand on 12 vs 4-6
        elif player_value == 12 and 4 <= dealer_value <= 6:
            return Action.STAND
        # Otherwise hit
        else:
            return Action.HIT
            
    def _plot_results(self, results_df, timestamp):
        """Plot comparison results.
        
        Args:
            results_df: DataFrame with results
            timestamp: Timestamp for filenames
        """
        # Create figures directory
        figures_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot ROI by agent and betting strategy
        plt.figure(figsize=(12, 8))
        
        # Group by agent and betting strategy
        grouped = results_df.groupby(['agent', 'betting_strategy'])['roi'].mean().unstack()
        
        # Plot heatmap
        ax = plt.gca()
        im = ax.imshow(grouped.values, cmap='RdYlGn')
        
        # Set labels
        ax.set_xticks(np.arange(len(grouped.columns)))
        ax.set_yticks(np.arange(len(grouped.index)))
        ax.set_xticklabels(grouped.columns)
        ax.set_yticklabels(grouped.index)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('ROI')
        
        # Add values to cells
        for i in range(len(grouped.index)):
            for j in range(len(grouped.columns)):
                if not np.isnan(grouped.values[i, j]):
                    ax.text(j, i, f"{grouped.values[i, j]:.2f}",
                           ha="center", va="center", color="black")
        
        plt.title('ROI by Agent and Betting Strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"roi_heatmap_{timestamp}.png"))
        
        # Plot win rate by agent
        plt.figure(figsize=(10, 6))
        win_rates = results_df.groupby('agent')['win_rate'].mean().sort_values(ascending=False)
        win_rates.plot(kind='bar', color='skyblue')
        plt.title('Average Win Rate by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Win Rate')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"win_rate_by_agent_{timestamp}.png"))
        
        # Plot average reward by agent and betting strategy
        plt.figure(figsize=(12, 8))
        pivot = results_df.pivot(index='agent', columns='betting_strategy', values='avg_reward')
        pivot.plot(kind='bar', figsize=(12, 8))
        plt.title('Average Reward by Agent and Betting Strategy')
        plt.xlabel('Agent')
        plt.ylabel('Average Reward per Hand')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Betting Strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"avg_reward_{timestamp}.png"))
        
        # Plot bankroll progression for top performers
        top_performers = results_df.sort_values('roi', ascending=False).head(3)
        
        plt.figure(figsize=(12, 8))
        for i, row in top_performers.iterrows():
            label = f"{row['agent']} + {row['betting_strategy']}"
            plt.plot(row['bankroll_history'], label=label)
            
        plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        plt.title('Bankroll Progression for Top Performers')
        plt.xlabel('Hand Number')
        plt.ylabel('Bankroll')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"bankroll_progression_{timestamp}.png"))
        
    def print_results(self):
        """Print comparison results in a readable format."""
        if self.results is None or len(self.results) == 0:
            print("No results available. Run comparison first.")
            return
            
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Print top 5 combinations
        print("\nTop 5 Strategies by ROI:")
        print("-"*80)
        print(f"{'Rank':<5}{'Agent':<20}{'Betting Strategy':<25}{'ROI':<10}{'Profit':<15}{'Win Rate':<10}")
        print("-"*80)
        
        top5 = self.results.head(5)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"{i:<5}{row['agent']:<20}{row['betting_strategy']:<25}"
                 f"{row['roi']:.2f}{'$' + str(int(row['net_profit'])):<15}{row['win_rate']:.2f}")
            
        # Print worst 3 combinations
        print("\nWorst 3 Strategies by ROI:")
        print("-"*80)
        print(f"{'Rank':<5}{'Agent':<20}{'Betting Strategy':<25}{'ROI':<10}{'Profit':<15}{'Win Rate':<10}")
        print("-"*80)
        
        bottom3 = self.results.tail(3)
        for i, (_, row) in enumerate(bottom3.iterrows(), len(self.results) - 2):
            print(f"{i:<5}{row['agent']:<20}{row['betting_strategy']:<25}"
                 f"{row['roi']:.2f}{'$' + str(int(row['net_profit'])):<15}{row['win_rate']:.2f}")
            
        # Print best agent overall
        best_agent = self.results.groupby('agent')['roi'].mean().sort_values(ascending=False).index[0]
        print(f"\nBest Agent Overall: {best_agent}")
        
        # Print best betting strategy overall
        best_strategy = self.results.groupby('betting_strategy')['roi'].mean().sort_values(ascending=False).index[0]
        print(f"Best Betting Strategy Overall: {best_strategy}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Compare blackjack AI models')
    parser.add_argument('--hands', type=int, default=10000, help='Number of hands to simulate')
    parser.add_argument('--bankroll', type=int, default=1000, help='Initial bankroll')
    parser.add_argument('--dqn-path', type=str, default=None, help='Path to DQN model')
    parser.add_argument('--mc-path', type=str, default=None, help='Path to Monte Carlo model')
    parser.add_argument('--pg-path', type=str, default=None, help='Path to Policy Gradient model')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-verbose', action='store_true', help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Create model comparison
    comparison = ModelComparison(output_dir=args.output_dir)
    
    # Load agents
    comparison.load_agents(
        dqn_path=args.dqn_path,
        mc_path=args.mc_path,
        pg_path=args.pg_path
    )
    
    # Start timer
    start_time = time.time()
    
    # Run comparison
    initial_bankroll = args.bankroll
    results = comparison.run_comparison(
        num_hands=args.hands,
        initial_bankroll=initial_bankroll,
        verbose=not args.no_verbose
    )
    
    # Print results
    comparison.print_results()
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total comparison time: {int(hours)}h {int(minutes)}m {int(seconds)}s") 