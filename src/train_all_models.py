"""
Train all reinforcement learning models for blackjack.

This script trains DQN, Monte Carlo, and Policy Gradient agents on the blackjack
environment and saves them for later comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_env import BlackjackEnv, Action
from q_learning_agent import DQNAgent
from monte_carlo_agent import MonteCarloAgent
from policy_network import PolicyGradientAgent
from card_counter import CardCounter
from config import TRAINING, SIMULATION, BETTING


def train_all_models(num_episodes=None, output_dir=None, checkpoint_interval=None, verbose=True):
    """Train all three reinforcement learning models for blackjack.
    
    Args:
        num_episodes: Number of episodes to train for
        output_dir: Directory to save models and metrics
        checkpoint_interval: Interval for saving checkpoints
        verbose: Whether to print progress
        
    Returns:
        Tuple of (DQN agent, Monte Carlo agent, Policy Gradient agent)
    """
    if num_episodes is None:
        num_episodes = SIMULATION['num_hands']
        
    if output_dir is None:
        output_dir = 'models'
        
    if checkpoint_interval is None:
        checkpoint_interval = SIMULATION['checkpoint_interval']
        
    # Create output directories
    models_dir = os.path.join(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'training_checkpoints')
    results_dir = os.path.join('results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = BlackjackEnv()
    env.reset()  # Initialize the environment first
    state_size = len(env.vectorize_state())
    action_size = len(list(Action))
    
    # Create agents
    dqn_agent = DQNAgent(state_size, action_size)
    mc_agent = MonteCarloAgent()
    pg_agent = PolicyGradientAgent(state_size, action_size)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train all agents
    if verbose:
        print(f"Starting training for all models ({num_episodes} episodes each)...")
        
    # Train DQN Agent
    if verbose:
        print("\n=== Training DQN Agent ===")
        
    dqn_agent = train_dqn_agent(
        agent=dqn_agent,
        env=env,
        num_episodes=num_episodes,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
        timestamp=timestamp,
        verbose=verbose
    )
    
    # Train Monte Carlo Agent
    if verbose:
        print("\n=== Training Monte Carlo Agent ===")
        
    mc_agent = train_monte_carlo_agent(
        agent=mc_agent,
        env=env,
        num_episodes=num_episodes,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
        timestamp=timestamp,
        verbose=verbose
    )
    
    # Train Policy Gradient Agent
    if verbose:
        print("\n=== Training Policy Gradient Agent ===")
        
    pg_agent = train_policy_gradient_agent(
        agent=pg_agent,
        env=env,
        num_episodes=num_episodes,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
        timestamp=timestamp,
        verbose=verbose
    )
    
    # Save final models
    dqn_path = os.path.join(models_dir, f"dqn_final_{timestamp}.pkl")
    mc_path = os.path.join(models_dir, f"monte_carlo_final_{timestamp}.pkl")
    pg_path = os.path.join(models_dir, f"policy_gradient_final_{timestamp}")
    
    dqn_agent.save(dqn_path)
    mc_agent.save(mc_path)
    pg_agent.save(pg_path)
    
    # Create model paths file for easy reference
    paths_file = os.path.join(output_dir, "model_paths.txt")
    with open(paths_file, 'w') as f:
        f.write(f"DQN Model: {dqn_path}\n")
        f.write(f"Monte Carlo Model: {mc_path}\n")
        f.write(f"Policy Gradient Model: {pg_path}\n")
        f.write(f"\nUsage with model_comparison.py:\n")
        f.write(f"python src/model_comparison.py --dqn-path {dqn_path} --mc-path {mc_path} --pg-path {pg_path}\n")
        
    if verbose:
        print("\nTraining completed!")
        print(f"Models saved to: {output_dir}")
        print(f"Model paths file created at: {paths_file}")
        
    return dqn_agent, mc_agent, pg_agent


def train_dqn_agent(agent, env, num_episodes, checkpoint_interval, output_dir, timestamp, verbose=True):
    """Train a DQN agent on the blackjack environment.
    
    Args:
        agent: DQN agent to train
        env: Blackjack environment
        num_episodes: Number of episodes to train for
        checkpoint_interval: Interval for saving checkpoints
        output_dir: Directory to save models and metrics
        timestamp: Timestamp for filenames
        verbose: Whether to print progress
        
    Returns:
        Trained DQN agent
    """
    # Initialize metrics tracking
    rewards = []
    win_rates = []
    win_count = 0
    loss_count = 0
    episode_rewards = []
    
    # Training loop
    progress_bar = tqdm(range(num_episodes)) if verbose else range(num_episodes)
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        done = False
        
        # Set bet amount
        bet_amount = BETTING['min_bet']
        
        # Consider true count for betting strategy
        true_count = state['true_count']
        if true_count > 1:
            # Increase bet based on count
            bet_amount = min(
                BETTING['max_bet'],
                BETTING['min_bet'] * (1 + true_count / 2)
            )
            
        env.set_bet(bet_amount)
        
        # Convert state to vector format for the agent
        state_vector = env.vectorize_state()
        episode_reward = 0
        
        # Play one episode
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action
            action = agent.act(state_vector, valid_actions)
            
            # Take action
            next_state, reward, done = env.step(action)
            next_state_vector = env.vectorize_state() if not done else np.zeros_like(state_vector)
            
            # Store experience in replay memory
            agent.remember(state_vector, action, reward, next_state_vector, done)
            
            # Learn from experiences
            agent.replay()
            
            # Update state
            state = next_state
            state_vector = next_state_vector
            episode_reward += reward
            
        # Update metrics
        rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        
        # Track win/loss
        if episode_reward > 0:
            win_count += 1
        elif episode_reward < 0:
            loss_count += 1
            
        # Calculate win rate over last 1000 episodes
        if (episode + 1) % 1000 == 0:
            window = min(1000, episode + 1)
            recent_rewards = rewards[-window:]
            win_rate = sum(r > 0 for r in recent_rewards) / window
            win_rates.append(win_rate)
            
            # Update progress bar
            if verbose:
                avg_reward = sum(recent_rewards) / window
                progress_bar.set_description(
                    f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.4f}"
                )
                
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                output_dir, 'training_checkpoints', f"dqn_checkpoint_ep{episode+1}_{timestamp}.pkl"
            )
            agent.save(checkpoint_path)
            
    # Plot learning curves
    figures_dir = os.path.join('results', 'visualizations')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot reward curve
    plt.figure(figsize=(10, 6))
    window_size = 1000
    smoothed_rewards = np.convolve(
        rewards, np.ones(window_size)/window_size, mode='valid'
    )
    plt.plot(smoothed_rewards)
    plt.title('DQN: Average Reward per Episode (Rolling Window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"dqn_rewards_{timestamp}.png"))
    
    # Plot win rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1000, len(win_rates) * 1000 + 1, 1000), win_rates)
    plt.title('DQN: Win Rate per 1000 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"dqn_win_rate_{timestamp}.png"))
    
    return agent


def train_monte_carlo_agent(agent, env, num_episodes, checkpoint_interval, output_dir, timestamp, verbose=True):
    """Train a Monte Carlo agent on the blackjack environment.
    
    Args:
        agent: Monte Carlo agent to train
        env: Blackjack environment
        num_episodes: Number of episodes to train for
        checkpoint_interval: Interval for saving checkpoints
        output_dir: Directory to save models and metrics
        timestamp: Timestamp for filenames
        verbose: Whether to print progress
        
    Returns:
        Trained Monte Carlo agent
    """
    # Initialize metrics tracking
    rewards = []
    win_rates = []
    win_count = 0
    loss_count = 0
    
    # Training loop
    progress_bar = tqdm(range(num_episodes)) if verbose else range(num_episodes)
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        
        # Start new episode for Monte Carlo agent
        agent.start_episode()
        
        # Set bet amount
        bet_amount = BETTING['min_bet']
        env.set_bet(bet_amount)
        
        # Play one episode
        done = False
        episode_rewards = []
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action
            action = agent.choose_action(state, valid_actions)
            
            # Record state-action pair
            agent.record_step(state, action)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Record reward
            episode_rewards.append(reward)
            
            # Update state
            state = next_state
            
        # Update policy based on episode
        agent.update_policy(episode_rewards)
        
        # Calculate total reward for the episode
        total_reward = sum(episode_rewards)
        rewards.append(total_reward)
        
        # Track win/loss
        if total_reward > 0:
            win_count += 1
        elif total_reward < 0:
            loss_count += 1
            
        # Calculate win rate over last 1000 episodes
        if (episode + 1) % 1000 == 0:
            window = min(1000, episode + 1)
            recent_rewards = rewards[-window:]
            win_rate = sum(r > 0 for r in recent_rewards) / window
            win_rates.append(win_rate)
            
            # Update progress bar
            if verbose:
                avg_reward = sum(recent_rewards) / window
                progress_bar.set_description(
                    f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.4f}"
                )
                
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                output_dir, 'training_checkpoints', f"monte_carlo_checkpoint_ep{episode+1}_{timestamp}.pkl"
            )
            agent.save(checkpoint_path)
            
            # Plot policy at checkpoints
            policy_plot_path = os.path.join(
                'results', 'visualizations', f"mc_policy_ep{episode+1}_{timestamp}.png"
            )
            agent.plot_policy(policy_plot_path)
            
    # Plot learning curves
    figures_dir = os.path.join('results', 'visualizations')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot reward curve
    plt.figure(figsize=(10, 6))
    window_size = 1000
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(
            rewards, np.ones(window_size)/window_size, mode='valid'
        )
        plt.plot(smoothed_rewards)
    else:
        plt.plot(rewards)
    plt.title('Monte Carlo: Average Reward per Episode (Rolling Window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"monte_carlo_rewards_{timestamp}.png"))
    
    # Plot win rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1000, len(win_rates) * 1000 + 1, 1000), win_rates)
    plt.title('Monte Carlo: Win Rate per 1000 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"monte_carlo_win_rate_{timestamp}.png"))
    
    # Plot final policy
    policy_plot_path = os.path.join(
        'results', 'visualizations', f"mc_policy_final_{timestamp}.png"
    )
    agent.plot_policy(policy_plot_path)
    
    return agent


def train_policy_gradient_agent(agent, env, num_episodes, checkpoint_interval, output_dir, timestamp, verbose=True):
    """Train a Policy Gradient agent on the blackjack environment.
    
    Args:
        agent: Policy Gradient agent to train
        env: Blackjack environment
        num_episodes: Number of episodes to train for
        checkpoint_interval: Interval for saving checkpoints
        output_dir: Directory to save models and metrics
        timestamp: Timestamp for filenames
        verbose: Whether to print progress
        
    Returns:
        Trained Policy Gradient agent
    """
    # Initialize metrics tracking
    rewards = []
    win_rates = []
    win_count = 0
    loss_count = 0
    
    # Training loop
    progress_bar = tqdm(range(num_episodes)) if verbose else range(num_episodes)
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        done = False
        
        # Set bet amount
        bet_amount = BETTING['min_bet']
        env.set_bet(bet_amount)
        
        # Convert state to vector format for the agent
        state_vector = env.vectorize_state()
        episode_total_reward = 0
        
        # Play one episode
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action
            action = agent.act(state_vector, valid_actions)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Remember the state, action, reward
            agent.remember(state_vector, action, reward)
            
            # Update state
            state = next_state
            if not done:
                state_vector = env.vectorize_state()
                
            episode_total_reward += reward
            
        # End of episode, update policy
        loss = agent.end_episode()
        
        # Update metrics
        rewards.append(episode_total_reward)
        
        # Track win/loss
        if episode_total_reward > 0:
            win_count += 1
        elif episode_total_reward < 0:
            loss_count += 1
            
        # Calculate win rate over last 1000 episodes
        if (episode + 1) % 1000 == 0:
            window = min(1000, episode + 1)
            recent_rewards = rewards[-window:]
            win_rate = sum(r > 0 for r in recent_rewards) / window
            win_rates.append(win_rate)
            
            # Update progress bar
            if verbose:
                avg_reward = sum(recent_rewards) / window
                progress_bar.set_description(
                    f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Loss: {loss:.4f}"
                )
                
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                output_dir, 'training_checkpoints', f"policy_gradient_checkpoint_ep{episode+1}_{timestamp}"
            )
            agent.save(checkpoint_path)
            
    # Plot learning curves
    figures_dir = os.path.join('results', 'visualizations')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot reward curve
    plt.figure(figsize=(10, 6))
    window_size = 1000
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(
            rewards, np.ones(window_size)/window_size, mode='valid'
        )
        plt.plot(smoothed_rewards)
    else:
        plt.plot(rewards)
    plt.title('Policy Gradient: Average Reward per Episode (Rolling Window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"policy_gradient_rewards_{timestamp}.png"))
    
    # Plot win rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1000, len(win_rates) * 1000 + 1, 1000), win_rates)
    plt.title('Policy Gradient: Win Rate per 1000 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"policy_gradient_win_rate_{timestamp}.png"))
    
    return agent


if __name__ == "__main__":
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Train all blackjack AI models')
    parser.add_argument('--episodes', type=int, default=None, 
                       help='Number of episodes to train for each model')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save models')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Interval for saving checkpoints')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Train all models
    dqn_agent, mc_agent, pg_agent = train_all_models(
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        verbose=not args.no_verbose
    )
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s") 