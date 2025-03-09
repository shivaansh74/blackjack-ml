"""
Train reinforcement learning models for blackjack.

This script trains the DQN agent on the blackjack environment and
saves performance metrics and model checkpoints.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_env import BlackjackEnv, Action
from q_learning_agent import DQNAgent
from config import TRAINING, SIMULATION, BETTING


def train_dqn(num_episodes=None, model_path=None, checkpoint_dir=None, log_dir=None):
    """Train a DQN agent on the blackjack environment.
    
    Args:
        num_episodes: Number of episodes to train for
        model_path: Path to save final model
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
    """
    if num_episodes is None:
        num_episodes = SIMULATION['num_hands']
        
    if model_path is None:
        model_path = os.path.join('models', 'trained_blackjack_ai.pkl')
        
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join('models', 'training_checkpoints')
        
    if log_dir is None:
        log_dir = os.path.join('results')
        
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment and agent
    env = BlackjackEnv()
    state_size = len(env.vectorize_state())
    action_size = len(list(Action))
    agent = DQNAgent(state_size, action_size)
    
    # Initialize metrics tracking
    rewards = []
    win_rates = []
    win_count = 0
    loss_count = 0
    episode_rewards = []
    evaluation_rewards = []
    evaluation_episodes = []
    
    # Training loop
    print(f"Starting training for {num_episodes} episodes...")
    progress_bar = tqdm(range(num_episodes))
    
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
            avg_reward = sum(recent_rewards) / window
            progress_bar.set_description(
                f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.4f}"
            )
            
        # Evaluate model
        if (episode + 1) % SIMULATION['evaluation_interval'] == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=100)
            evaluation_rewards.append(eval_reward)
            evaluation_episodes.append(episode + 1)
            
        # Save checkpoint
        if (episode + 1) % SIMULATION['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"dqn_checkpoint_ep{episode+1}.pkl"
            )
            agent.save(checkpoint_path)
            
    # Save final model
    agent.save(model_path)
    
    # Save metrics
    save_metrics(
        rewards=rewards,
        win_rates=win_rates,
        evaluation_rewards=evaluation_rewards,
        evaluation_episodes=evaluation_episodes,
        log_dir=log_dir
    )
    
    # Plot learning curves
    plot_learning_curves(
        rewards=rewards,
        win_rates=win_rates,
        evaluation_rewards=evaluation_rewards,
        evaluation_episodes=evaluation_episodes,
        log_dir=log_dir
    )
    
    # Final statistics
    total_win_rate = win_count / num_episodes
    total_avg_reward = sum(rewards) / num_episodes
    
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Final win rate: {total_win_rate:.4f}")
    print(f"Final average reward: {total_avg_reward:.4f}")
    print(f"Model saved to: {model_path}")
    
    return agent


def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate agent performance without exploration.
    
    Args:
        agent: DQN agent
        env: Blackjack environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward per episode
    """
    # Save epsilon value
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        # Set bet amount
        env.set_bet(BETTING['min_bet'])
        
        # Convert state to vector format
        state_vector = env.vectorize_state()
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state_vector, valid_actions)
            next_state, reward, done = env.step(action)
            
            if not done:
                state_vector = env.vectorize_state()
            
            total_reward += reward
            
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Return average reward
    return total_reward / num_episodes


def save_metrics(rewards, win_rates, evaluation_rewards, evaluation_episodes, log_dir):
    """Save training metrics to CSV files.
    
    Args:
        rewards: List of episode rewards
        win_rates: List of win rates
        evaluation_rewards: List of evaluation rewards
        evaluation_episodes: List of evaluation episodes
        log_dir: Directory to save metrics
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save episode rewards
    rewards_df = pd.DataFrame({
        'episode': range(1, len(rewards) + 1),
        'reward': rewards
    })
    rewards_df.to_csv(os.path.join(log_dir, f"dqn_rewards_{timestamp}.csv"), index=False)
    
    # Save win rates
    win_rates_df = pd.DataFrame({
        'episode': range(1000, len(win_rates) * 1000 + 1, 1000),
        'win_rate': win_rates
    })
    win_rates_df.to_csv(os.path.join(log_dir, f"dqn_win_rates_{timestamp}.csv"), index=False)
    
    # Save evaluation metrics
    eval_df = pd.DataFrame({
        'episode': evaluation_episodes,
        'reward': evaluation_rewards
    })
    eval_df.to_csv(os.path.join(log_dir, f"dqn_evaluation_{timestamp}.csv"), index=False)


def plot_learning_curves(rewards, win_rates, evaluation_rewards, evaluation_episodes, log_dir):
    """Plot and save learning curves.
    
    Args:
        rewards: List of episode rewards
        win_rates: List of win rates
        evaluation_rewards: List of evaluation rewards
        evaluation_episodes: List of evaluation episodes
        log_dir: Directory to save plots
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figures directory
    figures_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot episode rewards (rolling average)
    plt.figure(figsize=(10, 6))
    rolling_rewards = pd.Series(rewards).rolling(window=1000).mean()
    plt.plot(range(1, len(rewards) + 1), rolling_rewards)
    plt.title('Average Reward per Episode (Rolling Window of 1000)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"dqn_rewards_{timestamp}.png"))
    
    # Plot win rates
    plt.figure(figsize=(10, 6))
    plt.plot(range(1000, len(win_rates) * 1000 + 1, 1000), win_rates)
    plt.title('Win Rate per 1000 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"dqn_win_rates_{timestamp}.png"))
    
    # Plot evaluation rewards
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_episodes, evaluation_rewards)
    plt.title('Evaluation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Over 100 Episodes)')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"dqn_evaluation_{timestamp}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for blackjack')
    parser.add_argument('--episodes', type=int, default=None, 
                        help='Number of episodes to train for')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to save final model')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Train agent
    agent = train_dqn(
        num_episodes=args.episodes,
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s") 