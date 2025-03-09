# Blackjack AI Visualization Guide

This guide provides an overview of the visualization files generated during our blackjack AI model comparison. These visualizations help interpret the performance data and understand the strengths and weaknesses of different models and betting strategies.

## Available Visualizations

Based on the files in the `results/visualizations` directory, the following visualizations are available:

### Overall Comparison Visualizations

1. **`bankroll_progression_20250309_190842.png`**
   - Shows how the bankroll changes over time for different agent-strategy combinations
   - Use this to visualize bankroll stability and overall performance trajectory

2. **`avg_reward_20250309_170043.png`**
   - Displays the average reward per hand for each model and betting strategy
   - Higher values indicate better performance

3. **`win_rate_by_agent_20250309_170043.png`**
   - Compares the win rates across different agent types
   - Useful for isolating the performance of the playing strategy from the betting strategy

4. **`roi_heatmap_20250309_170043.png`**
   - A heatmap showing ROI for different combinations of agents and betting strategies
   - Darker colors typically indicate better performance

### DQN-Specific Visualizations

1. **`dqn_rewards_20250309_145852.png`**
   - Shows the rewards during DQN training over episodes
   - Helps understand the learning progress of the DQN model

2. **`dqn_win_rate_20250309_145852.png`**
   - Displays the win rate progression during DQN training
   - Should show improvement as training progresses

### Monte Carlo Visualizations

1. **`monte_carlo_rewards_20250309_145852.png`**
   - Shows the rewards during Monte Carlo training
   - Useful for comparing learning efficiency against DQN

2. **`monte_carlo_win_rate_20250309_145852.png`**
   - Displays the win rate progression during Monte Carlo training

3. **`mc_policy_*.png`** (Various episodes)
   - Visualizations of the Monte Carlo policy at different training stages
   - Shows how the policy evolves from early training to the final model

### Policy Gradient Visualizations

1. **`policy_gradient_rewards_20250309_174502.png`**
   - Shows the rewards during Policy Gradient training
   - Useful for comparing learning efficiency against other models

2. **`policy_gradient_win_rate_20250309_174502.png`**
   - Displays the win rate progression during Policy Gradient training

## How to Interpret the Visualizations

### Bankroll Progression

The bankroll progression chart is one of the most important visualizations as it shows the actual performance of each model-strategy combination in terms of money gained or lost over time.

**Key aspects to look for:**
- **Upward trends** indicate profitable strategies
- **Downward trends** show losing strategies
- **Steep drops** suggest high-risk betting strategies
- **Horizontal sections** indicate stable but not profitable play
- **Early termination** of a line indicates bankroll depletion

### Learning Curves

Learning curves (rewards and win rates over episodes) show how quickly and effectively each model learns.

**Key aspects to look for:**
- **Upward trend** indicates the model is learning
- **Plateau** suggests the model has reached its learning capacity
- **Volatility** (jagged lines) can indicate unstable learning or exploration
- **Comparison between models** shows relative learning efficiency

### Policy Visualizations

The Monte Carlo policy visualizations (mc_policy_*.png) show the actual strategy the model has learned for different hand values and dealer upcard combinations.

**Key aspects to look for:**
- **Consistency with Basic Strategy** in later episodes
- **Evolution of policy** from early to late episodes
- **Areas of disagreement** with Basic Strategy that might indicate innovative approaches or learning issues

### ROI Heatmap

The ROI heatmap provides a condensed view of performance across all combinations.

**Key aspects to look for:**
- **Dark areas** indicating high-performing combinations
- **Patterns** showing which agents or betting strategies consistently perform well
- **Outliers** that might suggest unique synergies between specific agents and betting strategies

## Recommended Visualization Analysis Workflow

1. Start with the **ROI heatmap** to get an overview of which combinations performed best
2. Look at the **bankroll progression** chart to understand performance over time
3. Compare **win rates by agent** to isolate playing strategy performance
4. Examine the **learning curves** to understand how each model improved during training
5. For Monte Carlo, review the **policy visualizations** to see how the learned strategy compares to Basic Strategy

## Creating Custom Visualizations

If you need additional insights, you can create custom visualizations using the raw data in the CSV files:
- `comparison_results_20250309_174501.csv`
- `model_comparison_20250309_190842.csv`
- `model_comparison_20250309_170043.csv`

These files contain detailed metrics for each agent-strategy combination, including:
- Final bankroll
- Net profit
- ROI
- Average reward
- Win rate
- Hands played
- Blackjack rate
- Complete bankroll history

## Conclusion

These visualizations provide valuable insights into the performance of different blackjack AI models and betting strategies. By analyzing them carefully, you can identify the most promising approaches and understand the trade-offs between risk and reward across different combinations.

For a comprehensive analysis of the results, please refer to the accompanying `Analysis_Results.md` document, which interprets these visualizations in the context of our research goals. 