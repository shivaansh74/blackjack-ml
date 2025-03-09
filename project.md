# Blackjack AI: A Machine Learning-Based Card Strategy System

## Project Overview
This project aims to develop a machine learning system capable of playing blackjack with an optimized strategy to maximize winnings. The AI will be trained using reinforcement learning techniques, specifically Deep Q-Learning, Monte Carlo simulations, and policy gradient methods. The final model will make real-time betting and playing decisions to achieve the highest expected margin of success. Additionally, an interactive module will allow users to play against the trained AI, simulating a casino experience.

## Key Features
- **Reinforcement Learning-Based AI**: Utilizes advanced RL techniques to learn optimal blackjack strategies.
- **Card Counting Awareness**: Incorporates memory of past cards to improve decision-making.
- **Interactive Gameplay**: Users can play blackjack against the AI just like in a real casino.
- **Simulation & Training**: Runs millions of hands to refine strategy and improve performance.
- **Dynamic Betting System**: Adjusts bet sizes based on confidence in outcomes.
- **Multiple AI Strategies**: Compares various RL models and selects the best-performing one.

---

## Project Structure
```
blackjack_ai/
├── data/
│   ├── game_logs/               # Stores logs of AI gameplay for analysis
│   ├── training_data/           # Stores historical game data for reinforcement learning
│   └── preprocessed_data/       # Processed data ready for model training
├── models/
│   ├── trained_blackjack_ai.pkl # Final trained model
│   └── training_checkpoints/    # Checkpoints for different stages of training
├── src/
│   ├── blackjack_env.py         # Custom blackjack environment for AI training
│   ├── model_training.py        # Machine learning model training scripts
│   ├── policy_network.py        # Deep learning policy model for RL
│   ├── q_learning_agent.py      # Deep Q-Network (DQN) agent implementation
│   ├── monte_carlo_agent.py     # Monte Carlo simulation-based agent
│   ├── card_counter.py          # Card counting module for strategy enhancement
│   ├── bet_optimizer.py         # Betting strategy optimizer
│   ├── game_simulator.py        # Runs AI vs. AI and AI vs. baseline strategies
│   ├── interactive_play.py      # Allows users to play against AI like in a casino
├── notebooks/
│   ├── exploratory_data_analysis.ipynb # Data analysis and visualization
│   ├── model_performance.ipynb        # Evaluation of different training approaches
├── results/
│   ├── ai_performance_metrics.txt     # Logs of AI's win rate, profit margin, etc.
│   ├── visualizations/                 # Graphs and insights from model training
├── config.py                           # Configuration file for hyperparameters
├── requirements.txt                     # Dependencies for the project
├── README.md                            # Documentation and setup guide
```

---

## Methodology
### 1. **Blackjack Environment Simulation**
- Implement a blackjack environment that follows standard casino rules.
- Allow multi-hand simulations to generate training data.
- Track card history for counting techniques.

### 2. **Feature Engineering & Game State Representation**
- Encode game states, including:
  - Player’s hand value and composition.
  - Dealer’s visible card.
  - Count of remaining high and low-value cards (card counting).
  - Running win/loss record for adaptive betting.
  
### 3. **Reinforcement Learning Training**
- **Deep Q-Network (DQN)**: Uses neural networks to approximate the optimal Q-values.
- **Monte Carlo Simulation**: Generates probability-based strategies by running thousands of hands.
- **Policy Gradient (PG) Methods**: Uses reinforcement learning to directly optimize betting and playing policies.

### 4. **Model Evaluation & Optimization**
- Train multiple models and compare performance based on:
  - **Win rate percentage**
  - **Expected value per hand**
  - **Betting profit/loss over time**
- Optimize hyperparameters such as:
  - Learning rate
  - Discount factor
  - Exploration-exploitation balance

### 5. **Interactive AI Gameplay Mode**
- Users can play blackjack against the AI in real-time.
- Simulates a real casino experience with:
  - **Graphical User Interface (GUI) or Terminal-based interaction**
  - **AI decision explanations**
  - **Card counting statistics displayed live**

---

## Skills Demonstrated
- **Machine Learning & AI**: Reinforcement learning, deep learning, Monte Carlo simulations.
- **Game Theory & Probability**: Understanding expected values and strategic decision-making.
- **Software Engineering**: Building modular, scalable Python codebases.
- **Data Science & Visualization**: Analyzing blackjack outcomes and model performance.

---

## Installation & Usage
### Requirements
- Python 3.9+
- Required packages listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blackjack_ai.git
   cd blackjack_ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the AI (optional if using a pre-trained model):
   ```bash
   python src/model_training.py
   ```
4. Play against the AI:
   ```bash
   python src/interactive_play.py
   ```

---

## Future Improvements
- **Neural Network-Based Opponent Modeling**: Learn to adjust strategy based on different player types.
- **Live Card Counting Tracker**: Provide real-time insights on deck composition.
- **Multi-Agent AI Training**: Train multiple AI agents to optimize against each other.
- **Web-Based UI**: Deploy an online version for users to play through a browser.

---

## Conclusion
This project builds a highly optimized AI capable of playing blackjack with an advanced reinforcement learning strategy. By integrating real-time interactive play, this AI not only maximizes winnings but also provides an engaging way for users to test their skills against a machine-learning-trained blackjack bot.

