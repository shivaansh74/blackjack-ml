"""
Bet Optimizer for Blackjack.

This module optimizes betting strategies based on card counting, 
bankroll management, and risk analysis.
"""

import numpy as np
import sys
import os
from enum import Enum

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BETTING, CARD_COUNTING
from card_counter import CardCounter


class BettingStrategy(Enum):
    """Different betting strategies for blackjack."""
    FLAT = 1         # Bet the same amount always
    PROPORTIONAL = 2  # Bet proportional to the true count
    KELLY = 3         # Kelly criterion for optimal bet sizing
    OSCAR = 4         # Oscar's Grind progression system
    MARTINGALE = 5    # Double after each loss
    ANTI_MARTINGALE = 6  # Double after each win
    FIBONACCI = 7     # Use Fibonacci sequence for progression


class BetOptimizer:
    """Bet optimizer for blackjack."""
    
    def __init__(self, initial_bankroll=None, strategy=BettingStrategy.PROPORTIONAL):
        """Initialize the bet optimizer.
        
        Args:
            initial_bankroll: Starting bankroll (default from config)
            strategy: Betting strategy to use
        """
        self.initial_bankroll = initial_bankroll or BETTING['initial_bankroll']
        self.current_bankroll = self.initial_bankroll
        self.min_bet = BETTING['min_bet']
        self.max_bet = BETTING['max_bet']
        self.strategy = strategy
        
        # Initialize card counter
        self.card_counter = CardCounter()
        
        # Strategy-specific variables
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_sequence = [1, 1]  # For Fibonacci
        self.sequence_position = 0
        self.session_start_bankroll = self.initial_bankroll
        self.win_threshold = 0  # For Oscar's Grind
        
        # Risk management parameters
        self.risk_factor = 0.01  # Percentage of bankroll to risk by default
        self.target_profit = 0.5  # Target profit as a percentage of initial bankroll
        self.stop_loss = 0.5     # Stop loss as a percentage of initial bankroll
        self.bet_spread = 1.0    # Maximum bet as multiple of minimum bet
        self.max_risk_per_hand = 0.05  # Maximum percentage of bankroll to risk on one hand
        
        # Session tracking
        self.session_hands = 0
        self.session_won = 0
        self.session_lost = 0
        self.session_profit = 0
        
    def reset_session(self):
        """Reset the current betting session."""
        self.session_start_bankroll = self.current_bankroll
        self.session_hands = 0
        self.session_won = 0
        self.session_lost = 0
        self.session_profit = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_sequence = [1, 1]
        self.sequence_position = 0
        self.win_threshold = 0
        
    def update_bankroll(self, result):
        """Update bankroll after a hand.
        
        Args:
            result: Amount won or lost in the hand
            
        Returns:
            Updated bankroll
        """
        self.current_bankroll += result
        self.session_profit += result
        self.session_hands += 1
        
        if result > 0:
            self.session_won += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif result < 0:
            self.session_lost += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        return self.current_bankroll
        
    def get_optimal_bet(self, true_count=None):
        """Calculate the optimal bet amount based on current strategy.
        
        Args:
            true_count: Current true count (if None, use card counter)
            
        Returns:
            Optimal bet amount
        """
        if true_count is None:
            true_count = self.card_counter.true_count
            
        if self.strategy == BettingStrategy.FLAT:
            # Always bet the same amount
            return self._flat_bet()
            
        elif self.strategy == BettingStrategy.PROPORTIONAL:
            # Bet proportional to the true count
            return self._proportional_bet(true_count)
            
        elif self.strategy == BettingStrategy.KELLY:
            # Kelly criterion for optimal bet sizing
            return self._kelly_bet(true_count)
            
        elif self.strategy == BettingStrategy.OSCAR:
            # Oscar's Grind progression system
            return self._oscars_grind_bet()
            
        elif self.strategy == BettingStrategy.MARTINGALE:
            # Double after each loss
            return self._martingale_bet()
            
        elif self.strategy == BettingStrategy.ANTI_MARTINGALE:
            # Double after each win
            return self._anti_martingale_bet()
            
        elif self.strategy == BettingStrategy.FIBONACCI:
            # Use Fibonacci sequence for progression
            return self._fibonacci_bet()
            
        else:
            # Default to flat betting
            return self._flat_bet()
            
    def _flat_bet(self):
        """Calculate flat bet amount.
        
        Returns:
            Bet amount
        """
        return max(self.min_bet, min(self.max_bet, int(self.current_bankroll * self.risk_factor)))
        
    def _proportional_bet(self, true_count):
        """Calculate bet proportional to true count.
        
        Args:
            true_count: Current true count
            
        Returns:
            Bet amount
        """
        # Get bet multiplier from card counter
        multiplier = self.card_counter.get_bet_multiplier()
        
        # Calculate base bet based on risk factor
        base_bet = self.current_bankroll * self.risk_factor
        
        # Apply multiplier
        bet = base_bet * multiplier
        
        # Apply constraints
        return max(self.min_bet, min(self.max_bet, int(bet)))
        
    def _kelly_bet(self, true_count):
        """Calculate bet using Kelly criterion.
        
        Args:
            true_count: Current true count
            
        Returns:
            Bet amount
        """
        # Estimate edge based on true count
        # For Hi-Lo system, each true count point is approximately 0.5% edge
        edge = 0.005 * true_count
        
        # Adjust for house edge in blackjack (approx -0.5% with basic strategy)
        adjusted_edge = edge - 0.005
        
        # Apply Kelly formula (edge/variance)
        # In blackjack, variance is approximately 1.15
        if adjusted_edge <= 0:
            kelly_fraction = 0
        else:
            kelly_fraction = adjusted_edge / 1.15
            
        # Use half Kelly to be more conservative
        half_kelly = kelly_fraction / 2
        
        # Calculate bet amount
        bet = self.current_bankroll * half_kelly
        
        # Apply constraints
        return max(self.min_bet, min(self.max_bet, int(bet)))
        
    def _oscars_grind_bet(self):
        """Calculate bet using Oscar's Grind system.
        
        Returns:
            Bet amount
        """
        # Start with minimum bet
        if self.session_hands == 0:
            return self.min_bet
            
        # Oscar's Grind: Increase bet by 1 unit after a win,
        # keep the same bet after a loss, and stop when profit target reached
        bet = self.min_bet
        
        if self.consecutive_wins > 0:
            bet = min(self.min_bet + self.consecutive_wins, self.max_bet)
            
        # Don't bet more than needed to reach profit target
        if self.session_profit + bet > self.win_threshold:
            bet = max(self.min_bet, self.win_threshold - self.session_profit)
            
        return bet
        
    def _martingale_bet(self):
        """Calculate bet using Martingale system.
        
        Returns:
            Bet amount
        """
        # Double bet after each loss, start with minimum
        base_bet = self.min_bet
        bet = base_bet * (2 ** self.consecutive_losses)
        
        # Apply bankroll percentage cap for risk management
        max_bet_allowed = self.current_bankroll * self.max_risk_per_hand
        
        return max(self.min_bet, min(self.max_bet, min(int(bet), int(max_bet_allowed))))
        
    def _anti_martingale_bet(self):
        """Calculate bet using Anti-Martingale system.
        
        Returns:
            Bet amount
        """
        # Double bet after each win, start with minimum
        base_bet = self.min_bet
        bet = base_bet * (2 ** self.consecutive_wins)
        
        # Apply bankroll percentage cap for risk management
        max_bet_allowed = self.current_bankroll * self.max_risk_per_hand
        
        return max(self.min_bet, min(self.max_bet, min(int(bet), int(max_bet_allowed))))
        
    def _fibonacci_bet(self):
        """Calculate bet using Fibonacci sequence.
        
        Returns:
            Bet amount
        """
        # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
        if self.session_hands == 0:
            self.current_sequence = [1, 1]
            self.sequence_position = 0
            return self.min_bet
            
        # Move forward in sequence after a loss
        if self.consecutive_losses > 0 and self.consecutive_losses > self.consecutive_wins:
            # Generate next Fibonacci number if needed
            if self.sequence_position + 1 >= len(self.current_sequence):
                self.current_sequence.append(
                    self.current_sequence[-1] + self.current_sequence[-2]
                )
            self.sequence_position = min(self.sequence_position + 1, len(self.current_sequence) - 1)
        
        # Move backward in sequence after a win
        elif self.consecutive_wins > 0:
            self.sequence_position = max(0, self.sequence_position - 1)
            
        # Calculate bet
        bet = self.min_bet * self.current_sequence[self.sequence_position]
        
        # Apply bankroll percentage cap for risk management
        max_bet_allowed = self.current_bankroll * self.max_risk_per_hand
        
        return max(self.min_bet, min(self.max_bet, min(int(bet), int(max_bet_allowed))))
        
    def should_continue_session(self):
        """Determine if the betting session should continue.
        
        Returns:
            Boolean indicating whether to continue
        """
        # Check for stop-loss
        if self.current_bankroll <= self.initial_bankroll * (1 - self.stop_loss):
            return False
            
        # Check for profit target
        if self.current_bankroll >= self.initial_bankroll * (1 + self.target_profit):
            return False
            
        # Continue if bankroll is still positive and above minimum bet
        return self.current_bankroll >= self.min_bet
        
    def get_session_stats(self):
        """Get statistics for the current session.
        
        Returns:
            Dictionary with session statistics
        """
        win_rate = self.session_won / max(1, self.session_hands)
        roi = self.session_profit / max(1, self.session_start_bankroll)
        
        return {
            'hands_played': self.session_hands,
            'hands_won': self.session_won,
            'hands_lost': self.session_lost,
            'win_rate': win_rate,
            'profit': self.session_profit,
            'roi': roi,
            'current_bankroll': self.current_bankroll,
            'starting_bankroll': self.session_start_bankroll
        }
        
    def get_strategy_name(self):
        """Get the name of the current betting strategy.
        
        Returns:
            String name of strategy
        """
        strategy_names = {
            BettingStrategy.FLAT: "Flat Betting",
            BettingStrategy.PROPORTIONAL: "Proportional to Count",
            BettingStrategy.KELLY: "Kelly Criterion",
            BettingStrategy.OSCAR: "Oscar's Grind",
            BettingStrategy.MARTINGALE: "Martingale",
            BettingStrategy.ANTI_MARTINGALE: "Anti-Martingale",
            BettingStrategy.FIBONACCI: "Fibonacci Progression"
        }
        
        return strategy_names.get(self.strategy, "Unknown Strategy")
        
    def set_strategy(self, strategy):
        """Set the betting strategy.
        
        Args:
            strategy: BettingStrategy enum value
            
        Returns:
            None
        """
        if not isinstance(strategy, BettingStrategy):
            raise ValueError("Strategy must be a BettingStrategy enum value")
            
        self.strategy = strategy
        
        # Reset progression-specific variables
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_sequence = [1, 1]
        self.sequence_position = 0


if __name__ == "__main__":
    # Test the bet optimizer
    optimizer = BetOptimizer(initial_bankroll=1000, strategy=BettingStrategy.KELLY)
    print(f"Using {optimizer.get_strategy_name()} strategy")
    
    # Simulate different count scenarios
    for true_count in [-3, -1, 0, 1, 3, 5]:
        # Update card counter true count
        optimizer.card_counter.true_count = true_count
        
        # Get optimal bet
        bet = optimizer.get_optimal_bet()
        print(f"True count: {true_count}, Optimal bet: ${bet}")
        
    # Change strategy and test again
    optimizer.set_strategy(BettingStrategy.PROPORTIONAL)
    print(f"\nSwitched to {optimizer.get_strategy_name()} strategy")
    
    for true_count in [-3, -1, 0, 1, 3, 5]:
        # Update card counter true count
        optimizer.card_counter.true_count = true_count
        
        # Get optimal bet
        bet = optimizer.get_optimal_bet()
        print(f"True count: {true_count}, Optimal bet: ${bet}") 