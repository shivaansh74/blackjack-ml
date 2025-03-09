"""
Card Counting Module for Blackjack.

This module implements various card counting systems to enhance 
betting strategies in blackjack.
"""

import numpy as np
from collections import defaultdict
import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CARD_COUNTING


class CardCounter:
    """Card counter for blackjack strategies."""
    
    # Card counting systems with card values
    # Each system assigns values to cards 2-10, J, Q, K, A
    COUNTING_SYSTEMS = {
        'hi_lo': {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1},
        'ko': {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1},
        'hi_opt_i': {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1},
        'hi_opt_ii': {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 0, 9: 0, 10: -2, 11: -2, 12: -2, 13: -2},
        'omega_ii': {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 0, 9: -1, 10: -2, 11: -2, 12: -2, 13: -2},
        'zen_count': {1: -1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 0, 9: 0, 10: -2, 11: -2, 12: -2, 13: -2},
    }
    
    def __init__(self, system=None, num_decks=6):
        """Initialize the card counter.
        
        Args:
            system: Card counting system to use ('hi_lo', 'ko', etc.)
            num_decks: Number of decks in the shoe
        """
        self.system = system or CARD_COUNTING['system']
        self.num_decks = num_decks
        
        if self.system not in self.COUNTING_SYSTEMS:
            raise ValueError(f"Unknown counting system: {self.system}")
            
        self.count_values = self.COUNTING_SYSTEMS[self.system]
        
        # Initialize counters
        self.reset()
        
    def reset(self):
        """Reset counters for a new shoe."""
        self.running_count = 0
        self.cards_seen = 0
        self.true_count = 0
        
        # Track seen cards for more advanced counting
        self.seen_cards = defaultdict(int)
        
        # Calculate initial count for balanced systems
        # (some systems like KO are unbalanced)
        self.initial_count = 0
        
    def update(self, card):
        """Update count based on the observed card.
        
        Args:
            card: Card object with rank attribute (1=A, 2-10, 11=J, 12=Q, 13=K)
            
        Returns:
            Updated true count
        """
        # Get card rank
        rank = card.rank
        
        # Update running count
        self.running_count += self.count_values[rank]
        
        # Track seen cards
        self.seen_cards[rank] += 1
        self.cards_seen += 1
        
        # Calculate true count (running count per deck remaining)
        decks_remaining = max(1, (self.num_decks * 52 - self.cards_seen) / 52)
        self.true_count = self.running_count / decks_remaining
        
        return self.true_count
        
    def update_multiple(self, cards):
        """Update count based on multiple observed cards.
        
        Args:
            cards: List of Card objects
            
        Returns:
            Updated true count
        """
        for card in cards:
            self.update(card)
        return self.true_count
        
    def get_bet_multiplier(self, base_factor=None):
        """Calculate bet multiplier based on the true count.
        
        Args:
            base_factor: Factor to multiply the true count by (default from config)
            
        Returns:
            Bet multiplier (1.0 for neutral count, higher for positive count)
        """
        if base_factor is None:
            base_factor = CARD_COUNTING['count_factor']
            
        # For most counting systems, a positive count favors the player
        # We increase bets linearly based on the true count
        if self.true_count <= 0:
            return 1.0  # Minimum bet for negative or zero count
        else:
            return 1.0 + (self.true_count * base_factor)
            
    def get_remaining_card_probability(self, rank):
        """Calculate probability of drawing a card of given rank.
        
        Args:
            rank: Card rank (1=A, 2-10, 11=J, 12=Q, 13=K)
            
        Returns:
            Probability of drawing the card
        """
        # Count initial cards of this rank in the shoe
        initial_count = 4 * self.num_decks  # 4 cards of each rank per deck
        
        # Adjust for face cards (10, J, Q, K all count as 10-value cards)
        if rank >= 10:
            initial_count = 16 * self.num_decks  # 16 ten-value cards per deck
            seen_ten_values = sum(self.seen_cards[r] for r in range(10, 14))
            remaining = initial_count - seen_ten_values
        else:
            remaining = initial_count - self.seen_cards[rank]
            
        # Calculate total remaining cards
        total_remaining = self.num_decks * 52 - self.cards_seen
        
        # Return probability
        return max(0, remaining) / total_remaining
        
    def get_bust_probability(self, current_value):
        """Calculate probability of busting if hitting.
        
        Args:
            current_value: Current hand value
            
        Returns:
            Probability of busting
        """
        # Calculate needed value to bust
        needed_to_bust = 22 - current_value
        
        # If already at 21 or higher, can't hit
        if needed_to_bust <= 0:
            return 1.0
            
        # Calculate probability of drawing a card that would bust
        bust_prob = sum(
            self.get_remaining_card_probability(rank) 
            for rank in range(1, 14) 
            if (rank == 1 and current_value + 11 > 21) or
               (rank >= 10 and current_value + 10 > 21) or
               (2 <= rank <= 9 and current_value + rank > 21)
        )
        
        return bust_prob
        
    def get_card_advantage_index(self):
        """Calculate a card advantage index based on remaining cards.
        
        Returns:
            Advantage index (-1 to +1, higher is better for player)
        """
        # Calculate advantage based on ratio of favorable to unfavorable cards
        high_cards_prob = sum(
            self.get_remaining_card_probability(rank)
            for rank in [1, 10, 11, 12, 13]  # A, 10, J, Q, K
        )
        
        low_cards_prob = sum(
            self.get_remaining_card_probability(rank)
            for rank in range(2, 7)  # 2-6
        )
        
        # Normalize to -1 to +1 range
        # High cards favor player, low cards favor dealer
        advantage = (high_cards_prob - low_cards_prob) * 2
        
        return advantage
        
    def get_strategy_adjustment(self, player_value, dealer_upcard_value, is_soft=False):
        """Get strategy adjustment based on count.
        
        Args:
            player_value: Player's hand value
            dealer_upcard_value: Dealer's upcard value
            is_soft: Whether the player's hand is soft
            
        Returns:
            Dictionary with strategy adjustments for actions
        """
        adjustments = {}
        
        # Basic strategy adjustments based on true count
        if self.true_count >= 3:
            # High true count favors player - take more risks
            
            # Insurance becomes profitable at true count >= 3 for Hi-Lo
            adjustments['take_insurance'] = True
            
            # Stand on more hands with high count
            if 12 <= player_value <= 16 and dealer_upcard_value >= 7:
                adjustments['prefer_stand'] = True
                
            # Double down more aggressively
            if 9 <= player_value <= 11:
                adjustments['prefer_double'] = True
                
        elif self.true_count <= -3:
            # Low true count favors dealer - be more conservative
            
            # Hit more hands with low count
            if 12 <= player_value <= 16 and dealer_upcard_value >= 7:
                adjustments['prefer_hit'] = True
                
            # Surrender more hands
            if player_value == 15 and dealer_upcard_value == 10:
                adjustments['prefer_surrender'] = True
                
        return adjustments
        
    def get_count_status(self):
        """Get the current count status as a human-readable string.
        
        Returns:
            String describing the count status
        """
        status_map = {
            (-float('inf'), -3): "Very negative - dealer advantage, bet minimum",
            (-3, -1): "Negative - slight dealer advantage, bet minimum",
            (-1, 1): "Neutral - no significant advantage, bet normal",
            (1, 3): "Positive - slight player advantage, increase bets",
            (3, 5): "Very positive - player advantage, bet aggressively",
            (5, float('inf')): "Extremely positive - strong player advantage, bet maximum"
        }
        
        for (lower, upper), description in status_map.items():
            if lower < self.true_count <= upper:
                return description
                
        return "Neutral - no significant advantage"


if __name__ == "__main__":
    # Test the card counter
    from blackjack_env import Card
    
    counter = CardCounter(system='hi_lo', num_decks=6)
    print(f"Card counter created with {counter.system} system")
    
    # Simulate seeing some cards
    cards = [
        Card(10, "Hearts"),  # 10 heart
        Card(5, "Diamonds"),  # 5 diamond
        Card(2, "Clubs"),    # 2 club
        Card(1, "Spades"),   # Ace spade
        Card(13, "Hearts"),  # King heart
    ]
    
    for card in cards:
        true_count = counter.update(card)
        print(f"Saw {card}, Running Count: {counter.running_count}, True Count: {true_count:.2f}")
        
    # Get betting recommendation
    bet_multiplier = counter.get_bet_multiplier()
    print(f"Recommended bet multiplier: {bet_multiplier:.2f}x")
    
    # Get advantage index
    advantage = counter.get_card_advantage_index()
    print(f"Card advantage index: {advantage:.2f}")
    
    # Get count status
    status = counter.get_count_status()
    print(f"Count status: {status}") 