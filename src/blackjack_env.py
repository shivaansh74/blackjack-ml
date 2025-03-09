"""
Blackjack environment for AI training and gameplay.
This module implements a custom blackjack environment that follows casino rules
and provides interfaces for AI agents to interact with.
"""

import random
import numpy as np
from enum import Enum, auto
import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLACKJACK_RULES


class Action(Enum):
    """Available actions in blackjack."""
    HIT = auto()
    STAND = auto()
    DOUBLE = auto()
    SPLIT = auto()
    SURRENDER = auto()


class Card:
    """Representation of a playing card."""
    
    def __init__(self, rank, suit):
        self.rank = rank  # 1=A, 2-10, 11=J, 12=Q, 13=K
        self.suit = suit  # "Hearts", "Diamonds", "Clubs", "Spades"
        
    @property
    def value(self):
        """Return the blackjack value of the card."""
        if self.rank == 1:  # Ace
            return 11
        elif self.rank >= 10:  # Face cards
            return 10
        else:
            return self.rank
            
    def __str__(self):
        """String representation of the card."""
        ranks = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        rank_str = ranks.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit[0]}"
        
    def __repr__(self):
        return self.__str__()


class Hand:
    """Representation of a blackjack hand."""
    
    def __init__(self, cards=None):
        self.cards = cards or []
        self.bet = 0
        self.doubled = False
        self.is_split = False
        self.is_surrender = False
        
    @property
    def values(self):
        """Return all possible values of the hand, accounting for aces."""
        total = sum(card.value for card in self.cards)
        num_aces = sum(1 for card in self.cards if card.rank == 1)
        
        # If we have aces and total > 21, convert aces from 11 to 1
        possible_values = [total]
        for _ in range(num_aces):
            if possible_values[0] > 21:
                possible_values[0] -= 10
                
        return possible_values
        
    @property
    def value(self):
        """Return the best value of the hand."""
        return min(self.values)
        
    @property
    def is_bust(self):
        """Check if the hand is bust (>21)."""
        return self.value > 21
        
    @property
    def is_blackjack(self):
        """Check if the hand is a blackjack (21 with 2 cards)."""
        return len(self.cards) == 2 and self.value == 21
        
    @property
    def is_soft(self):
        """Check if the hand is soft (contains an ace counted as 11)."""
        for card in self.cards:
            if card.rank == 1 and self.value <= 21:
                # Check if the ace is counted as 11 by seeing if value - 10 is still valid
                if self.value - 10 <= 21:
                    return True
        return False
        
    def add_card(self, card):
        """Add a card to the hand."""
        self.cards.append(card)
        
    def can_split(self):
        """Check if the hand can be split."""
        if len(self.cards) != 2:
            return False
        return self.cards[0].rank == self.cards[1].rank
        
    def can_double(self):
        """Check if the hand can be doubled."""
        return len(self.cards) == 2 and not self.doubled
        
    def can_surrender(self):
        """Check if the hand can be surrendered."""
        return len(self.cards) == 2 and not self.doubled
        
    def __str__(self):
        """String representation of the hand."""
        return f"Cards: {self.cards}, Value: {self.value}, {'Soft' if self.is_soft else 'Hard'}"
        
    def __repr__(self):
        return self.__str__()


class Deck:
    """Representation of a deck of cards."""
    
    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.cards = []
        self.reset()
        
    def reset(self):
        """Reset and shuffle the deck."""
        self.cards = []
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        ranks = list(range(1, 14))  # 1=A, 2-10, 11=J, 12=Q, 13=K
        
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(rank, suit))
                    
        self.shuffle()
        
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
        
    def draw(self):
        """Draw a card from the deck."""
        if not self.cards:
            raise ValueError("Cannot draw from an empty deck")
        return self.cards.pop()
        
    def __len__(self):
        return len(self.cards)


class BlackjackEnv:
    """Blackjack environment for AI training and gameplay."""
    
    def __init__(self, config=None):
        self.config = config or BLACKJACK_RULES
        self.deck = Deck(num_decks=self.config['num_decks'])
        self.player_hands = []
        self.dealer_hand = None
        self.current_hand_idx = 0
        
        # Initialize card counting
        self.running_count = 0
        self.true_count = 0
        self.cards_seen = 0
        
    def reset(self):
        """Reset the environment for a new hand."""
        # Reset if less than 25% of cards remain
        if len(self.deck) < (52 * self.config['num_decks'] * 0.25):
            self.deck.reset()
            self.running_count = 0
            self.cards_seen = 0
            
        self.player_hands = [Hand()]
        self.dealer_hand = Hand()
        self.current_hand_idx = 0
        
        # Deal initial cards
        for _ in range(2):
            self.player_hands[0].add_card(self._draw_card())
            
        self.dealer_hand.add_card(self._draw_card())
        
        # Dealer's hole card is drawn but not visible to player
        dealer_hole_card = self._draw_card()
        self.dealer_hand.add_card(dealer_hole_card)
        
        return self._get_state()
        
    def _draw_card(self):
        """Draw a card and update card count."""
        card = self.deck.draw()
        self.cards_seen += 1
        
        # Update running count (Hi-Lo system)
        if card.rank >= 10 or card.rank == 1:  # 10, J, Q, K, A
            self.running_count -= 1
        elif 2 <= card.rank <= 6:  # 2-6
            self.running_count += 1
        
        # Calculate true count
        decks_remaining = max(1, (len(self.deck) / 52))
        self.true_count = self.running_count / decks_remaining
        
        return card
        
    def _get_state(self):
        """Get the current state of the game."""
        # If all hands are played, return state for the last hand
        if self.current_hand_idx >= len(self.player_hands):
            self.current_hand_idx = len(self.player_hands) - 1
            
        player_hand = self.player_hands[self.current_hand_idx]
        
        state = {
            'player_hand': player_hand,
            'dealer_upcard': self.dealer_hand.cards[0],
            'can_hit': not player_hand.is_bust and not player_hand.doubled,
            'can_stand': not player_hand.is_bust,
            'can_double': player_hand.can_double(),
            'can_split': player_hand.can_split() and len(self.player_hands) < self.config['max_split_hands'],
            'can_surrender': player_hand.can_surrender() and self.config['allow_surrender'],
            'running_count': self.running_count,
            'true_count': self.true_count,
            'cards_seen': self.cards_seen,
        }
        
        return state
        
    def step(self, action):
        """Take an action in the environment."""
        player_hand = self.player_hands[self.current_hand_idx]
        reward = 0
        done = False
        
        if action == Action.HIT:
            if player_hand.is_bust or player_hand.doubled:
                raise ValueError("Cannot hit on a bust or doubled hand")
                
            player_hand.add_card(self._draw_card())
            
            if player_hand.is_bust:
                reward = -player_hand.bet
                self._move_to_next_hand()
                
        elif action == Action.STAND:
            self._move_to_next_hand()
            
        elif action == Action.DOUBLE:
            if not player_hand.can_double():
                raise ValueError("Cannot double on this hand")
                
            player_hand.bet *= 2
            player_hand.doubled = True
            player_hand.add_card(self._draw_card())
            
            if player_hand.is_bust:
                reward = -player_hand.bet
                
            self._move_to_next_hand()
            
        elif action == Action.SPLIT:
            if not player_hand.can_split():
                raise ValueError("Cannot split this hand")
                
            if len(self.player_hands) >= self.config['max_split_hands']:
                raise ValueError("Maximum number of split hands reached")
                
            # Create a new hand with the second card
            new_hand = Hand([player_hand.cards.pop()])
            new_hand.bet = player_hand.bet
            new_hand.is_split = True
            
            # Add a card to each hand
            player_hand.add_card(self._draw_card())
            new_hand.add_card(self._draw_card())
            
            # Add the new hand to the player's hands
            self.player_hands.insert(self.current_hand_idx + 1, new_hand)
            
        elif action == Action.SURRENDER:
            if not player_hand.can_surrender():
                raise ValueError("Cannot surrender this hand")
                
            player_hand.is_surrender = True
            reward = -player_hand.bet / 2
            self._move_to_next_hand()
            
        else:
            raise ValueError(f"Invalid action: {action}")
            
        # Check if all hands are played
        if self.current_hand_idx >= len(self.player_hands):
            done = True
            
            # If game is done, play dealer's hand and calculate rewards
            if not all(hand.is_bust or hand.is_surrender for hand in self.player_hands):
                self._play_dealer_hand()
                
            rewards = self._calculate_rewards()
            reward = sum(rewards)
            
        return self._get_state(), reward, done
        
    def _move_to_next_hand(self):
        """Move to the next hand if available."""
        self.current_hand_idx += 1
        
    def _play_dealer_hand(self):
        """Play the dealer's hand according to casino rules."""
        # Dealer plays hand only if player hasn't busted or surrendered
        active_hands = [hand for hand in self.player_hands 
                      if not hand.is_bust and not hand.is_surrender]
        
        if not active_hands:
            return
            
        # Dealer hits until hand value is at least 17
        while True:
            # Dealer must hit on soft 17 if configured
            if self.dealer_hand.value < 17 or (self.dealer_hand.value == 17 and 
                                           self.dealer_hand.is_soft and 
                                           self.config['dealer_hit_soft_17']):
                self.dealer_hand.add_card(self._draw_card())
            else:
                break
                
    def _calculate_rewards(self):
        """Calculate rewards for all hands."""
        rewards = []
        
        for hand in self.player_hands:
            if hand.is_bust:
                rewards.append(-hand.bet)
            elif hand.is_surrender:
                rewards.append(-hand.bet / 2)
            elif hand.is_blackjack and not self.dealer_hand.is_blackjack:
                rewards.append(hand.bet * self.config['blackjack_payout'])
            elif self.dealer_hand.is_bust:
                rewards.append(hand.bet)
            elif hand.value > self.dealer_hand.value:
                rewards.append(hand.bet)
            elif hand.value < self.dealer_hand.value:
                rewards.append(-hand.bet)
            elif hand.is_blackjack and self.dealer_hand.is_blackjack:
                rewards.append(0)  # Push on both blackjack
            else:
                rewards.append(0)  # Push
                
        return rewards
        
    def set_bet(self, bet_amount):
        """Set the bet for the current hand."""
        if not self.player_hands:
            self.player_hands = [Hand()]
            
        self.player_hands[0].bet = bet_amount
        
    def get_valid_actions(self):
        """Get the valid actions for the current state."""
        state = self._get_state()
        valid_actions = []
        
        if state['can_hit']:
            valid_actions.append(Action.HIT)
            
        if state['can_stand']:
            valid_actions.append(Action.STAND)
            
        if state['can_double']:
            valid_actions.append(Action.DOUBLE)
            
        if state['can_split']:
            valid_actions.append(Action.SPLIT)
            
        if state['can_surrender']:
            valid_actions.append(Action.SURRENDER)
            
        return valid_actions
        
    def vectorize_state(self):
        """Convert the state to a vector representation for ML models."""
        state = self._get_state()
        player_hand = state['player_hand']
        
        # Player's hand features
        player_sum = player_hand.value
        player_has_usable_ace = 1 if player_hand.is_soft else 0
        player_cards_count = len(player_hand.cards)
        
        # Dealer's upcard feature
        dealer_upcard_value = state['dealer_upcard'].value
        
        # Card counting features
        running_count = state['running_count']
        true_count = state['true_count']
        
        # Additional game state features
        can_double = 1 if state['can_double'] else 0
        can_split = 1 if state['can_split'] else 0
        can_surrender = 1 if state['can_surrender'] else 0
        
        features = [
            player_sum / 21.0,  # Normalize player sum
            player_has_usable_ace,
            player_cards_count / 10.0,  # Normalize card count
            dealer_upcard_value / 11.0,  # Normalize dealer upcard
            true_count / 10.0,  # Normalize true count
            can_double,
            can_split,
            can_surrender
        ]
        
        return np.array(features)


if __name__ == "__main__":
    # Test the environment
    env = BlackjackEnv()
    state = env.reset()
    env.set_bet(10)
    
    print(f"Player hand: {state['player_hand']}")
    print(f"Dealer upcard: {state['dealer_upcard']}")
    
    # Play a test hand
    done = False
    while not done:
        valid_actions = env.get_valid_actions()
        print(f"Valid actions: {valid_actions}")
        
        # Choose random action
        action = random.choice(valid_actions)
        print(f"Taking action: {action}")
        
        state, reward, done = env.step(action)
        
        if not done:
            print(f"Player hand: {state['player_hand']}")
        
    print(f"Game done! Reward: {reward}")
    print(f"Dealer hand: {env.dealer_hand}") 