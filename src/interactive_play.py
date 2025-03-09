"""
Interactive blackjack game against the trained AI.

This script allows users to play blackjack against the trained AI agent
in a terminal-based interface.
"""

import os
import sys
import time
import argparse
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_env import BlackjackEnv, Action, Card
from q_learning_agent import DQNAgent
from config import INTERACTIVE, BETTING


class InteractiveBlackjack:
    """Interactive blackjack game against AI."""
    
    def __init__(self, model_path=None):
        """Initialize the interactive game.
        
        Args:
            model_path: Path to the trained model
        """
        self.env = BlackjackEnv()
        
        # Initialize agent
        state_size = len(self.env.vectorize_state())
        action_size = len(list(Action))
        self.agent = DQNAgent(state_size, action_size)
        
        # Try to load model
        if model_path is None:
            model_path = os.path.join('models', 'trained_blackjack_ai.pkl')
            
        try:
            self.agent.load(model_path)
            self.model_loaded = True
            print(f"Loaded trained model from {model_path}")
        except Exception as e:
            self.model_loaded = False
            print(f"Could not load model from {model_path}. Using untrained agent.")
            print(f"Error: {e}")
            
        # Set exploration to 0 for deterministic behavior
        self.agent.epsilon = 0
        
        # Game state
        self.player_bankroll = BETTING['initial_bankroll']
        self.bet_amount = INTERACTIVE['default_bet']
        self.show_ai_reasoning = INTERACTIVE['show_ai_reasoning']
        self.show_count = INTERACTIVE['show_count']
        
    def play(self):
        """Start the interactive game."""
        self._print_welcome()
        
        playing = True
        while playing and self.player_bankroll > 0:
            # Start a new hand
            self._play_hand()
            
            # Ask to play again
            playing = self._play_again()
            
        self._print_goodbye()
        
    def _play_hand(self):
        """Play a single hand of blackjack."""
        # Reset environment
        state = self.env.reset()
        
        # Place bets
        self._place_bets()
        
        # Deal initial cards
        self._show_game_state(initial_deal=True)
        
        # Check for player blackjack
        player_hand = state['player_hand']
        if player_hand.is_blackjack:
            print("\nYou have blackjack!")
            self._finish_hand()
            return
            
        # Player's turn
        self._player_turn()
        
        # Finish hand (dealer's turn and payout)
        self._finish_hand()
        
    def _place_bets(self):
        """Place bets for the hand."""
        print(f"\nYour bankroll: ${self.player_bankroll}")
        
        # Show card counting information if enabled
        if self.show_count:
            state = self.env._get_state()
            print(f"Running count: {state['running_count']}")
            print(f"True count: {state['true_count']:.2f}")
            
        # Get bet amount from player
        while True:
            try:
                bet_input = input(f"Enter bet amount (default: ${self.bet_amount}): ")
                if bet_input.strip() == "":
                    bet_amount = self.bet_amount
                else:
                    bet_amount = int(bet_input)
                    
                if bet_amount <= 0:
                    print("Bet amount must be positive.")
                    continue
                    
                if bet_amount > self.player_bankroll:
                    print(f"You don't have enough money. Your bankroll is ${self.player_bankroll}.")
                    continue
                    
                self.bet_amount = bet_amount
                break
            except ValueError:
                print("Please enter a valid number.")
                
        # Set bet in environment
        self.env.set_bet(self.bet_amount)
        print(f"You bet: ${self.bet_amount}")
        
    def _show_game_state(self, initial_deal=False):
        """Display the current game state."""
        # Get the current state from environment
        state = self.env._get_state()
        player_hand = state['player_hand']
        dealer_upcard = state['dealer_upcard']
        
        # Clear screen for better UI
        self._clear_screen()
        
        # Dealer's cards
        print("\n=== Blackjack ===\n")
        if initial_deal:
            print(f"Dealer shows: {dealer_upcard} [?]")
        else:
            dealer_cards = ", ".join(str(card) for card in self.env.dealer_hand.cards)
            print(f"Dealer: {dealer_cards} (Value: {self.env.dealer_hand.value})")
            
        # Player's cards
        player_cards = ", ".join(str(card) for card in player_hand.cards)
        print(f"Your hand: {player_cards} (Value: {player_hand.value})")
        
        # Card counting info (if enabled)
        if self.show_count:
            print(f"\nRunning count: {state['running_count']}")
            print(f"True count: {state['true_count']:.2f}")
            
    def _player_turn(self):
        """Handle the player's turn."""
        done = False
        
        while not done:
            # Show game state
            self._show_game_state()
            
            # Get valid actions
            valid_actions = self.env.get_valid_actions()
            action_map = {
                'h': Action.HIT,
                's': Action.STAND,
                'd': Action.DOUBLE,
                'p': Action.SPLIT,
                'r': Action.SURRENDER
            }
            
            # Map valid actions to user inputs
            valid_inputs = []
            action_display = []
            
            if Action.HIT in valid_actions:
                valid_inputs.append('h')
                action_display.append("(H)it")
                
            if Action.STAND in valid_actions:
                valid_inputs.append('s')
                action_display.append("(S)tand")
                
            if Action.DOUBLE in valid_actions:
                valid_inputs.append('d')
                action_display.append("(D)ouble")
                
            if Action.SPLIT in valid_actions:
                valid_inputs.append('p')
                action_display.append("S(p)lit")
                
            if Action.SURRENDER in valid_actions:
                valid_inputs.append('r')
                action_display.append("Su(r)render")
                
            # Get AI's recommendation
            if self.show_ai_reasoning and self.model_loaded:
                self._show_ai_recommendation(valid_actions)
                
            # Get player's action
            print("\nAvailable actions:", " | ".join(action_display))
            action_input = input("Your action: ").lower()
            
            while action_input not in valid_inputs:
                print(f"Invalid action. Choose from: {', '.join(action_display)}")
                action_input = input("Your action: ").lower()
                
            # Execute the action
            action = action_map[action_input]
            print(f"You chose to {action.name.lower()}.")
            
            # Take the action
            _, reward, done = self.env.step(action)
            
            # Check if player busted
            if self.env.player_hands[self.env.current_hand_idx-1].is_bust:
                print("Bust! You went over 21.")
                done = True
                
            time.sleep(1)
            
    def _show_ai_recommendation(self, valid_actions):
        """Show AI agent's recommended action and reasoning."""
        state_vector = self.env.vectorize_state()
        action = self.agent.act(state_vector, valid_actions)
        q_values = self.agent.get_q_values(state_vector)
        
        # Get Q-values for valid actions
        valid_q_values = {}
        for a in valid_actions:
            action_idx = a.value - 1
            valid_q_values[a.name] = q_values[action_idx]
            
        # Sort by expected value
        sorted_actions = sorted(valid_q_values.items(), key=lambda x: x[1], reverse=True)
        
        # Display recommendation
        print("\nAI recommendation:")
        print(f"Recommended action: {action.name}")
        print("Expected values:")
        
        for action_name, value in sorted_actions:
            if action_name == action.name:
                print(f"  * {action_name}: {value:.2f}")
            else:
                print(f"    {action_name}: {value:.2f}")
            
    def _finish_hand(self):
        """Finish the hand (dealer's turn and payout)."""
        # Show all cards
        self._show_game_state(initial_deal=False)
        
        # Check if all player hands busted or surrendered
        all_busted = all(hand.is_bust or hand.is_surrender 
                       for hand in self.env.player_hands)
                       
        if not all_busted:
            # Play dealer's hand
            print("\nDealer's turn:")
            self.env._play_dealer_hand()
            
            # Show final state
            self._show_game_state(initial_deal=False)
            
            # Show dealer's final hand
            dealer_hand = self.env.dealer_hand
            if dealer_hand.is_bust:
                print("Dealer busts!")
                
        # Calculate rewards
        rewards = self.env._calculate_rewards()
        total_reward = sum(rewards)
        
        # Update bankroll
        self.player_bankroll += total_reward
        
        # Show result
        if total_reward > 0:
            print(f"\nYou won ${total_reward:.2f}!")
        elif total_reward < 0:
            print(f"\nYou lost ${abs(total_reward):.2f}.")
        else:
            print("\nPush (tie).")
            
        print(f"Your bankroll is now ${self.player_bankroll:.2f}")
        input("\nPress Enter to continue...")
        
    def _play_again(self):
        """Ask if the player wants to play another hand."""
        if self.player_bankroll <= 0:
            print("\nYou're out of money! Game over.")
            return False
            
        while True:
            answer = input("\nPlay another hand? (y/n): ").lower()
            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'.")
                
    def _print_welcome(self):
        """Print welcome message."""
        self._clear_screen()
        print("=" * 60)
        print("Welcome to Blackjack AI!")
        print("=" * 60)
        print("Play against a reinforcement learning trained blackjack AI.")
        print(f"You're starting with a bankroll of ${self.player_bankroll}.")
        
        # Training status
        if self.model_loaded:
            print("The AI has been trained and will use optimal strategy.")
        else:
            print("The AI is using an untrained model - it might not play optimally.")
            
        # Settings
        print("\nGame settings:")
        print(f"- Show AI reasoning: {'Yes' if self.show_ai_reasoning else 'No'}")
        print(f"- Show card counting: {'Yes' if self.show_count else 'No'}")
        
        print("\nGood luck!")
        print("=" * 60)
        input("Press Enter to start playing...")
        
    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThanks for playing Blackjack AI!")
        print(f"Your final bankroll: ${self.player_bankroll:.2f}")
        
        if self.player_bankroll > BETTING['initial_bankroll']:
            profit = self.player_bankroll - BETTING['initial_bankroll']
            print(f"You made a profit of ${profit:.2f}!")
        elif self.player_bankroll < BETTING['initial_bankroll']:
            loss = BETTING['initial_bankroll'] - self.player_bankroll
            print(f"You lost ${loss:.2f}.")
        else:
            print("You broke even.")
            
        print("\nCome back soon!")
        
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play interactive blackjack against AI')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--no-ai-reasoning', action='store_true',
                        help='Disable AI reasoning display')
    parser.add_argument('--no-card-counting', action='store_true',
                        help='Disable card counting display')
    parser.add_argument('--bankroll', type=float, default=None,
                        help='Starting bankroll')
    
    args = parser.parse_args()
    
    # Create interactive game
    game = InteractiveBlackjack(model_path=args.model_path)
    
    # Override settings if specified
    if args.no_ai_reasoning:
        game.show_ai_reasoning = False
        
    if args.no_card_counting:
        game.show_count = False
        
    if args.bankroll is not None:
        game.player_bankroll = args.bankroll
        
    # Start the game
    game.play() 