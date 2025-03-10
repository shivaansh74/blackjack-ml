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
import random

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
        
        # Reset the environment first to ensure we have a valid state
        self.env.reset()
        
        # Initialize the player's advisor agent
        state_size = len(self.env.vectorize_state())
        action_size = len(list(Action))
        self.agent = DQNAgent(state_size, action_size)
        
        # Try to load model for player's advisor
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
        
        # Create AI opponents
        self.ai_opponents = []
        self.ai_names = ["Monte Carlo Mia", "DQN Dan", "Gradient Gary", "Basic Betty"]
        self.ai_opponent_count = random.randint(1, 3)  # Random number of opponents
        
        # Initialize AI opponent agents (between 1-3 opponents)
        for i in range(self.ai_opponent_count):
            # Create a new agent with random behavior
            ai_agent = DQNAgent(state_size, action_size)
            ai_agent.epsilon = 0  # Deterministic behavior
            
            # Create a unique environment for this AI
            ai_env = BlackjackEnv()
            
            # Add opponent to the table
            self.ai_opponents.append({
                "name": self.ai_names[i],
                "agent": ai_agent,
                "env": ai_env,
                "bankroll": BETTING['initial_bankroll'],
                "bet": INTERACTIVE['default_bet'],
                "strategy": random.choice(["basic", "dqn", "random"])  # Random strategy type
            })
            
        # Game state
        self.player_bankroll = BETTING['initial_bankroll']
        self.bet_amount = INTERACTIVE['default_bet']
        self.show_ai_reasoning = INTERACTIVE['show_ai_reasoning']
        self.show_count = INTERACTIVE['show_count']
        # Default to basic strategy for recommendations
        self.recommendation_strategy = "basic"
        
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
        # Reset environment for player
        state = self.env.reset()
        
        # Reset environments for all AI opponents
        for opponent in self.ai_opponents:
            opponent["env"].reset()
        
        # Place bets
        self._place_bets()
        
        # Deal initial cards
        self._show_game_state(initial_deal=True)
        
        # Check for player blackjack
        player_hand = state['player_hand']
        if player_hand.is_blackjack:
            print("\n🎉 You have blackjack!")
            self._finish_hand()
            return
            
        # Player's turn
        self._player_turn()
        
        # AI opponents' turns
        for opponent in self.ai_opponents:
            self._ai_opponent_turn(opponent)
            
        # Finish hand (dealer's turn and payout)
        self._finish_hand()
        
    def _place_bets(self):
        """Place bets for the hand."""
        print(f"\n💰 Your bankroll: ${self.player_bankroll:.2f}")
        
        # Show card counting information if enabled
        if self.show_count:
            state = self.env._get_state()
            print(f"📊 Running count: {state['running_count']}")
            print(f"📊 True count: {state['true_count']:.2f}")
            
        print("\n┌───────────────────────────┐")
        print("│        PLACE YOUR BET      │")
        print("└───────────────────────────┘")
            
        # Get bet amount from player
        while True:
            try:
                bet_input = input(f"💲 Enter bet amount (default: ${self.bet_amount}): ")
                if bet_input.strip() == "":
                    bet_amount = self.bet_amount
                else:
                    bet_amount = int(bet_input)
                    
                if bet_amount <= 0:
                    print("❌ Bet amount must be positive.")
                    continue
                    
                if bet_amount > self.player_bankroll:
                    print(f"❌ Not enough funds. Your bankroll is ${self.player_bankroll:.2f}.")
                    continue
                    
                self.bet_amount = bet_amount
                break
            except ValueError:
                print("❌ Please enter a valid number.")
                
        # Set bet in environment
        self.env.set_bet(self.bet_amount)
        print(f"\n🎲 You bet: ${self.bet_amount:.2f}")
        
        # AI opponents place their bets
        if self.ai_opponents:
            print("\n🤖 AI OPPONENTS PLACING BETS:")
            
            # Define default bet limits if not in config
            min_bet = 5  # Default minimum bet
            max_bet = 100  # Default maximum bet
            
            # Try to get values from config if available
            if 'min_bet' in INTERACTIVE:
                min_bet = INTERACTIVE['min_bet']
            if 'max_bet' in INTERACTIVE:
                max_bet = INTERACTIVE['max_bet']
            
            for opponent in self.ai_opponents:
                # AI betting strategy based on count or bankroll
                if opponent["strategy"] == "basic" and self.show_count:
                    # Basic strategy will use the count for betting
                    true_count = self.env._get_state()["true_count"]
                    if true_count > 2:
                        # Increase bet when count is favorable
                        bet = min(opponent["bankroll"], opponent["bet"] * (1 + (true_count - 2) * 0.5))
                    elif true_count < -1:
                        # Decrease bet when count is unfavorable
                        bet = max(min_bet, opponent["bet"] * 0.5)
                    else:
                        # Default bet
                        bet = opponent["bet"]
                elif opponent["strategy"] == "dqn":
                    # DQN uses a moderate betting strategy
                    bet = opponent["bet"] + random.randint(-10, 10)
                    bet = max(min_bet, min(opponent["bankroll"] * 0.25, bet))
                else:
                    # Random betting strategy
                    bet = random.randint(
                        min_bet, 
                        min(int(opponent["bankroll"] * 0.3), max_bet)
                    )
                    
                # Ensure bet is valid
                bet = max(min_bet, min(opponent["bankroll"], bet))
                opponent["bet"] = bet
                
                # Set bet in AI environment
                opponent["env"].set_bet(bet)
                print(f"🤖 {opponent['name']} bets: ${bet:.2f}")
                
            print("───────────────────────────────")
        
    def _show_game_state(self, initial_deal=False):
        """Display the current game state."""
        # Get the current state from environment
        state = self.env._get_state()
        player_hand = state['player_hand']
        dealer_upcard = state['dealer_upcard']
        
        # Clear screen for better UI
        self._clear_screen()
        
        # Show table border and game title
        print("\n┌─────────────────────────────────────────┐")
        print("│             🎲 BLACKJACK 21 🎲            │")
        print("└─────────────────────────────────────────┘\n")
        
        # Show number of players at the table
        players_count = 1 + len(self.ai_opponents)  # Player + AI opponents
        print(f"🎮 Players at table: {players_count} (You + {len(self.ai_opponents)} AI opponents)")
        
        # Show bet amount and bankroll
        print(f"💰 Your bankroll: ${self.player_bankroll:.2f}  |  🎲 Your bet: ${self.bet_amount:.2f}\n")
        
        # Dealer's cards with fancy formatting
        print("─────────── 🎩 DEALER ───────────")
        if initial_deal:
            # Only show one card during initial deal
            print(f"Cards: {dealer_upcard} [🂠]")
            print(f"Value: ?")
        else:
            # Show all dealer cards
            dealer_cards = ", ".join(str(card) for card in self.env.dealer_hand.cards)
            print(f"Cards: {dealer_cards}")
            print(f"Value: {self.env.dealer_hand.value}")
            
        print("───────────────────────────────")
        
        # Player's cards with fancy formatting
        print("\n─────────── 👤 YOU ───────────")
        player_cards = ", ".join(str(card) for card in player_hand.cards)
        print(f"Cards: {player_cards}")
        print(f"Value: {player_hand.value}")
        print("───────────────────────────────")
        
        # AI opponents
        if self.ai_opponents:
            print("\n─────────── 🤖 AI OPPONENTS ───────────")
            for i, opponent in enumerate(self.ai_opponents):
                opponent_state = opponent["env"]._get_state()
                opponent_hand = opponent_state['player_hand']
                
                # Show opponent's hand
                cards = ", ".join(str(card) for card in opponent_hand.cards)
                print(f"{i+1}. {opponent['name']} (${opponent['bankroll']:.2f})")
                print(f"   Bet: ${opponent['bet']:.2f}  |  Cards: {cards}  |  Value: {opponent_hand.value}")
                
                if i < len(self.ai_opponents) - 1:
                    print("   ---")
            print("───────────────────────────────")
        
        # Card counting info (if enabled)
        if self.show_count:
            print(f"\n📊 Card Counting Stats:")
            print(f"   Running count: {state['running_count']}")
            print(f"   True count: {state['true_count']:.2f}")
            print("───────────────────────────────")
            
    def _player_turn(self):
        """Handle the player's turn."""
        done = False
        
        while not done:
            # Show game state
            self._show_game_state()
            
            # Check if player has exactly 21 (not just blackjack)
            if self.env.player_hands[self.env.current_hand_idx].value == 21:
                print("\n🎉 You have 21! Perfect hand!")
                time.sleep(1)
                return
            
            # Get valid actions
            valid_actions = self.env.get_valid_actions()
            action_map = {
                'h': Action.HIT,
                's': Action.STAND,
                'd': Action.DOUBLE,
                'p': Action.SPLIT,
                'r': Action.SURRENDER,
                'b': None  # Special key for using basic strategy
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
                
            # Add basic strategy option
            valid_inputs.append('b')
            action_display.append("(B)asic Strategy")
                
            # Get AI's recommendation
            if self.show_ai_reasoning and self.model_loaded:
                self._show_ai_recommendation(valid_actions)
            
            # Show condensed game summary right above the action section
            state = self.env._get_state()
            player_hand = state['player_hand']
            dealer_upcard = state['dealer_upcard']
            
            print("\n┌───────────────────────────────────────────────────┐")
            print("│                  CURRENT SITUATION                │")
            print("├───────────────────────────────────────────────────┤")
            if self.env.dealer_hand.cards[1:]:  # If dealer's second card is visible
                dealer_cards = ", ".join(str(card) for card in self.env.dealer_hand.cards)
                print(f"│  🎩 Dealer: {dealer_cards.ljust(35)} │")
                print(f"│     Value: {str(self.env.dealer_hand.value).ljust(38)} │")
            else:
                print(f"│  🎩 Dealer shows: {str(dealer_upcard) + ' [?]'.ljust(30)} │")
            
            player_cards = ", ".join(str(card) for card in player_hand.cards)
            print(f"│  👤 Your hand: {player_cards.ljust(33)} │")
            print(f"│     Value: {str(player_hand.value).ljust(38)} │")
            print("└───────────────────────────────────────────────────┘")
                
            # Get player's action
            print("\n┌───────────────────────────────────────┐")
            print("│           AVAILABLE ACTIONS           │")
            print("└───────────────────────────────────────┘")
            print(" | ".join(action_display))
            action_input = input("\n👉 Your action: ").lower()
            
            # Special handling for basic strategy
            if action_input == 'b':
                # Get the current state
                state = self.env._get_state()
                player_hand = state['player_hand']
                dealer_upcard = state['dealer_upcard']
                
                # Get recommendation
                recommended_action = self._get_basic_strategy_recommendation(player_hand, dealer_upcard)
                
                # Make sure it's valid
                if recommended_action not in valid_actions:
                    if recommended_action == Action.DOUBLE and Action.HIT in valid_actions:
                        recommended_action = Action.HIT
                        print(f"\n✓ Basic strategy recommends DOUBLE, but that's not available now.")
                        print(f"   Using HIT instead as per basic strategy guidelines.")
                    elif recommended_action == Action.SPLIT and Action.HIT in valid_actions:
                        recommended_action = Action.HIT
                        print(f"\n✓ Basic strategy recommends SPLIT, but that's not available now.")
                        print(f"   Using HIT instead as per basic strategy guidelines.")
                    else:
                        valid_action = list(valid_actions)[0]
                        print(f"\n✓ Basic strategy recommends {recommended_action.name}, but that's not available.")
                        print(f"   Using {valid_action.name} instead.")
                        recommended_action = valid_action
                
                print(f"\n✓ Using basic strategy: {recommended_action.name}")
                action = recommended_action
            else:
                # Make sure it's a valid input
                while action_input not in valid_inputs or (action_input == 'b' and len(valid_actions) == 0):
                    if action_input == 'b' and len(valid_actions) == 0:
                        print("No valid actions available for basic strategy to choose from.")
                    elif action_input not in valid_inputs:
                        print(f"Invalid action. Choose from: {', '.join(action_display)}")
                    action_input = input("👉 Your action: ").lower()
                
                # If it's not basic strategy, execute the action directly
                if action_input != 'b':
                    action = action_map[action_input]
                else:
                    # Handle basic strategy again (this shouldn't happen but just in case)
                    state = self.env._get_state()
                    recommended_action = self._get_basic_strategy_recommendation(
                        state['player_hand'], state['dealer_upcard']
                    )
                    if recommended_action in valid_actions:
                        action = recommended_action
                    else:
                        action = list(valid_actions)[0]
            
            print(f"\n👉 You chose to {action.name.lower()}.")
            
            # Take the action
            _, reward, done = self.env.step(action)
            
            # Check if player busted
            if self.env.player_hands[self.env.current_hand_idx-1].is_bust:
                print("💥 Bust! You went over 21.")
                done = True
                
            # Check if player hit exactly 21 after an action
            elif self.env.player_hands[self.env.current_hand_idx-1].value == 21:
                print("\n🎉 You have 21! Perfect hand!")
                done = True
                
            time.sleep(1)
            
    def _get_basic_strategy_recommendation(self, player_hand, dealer_upcard):
        """Get recommendation based on basic blackjack strategy."""
        # Convert dealer's card value (Ace = 11, face cards = 10)
        dealer_value = dealer_upcard.value
        
        # Check for pairs
        if len(player_hand.cards) == 2 and player_hand.cards[0].rank == player_hand.cards[1].rank:
            rank = player_hand.cards[0].rank
            # Pair strategy
            if rank == 'A':
                return Action.SPLIT  # Always split Aces
            elif rank == '8':
                return Action.SPLIT  # Always split 8s
            elif rank in ['2', '3', '6', '7', '9']:
                # Split against dealer 2-6, otherwise hit (except 9s)
                if 2 <= dealer_value <= 6:
                    return Action.SPLIT
                elif rank == '9' and dealer_value in [7, 10, 11]:
                    return Action.STAND
                else:
                    return Action.HIT
            elif rank == '4':
                # Split 4s only against 5-6, otherwise hit
                if 5 <= dealer_value <= 6:
                    return Action.SPLIT
                else:
                    return Action.HIT
            elif rank == '5':
                # Never split 5s, double against 2-9, otherwise hit
                if 2 <= dealer_value <= 9:
                    return Action.DOUBLE
                else:
                    return Action.HIT
            elif rank == '10':
                return Action.STAND  # Never split 10s
                
        # Soft totals (hand with an Ace counted as 11)
        if player_hand.is_soft:
            soft_total = player_hand.value
            if soft_total >= 19:
                return Action.STAND  # A,8+ always stand
            elif soft_total == 18:
                if dealer_value in [2, 7, 8]:
                    return Action.STAND
                elif 3 <= dealer_value <= 6:
                    return Action.DOUBLE if len(player_hand.cards) == 2 else Action.STAND
                else:
                    return Action.HIT
            elif soft_total == 17:
                if 3 <= dealer_value <= 6:
                    return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
                else:
                    return Action.HIT
            elif soft_total in [15, 16]:
                if 4 <= dealer_value <= 6:
                    return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
                else:
                    return Action.HIT
            elif soft_total in [13, 14]:
                if 5 <= dealer_value <= 6:
                    return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
                else:
                    return Action.HIT
                    
        # Hard totals
        hard_total = player_hand.value
        if hard_total >= 17:
            return Action.STAND
        elif hard_total >= 13:
            if dealer_value <= 6:
                return Action.STAND
            else:
                return Action.HIT
        elif hard_total == 12:
            if 4 <= dealer_value <= 6:
                return Action.STAND
            else:
                return Action.HIT
        elif hard_total == 11:
            return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
        elif hard_total == 10:
            if dealer_value <= 9:
                return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
            else:
                return Action.HIT
        elif hard_total == 9:
            if 3 <= dealer_value <= 6:
                return Action.DOUBLE if len(player_hand.cards) == 2 else Action.HIT
            else:
                return Action.HIT
        else:
            return Action.HIT  # 8 or less, always hit
            
    def _show_ai_recommendation(self, valid_actions):
        """Show AI's recommended action with basic strategy comparison."""
        # Get current game state
        state = self.env._get_state()
        player_hand = state['player_hand']
        dealer_upcard = state['dealer_upcard']
        
        # Get basic strategy recommendation
        basic_strategy_action = self._get_basic_strategy_recommendation(player_hand, dealer_upcard)
        
        # Ensure recommended action is valid
        if basic_strategy_action not in valid_actions:
            # Find closest alternative
            if basic_strategy_action == Action.DOUBLE and Action.HIT in valid_actions:
                basic_strategy_action = Action.HIT
            elif basic_strategy_action == Action.SPLIT and Action.HIT in valid_actions:
                basic_strategy_action = Action.HIT
            else:
                basic_strategy_action = valid_actions[0]  # Fallback to first valid action
        
        # Only get DQN recommendation if model is loaded
        if self.model_loaded:
            # Get player state vector for DQN
            state_vector = self.env.vectorize_state()
            
            # Get DQN recommendation
            dqn_action = self.agent.act(state_vector, valid_actions)
            
            # Display recommendations with improved formatting - CONDENSED VERSION
            print("\n┌────────────────────────────────┐")
            print("│       STRATEGY SUGGESTION      │")
            print("├────────────────────────────────┤")
            print(f"│  ✓ Basic: {basic_strategy_action.name.ljust(10)}         │")
            
            # Alert if there's a discrepancy
            if basic_strategy_action != dqn_action:
                print(f"│  🤖 AI:    {dqn_action.name.ljust(10)} (differs) │")
                print("│                                │")
                print("│  ⚠️  Consider basic strategy!   │")
            else:
                print(f"│  🤖 AI:    {dqn_action.name.ljust(10)} (agrees)  │")
                
        else:
            # Just show basic strategy - CONDENSED VERSION
            print("\n┌────────────────────────────────┐")
            print("│       STRATEGY SUGGESTION      │")
            print("├────────────────────────────────┤")
            print(f"│  ✓ Recommended: {basic_strategy_action.name.ljust(10)}   │")
            
        print("└────────────────────────────────┘")
        
    def _finish_hand(self):
        """Finish the hand (dealer's turn and payout)."""
        # Show all cards
        self._show_game_state(initial_deal=False)
        
        # Check if all player hands busted or surrendered
        all_busted = all(hand.is_bust or hand.is_surrender 
                       for hand in self.env.player_hands)
                       
        # Also check if all AI opponents busted
        all_ai_busted = True
        for opponent in self.ai_opponents:
            all_ai_busted = all_ai_busted and all(
                hand.is_bust or hand.is_surrender 
                for hand in opponent["env"].player_hands
            )
                       
        if not (all_busted and all_ai_busted):
            # Play dealer's turn
            print("\n🎲 Dealer's turn:")
            self.env._play_dealer_hand()
            
            # Apply the same dealer cards to all AI opponents' environments
            for opponent in self.ai_opponents:
                # Create a new dealer hand for each opponent with the same cards
                opponent_dealer_hand = opponent["env"].dealer_hand
                # Clear opponent's dealer hand
                opponent_dealer_hand.cards = []
                # Add each card from the main dealer's hand
                for card in self.env.dealer_hand.cards:
                    opponent_dealer_hand.add_card(card)
            
            # Show final state
            self._show_game_state(initial_deal=False)
            
            # Show dealer's final hand
            dealer_hand = self.env.dealer_hand
            if dealer_hand.is_bust:
                print("🎉 Dealer busts!")
                
        # Calculate rewards for human player
        rewards = self.env._calculate_rewards()
        total_reward = sum(rewards)
        
        # Update bankroll
        self.player_bankroll += total_reward
        
        # Show player result
        if total_reward > 0:
            if any(hand.is_blackjack for hand in self.env.player_hands):
                print(f"\n🎰 BLACKJACK! You won ${total_reward:.2f}!")
            else:
                original_bet = self.bet_amount
                print(f"\n🎉 You won! Your ${original_bet:.2f} bet returns ${total_reward:.2f}")
                print(f"(Your original bet + ${total_reward - original_bet:.2f} winnings)")
        elif total_reward < 0:
            print(f"\n😔 You lost ${abs(total_reward):.2f}.")
        else:
            print("\n🤝 Push (tie). Your bet is returned.")
            
        print(f"\n💰 Your bankroll is now ${self.player_bankroll:.2f}")
        
        # Calculate results for AI opponents
        print("\n🤖 AI OPPONENTS RESULTS:")
        print("┌─────────────────────────────────────────────────┐")
        
        for opponent in self.ai_opponents:
            # Calculate opponent's rewards
            opponent_rewards = opponent["env"]._calculate_rewards()
            opponent_total_reward = sum(opponent_rewards)
            
            # Update opponent bankroll
            opponent["bankroll"] += opponent_total_reward
            
            # Show result
            opponent_name = opponent["name"]
            if opponent_total_reward > 0:
                if any(hand.is_blackjack for hand in opponent["env"].player_hands):
                    print(f"│ 🎰 {opponent_name} got BLACKJACK and won ${opponent_total_reward:.2f}!")
                else:
                    print(f"│ 🎉 {opponent_name} won ${opponent_total_reward:.2f}")
            elif opponent_total_reward < 0:
                print(f"│ 😔 {opponent_name} lost ${abs(opponent_total_reward):.2f}")
            else:
                print(f"│ 🤝 {opponent_name} pushed (tie)")
                
            print(f"│   Bankroll now: ${opponent['bankroll']:.2f}")
            
        print("└─────────────────────────────────────────────────┘")
        time.sleep(2)  # Give player time to read results
        
    def _play_again(self):
        """Ask if player wants to play another hand."""
        if self.player_bankroll <= 0:
            print("\n💸 You're out of money! Game over.")
            return False
            
        # Add a small delay
        time.sleep(0.5)
        
        while True:
            choice = input("\n🎮 Play another hand? (Y/N): ").lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter Y or N.")
                
    def _print_welcome(self):
        """Print welcome message."""
        self._clear_screen()
        print("\n┌─────────────────────────────────────────────────┐")
        print("│             🎲 BLACKJACK AI CASINO 🎲            │")
        print("└─────────────────────────────────────────────────┘")
        print("\nWelcome to the ultimate AI Blackjack experience!")
        
        # Show AI opponents
        print(f"\n🤖 Today you'll be playing against {len(self.ai_opponents)} AI players:")
        for i, opponent in enumerate(self.ai_opponents):
            print(f"  {i+1}. {opponent['name']} (Strategy: {opponent['strategy']})")
            
        print(f"\nYou're playing against a reinforcement learning trained dealer.")
        print(f"Starting bankroll: ${self.player_bankroll:.2f}")
        
        if self.model_loaded:
            print("\n🧠 Your advisor AI is using optimal strategy based on its training.")
            print("   But we recommend Basic Strategy for most reliable results!")
        else:
            print("\n⚠️  No trained model loaded. Your advisor AI will make random decisions.")
            print("   We strongly recommend using the Basic Strategy option!")
            
        print("\n🎮 Game Controls:")
        print("  • (H)it - Take another card")
        print("  • (S)tand - End your turn")
        print("  • (D)ouble - Double your bet and take one more card")
        print("  • S(p)lit - Split eligible pairs into two hands")
        print("  • Su(r)render - Give up half your bet")
        print("  • (B)asic Strategy - Let the proven optimal strategy decide")
        
        print("\n💰 Payouts:")
        print("  • Win: 1:1 (double your bet)")
        print("  • Blackjack: 3:2 (bet $10, win $15)")
        print("  • Push: Your bet is returned")
        
        input("\nPress Enter to start playing...")
        
    def _print_goodbye(self):
        """Print goodbye message."""
        self._clear_screen()
        print("\n┌─────────────────────────────────────────────────┐")
        print("│            🎲 BLACKJACK SESSION OVER 🎲          │")
        print("└─────────────────────────────────────────────────┘")
        
        print(f"\n💰 Final bankroll: ${self.player_bankroll:.2f}")
        
        initial_bankroll = BETTING['initial_bankroll']
        if self.player_bankroll > initial_bankroll:
            profit = self.player_bankroll - initial_bankroll
            print(f"🎉 You made a profit of ${profit:.2f}!")
            if profit > initial_bankroll * 0.5:
                print("🔥 Impressive winning streak! The casino is watching you...")
        elif self.player_bankroll < initial_bankroll:
            loss = initial_bankroll - self.player_bankroll
            print(f"😔 You lost ${loss:.2f}.")
            if self.player_bankroll == 0:
                print("💸 Looks like you're out of chips! Better luck next time.")
        else:
            print("🤝 You broke even. Not bad!")
            
        print("\n👋 Thanks for playing! Come back soon!")
        print("   Remember: The house always wins... eventually.\n")
        
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _ai_opponent_turn(self, opponent):
        """Handle the turn for an AI opponent."""
        opponent_env = opponent["env"]
        opponent_state = opponent_env._get_state()
        opponent_hand = opponent_state['player_hand']
        
        # Skip if opponent has blackjack or busted
        if opponent_hand.is_blackjack or opponent_hand.is_bust:
            return
        
        # Set opponent's bet
        opponent_env.set_bet(opponent["bet"])
        
        # Show opponent's hand
        print(f"\n🤖 {opponent['name']}'s turn (Strategy: {opponent['strategy']})")
        print(f"Cards: {', '.join(str(card) for card in opponent_hand.cards)}")
        print(f"Value: {opponent_hand.value}")
        
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = opponent_env.get_valid_actions()
            
            # Determine action based on strategy
            if opponent["strategy"] == "basic":
                # Use basic strategy
                action = self._get_basic_strategy_recommendation(
                    opponent_hand, opponent_state['dealer_upcard']
                )
                # Ensure action is valid
                if action not in valid_actions:
                    # Fallback to hit or stand
                    if Action.HIT in valid_actions:
                        action = Action.HIT
                    else:
                        action = Action.STAND
            elif opponent["strategy"] == "dqn":
                # Use DQN model
                state_vector = opponent_env.vectorize_state()
                action = opponent["agent"].act(state_vector, valid_actions)
            else:
                # Random strategy
                action = random.choice(list(valid_actions))
                
            # Take action
            print(f"🤖 {opponent['name']} chooses to {action.name.lower()}")
            _, reward, done = opponent_env.step(action)
            
            # Show updated hand if not done
            if not done:
                updated_state = opponent_env._get_state()
                updated_hand = updated_state['player_hand']
                print(f"New hand: {', '.join(str(card) for card in updated_hand.cards)}")
                print(f"New value: {updated_hand.value}")
            
            # Check for bust
            if opponent_env.player_hands[opponent_env.current_hand_idx-1].is_bust:
                print(f"💥 {opponent['name']} busts!")
                done = True
                
            # Check for 21
            elif opponent_env.player_hands[opponent_env.current_hand_idx-1].value == 21:
                print(f"🎯 {opponent['name']} has 21!")
                done = True
                
            time.sleep(0.5)  # Pause for effect


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