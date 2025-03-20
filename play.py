import os
import sys
import time
from datetime import datetime, timedelta

import colorama
import numpy as np
from colorama import Back, Fore, Style

from alpha_zero import AlphaZero

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)


class ThreeMensMorris:
    """Implementation of 成三棋 (Three Men's Morris) game for AlphaZero"""

    def __init__(self):
        # Board has 24 possible positions
        self.num_positions = 24
        # Action space:
        # During placement: place a piece at one of 24 positions
        # (encoded as 0-23)
        # During movement: move from one of 24 positions to an adjacent position
        # (encoded as 24 to 24*24+23)
        self.action_size = self.num_positions + (
            self.num_positions * self.num_positions
        )
        self.pieces_per_player = 9
        self.max_steps = 100  # Max steps to avoid infinite games
        self.is_two_player = True

        # For compatibility with AlphaZero play_game method
        self.board_size = 5  # We're using a 5x5 grid

        # Board representation: 24 positions with potential connections
        # First represent the adjacency list for valid moves
        self.adjacent_positions = {
            # Outer square (Indices: 0,1,2,3,4,5,6,7,8,12,13,14)
            0: [1, 3],
            1: [0, 2, 9],
            2: [1, 5],
            3: [0, 4, 10],
            4: [3, 5],
            5: [2, 4, 13],
            6: [7, 11],
            7: [6, 8, 14],
            8: [7, 12],
            12: [8, 13],
            13: [5, 12, 14],
            14: [7, 13],
            # Middle square (Top: 9,10,11; Bottom: 21,22,23)
            9: [10, 15],
            10: [9, 11, 16],
            11: [10, 17],
            21: [22, 18],
            22: [21, 23, 19],
            23: [22, 20],
            # Inner square (Indices: 15,16,17,18,19,20)
            15: [16, 9],
            16: [15, 17, 10],
            17: [16, 11],
            18: [19, 21],
            19: [18, 20, 22],
            20: [19, 23],
        }

        # Define the lines (mill combinations) where three pieces can form a "mill"
        self.mill_combinations = [
            # Horizontal lines - outer square
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [12, 13, 14],
            # Horizontal lines - middle square
            [9, 10, 11],
            [21, 22, 23],
            # Horizontal lines - inner square
            [15, 16, 17],
            [18, 19, 20],
            # Vertical lines - outer
            [0, 3, 12],
            [1, 10, 13],
            [2, 5, 14],
            [8, 7, 12],
            # Vertical lines - connecting squares
            [3, 4, 12],
            [4, 9, 21],
            [5, 11, 23],
            [6, 18, 15],
            [7, 19, 16],
            [8, 20, 17],
        ]

        # State representation:
        # Channel 0: Current player's pieces (1 = piece present)
        # Channel 1: Opponent's pieces (1 = piece present)
        # Channel 2: Blocked positions during placement phase (1 = blocked)
        # Channel 3: Phase indicator (1 = placement phase, 0 = movement phase)
        # Channel 4: Current player's pieces remaining to place
        # Channel 5: Opponent's pieces remaining to place
        self.observation_shape = (
            6,
            5,
            5,
        )  # We'll use a 5x5 grid to represent the board

        # Mapping from linear positions (0-23) to 2D positions on 5x5 grid
        self.pos_to_2d = {
            # Outer square
            0: (0, 0),
            1: (0, 2),
            2: (0, 4),
            3: (2, 0),
            4: (2, 2),
            5: (2, 4),
            6: (4, 0),
            7: (4, 2),
            8: (4, 4),
            # Middle square
            9: (1, 1),
            10: (1, 2),
            11: (1, 3),
            12: (3, 1),
            13: (3, 2),
            14: (3, 3),
            # Inner square
            15: (2, 1),
            16: (2, 3),
            17: (3, 0),
            18: (1, 0),
            19: (1, 4),
            20: (3, 4),
            21: (4, 1),
            22: (4, 3),
            23: (0, 1),
        }

        # Reverse mapping from 2D to linear positions
        self.grid_2d_to_pos = {v: k for k, v in self.pos_to_2d.items()}

    def get_initial_state(self):
        """Return initial game state"""
        state = np.zeros(self.observation_shape, dtype=np.float32)

        # Set initial phase to placement phase
        state[3] = np.ones((5, 5), dtype=np.float32)

        # Set initial pieces to place
        state[4] = np.full((5, 5), self.pieces_per_player, dtype=np.float32)
        state[5] = np.full((5, 5), self.pieces_per_player, dtype=np.float32)

        return state

    def get_valid_moves(self, state):
        """Return valid moves as a binary mask"""
        valid_moves = np.zeros(self.action_size, dtype=np.int8)

        # Extract board state
        player_pieces = state[0]
        opponent_pieces = state[1]
        blocked_positions = state[2]
        is_placement_phase = np.any(state[3])
        player_pieces_to_place = state[4][0, 0]

        # Convert 2D representation to linear for easier processing
        player_pos = set()
        opponent_pos = set()
        blocked_pos = set()

        for pos, (i, j) in self.pos_to_2d.items():
            if player_pieces[i, j] == 1:
                player_pos.add(pos)
            if opponent_pieces[i, j] == 1:
                opponent_pos.add(pos)
            if blocked_positions[i, j] == 1:
                blocked_pos.add(pos)

        if is_placement_phase:
            # Placement phase: can place pieces on any empty, unblocked position
            if player_pieces_to_place > 0:
                for pos in range(self.num_positions):
                    if (
                        pos not in player_pos
                        and pos not in opponent_pos
                        and pos not in blocked_pos
                    ):
                        # Direct placement actions (0-23)
                        valid_moves[pos] = 1
        else:
            # Movement phase: can move pieces to adjacent empty positions
            for from_pos in player_pos:
                for to_pos in self.adjacent_positions[from_pos]:
                    if to_pos not in player_pos and to_pos not in opponent_pos:
                        # Movement actions (24 to 24*24+23)
                        movement_action = self.num_positions + (
                            from_pos * self.num_positions + to_pos
                        )
                        valid_moves[movement_action] = 1

        return valid_moves

    def _has_mill(self, positions, pos):
        """Check if a position is part of a mill"""
        for mill in self.mill_combinations:
            if pos in mill and all(p in positions for p in mill):
                return True
        return False

    def _get_new_mills(self, old_positions, new_positions, pos):
        """Check if a new mill was formed by adding a piece at pos"""
        if pos not in new_positions:
            return False

        # Track all mills before and after the move
        old_mills = set()
        new_mills = set()

        # Get all mills before the move
        for mill in self.mill_combinations:
            if all(p in old_positions for p in mill):
                old_mills.add(tuple(sorted(mill)))

        # Get all mills after the move
        for mill in self.mill_combinations:
            if pos in mill and all(p in new_positions for p in mill):
                mill_tuple = tuple(sorted(mill))
                if mill_tuple not in old_mills:
                    new_mills.add(mill_tuple)

        # Return True only if there are genuinely new mills
        return len(new_mills) > 0

    def _can_remove_piece(self, state, pos):
        """Check if opponent's piece at pos can be removed (not part of a mill)"""
        # Convert 2D representation to linear
        opponent_pos = set()
        for p, (i, j) in self.pos_to_2d.items():
            if state[1, i, j] == 1:
                opponent_pos.add(p)

        # If position isn't occupied by opponent, can't remove
        if pos not in opponent_pos:
            return False

        # Can't remove piece that's in a mill, unless all opponents are in mills
        if self._has_mill(opponent_pos, pos):
            # Check if ALL opponent pieces are in mills
            all_in_mills = True
            for opp_pos in opponent_pos:
                if not self._has_mill(opponent_pos, opp_pos):
                    all_in_mills = False
                    break
            return all_in_mills

        return True

    def step(self, state, action):
        """Execute action and return next state, reward, done, and info"""
        # Initialize move counter if not exists
        if not hasattr(self, 'move_counter'):
            self.move_counter = 0
        self.move_counter += 1

        # Copy state to avoid modifying the original
        next_state = state.copy()

        # Extract board state
        player_pieces = next_state[0].copy()
        opponent_pieces = next_state[1].copy()
        blocked_positions = next_state[2].copy()
        is_placement_phase = np.any(next_state[3])
        player_pieces_to_place = next_state[4][0, 0]
        opponent_pieces_to_place = next_state[5][0, 0]

        # Convert to linear representation for easier processing
        player_pos = set()
        opponent_pos = set()
        for pos, (i, j) in self.pos_to_2d.items():
            if player_pieces[i, j] == 1:
                player_pos.add(pos)
            if opponent_pieces[i, j] == 1:
                opponent_pos.add(pos)

        # Record position before the move
        old_player_pos = player_pos.copy()
        to_pos = None

        # Decode action based on phase
        if is_placement_phase:
            # Placement phase: action is the position to place (0-23)
            to_pos = action
            i, j = self.pos_to_2d[to_pos]
            player_pieces[i, j] = 1
            player_pos.add(to_pos)

            # Decrement pieces to place
            next_state[4] = np.full((5, 5), player_pieces_to_place - 1)

            # Check for mill formation
            mill_formed = self._get_new_mills(old_player_pos, player_pos, to_pos)

            # Handle mill formation (capture opponent's piece)
            if mill_formed and opponent_pos:
                # We choose a piece that's not part of a mill if possible
                capturable_pieces = []
                for opp_pos in opponent_pos:
                    if self._can_remove_piece(next_state, opp_pos):
                        capturable_pieces.append(opp_pos)

                if capturable_pieces:
                    capture_pos = capturable_pieces[0]
                    i, j = self.pos_to_2d[capture_pos]
                    opponent_pieces[i, j] = 0
                    opponent_pos.remove(capture_pos)
                    next_state[1] = opponent_pieces

                    if is_placement_phase:
                        blocked_positions[i, j] = 1
                        next_state[2] = blocked_positions

                    # Only print mill formation messages when not in training
                    if not hasattr(self, 'is_training'):
                        print(
                            f"{Fore.MAGENTA}Mill formed! {Fore.RED}X{Fore.MAGENTA} captured "
                            f"{Fore.BLUE}O{Fore.MAGENTA} at position {capture_pos}{Style.RESET_ALL}"
                        )

            # Check if placement phase is complete
            if player_pieces_to_place - 1 <= 0 and opponent_pieces_to_place <= 0:
                # Transition to movement phase
                next_state[3] = np.zeros((5, 5))
                # Clear blocked positions
                next_state[2] = np.zeros((5, 5))
        else:
            # Movement phase: action is movement action (24 to 24*24+23)
            movement_action = action - self.num_positions
            from_pos = movement_action // self.num_positions
            to_pos = movement_action % self.num_positions

            i_from, j_from = self.pos_to_2d[from_pos]
            i_to, j_to = self.pos_to_2d[to_pos]

            # Move piece
            player_pieces[i_from, j_from] = 0
            player_pieces[i_to, j_to] = 1
            player_pos.remove(from_pos)
            player_pos.add(to_pos)

        # Update state with new pieces
        next_state[0] = player_pieces

        # Check for win/loss conditions
        is_terminal, reward = self.is_terminal(next_state)

        if not is_terminal:
            # Swap players by swapping the channels
            next_state = self._swap_players(next_state)

        # Reset counter on game end
        if is_terminal:
            self.move_counter = 0

        # Add debug output only if debug_mode is True AND we're not in training
        if hasattr(self, 'debug_mode') and self.debug_mode and not hasattr(self, 'is_training'):
            print(f"\nMove {self.move_counter}")
            print(f"Phase: {'Placement' if is_placement_phase else 'Movement'}")
            if mill_formed and opponent_pos:
                print(
                    f"{Fore.MAGENTA}Mill formed! {Fore.RED}X{Fore.MAGENTA} captured "
                    f"{Fore.BLUE}O{Fore.MAGENTA} at position {capture_pos}{Style.RESET_ALL}"
                )
            self.render(state)

        return next_state, next_state, is_terminal, reward

    def _swap_players(self, state):
        """Swap player and opponent perspectives"""
        new_state = state.copy()

        # Swap piece positions
        new_state[0], new_state[1] = state[1].copy(), state[0].copy()

        # Swap pieces to place
        new_state[4], new_state[5] = state[5].copy(), state[4].copy()

        return new_state

    def is_terminal(self, state):
        """Check if the game is over and return reward"""
        # Extract board state
        player_pieces = state[0]
        opponent_pieces = state[1]
        is_placement_phase = np.any(state[3])

        # Count pieces
        player_pos = set()
        opponent_pos = set()
        for pos, (i, j) in self.pos_to_2d.items():
            if player_pieces[i, j] == 1:
                player_pos.add(pos)
            if opponent_pieces[i, j] == 1:
                opponent_pos.add(pos)

        player_piece_count = len(player_pos)
        opponent_piece_count = len(opponent_pos)

        # If a player has fewer than 3 pieces in movement phase, they lose
        if not is_placement_phase:
            if player_piece_count < 3:
                return True, -1.0
            if opponent_piece_count < 3:
                return True, 1.0

            # Check if current player has no valid moves
            valid_moves_exist = False
            for from_pos in player_pos:
                for to_pos in self.adjacent_positions[from_pos]:
                    if to_pos not in player_pos and to_pos not in opponent_pos:
                        valid_moves_exist = True
                        break
                if valid_moves_exist:
                    break

            if not valid_moves_exist:
                return True, -1.0

        # Check for maximum moves exceeded (to prevent infinite games)
        if hasattr(self, 'move_counter'):
            if self.move_counter >= self.max_steps:
                return True, 0.0  # Draw if max moves reached

        return False, 0.0

    def get_draw_reward(self):
        """Reward for a draw"""
        return 0.0

    def render(self, state):
        """Display the board in a more traditional layout matching the actual game"""
        # Extract board state
        player_pieces = state[0]
        opponent_pieces = state[1]
        blocked_positions = state[2]
        is_placement_phase = np.any(state[3])
        player_pieces_to_place = state[4][0, 0]
        opponent_pieces_to_place = state[5][0, 0]

        # Create a symbol map for pieces with colors
        board_symbols = {}
        for pos, (i, j) in self.pos_to_2d.items():
            if player_pieces[i, j] == 1:
                board_symbols[pos] = f"{Fore.RED}X{Style.RESET_ALL}"
            elif opponent_pieces[i, j] == 1:
                board_symbols[pos] = f"{Fore.BLUE}O{Style.RESET_ALL}"
            elif blocked_positions[i, j] == 1:
                board_symbols[pos] = f"{Fore.YELLOW}×{Style.RESET_ALL}"
            else:
                board_symbols[pos] = f"{Fore.WHITE}•{Style.RESET_ALL}"

        # Revised ASCII art with strict vertical alignment
        print()
        print(f" {board_symbols[0]}───────────────{board_symbols[1]}───────────────{board_symbols[2]}")
        print(" │               │               │")
        print(f" │     {board_symbols[9]}─────────{board_symbols[10]}─────────{board_symbols[11]}     │")
        print(" │     │         │         │     │")
        print(f" │     │     {board_symbols[15]}───{board_symbols[16]}───{board_symbols[17]}     │     │")
        print(" │     │     │       │     │     │")
        print(f" {board_symbols[3]}─────{board_symbols[4]}─────{board_symbols[5]}       {board_symbols[6]}─────{board_symbols[7]}─────{board_symbols[8]}")
        print(" │     │     │       │     │     │")
        print(f" │     │     {board_symbols[18]}───{board_symbols[19]}───{board_symbols[20]}     │     │")
        print(" │     │         │         │     │")
        print(f" │     {board_symbols[21]}─────────{board_symbols[22]}─────────{board_symbols[23]}     │")
        print(" │               │               │")
        print(f" {board_symbols[12]}───────────────{board_symbols[13]}───────────────{board_symbols[14]}")
        print()

        # Print reference board only the first time
        if not hasattr(self, "_board_shown"):
            self._print_reference_board()
            self._board_shown = True

        # Print game state info
        if is_placement_phase:
            print(
                f"Phase: {Fore.GREEN}PLACEMENT{Style.RESET_ALL} - Pieces to place: "
                f"{Fore.RED}X:{int(player_pieces_to_place)}{Style.RESET_ALL} "
                f"{Fore.BLUE}O:{int(opponent_pieces_to_place)}{Style.RESET_ALL}"
            )
        else:
            print(f"Phase: {Fore.GREEN}MOVEMENT{Style.RESET_ALL}")

        # Count pieces on board
        player_count = int(np.sum(player_pieces))
        opponent_count = int(np.sum(opponent_pieces))
        print(
            f"Pieces on board: {Fore.RED}X:{player_count}{Style.RESET_ALL} "
            f"{Fore.BLUE}O:{opponent_count}{Style.RESET_ALL}"
        )
        print()

    def _print_reference_board(self):
        """Print the board position reference"""
        print(f"{Fore.CYAN}Board positions reference:{Style.RESET_ALL}")
        print(" 0────────────────1────────────────2")
        print(" │                │                │")
        print(" │      9────────10──────────11    │")
        print(" │      │         │          │     │")
        print(" │      │   15───16────17    │     │")
        print(" │      │   │           │    │     │")
        print(" 3──────4───5           6────7─────8")
        print(" │      │   │           │    │     │")
        print(" │      │   18────19───20    │     │")
        print(" │      │         │          │     │")
        print(" │      21────────22────────23     │")
        print(" │                │                │")
        print(" 12───────────────13──────────────14")
        print()

    def reset(self):
        """Reset the game state"""
        self.move_counter = 0
        self._board_shown = False  # Reset board reference flag
        return self.get_initial_state()


class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        valid_moves = self.game.get_valid_moves(state)

        print(f"\n{Fore.CYAN}=== YOUR TURN ==={Style.RESET_ALL}")
        self.game.render(state)

        # Extract game phase
        is_placement_phase = np.any(state[3])

        if is_placement_phase:
            return self._handle_placement_phase(state, valid_moves)
        else:
            return self._handle_movement_phase(state, valid_moves)

    def _handle_placement_phase(self, state, valid_moves):
        """Handle user input during the placement phase"""
        while True:
            try:
                print(
                    f"Enter the position number (0-23) where you want to place your piece:"
                )
                position = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

                # Allow the player to see the position reference again
                if position.lower() in ["h", "help", "?"]:
                    print("\nBoard position reference:")
                    self._print_position_reference()
                    continue

                position = int(position)
                if not (0 <= position < self.game.num_positions):
                    print(
                        f"{Fore.YELLOW}Invalid position! Please enter a number between 0 and 23.{Style.RESET_ALL}"
                    )
                    continue

                # Placement actions are directly the position (0-23)
                action = position

                if valid_moves[action]:
                    i, j = self.game.pos_to_2d[position]
                    print(
                        f"{Fore.GREEN}You placed a piece at position {position}{Style.RESET_ALL}"
                    )
                    return action
                else:
                    print(
                        f"{Fore.YELLOW}Invalid move! This position is already occupied or blocked.{Style.RESET_ALL}"
                    )
            except ValueError:
                print(
                    f"{Fore.YELLOW}Please enter a valid number or 'h' for help.{Style.RESET_ALL}"
                )

    def _handle_movement_phase(self, state, valid_moves):
        """Handle user input during the movement phase"""
        while True:
            try:
                print("Enter the position number (0-23) of the piece you want to move:")
                from_pos = input(f"{Fore.CYAN}From > {Style.RESET_ALL}")

                # Allow the player to see the position reference again
                if from_pos.lower() in ["h", "help", "?"]:
                    print("\nBoard position reference:")
                    self._print_position_reference()
                    continue

                from_pos = int(from_pos)
                if not (0 <= from_pos < self.game.num_positions):
                    print(
                        f"{Fore.YELLOW}Invalid position! Please enter a number between 0 and 23.{Style.RESET_ALL}"
                    )
                    continue

                print("Enter the position number (0-23) where you want to move to:")
                to_pos = input(f"{Fore.CYAN}To > {Style.RESET_ALL}")

                if to_pos.lower() in ["h", "help", "?"]:
                    print("\nBoard position reference:")
                    self._print_position_reference()
                    continue

                to_pos = int(to_pos)
                if not (0 <= to_pos < self.game.num_positions):
                    print(
                        f"{Fore.YELLOW}Invalid position! Please enter a number between 0 and 23.{Style.RESET_ALL}"
                    )
                    continue

                # Movement actions are offset by num_positions
                action = self.game.num_positions + (
                    from_pos * self.game.num_positions + to_pos
                )

                if valid_moves[action]:
                    print(
                        f"{Fore.GREEN}You moved from position {from_pos} to position {to_pos}{Style.RESET_ALL}"
                    )
                    return action
                else:
                    # Check if the piece exists
                    player_pos = set()
                    for pos, (i, j) in self.game.pos_to_2d.items():
                        if state[0, i, j] == 1:
                            player_pos.add(pos)

                    if from_pos not in player_pos:
                        print(
                            f"{Fore.YELLOW}You don't have a piece at that position!{Style.RESET_ALL}"
                        )
                    elif to_pos not in self.game.adjacent_positions[from_pos]:
                        print(
                            f"{Fore.YELLOW}Invalid move! You can only move to adjacent positions.{Style.RESET_ALL}"
                        )
                    else:
                        print(
                            f"{Fore.YELLOW}Invalid move! The destination position is already occupied.{Style.RESET_ALL}"
                        )
            except ValueError:
                print(
                    f"{Fore.YELLOW}Please enter a valid number or 'h' for help.{Style.RESET_ALL}"
                )

    def _print_position_reference(self):
        """Print the board position reference"""
        print(f"{Fore.CYAN}Board positions reference:{Style.RESET_ALL}")
        print(" 0──────────────1──────────────────2")
        print(" │              │                 │")
        print(" │      9───────10────────11      │")
        print(" │      │       │          │      │")
        print(" │      │   15───16───17   │      │")
        print(" │      │   │          │   │      │")
        print(" 3──────4───5          6───7────8")
        print(" │      │   │          │   │      │")
        print(" │      │   18───19───20   │      │")
        print(" │      │       │          │      │")
        print(" │      21──────22─────────23     │")
        print(" │              │                 │")
        print(" 12─────────────13───────────────14")
        print()


class CustomAlphaZero(AlphaZero):
    """Extended AlphaZero with custom play_game method for Three Men's Morris"""

    def play_game(self, opponent=None, render=False):
        """Play a game against an opponent (or against itself if opponent is None)"""
        self.neural_net.eval()

        # Initialize MCTS
        from alpha_zero import MCTS
        mcts = MCTS(self.game, self.neural_net, **self.mcts_config)

        # Initialize game state
        state = self.game.get_initial_state()
        done = False
        turn = 0
        move_count = 0
        game_start = time.time()

        print(f"\n{Fore.CYAN}" + "-" * 30)
        print("Game starting")
        print("-" * 30 + f"{Style.RESET_ALL}")

        # Show initial board state
        if render:
            self.game.render(state)

        while not done:
            move_count += 1
            move_start = time.time()

            # AI's turn
            if turn == 0:
                print(
                    f"\n{Fore.CYAN}[Move {move_count}] {Fore.RED}Player X (AI){Fore.CYAN} thinking...{Style.RESET_ALL}"
                )
                time.sleep(0.5)  # Short delay for better readability

                pi, _ = mcts.search(state, temperature=0.01)
                action = np.argmax(pi)

                # Show action in human-readable form
                if isinstance(self.game, ThreeMensMorris):
                    is_placement_phase = np.any(state[3])
                    if is_placement_phase:
                        to_pos = action
                        print(
                            f"{Fore.RED}Player X (AI) places a piece at position {to_pos}{Style.RESET_ALL}"
                        )
                    else:
                        movement_action = action - self.game.num_positions
                        from_pos = movement_action // self.game.num_positions
                        to_pos = movement_action % self.game.num_positions
                        print(
                            f"{Fore.RED}Player X (AI) moves from position {from_pos} to {to_pos}{Style.RESET_ALL}"
                        )

            # Opponent's turn (AI vs AI in self-play)
            else:
                print(
                    f"\n{Fore.CYAN}[Move {move_count}] {Fore.BLUE}Player O (AI){Fore.CYAN} thinking...{Style.RESET_ALL}"
                )
                if opponent:
                    action = opponent.get_action(state)
                else:
                    time.sleep(0.5)  # Short delay for better readability
                    pi, _ = mcts.search(state, temperature=0.01)
                    action = np.argmax(pi)

                    # Show action in human-readable form
                    if isinstance(self.game, ThreeMensMorris):
                        is_placement_phase = np.any(state[3])
                        if is_placement_phase:
                            to_pos = action
                            print(
                                f"{Fore.BLUE}Player O (AI) places a piece at position {to_pos}{Style.RESET_ALL}"
                            )
                        else:
                            movement_action = action - self.game.num_positions
                            from_pos = movement_action // self.game.num_positions
                            to_pos = movement_action % self.game.num_positions
                            print(
                                f"{Fore.BLUE}Player O (AI) moves from position {from_pos} to {to_pos}{Style.RESET_ALL}"
                            )

            # Execute action
            next_state, _, done, reward = self.game.step(state, action)
            state = next_state

            # Print move time
            move_time = time.time() - move_start
            print(f"{Fore.CYAN}Move took: {move_time:.1f}s{Style.RESET_ALL}")

            # Always render the board after each move in self-play
            if render:
                self.game.render(state)

            # Switch turns (0 = player 1, 1 = player 2)
            turn = 1 - turn

            # Add a pause between moves in self-play
            if not done and opponent is None:
                input(f"{Fore.CYAN}Press Enter to continue to the next move... {Style.RESET_ALL}")

        # Show final board state and game statistics
        if render:
            print(f"\n{Fore.CYAN}Final board state:{Style.RESET_ALL}")
            self.game.render(state)
            
        game_time = time.time() - game_start
        print(f"\n{Fore.CYAN}Game Statistics:{Style.RESET_ALL}")
        print(f"├─ Total moves: {move_count}")
        print(f"└─ Total time: {timedelta(seconds=int(game_time))}")

        return reward


def main():
    # Initialize game
    game = ThreeMensMorris()

    # Print welcome message
    print(f"\n{Fore.CYAN}" + "=" * 60)
    print(
        f"{Fore.WHITE}{Back.BLUE}   THREE MEN'S MORRIS (成三棋) - ALPHAZERO IMPLEMENTATION   {Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}" + "=" * 60 + f"{Style.RESET_ALL}")
    print(f"""
{Fore.WHITE}Three Men's Morris is an ancient strategy game played on a grid with 24 points.
Each player has 9 pieces and tries to form "mills" (3 pieces in a row).

{Fore.YELLOW}Game Rules:{Style.RESET_ALL}
1. {Fore.GREEN}Placement Phase:{Style.RESET_ALL} Players take turns placing pieces until all pieces are placed.
   When a player forms a mill, they can remove one of the opponent's pieces.
2. {Fore.GREEN}Movement Phase:{Style.RESET_ALL} Players take turns moving pieces to adjacent positions.
   Movement continues until one player has fewer than 3 pieces or can't move.
3. A player wins by reducing the opponent to fewer than 3 pieces or blocking
   all their possible moves.

Type '{Fore.CYAN}h{Style.RESET_ALL}' at any input prompt to see the board position reference.
""")

    # Initialize AlphaZero with a smaller network for this game
    alphazero = CustomAlphaZero(
        game,
        model_config={"num_resblocks": 3, "num_hidden": 64},
        mcts_config={"num_simulations": 100, "c_puct": 2.0},
        num_iterations=5,
        num_self_play_games=20,
        num_epochs=5,
        replay_buffer_size=1000,
        batch_size=32,
    )

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print(f"{Fore.CYAN}Training AlphaZero on Three Men's Morris...{Style.RESET_ALL}")
            # Enable debug mode for training
            game.debug_mode = True
            alphazero.train()
            game.debug_mode = False
            # Save the final model
            alphazero._save_checkpoint("threemenmorris_final.pt")
            print(
                f"{Fore.GREEN}Training complete. Model saved as 'threemenmorris_final.pt'{Style.RESET_ALL}"
            )
        elif sys.argv[1] == "load" and len(sys.argv) > 2:
            try:
                alphazero.load_checkpoint(sys.argv[2])
                print(f"{Fore.GREEN}Model loaded from {sys.argv[2]}{Style.RESET_ALL}")
            except Exception as e:
                print(
                    f"{Fore.RED}Failed to load model from {sys.argv[2]}: {e}{Style.RESET_ALL}"
                )
                sys.exit(1)

    # Add this while loop
    while True:
        print(f"\n{Fore.CYAN}What would you like to do?{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1. Play against AlphaZero")
        print("2. Watch AlphaZero play against itself")
        print("3. Watch trained best model play")
        print(f"4. Exit{Style.RESET_ALL}")
        choice = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

        if choice == "1":
            human = HumanPlayer(game)
            print(
                f"\n{Fore.WHITE}You'll play as '{Fore.BLUE}O{Fore.WHITE}', AlphaZero will play as '{Fore.RED}X{Fore.WHITE}'{Style.RESET_ALL}"
            )
            reward = alphazero.play_game(opponent=human, render=True)

            print(f"\n{Fore.CYAN}" + "=" * 30)
            if reward > 0:
                print(f"{Fore.RED}GAME OVER: AlphaZero wins!{Style.RESET_ALL}")
            elif reward < 0:
                print(f"{Fore.GREEN}GAME OVER: You win!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}GAME OVER: It's a draw!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}" + "=" * 30 + f"{Style.RESET_ALL}")

        elif choice == "2":
            print(f"\n{Fore.CYAN}AlphaZero playing against itself:{Style.RESET_ALL}")
            reward = alphazero.play_game(render=True)

            print(f"\n{Fore.CYAN}" + "=" * 30)
            if reward > 0:
                print(f"{Fore.RED}GAME OVER: Player 1 (X) wins!{Style.RESET_ALL}")
            elif reward < 0:
                print(f"{Fore.BLUE}GAME OVER: Player 2 (O) wins!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}GAME OVER: It's a draw!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}" + "=" * 30 + f"{Style.RESET_ALL}")
        
        elif choice == "3":
            try:
                if not os.path.exists("threemenmorris_final.pt"):
                    print(f"\n{Fore.YELLOW}No trained model found. Starting training process...{Style.RESET_ALL}")
                    
                    # Set training parameters
                    game.debug_mode = True
                    game.is_training = True  # Add this flag
                    alphazero.num_iterations = 2
                    alphazero.num_self_play_games = 5
                    alphazero.num_epochs = 3
                    
                    try:
                        alphazero.train()
                        alphazero._save_checkpoint("threemenmorris_final.pt")
                        print(f"{Fore.GREEN}Training complete. Model saved.{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Training failed: {str(e)}{Style.RESET_ALL}")
                        return
                    finally:
                        game.debug_mode = False
                        if hasattr(game, 'is_training'):
                            delattr(game, 'is_training')  # Remove training flag

                # Load and play with the trained model
                alphazero.load_checkpoint("threemenmorris_final.pt")
                print(f"\n{Fore.GREEN}Starting game with trained model{Style.RESET_ALL}")
                reward = alphazero.play_game(render=True)
                
                print(f"\n{Fore.CYAN}" + "=" * 30)
                if reward > 0:
                    print(
                        f"{Fore.RED}GAME OVER: Expert Player 1 (X) wins!"
                        f"{Style.RESET_ALL}"
                    )
                elif reward < 0:
                    print(
                        f"{Fore.BLUE}GAME OVER: Expert Player 2 (O) wins!"
                        f"{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"{Fore.YELLOW}GAME OVER: Expert game is a draw!"
                        f"{Style.RESET_ALL}"
                    )
                print(f"{Fore.CYAN}" + "=" * 30 + f"{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

        elif choice == "4":
            print(f"{Fore.GREEN}Thanks for playing!{Style.RESET_ALL}")
            break  # Now this break is inside the while loop
        else:
            print(f"{Fore.YELLOW}Invalid choice. Please try again.{Style.RESET_ALL}")

        # Add a small pause before showing menu again
        input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
