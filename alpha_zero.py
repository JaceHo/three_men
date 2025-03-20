import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNetwork(nn.Module):
    """Neural network for AlphaZero: policy + value network"""

    def __init__(self, game, num_resblocks=19, num_hidden=256):
        super().__init__()
        self.game = game
        self.num_resblocks = num_resblocks
        self.num_hidden = num_hidden
        action_size = game.action_size

        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(game.observation_shape[0], num_hidden, 3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resblocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                2 * game.observation_shape[1] * game.observation_shape[2], action_size
            ),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                game.observation_shape[1] * game.observation_shape[2], num_hidden
            ),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Input shape: (batch_size, observation_shape)
        x = self.conv_block(x)

        for block in self.res_blocks:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResBlock(nn.Module):
    """Residual block for the neural network"""

    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class MCTS:
    """Monte Carlo Tree Search for finding the best actions"""

    def __init__(
        self,
        game,
        neural_net,
        num_simulations=800,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    ):
        self.game = game
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Tree nodes dictionary with state hash as key
        self.Q = {}  # mean action value Q(s,a)
        self.P = {}  # policy P(s,a) from neural network
        self.N = {}  # visit count for (s,a)
        self.N_s = {}  # visit count for state s
        self.valid_moves = {}  # valid moves from state s
        self.game_ended = {}  # whether state s is terminal
        self.terminal_value = {}  # terminal value of state s if game has ended

    def _get_state_hash(self, state):
        """Convert state to hashable representation"""
        return state.tobytes()

    def search(self, state, temperature=1.0):
        """Execute MCTS search and return action probabilities and values"""
        state_hash = self._get_state_hash(state)

        # Initialize tree with root state if not already initialized
        if state_hash not in self.P:
            self._expand_node(state)

        # Add Dirichlet noise to the root node for exploration
        if self.dirichlet_epsilon > 0:
            valid_moves = self.valid_moves[state_hash]
            noise = np.random.dirichlet([self.dirichlet_alpha] * np.sum(valid_moves))
            noise_idx = 0
            noisy_P = self.P[state_hash].copy()

            for i in range(len(noisy_P)):
                if valid_moves[i]:
                    noisy_P[i] = (1 - self.dirichlet_epsilon) * noisy_P[
                        i
                    ] + self.dirichlet_epsilon * noise[noise_idx]
                    noise_idx += 1

            self.P[state_hash] = noisy_P

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(state.copy(), state_hash)

        # Calculate action probabilities based on visit counts
        counts = np.array(
            [self.N.get((state_hash, a), 0) for a in range(self.game.action_size)]
        )

        # Apply temperature
        if temperature == 0:  # Deterministic choice
            best_action = np.argmax(counts)
            probs = np.zeros(self.game.action_size)
            probs[best_action] = 1
        else:  # Temperature-controlled distribution
            counts = counts ** (1 / temperature)
            probs = counts / np.sum(counts)

        return probs, [
            self.Q.get((state_hash, a), 0) for a in range(self.game.action_size)
        ]

    def _simulate(self, state, state_hash):
        """Simulate a single MCTS iteration using iterative approach"""
        path = []
        current_state = state.copy()
        current_hash = state_hash
        value = 0
        
        while True:
            # Check if we need to expand this node
            if current_hash not in self.P:
                value = self._expand_node(current_state)
                break
                
            # Check if state is terminal
            if self.game_ended.get(current_hash, False):
                value = self.terminal_value[current_hash]
                break

            # Select action with highest UCB
            valid_moves = self.valid_moves[current_hash]
            best_action = -1
            best_value = -float('inf')
            
            for action in range(self.game.action_size):
                if valid_moves[action]:
                    q = self.Q.get((current_hash, action), 0)
                    n = self.N.get((current_hash, action), 0)
                    p = self.P[current_hash][action]
                    ucb = q + self.c_puct * p * math.sqrt(self.N_s[current_hash]) / (1 + n)
                    
                    if ucb > best_value:
                        best_value = ucb
                        best_action = action

            if best_action == -1:  # No valid moves
                value = 0
                break

            # Store the path for backpropagation
            path.append((current_hash, best_action))

            # Move to next state
            next_state, _, done, reward = self.game.step(current_state, best_action)
            current_state = next_state
            current_hash = self._get_state_hash(next_state)
            
            if done:
                value = reward
                break

        # Backpropagate the value through the path
        for (node_hash, action) in path:
            self.N_s[node_hash] += 1
            self.N[(node_hash, action)] = self.N.get((node_hash, action), 0) + 1
            q_old = self.Q.get((node_hash, action), 0)
            self.Q[(node_hash, action)] = q_old + (value - q_old) / self.N[(node_hash, action)]

        return value

    def _expand_node(self, state):
        """Expand a new node and evaluate it with the neural network"""
        state_hash = self._get_state_hash(state)

        # Check if state is terminal
        is_terminal, reward = self.game.is_terminal(state)
        self.game_ended[state_hash] = is_terminal

        if is_terminal:
            self.terminal_value[state_hash] = reward
            return reward

        # Get valid moves for this state
        valid_moves = self.game.get_valid_moves(state)
        self.valid_moves[state_hash] = valid_moves

        # Use neural network to evaluate the state
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy, value = self.neural_net(state_tensor)
            policy = F.softmax(policy.squeeze(0), dim=0).cpu().numpy()
            value = value.item()

        # Mask invalid moves and renormalize
        policy = policy * valid_moves
        policy_sum = np.sum(policy)

        if policy_sum > 0:
            policy /= policy_sum
        else:
            # If all valid moves were masked, use a uniform distribution over valid moves
            policy = valid_moves / np.sum(valid_moves)

        # Initialize node in search tree
        self.P[state_hash] = policy
        self.N_s[state_hash] = 0

        return value


class AlphaZero:
    """Main AlphaZero implementation that handles training and self-play"""

    def __init__(
        self,
        game,
        model_config=None,
        mcts_config=None,
        replay_buffer_size=10000,
        batch_size=128,
        num_iterations=100,
        num_self_play_games=100,
        num_epochs=10,
        checkpoint_interval=10,
    ):
        self.game = game

        # Neural network
        self.model_config = model_config or {}
        self.neural_net = NeuralNetwork(game, **self.model_config)

        # MCTS configuration
        self.mcts_config = mcts_config or {"num_simulations": 800, "c_puct": 1.0}

        # Training parameters
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval

        # Experience replay buffer: (state, policy target, value target)
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Optimizer
        self.optimizer = optim.Adam(
            self.neural_net.parameters(), lr=0.001, weight_decay=1e-4
        )

    def train(self):
        """Main training loop for AlphaZero"""
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")

            # Generate self-play data
            print("Self-play phase...")
            self._self_play()

            # Train the neural network
            print("Training phase...")
            self._train_neural_network()

            # Save checkpoint
            if (iteration + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(f"alphazero_iter_{iteration + 1}.pt")

    def _self_play(self):
        """Generate self-play games and add experiences to replay buffer"""
        self.neural_net.eval()

        for game_num in range(self.num_self_play_games):
            if game_num % 10 == 0:
                print(f"Self-play game {game_num}/{self.num_self_play_games}")

            # Initialize MCTS for this game
            mcts = MCTS(self.game, self.neural_net, **self.mcts_config)

            # Initialize game state
            state = self.game.get_initial_state()
            game_history = []

            # Play until game terminates
            done = False
            step = 0

            while not done:
                # For the first 30 moves, use temperature = 1; then temperature = 0.01
                temperature = 1.0 if step < 30 else 0.01

                # Get action probabilities from MCTS
                pi, _ = mcts.search(state, temperature)

                # Store (state, policy) pair
                game_history.append((state.copy(), pi))

                # Choose action based on the policy
                if temperature == 0:
                    action = np.argmax(pi)
                else:
                    action = np.random.choice(len(pi), p=pi)

                # Execute action
                next_state, _, done, reward = self.game.step(state, action)
                state = next_state
                step += 1

                if step > self.game.max_steps:
                    done = True
                    reward = self.game.get_draw_reward()

            # Update game history with final reward
            for idx, (hist_state, hist_pi) in enumerate(game_history):
                # For zero-sum two-player games, invert value for opponent's moves
                # if idx % 2 == 1 and hasattr(self.game, 'is_two_player') and self.game.is_two_player:
                #    value_target = -reward
                # else:
                value_target = reward

                # Add experience to replay buffer
                self.replay_buffer.append((hist_state, hist_pi, value_target))

    def _train_neural_network(self):
        """Train neural network on collected experiences"""
        if len(self.replay_buffer) < self.batch_size:
            print(
                f"Not enough experiences in buffer ({len(self.replay_buffer)}). Skipping training."
            )
            return

        self.neural_net.train()

        for epoch in range(self.num_epochs):
            # Sample batch from replay buffer
            minibatch = random.sample(
                self.replay_buffer, min(self.batch_size, len(self.replay_buffer))
            )
            states, pis, values = zip(*minibatch)

            # Convert to tensors
            state_batch = torch.FloatTensor(np.array(states))
            pi_batch = torch.FloatTensor(np.array(pis))
            value_batch = torch.FloatTensor(np.array(values).reshape(-1, 1))

            # Train model
            self.optimizer.zero_grad()

            # Forward pass
            policy_output, value_output = self.neural_net(state_batch)

            # Calculate loss
            policy_loss = F.cross_entropy(policy_output, pi_batch)
            value_loss = F.mse_loss(value_output, value_batch)
            total_loss = policy_loss + value_loss

            # Backward pass and optimize
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{self.num_epochs}, Loss: {total_loss.item():.4f} "
                    f"(Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})"
                )

    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.neural_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.neural_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {filename}")

    def play_game(self, opponent=None, render=False):
        """Play a game against an opponent (or against itself if opponent is None)"""
        self.neural_net.eval()

        # Initialize MCTS
        mcts = MCTS(self.game, self.neural_net, **self.mcts_config)

        # Initialize game state
        state = self.game.get_initial_state()
        done = False
        turn = 0

        while not done:
            if render and (opponent is None or turn == 0):
                self.game.render(state)

            # AI's turn
            if turn == 0:
                print("AlphaZero is thinking...")
                pi, _ = mcts.search(state, temperature=0.01)
                action = np.argmax(pi)

                # Show action in human-readable form
                row, col = action // self.game.board_size, action % self.game.board_size
                print(f"AlphaZero plays: row {row}, column {col}")

            # Opponent's turn
            else:
                if opponent:
                    action = opponent.get_action(state)
                else:
                    # Self-play - use AlphaZero again
                    print("AlphaZero (player 2) is thinking...")
                    pi, _ = mcts.search(state, temperature=0.01)
                    action = np.argmax(pi)
                    # Show action in human-readable form
                    row, col = (
                        action // self.game.board_size,
                        action % self.game.board_size,
                    )
                    print(f"AlphaZero (player 2) plays: row {row}, column {col}")

            # Execute action
            next_state, _, done, reward = self.game.step(state, action)
            state = next_state

            # Switch turns (0 = player 1, 1 = player 2)
            turn = 1 - turn

        if render:
            self.game.render(state)

        return reward


# Example game environment - TicTacToe
class TicTacToe:
    """TicTacToe game environment for AlphaZero"""

    def __init__(self):
        self.board_size = 3
        self.action_size = self.board_size * self.board_size
        self.is_two_player = True
        self.max_steps = self.action_size

        # State representation: first channel for player's pieces, second for opponent's
        self.observation_shape = (2, self.board_size, self.board_size)

    def get_initial_state(self):
        """Return initial game state"""
        return np.zeros(self.observation_shape, dtype=np.float32)

    def get_valid_moves(self, state):
        """Return valid moves as a binary mask"""
        # The board positions that are empty (not occupied by either player)
        player_pieces = state[0]
        opponent_pieces = state[1]

        valid = np.ones(self.action_size, dtype=np.int8)

        for i in range(self.board_size):
            for j in range(self.board_size):
                idx = i * self.board_size + j
                if player_pieces[i, j] != 0 or opponent_pieces[i, j] != 0:
                    valid[idx] = 0

        return valid

    def step(self, state, action):
        """Execute action and return next state, reward, done, and info"""
        # Copy state to avoid modifying the original
        next_state = state.copy()

        # Convert action index to board position
        row = action // self.board_size
        col = action % self.board_size

        # Place piece
        next_state[0, row, col] = 1

        # Check if game is over
        is_terminal, reward = self.is_terminal(next_state)

        if not is_terminal:
            # Swap players by swapping the channels
            next_state = np.flip(next_state, axis=0).copy()

        return next_state, next_state, is_terminal, reward

    def is_terminal(self, state):
        """Check if the game is over and return reward"""
        player_pieces = state[0]

        # Check rows, columns, and diagonals for a win
        for i in range(self.board_size):
            # Check rows
            if np.all(player_pieces[i, :] == 1):
                return True, 1.0

            # Check columns
            if np.all(player_pieces[:, i] == 1):
                return True, 1.0

        # Check diagonals
        if np.all(np.diag(player_pieces) == 1):
            return True, 1.0

        if np.all(np.diag(np.fliplr(player_pieces)) == 1):
            return True, 1.0

        # Check for draw (board is full)
        if np.all((state[0] + state[1]) != 0):
            return True, 0.0

        # Game is not over
        return False, 0.0

    def get_draw_reward(self):
        """Reward for a draw"""
        return 0.0

    def render(self, state):
        """Display the board"""
        player_pieces = state[0]
        opponent_pieces = state[1]

        print("  0 1 2")
        print(" -------")
        for i in range(self.board_size):
            print(f"{i}|", end="")
            for j in range(self.board_size):
                if player_pieces[i, j] == 1:
                    print("X ", end="")
                elif opponent_pieces[i, j] == 1:
                    print("O ", end="")
                else:
                    print("· ", end="")
            print()
        print()


# Usage example
if __name__ == "__main__":
    import sys

    # Initialize game
    game = TicTacToe()

    # Initialize AlphaZero with a smaller network and fewer simulations for demonstration
    alphazero = AlphaZero(
        game,
        model_config={"num_resblocks": 3, "num_hidden": 64},
        mcts_config={"num_simulations": 100, "c_puct": 2.0},
        num_iterations=5,  # Reduced for demonstration
        num_self_play_games=20,  # Reduced for demonstration
        num_epochs=5,  # Reduced for demonstration
        replay_buffer_size=1000,
        batch_size=32,
    )

    # Human player class
    class HumanPlayer:
        def __init__(self, game):
            self.game = game

        def get_action(self, state):
            valid_moves = self.game.get_valid_moves(state)

            print("\nYour turn! Current board:")
            self.game.render(state)

            while True:
                try:
                    print("\nEnter your move:")
                    row = int(input("Row (0-2): "))
                    col = int(input("Column (0-2): "))

                    if not (
                        0 <= row < self.game.board_size
                        and 0 <= col < self.game.board_size
                    ):
                        print("Invalid coordinates! Must be between 0-2.")
                        continue

                    action = row * self.game.board_size + col

                    if valid_moves[action]:
                        print(f"You played: row {row}, column {col}")
                        return action
                    else:
                        print("That position is already taken! Try again.")
                except ValueError:
                    print("Please enter numbers between 0-2.")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training AlphaZero...")
            alphazero.train()
            # Save the final model
            alphazero._save_checkpoint("alphazero_final.pt")
            print("Training complete. Model saved as 'alphazero_final.pt'")
        elif sys.argv[1] == "load" and len(sys.argv) > 2:
            try:
                alphazero.load_checkpoint(sys.argv[2])
                print(f"Model loaded from {sys.argv[2]}")
            except:
                print(f"Failed to load model from {sys.argv[2]}")
                sys.exit(1)

    # Play mode
    print("\nDo you want to play against AlphaZero? (y/n)")
    choice = input().lower()

    if choice == "y" or choice == "yes":
        human = HumanPlayer(game)
        print("\nYou'll play as 'O', AlphaZero will play as 'X'")
        reward = alphazero.play_game(opponent=human, render=True)

        if reward > 0:
            print("AlphaZero wins!")
        elif reward < 0:
            print("You win!")
        else:
            print("It's a draw!")
    else:
        # Self-play demonstration
        print("\nAlphaZero playing against itself:")
        reward = alphazero.play_game(render=True)
        print(f"Game finished with reward: {reward}")
