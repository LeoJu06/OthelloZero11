from src.othello.game_constants import PlayerColor
from src.othello.othello_game import OthelloGame
from src.neural_net.model import OthelloZeroModel
from src.mcts.node import Node
import os

import pickle
import random
from collections import deque
import pickle
import torch
from pathlib import Path

class DataManager:
    def __init__(self):
        """Initializes the DataManager."""
        self.data = []
        self.game = OthelloGame()

    def create_example(self, current_state, player, root: Node, temperature):
        """Creates a training example.

        Args:
            current_state: The current game state.
            player: The current player.
            root: The root node of the MCTS tree.
            temperature: The temperature for policy calculation.

        Returns:
            A list containing the canonical board, policy, and reward (initially None).
        """
        if player == PlayerColor.BLACK.value:
            current_state = self.game.get_canonical_board(current_state, player)
        return [current_state, root.pi(temperature), None]

    def assign_rewards(self, examples, game_outcome):
        """Assigns rewards to the examples based on the game outcome.

        Args:
            examples: A list of training examples.
            game_outcome: The outcome of the game (1 for win, -1 for loss).

        Returns:
            The updated list of examples with rewards assigned.
        """
        for example in examples:
            example[2] = game_outcome
            game_outcome *= -1  # Alternate rewards for players
        return examples

    def _path_to_data_dir(self):
        """Returns the path to the data directory."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

    def get_iter_number(self):
        """Returns the current iteration number.

        Returns:
            The current iteration number as an integer.
        """
        data_dir = self._path_to_data_dir()
        path = os.path.join(data_dir, "iteration_number.txt")

        # Ensure the file exists and is initialized
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("0")  # Initialize with 0 if the file doesn't exist

        with open(path, "r") as f:
            n = f.read().strip()

        # Validate the content of the file
        if not n.isdigit():
            raise ValueError(f"Invalid content in {path}. Expected a number, got '{n}'.")

        return int(n)

    def increment_iteration(self):
        """Increments the iteration number by 1."""
        n = self.get_iter_number()
        data_dir = self._path_to_data_dir()
        path = os.path.join(data_dir, "iteration_number.txt")

        with open(path, "w") as f:
            f.write(str(n + 1))  # Write the incremented value

    def save_training_examples(self, examples):
        """Saves training examples to a file.

        Args:
            examples: A list of training examples to save.
        """
        data_dir = self._path_to_data_dir()
        n = self.get_iter_number()
        filename = f"examples/examples_iteration_{n}.pkl"
        path = os.path.join(data_dir, filename)

        with open(path, "wb") as f:
            pickle.dump(examples, f)


    def load_examples(self, n=None):

        data_dir = self._path_to_data_dir()

        if n is not None:

            filename = f"examples_iteration_{n}.pkl"
            path = os.path.join(data_dir, filename)

            with open(path, "rb") as f:
                examples = pickle.load(f)

            return examples
        
        else:
            n = self.get_iter_number()  # +1, because range() exludes the last number
            combinded = []

            for ex_n in (range(max(n-5, 0), max(n, 1))):
                print("File number", ex_n)
                filename = f"examples/examples_iteration_{ex_n}.pkl"
                path = os.path.join(data_dir, filename)

                with open(path, "rb") as f:
                    examples = pickle.load(f)
                    combinded.extend(examples)

            return combinded

        


    
    def save_model(self, model:OthelloZeroModel):
        n = self.get_iter_number()
        torch.save(model.state_dict(), f"data/models/othello_zero_model_{n}")

    def load_model(self, latest_model=True):
        """Loads a model. If n is None the best model (last iter) is being returned"""

        if latest_model:
            n = self.get_iter_number()
        model = OthelloZeroModel(board_size=8, action_size=65, device="cuda")
        model.load_state_dict(torch.load(f"data/models/othello_zero_model_{n}"))
        return model


    def collect(self, training_example):
        """Collects training examples.

        Args:
            training_example: A single training example to add to the data list.
        """
        self.data.append(training_example)



class ReplayBuffer:
    def __init__(self, max_size=500_000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples):
        self.buffer.extend(examples)
    
    def sample(self, num_samples):
        # Gibt alle Beispiele zurück, wenn weniger als num_samples vorhanden sind
        return random.sample(self.buffer, min(num_samples, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    print(f"Data directory: {data_dir}")

    da = DataManager()
    n = da.get_iter_number()

    e = da.load_examples(0)
    print(len(e))
