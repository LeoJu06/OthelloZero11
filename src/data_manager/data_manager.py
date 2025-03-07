from src.othello.game_constants import PlayerColor
from src.othello.othello_game import OthelloGame
import numpy as np
from src.mcts.node import Node
import os
import pickle

class DataManager:

    def __init__(self):

        self.data = []
        self.game = OthelloGame()
        
        pass

    def create_example(self, current_state, player, root:Node, temperature):
        """Creates an example"""
        
        if player == PlayerColor.BLACK.value:
            current_state = self.game.get_canonical_board(current_state, player)

        return [current_state, root.pi(temperature), None]
    
    def assign_rewards(self, examples, game_outcome):

        for situation in examples:

            situation[2] = game_outcome
            game_outcome *= -1

        return examples
    
    def increment_iteration(self):

        pass

    def save_training_examples(self, examples, idx):

       
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

        new_folder = os.path.join(data_dir, 'Iter1')
       # os.makedirs(new_folder)
        filename = f"examples_{idx}.pkl"
        path = os.path.join(data_dir,filename)

        with open(path, "wb") as f:
            pickle.dump(examples, f)


        print()
        pass

    def load_best_model(self):
        best_model = None
        return best_model

        
        




    def collect(self, training_example):
        """Collects data examples"""

        self.data.append(training_example)



if __name__ == "__main__":

    import os
    import os
    import os

    from pathlib import Path

    import os

    # Pfad zum 'data' Ordner im Root-Verzeichnis
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

    print(data_dir)


