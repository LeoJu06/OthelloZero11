"Module to hold the Hyperparamter class."
import torch


class Hyperparameters:

    """The Hyperparameter class contains dictionnaries with
    hyperparameters needed for certain tasks.
    Simply address them by writing Hyperparamerts.Name["key"]

    The Hyperparameter class contains:
        - MCTS with keys ["num_simulations", "exploration_weight", "temp_threshold", temperature]
        - Coach with keys ["iterations", "episodes", "num_workers", "episodes_per_worker"]
        - Neural_Network with keys ["device"]
        - Node with keys ["key_passsing, prior_passing]"""

    MCTS = {
        "num_simulations": 200,
        "exploration_weight": 1.5,
        "temp_threshold": 14,
        "temp": 1,
    }

    Coach = {"iterations": 50, 
             "episodes": 22*100,
             "num_workers" :22, }
    Coach["episodes_per_worker"] = Coach["episodes"] // Coach["num_workers"]

    Neural_Network = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        "epochs" : 20, 
        "batch_size" : 128,
        "learning_rate" : 0.001
    }

    Node = {"key_passing": -1, "prior_passing": 1}

    Arena = {"treshold": 0.55, "arena_games": 10}

        
    