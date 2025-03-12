from src.config.hyperparameters import Hyperparameters
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MultiprocessedMCTS
from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager
from src.mcts.manager import init_manager, init_manager_process, terminate_manager_process, create_multiprocessed_mcts
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates
from src.arena.arena import Arena
from src.neural_net.train_model import train
import torch.multiprocessing as mp

from tqdm import tqdm
import time
import logging
logging.basicConfig(level=logging.CRITICAL)
class Coach:
    """
    The Coach class manages self-play episodes, data collection, and training 
    for the Othello Zero model.
    """

    mp.set_start_method("spawn", force=True)  # Set multiprocessing start method
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def __init__(self):
        """
        Initializes the Coach with hyperparameters and a data manager.
        """
        self.hyperparams = Hyperparameters()
        self.data_manager = DataManager()
        self.arena = Arena()

    def execute_single_episode(self, mcts: MultiprocessedMCTS):
        """
        Runs a single self-play episode using MCTS.

        Args:
            mcts (MultiprocessedMCTS): The Monte Carlo Tree Search instance.

        Returns:
            list: Training examples generated during the episode.
        """
        game = OthelloGame()
        state = game.get_init_board()
        current_player = PlayerColor.BLACK.value  # Black always starts
        examples = []  # Training data storage
        episode_step = 0

        while not game.is_terminal_state(state):
            temp = int(episode_step < self.hyperparams.MCTS["temp_threshold"])
            root = mcts.run_search(state, current_player)

            # Store training example
            examples.append(self.data_manager.create_example(state, current_player, root, temp))

            # Select action and update game state
            x_pos, y_pos = index_to_coordinates(root.select_action(temp))
            state, current_player = game.get_next_state(state, current_player, x_pos, y_pos)

            episode_step += 1
            root.reset()

        # Assign rewards based on the game outcome
        game_outcome = game.get_reward_for_player(state, PlayerColor.BLACK.value)
        return self.data_manager.assign_rewards(examples, game_outcome)

    def self_play_worker(self, queue, worker_id, manager):
        """
        Worker function to run multiple self-play episodes.
        """
        logging.info(f"Worker {worker_id} started.")
        multi_mcts = create_multiprocessed_mcts(worker_id, manager)
        episodes = self.hyperparams.Coach["episodes_per_worker"]

        for _ in range(episodes):
            episode_data = self.execute_single_episode(multi_mcts)
            queue.put(episode_data)  # Send data to the main process

        logging.info(f"Worker {worker_id} finished.")
        queue.put(None)  # Signal completion

    def self_play(self, manager):
        """
        Runs multiple self-play episodes in parallel.
        """
        queue = mp.Queue()
        num_workers = self.hyperparams.Coach["num_workers"]
        workers = []

        for worker_id in range(num_workers):
            worker_process = mp.Process(target=self.self_play_worker, args=(queue, worker_id, manager))
            workers.append(worker_process)
            worker_process.start()

        # Collect data from workers
        total_examples = []
        completed_workers = 0
        while completed_workers < num_workers:
            data = queue.get()
            if data is None:
                completed_workers += 1
            else:
                total_examples.extend(data)

        # Save data once all workers are done
        self.data_manager.save_training_examples(total_examples) # save training
      
        logging.info("Self-play data saved.")

        for worker in workers:
            worker.join()

    def learn(self):
        """
        Runs the reinforcement learning loop, collecting self-play data and training the model.
        """
        game = OthelloGame()
        hyperparams = self.hyperparams
        model = OthelloZeroModel(game.rows, game.get_action_size(), hyperparams.Neural_Network["device"])
        self.data_manager.save_model(model)

        for iteration in range(1, hyperparams.Coach["iterations"] + 1):
            start_time = time.time()
            logging.info(f"Iteration {iteration}/{hyperparams.Coach['iterations']} - Starting self-play...")

            manager = init_manager(model, hyperparams)
            manager_process = init_manager_process(manager)

            self.self_play(manager)
            

            terminate_manager_process(manager_process)

            logging.info(f"Iteration {iteration} - Self-play complete. Training model...")

            examples = self.data_manager.load_examples(self.data_manager.get_iter_number())
            

            new_model = self.train(model, examples)
            old_model = self.data_manager.load_model(latest_model=True)

            won, lost = self.arena.let_compete(new_model, old_model)
            if won / self.hyperparams.Arena["arena_games"] >= self.hyperparams.Arena["treshold"]:
                print("Accepting new model")
                model = new_model
            else:
                print("New model declined")
                model = old_model

            self.data_manager.increment_iteration() # increment interation number in txt file
            self.data_manager.save_model(model)
            logging.info(f"Iteration {iteration} completed in {time.time() - start_time:.2f}s.")

    def train(self,model,  examples):
        """
        Trains the neural network using collected self-play data.
        """
        logging.info("Training process started.")

        model = train(model, examples, epochs=40, batch_size=256)
        # TODO: Implement training mechanism
        logging.info("Training complete.")
        return model

if __name__ == "__main__":
    coach = Coach()
    coach.learn()
