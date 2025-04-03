from src.config.hyperparameters import Hyperparameters
import os
from src.othello.othello_game import OthelloGame
from src.mcts.mcts import MultiprocessedMCTS
from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager
from src.mcts.manager import init_manager, init_manager_process, terminate_manager_process, create_multiprocessed_mcts
from src.othello.game_constants import PlayerColor
from src.utils.index_to_coordinates import index_to_coordinates
from src.arena.arena import Arena
from src.neural_net.train_model import train
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
import torch.multiprocessing as mp
import src.utils.logger_config as lg
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm
import time



class Coach:
    """
    The Coach class manages self-play episodes, data collection, and training 
    for the Othello Zero model.
    """

    mp.set_start_method("spawn", force=True)  # Set multiprocessing start method
    

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
        lg.logger_coach.info(f"Worker[{worker_id}] Started executing Episodes")

        multi_mcts = create_multiprocessed_mcts(worker_id, manager)
        episodes = self.hyperparams.Coach["episodes_per_worker"]

        for _ in tqdm(range(episodes)):
            episode_data = self.execute_single_episode(multi_mcts)
            queue.put(episode_data)  # Send data to the main process

        lg.logger_coach.info(f"Worker[{worker_id}] Finished executing Episodes")
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

        lg.logger_coach.info("Self-PLay Finished")
        lg.logger_coach.info(f"Collected {len(total_examples)} Examples")

        # Save data once all workers are done
        self.data_manager.save_training_examples(total_examples) # save training
      

        lg.logger_coach.info("Data was successfully saved")

        for worker in workers:
            worker.join()

    def learn(self):
        """
        Runs the reinforcement learning loop, collecting self-play data and training the model.
        """
        game = OthelloGame()
        hyperparams = self.hyperparams
        #model = OthelloZeroModel(game.rows, game.get_action_size(), device=hyperparams.Neural_Network["device"])
        model = self.data_manager.load_model()
        self.data_manager.save_model(model)

        for iteration in (range(1, hyperparams.Coach["iterations"] + 1)):
            lg.logger_coach.info(f"---Starting Iteration {iteration}---")
          
            start_time = time.time()
            lg.logger_coach.info(f"Iteration {iteration}/{hyperparams.Coach['iterations']} - Starting self-play...")

            manager = init_manager(model, hyperparams)
            manager_process = init_manager_process(manager)

            self.self_play(manager)
            

            terminate_manager_process(manager_process)

            lg.logger_coach.info(f"Iteration {iteration} - Self-play complete. Training model...")

            examples = self.data_manager.load_examples()
            
            lg.logger_coach.info("Start Training Model.")
            new_model = self.train(model, examples)
            lg.logger_coach.info("Training Model complete.")

            old_model =  self.data_manager.load_model(latest_model=True)
            lg.logger_coach.info("Arena - New Model vs. old Model.")

            won, lost = self.arena.let_compete(new_model, old_model)
            lg.logger_coach.info("Battle Completed.")


            if self.accept_new_model(won):
                lg.logger_coach.info(f"New Model was accepted [win={won}, lost={lost}]")

                print("New model was accepted")
                model = new_model
            else:
                print("New model was Rejected")
                lg.logger_coach.info(f"New Model was rejected [win={won}, lost={lost}]")
                model = old_model

            

          
            self.report(won, lost, examples, duration=time.time()-start_time)
            self.data_manager.increment_iteration() # increment interation number in txt file
            self.data_manager.save_model(model) # save new model
            self.hyperparams.MCTS["num_simulations"] += 30
            self.hyperparams.MCTS["num_simulations"] = min(800, self.hyperparams.MCTS["num_simulations"])
            self.hyperparams.Neural_Network["learning_rate"] = max(self.hyperparams.Neural_Network["learning_rate"]-0.001, 0.001)
            
            lg.logger_coach.info(f"Iteration {iteration} completed in {time.time() - start_time:.2f}s.")

            
            lg.logger_coach.info("*** --- ***")
            lg.logger_coach.info("")

    def train(self,model,  examples):
        """
        Trains the neural network using collected self-play data.
        """
        epochs = self.hyperparams.Neural_Network["epochs"]
        batch_size = self.hyperparams.Neural_Network["batch_size"]
        learning_rate = self.hyperparams.Neural_Network["learning_rate"]
       

        model, policy_losses, value_losses = train(model, examples, epochs, batch_size, learning_rate)

        iter_number = self.data_manager.get_iter_number()
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), policy_losses, label="Policy Loss")
        plt.plot(range(1, epochs + 1), value_losses, label="Value Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Iteration {iter_number} ")
        plt.legend()

        plt.savefig(f"data/losses_plotted/Training_loss_{iter_number}", dpi=300)

        return model
    
    def accept_new_model(self, won):
      
        return won / self.hyperparams.Arena["arena_games"] >= self.hyperparams.Arena["treshold"]
    

    def report(self, won, lost, examples, duration):
        n = self.data_manager.get_iter_number()
        pdf_filename = f"data/reports/report_iteration_{n}.pdf"
        report_image = f"data/losses_plotted/Training_loss_{n}.png"

        # PDF-Dokument erstellen
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 770, f"Training Report - Iteration {n}")

        # Horizontale Linie für Struktur
        c.setStrokeColor(colors.black)
        c.line(100, 760, 500, 760)

        # Informationen über Trainingsergebnisse
        c.setFont("Helvetica", 12)
        info_text = [
            f"Games Won: {won}",
            f"Games Lost: {lost}",
            f"Number of Training Examples: {len(examples)}",
            f"Model Accepted: {self.accept_new_model(won)}",
            f"Training Duration: {duration:.2f} seconds",
        ]

        y_position = 730
        for line in info_text:
            c.drawString(100, y_position, line)
            y_position -= 20  # Abstand zwischen den Zeilen

        # Loss-Plot einfügen (falls vorhanden)
        if os.path.exists(report_image):
            c.drawString(100, y_position - 10, "Training Loss Plot:")
            c.drawImage(report_image, 100, y_position - 250, width=400, height=250)
        else:
            c.drawString(100, y_position - 10, "Training Loss Plot: (Not Found)")

        # PDF speichern
        c.save()
        print(f"📄 PDF gespeichert als {pdf_filename}")

if __name__ == "__main__":
    coach = Coach()
    coach.learn()
