import threading
import pygame
from src.othello.othello_game import OthelloGame
import src.othello.game_constants as const
from src.othello.game_settings import (
    WIDTH,
    HEIGHT,
    SQUARE_SIZE,
    FPS,
)
from src.othello.game_visuals import GameVisuals
from src.data_manager.data_manager import DataManager
from src.mcts.mcts import MCTS
from src.utils.index_to_coordinates import index_to_coordinates



class GamePvsAi:
    """
    Handles a Player vs Random AI game of Othello.
    """

    def __init__(self, screen):
        """
        Initializes the game instance.

        Args:
            screen: Pygame screen object for rendering.
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.board = OthelloGame()
        self.visuals = GameVisuals(screen, self.clock)
        self.data_manager = DataManager()
        self.mcts = MCTS(model=self.data_manager.load_model())

        # Game state variables
        self.running = True
        self.current_player = const.PlayerColor.BLACK.value
        self.is_ai_turn = False  # Player starts by default
        self.ai_action = None 
        self.game_state = self.board.get_init_board()

    def run_game_loop(self, fps=FPS):
        """
        Main game loop that alternates turns between the player and the AI.

        Args:
            fps: Frames per second for the game loop.
        """
        while self.running:
            if self.board.is_terminal_state(self.game_state):
                self.running = False
                break

            self.process_turn()
            self.update_display()
            self.clock.tick(fps)

        self.display_winner()

    def process_turn(self):
        """
        Processes the current turn, either for the player or the AI.
        """
        if self.is_ai_turn:
            self.execute_ai_turn()
        else:
            self.handle_player_input()

    def handle_player_input(self):
        """
        Handles the player's turn, allowing them to make a valid move.
        """
        valid_moves = self.board.get_valid_moves(self.game_state, self.current_player)

        if not valid_moves:
            print(f"Player {self.current_player} has no valid moves. Passing turn.")
            self.switch_turn()
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.process_player_move(event.pos, valid_moves):
                    self.switch_turn()

    def process_player_move(self, position, valid_moves):
        """
        Processes the player's move based on their mouse click.

        Args:
            position: Tuple (x, y) of mouse click coordinates.
            valid_moves: List of valid move positions.

        Returns:
            bool: True if the move was valid, False otherwise.
        """
        x, y = position
        row, col = y // SQUARE_SIZE, x // SQUARE_SIZE

        if (row, col) in valid_moves:
            # Identify the stones flipped during this move
            flipped_stones = self.board._find_stones_to_flip(
                self.game_state, self.current_player, row, col
            )
            # Note that it's important to firstly calculate the flipped_stones
            # than process the move
            # and than displaying the flip animation, otherwise it might look strange

            self.game_state, self.current_player = self.board.get_next_state(
                self.game_state, self.current_player, row, col
            )

            # Play the flip animation
            self.visuals.play_flip_animation(
                self.game_state, flipped_stones, self.current_player
            )

            return True

        print("Invalid move attempted.")
        return False

    def execute_ai_turn(self):
        """
        Executes the AI's turn by making a random valid move.
        """
        print("AI's turn")

        value  = self.mcts.model.predict(self.game_state)[1]
        self.visuals.append_value(value)

        self._run_mcts_search()



        x, y = index_to_coordinates(self.ai_action)

        flipped_stones = self.board._find_stones_to_flip(self.game_state, self.current_player, x, y)


        self.game_state, self.current_player = self.board.get_next_state(self.game_state, self.current_player, x, y)
        self.visuals.play_flip_animation(self.game_state, flipped_stones, self.current_player)
        #self.game_state, self.current_player = self.board.play_random_move(
        #    self.game_state, self.current_player
        #)
        self.mcts.root.reset()
        self.switch_turn()

    def _run_mcts_search(self):
        root = self.mcts.run_search(self.game_state, self.current_player, False)
        action = root.select_action(0)
        self.ai_action = action


    def switch_turn(self):
        """
        Switches the turn between the player and the AI.
        """
        self.is_ai_turn = not self.is_ai_turn

    def update_display(self):
        """
        Updates the game display, showing the current board and valid moves.
        """
        self.visuals.draw_board(self.game_state)
        valid_moves = self.board.get_valid_moves(self.game_state, self.current_player)
        self.visuals.mark_valid_fields(valid_moves)
        self.visuals.draw_plot()
        pygame.display.flip()

    def display_winner(self):
        """
        Determines and displays the winner of the game.
        """
        winner = self.board.determine_winner(self.game_state)
        if winner == 0:
            print("The game is a draw!")
        else:
            player = "Black" if winner == const.PlayerColor.BLACK.value else "White"
            print(f"Player {player} wins!")


def main():
    """
    Entry point to start the Player vs AI game.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    game = GamePvsAi(screen)
    game.run_game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
