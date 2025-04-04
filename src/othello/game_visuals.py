import pygame
import os
from src.othello.game_constants import PlayerColor
from src.othello.game_settings import FPS

from src.othello.game_settings import BACKGROUND_COLOR, ROWS, COLS
from src.othello.game_settings import GRID_COLOR, SQUARE_SIZE, COLOR_VALID_FIELDS
import src.othello.game_constants as const

import matplotlib.pyplot as plt
import numpy as np
import pygame
import io

import matplotlib.pyplot as plt
import numpy as np
import io
import pygame
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np
import io
import pygame
from matplotlib.ticker import MaxNLocator

def generate_plot_image(data_points):
    """
    Erstellt ein Matplotlib-Diagramm mit Zugnummern auf der X-Achse und gibt es als Pygame-Bild zurück.

    Args:
        data_points (list): Eine Liste von Zahlen, die geplottet werden sollen.

    Returns:
        pygame.Surface: Das gerenderte Diagramm als Bild.
    """
    fig, ax = plt.subplots(figsize=(4, 2))  # Größeren Plot anpassen
    
    # Erstellen des Plots mit X-Achse als Züge (0,1,2,...)
    turns = list(range(len(data_points)))  # X-Achse: Zugnummern
    ax.plot(turns, data_points, marker="o", linestyle="-", color="blue", markersize=4)

    # Achsentitel setzen
    ax.set_title("Value estimation of the neural network")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Value")

    # Maximale Anzahl der Ticks auf der X-Achse steuern (z.B. alle 5 Züge)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', steps=[5]))

    # Y-Achse anpassen, um den Bereich (-1 bis 1) zu kontrollieren
    ax.set_ylim(-1.1, 1.1)

    # X-Achsen-Beschriftungen horizontal ausrichten
    plt.xticks(rotation=0)

    # Letzten Wert als Text an der X-Achse hinzufügen
    ax.text(
        turns[-1],  # Position auf der X-Achse (letzter Wert)
        data_points[-1],  # Position auf der Y-Achse (entsprechender Wert)
        f"{data_points[-1]:.2f}",  # Text (der Wert mit 2 Dezimalstellen)
        ha='center',  # Horizontale Ausrichtung
        va='top',  # Vertikale Ausrichtung (Text etwas oberhalb des Punktes)
        color='black',  # Textfarbe
        fontsize=10,  # Schriftgröße
    )

    # Layout anpassen
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Konvertiere den Plot in ein Bild
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)

    # Lade das Bild in Pygame
    buf.seek(0)
    image = pygame.image.load(buf)
    
    return image


class GameVisuals:
    """
    Handles the rendering and animation for the Othello game.

    Attributes:
        screen (pygame.Surface): The Pygame surface for rendering.
        board (list): Current state of the board.
        clock (pygame.time.Clock): Clock for managing frame rate.
        rotation_images_black_to_white (list): Preloaded frames for black-to-white flipping animation.
        rotation_images_white_to_black (list): Preloaded frames for white-to-black flipping animation.
        image_black_stone (pygame.Surface): Scaled image for black stones.
        image_white_stone (pygame.Surface): Scaled image for white stones.
    """

    def __init__(self, screen, clock):
        """
        Initializes the visual manager and preloads images for animations and rendering.

        Args:
            screen (pygame.Surface): The Pygame surface for rendering.
            board (list): Current board state.
            clock (pygame.time.Clock): Clock for managing frame rate.
        """
        self.square_size = SQUARE_SIZE  # Square size for each tile on the board
        self.screen = screen  # Screen to render the visuals
        self.clock = clock  # Clock to control the frame rate
        self.values = [0]

        # Load the images for black and white stones
        self.image_black_stone, self.image_white_stone = self._load_stone_images()

        # Load the transition images for the flipping animation
        (
            self.rotation_images_black_to_white,
            self.rotation_images_white_to_black,
        ) = self._load_transition_images()

    def _load_stone_images(self):
        """
        Loads and scales the images for black and white stones.

        Returns:
            tuple: Scaled images for black and white stones.
        """
        path_to_stone_images = os.path.join(
            os.path.dirname(__file__), "assets"
        )  # Path to images folder

        # Load and scale the black stone image
        image_black_stone = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(path_to_stone_images, "black_stone.png")),
            (SQUARE_SIZE, SQUARE_SIZE),  # Scale to fit the square size
        )

        # Load and scale the white stone image
        image_white_stone = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(path_to_stone_images, "white_stone.png")),
            (SQUARE_SIZE, SQUARE_SIZE),  # Scale to fit the square size
        )

        return image_black_stone, image_white_stone

    def _load_transition_images(self):
        """
        Loads the images for the flipping animation.

        Returns:
            tuple: Two lists of images for black-to-white and white-to-black transitions.
        """
        path_to_images = os.path.join(
            os.path.dirname(__file__), "assets", "transition_images"
        )  # Path to transition images folder

        # Load and scale the black-to-white flipping images
        rotation_images_black_to_white = [
            pygame.image.load(os.path.join(path_to_images, f"black_to_white_{i}.png"))
            for i in range(14)  # Assuming 14 frames of transition
        ]
        rotation_images_black_to_white = [
            pygame.transform.smoothscale(image, (self.square_size, self.square_size))
            for image in rotation_images_black_to_white
        ]

        # Reverse and flip the images for the white-to-black transition
        rotation_images_white_to_black = [
            pygame.transform.flip(frame, False, True)
            for frame in reversed(rotation_images_black_to_white)
        ]

        return rotation_images_black_to_white, rotation_images_white_to_black

    def play_flip_animation(self, board, flipped_stones, player):
        """
        Plays the flip animation for a set of stones.

        Args:
            flipped_stones (list of tuples): Coordinates of stones being flipped.
            player (int): The player initiating the flip (BLACK or WHITE).
        """
        # Choose the correct animation images based on the player color
        animation_images = (
            self.rotation_images_white_to_black
            if player == PlayerColor.WHITE.value
            else self.rotation_images_black_to_white
        )

        # Iterate through each frame of the animation
        for frame in animation_images:
            # Redraw the board in each frame to keep static elements visible
            self.draw_board(board)

            # Overlay the current animation frame on the flipping stones
            for row, col in flipped_stones:
                self.screen.blit(
                    frame, (col * self.square_size, row * self.square_size)
                )

            pygame.display.flip()  # Refresh the display to show the updated frame
            self.clock.tick(FPS)  # Maintain consistent frame rate as defined by FPS

    def draw_board(self, board):
        """
        Draws the game board and pieces.

        Args:
            board (list): The current state of the game board.
        """
        self.screen.fill(
            BACKGROUND_COLOR
        )  # Fill the background with the specified color

        # Loop through each row and column of the board to draw the grid and pieces
        for row in range(ROWS):
            for col in range(COLS):
                # Draw the grid square (the border around each tile)
                pygame.draw.rect(
                    self.screen,
                    GRID_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,  # Thickness of the grid line
                )

                # Draw the piece (stone) based on the current board state
                piece = board[row][col]
                if piece == const.PlayerColor.BLACK.value:
                    self.screen.blit(
                        self.image_black_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )  # Draw black stone
                elif piece == const.PlayerColor.WHITE.value:
                    self.screen.blit(
                        self.image_white_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )  # Draw white stone

    def mark_valid_fields(self, valid_moves: list):
        for x, y in valid_moves:
            # Calculate the position of the rectangle
            rect_position = (
                y * SQUARE_SIZE + 1,
                x * SQUARE_SIZE + 1,
            )  # adding a tiny value for not overdraing the grid lines
            rect_size = (
                SQUARE_SIZE - 2,
                SQUARE_SIZE - 2,
            )  # subtracting a but for not overdrawing the grid lines

            # Create the rectangle and draw it
            pygame.draw.rect(
                self.screen, COLOR_VALID_FIELDS, pygame.Rect(rect_position, rect_size)
            )

    def draw_plot(self,position=(900, 50)):
        """
        Zeichnet einen Live-Plot in das Pygame-Fenster.

        Args:
            data_points (list): Eine Liste von Zahlen, die geplottet werden sollen.
            position (tuple): Die (x, y)-Position des Plots im Fenster.
        """
        plot_image = generate_plot_image(self.values)
        self.screen.blit(plot_image, position)
        pygame.display.flip()

    def append_value(self, value):
        self.values.append(value)
