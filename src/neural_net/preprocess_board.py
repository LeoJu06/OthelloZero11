import numpy as np

def preprocess_board(board):
    """
    Converts a canonical Othello board [8, 8] into two-channel format [2, 8, 8].

    Args:
        board (np.ndarray): The canonical Othello board where 1 = current player, -1 = opponent.

    Returns:
        np.ndarray: A board with shape (2, 8, 8) for model input.
    """
    board = np.array(board, dtype=np.float32)

    # Channel 1: Current player's pieces
    player_channel = (board == 1).astype(np.float32)

    # Channel 2: Opponent's pieces
    opponent_channel = (board == -1).astype(np.float32)

    return np.stack([player_channel, opponent_channel], axis=0)  # Shape: [2, 8, 8]