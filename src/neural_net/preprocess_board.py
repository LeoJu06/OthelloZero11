import numpy as np
from src.othello.othello_game import OthelloGame


def preprocess_board(board, current_player=1):
    """
    Converts a canonical Othello board [8, 8] into three-channel format [3, 8, 8].
    
    Args:
        board (np.ndarray): The Othello board where 1 = player1, -1 = player2.
        current_player (int): 1 or -1, indicating which player is acting.
    
    Returns:
        np.ndarray: A board with shape (3, 8, 8) for model input.
                   Channels: [current_player, opponent, valid_moves]
    """
    board = np.array(board, dtype=np.float32)
    
    # Channel 1: Current player's pieces
    player_channel = (board == current_player).astype(np.float32)
    
    # Channel 2: Opponent's pieces
    opponent_channel = (board == -current_player).astype(np.float32)
    
    # Channel 3: Valid moves (1 = legal, 0 = illegal)
    valid_moves = OthelloGame().get_valid_moves(board, current_player)
    valid_moves_channel = OthelloGame().legal_moves_to_2d(valid_moves)
    
    return np.stack([player_channel, opponent_channel, valid_moves_channel], axis=0)  # Shape: [3, 8, 8]