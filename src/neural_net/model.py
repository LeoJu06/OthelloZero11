import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.neural_net.preprocess_board import preprocess_board


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class OthelloZeroModel(nn.Module):
    def __init__(self, board_size=8, action_size=64, device="cpu", num_res_blocks=13):
        super().__init__()
        self.device = device

        # Initial Convolution
        self.conv1 = nn.Conv2d(2, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # Residual Tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # Policy Head (mehr Kanäle)
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_size)  # Logits!

        # Value Head (mehr Kanäle)
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)  # Logits!

        self.to(device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # Policy Head (Logits)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # Keine Softmax hier!

        # Value Head (Logits)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Tanh für [-1, 1]

        return p, v

       
    def predict(self, board):
        """
        Makes predictions for a single board.

        Args:
            board (np.ndarray): Shape (8, 8), single board.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and value.
        """
        board = preprocess_board(board)  # Convert to [2, 8, 8]
        board = torch.FloatTensor(board).to(self.device).unsqueeze(0)  # Add batch dim -> [1, 2, 8, 8]

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)
            pi = F.softmax(pi, dim=1)

        return pi.data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()


    def predict_batch(self, boards):
        """
        Makes predictions for a batch of boards.

        Args:
            boards (np.ndarray): Shape (batch, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and values for the batch.
        """
        boards = np.array([preprocess_board(board) for board in boards])  # Convert all to [batch, 2, 8, 8]
        boards = torch.FloatTensor(boards).to(self.device)

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(boards)
            pi = F.softmax(pi, dim=1)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()


