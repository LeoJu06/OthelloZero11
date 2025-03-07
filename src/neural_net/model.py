import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        # Save the input to add later
        identity = x
        
        # First layer
        x = F.relu(self.fc1(x))
        
        # Second layer
        x = self.fc2(x)
        
        # Skip connection (add the original input to the output)
        x += identity
        
        # Apply ReLU after adding
        x = F.relu(x)
        
        return x


# OthelloZeroModel with Residual Blocks
class OthelloZeroModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(OthelloZeroModel, self).__init__()

        self.device = device
        self.board_size = board_size  # Expected 8x8
        self.action_size = action_size

        # Initial fully connected layers
        self.fc1 = nn.Linear(in_features=self.board_size * self.board_size, out_features=64)
        
        # Stack of Residual Blocks (9 blocks)
        self.residual_blocks = nn.ModuleList([ResidualBlock(64, 64) for _ in range(9)])
        
        # Output heads: one for actions and one for the value
        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)

        self.to(device)

    def forward(self, x):
        # Flatten the board(s) for the fully connected layers
        x = x.view(x.size(0), -1)  # Batch size stays the same, flatten board

        # Initial fully connected layer
        x = F.relu(self.fc1(x))

        # Pass through all the residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Action and value heads
        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        # Softmax for actions, Tanh for the value
        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    def predict(self, board):
        """
        Makes predictions for a single board.

        Args:
            board (np.ndarray): 8x8 Board (single).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and value.
        """
        # Convert to tensor and ensure batch dimension
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device).unsqueeze(0)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()

    def predict_batch(self, boards):
        """
        Makes predictions for a batch of boards.

        Args:
            boards (np.ndarray): Batch of boards (N, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and values for the batch.
        """
        # Convert to tensor
        boards = boards.to(self.device)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(boards)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()
    


if __name__ == "__main__":

    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_loss(losses):
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.show()

    # Define the loss function
    def compute_loss(policy_logits, value_logit, target_policy, target_value):
        """
        Compute the combined loss for both action probabilities and value.
        Args:
            policy_logits (Tensor): Predicted policy (action probabilities)
            value_logit (Tensor): Predicted value (single scalar per board)
            target_policy (Tensor): True policy (target action probabilities)
            target_value (Tensor): True value (target value)
        
        Returns:
            Tensor: Combined loss
        """
        # Cross-entropy loss for action probabilities (policy loss)
        # target_policy should be indices for cross-entropy, not probabilities
        policy_loss = F.cross_entropy(policy_logits, target_policy)

        # Mean squared error loss for value prediction
        value_loss = F.mse_loss(value_logit.squeeze(), target_value)

        # Combine the two losses
        total_loss = policy_loss + value_loss
        return total_loss

    # Training function
    def train(model, train_loader, optimizer, epochs, device):
        model.train()  # Stelle sicher, dass das Modell im Trainingsmodus ist
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (board_state, target_policy, target_value) in enumerate(train_loader):
                board_state = board_state.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                
                # Optimieren
                optimizer.zero_grad()

                # Vorwärtsdurchlauf
                pi, v = model(board_state)
                
                # Policy Loss (Cross Entropy)
                policy_loss = F.cross_entropy(pi, target_policy.argmax(dim=1))  # `target_policy.argmax()` gibt den Index der bevorzugten Aktion
                
                # Value Loss (Mean Squared Error)
                value_loss = F.mse_loss(v.view(-1), target_value)
                
                # Gesamtverlust
                loss = policy_loss + value_loss
                total_loss += loss.item()
                
                # Rückwärtsdurchlauf
                loss.backward()
                optimizer.step()

            average_loss = total_loss / len(train_loader)
            losses.append(average_loss)  # Speichern des Durchschnitts des Losses für jede Epoche

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
        
        plot_loss(losses)


    # Example data loader (assuming you have the data)
    import pickle
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    path = os.path.join(data_dir, "examples_0.pkl")
    # Laden der Daten aus der .pkl-Datei
    with open(path, 'rb') as f:
        train_data = pickle.load(f)

    # Überprüfen Sie den Inhalt der geladenen Daten (optional)
    print(f"Beispielhafte Daten: {train_data[:2]}")  # Zeigt die ersten 2 Einträge an


    class OthelloDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            """
            Angenommen, dass `data` eine Liste von Tupeln ist: (board_state, target_policy, target_value)
            - board_state: 8x8-Array des Othello-Bretts
            - target_policy: Array der Wahrscheinlichkeiten für jede mögliche Aktion (Größe 64)
            - target_value: Wert des Spiels (z. B. -1 für Verlust, 0 für Unentschieden, 1 für Sieg)
            """
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            board_state, target_policy, target_value = self.data[idx]
            board_state = torch.tensor(board_state, dtype=torch.float32)  # Umwandlung des Bretts in einen Tensor
            
            # target_policy bleibt als Wahrscheinlichkeitsverteilung
            target_policy = torch.tensor(target_policy, dtype=torch.float32)  # Wahrscheinlichkeitsverteilung als Tensor
            
            target_value = torch.tensor(target_value, dtype=torch.float32)  # Ziel für den Wert
            return board_state, target_policy, target_value


    # Umwandlung der geladenen Daten in das richtige Format
    # Angenommen, die geladenen Daten sind in einer Liste gespeichert:
    # Beispiel: [(board_state, target_policy, target_value), ...]
    # `board_state` ist ein 8x8-Array, `target_policy` ist der Index der gewählten Aktion (Integer),
    # `target_value` ist der Wert des Spiels (-1, 0, 1).

    # Initialisieren des Datasets
    train_dataset = OthelloDataset(train_data)

    # Erstellen des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Trainingsfunktion und Trainingseinrichtung wie zuvor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OthelloZeroModel(board_size=8, action_size=64, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Trainieren des Modells
    train(model, train_loader, optimizer, epochs=80, device=device)



    """
    if __name__ == "__main__":
        # Example: Prediction for Othello game model
        model = OthelloZeroModel(
            board_size=8, action_size=64, device="cuda"
        )  # Example device: 'cuda'

        # Single board example
        single_board = np.zeros((8, 8))  # An empty 8x8 board
        single_board[3, 3] = 1  # Example piece placement

        # Batch of boards example
        batch_boards = torch.tensor(np.stack([single_board, single_board * -1]), dtype=torch.float32)  # Two boards in a batch

        # Single board prediction
        pi_single, v_single = model.predict(single_board)
        print("Single Board - Action Probabilities:", pi_single)
        print("Single Board - Value:", v_single)

        # Batch board prediction
        pi_batch, v_batch = model.predict_batch(batch_boards)
        print("Batch Boards - Action Probabilities:", pi_batch)
        print("Batch Boards - Values:", v_batch)
    """