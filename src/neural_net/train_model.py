import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from random import shuffle
from src.neural_net.preprocess_board import preprocess_board
from src.data_manager.data_manager import DataManager

def train(model, data, epochs=10, batch_size=128, lr=0.001, accumulation_steps=4):
    """Train AlphaZero model with gradient accumulation to simulate a larger batch size"""
    device = next(model.parameters()).device
    
    # Convert data to tensors
    boards = torch.stack([torch.tensor(preprocess_board(d[0])) for d in data]).float()
    policies = torch.tensor(np.array([d[1] for d in data])).float()
    values = torch.tensor(np.array([d[2] for d in data])).float()
    
    # Normalize policies to ensure sum=1
    policies = policies / policies.sum(dim=1, keepdim=True)
    
    # Move to device
    boards, policies, values = boards.to(device), policies.to(device), values.to(device)

    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    # Create DataLoader
    dataset = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    policy_losses, value_losses = [], []

    for epoch in range(epochs):
        epoch_policy_loss, epoch_value_loss = 0, 0
        optimizer.zero_grad()  # Initialize the gradient accumulator

        for i, (batch_boards, batch_policies, batch_values) in enumerate(dataloader):
            # Forward pass
            logits, value_pred = model(batch_boards)

            # Policy loss (KL-Divergence)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = policy_loss_fn(log_probs, batch_policies)
            
            # Value loss
            value_loss = value_loss_fn(value_pred.squeeze(), batch_values)
            
            # Total loss
            loss = policy_loss + value_loss
            loss.backward()

            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after the update

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

        # Calculate epoch averages
        avg_policy = epoch_policy_loss / len(dataloader)
        avg_value = epoch_value_loss / len(dataloader)
        policy_losses.append(avg_policy)
        value_losses.append(avg_value)

        print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {avg_policy:.4f} | Value Loss: {avg_value:.4f}")

    return model, policy_losses, value_losses




if __name__ == "__main__":

# Load data
    data_manager = DataManager()
    examples = data_manager.load_examples()
    print(len(examples))

    # Initialize the model
    model = data_manager.load_model()
    
    train(model, examples, lr=0.004, batch_size=2048, epochs=100)

