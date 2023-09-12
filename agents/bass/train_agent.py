import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE_BASS,
    LEARNING_RATE_BASS,
    NUM_EPOCHS_BASS,
)


def train_bass(model: nn.Module, dataset: Dataset):
    # Hyperparameters

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_BASS, shuffle=True)

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_BASS)
    loss_list = []
    # Training loop
    for epoch in range(NUM_EPOCHS_BASS):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Compute loss
            loss = criterion(output, target)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS_BASS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
    plot_loss(loss_list)
    torch.save(model.state_dict(), "models/bass/bass_model.pth")


def plot_loss(loss_values):
    plt.plot(loss_values)
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()
