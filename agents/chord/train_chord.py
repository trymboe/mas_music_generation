import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE_CHORD,
    LEARNING_RATE_CHORD,
    NUM_EPOCHS_CHORD,
    MODEL_PATH_CHORD,
)


def train_chord(model: nn.Module, dataset: Dataset):
    # Hyperparameters

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_CHORD, shuffle=True)

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_CHORD)
    loss_list = []

    # Training loop
    for epoch in range(NUM_EPOCHS_CHORD):
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS_CHORD}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    plot_loss(loss_list)
    torch.save(model, MODEL_PATH_CHORD)


def plot_loss(loss_values):
    plt.figure()
    plt.plot(loss_values)
    plt.title("Training Loss Chord")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("figures/chord_training_loss.png")
