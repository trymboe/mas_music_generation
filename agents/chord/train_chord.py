import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

from config import (
    BATCH_SIZE_CHORD,
    LEARNING_RATE_CHORD,
    NUM_EPOCHS_CHORD,
    MODEL_PATH_CHORD,
    TRAIN_DATASET_PATH_CHORD,
    VAL_DATASET_PATH_CHORD,
    MAX_BATCHES_CHORD,
    DEVICE,
)


def train_chord(model: nn.Module) -> None:
    """
    Trains the chord agent using the provided dataset.

    Parameters
    ----------
    model : nn.Module
        The chord network model to be trained.
    dataset : Dataset
        The dataset to be used for training the model.

    Returns
    -------
    None
    """

    chord_dataset_train = torch.load(TRAIN_DATASET_PATH_CHORD)
    chord_dataset_val = torch.load(VAL_DATASET_PATH_CHORD)

    # Create DataLoader
    dataloader_train = DataLoader(
        chord_dataset_train, batch_size=BATCH_SIZE_CHORD, shuffle=True
    )
    dataloader_val = DataLoader(
        chord_dataset_val, batch_size=BATCH_SIZE_CHORD, shuffle=True
    )

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_CHORD)
    loss_list = []
    val_loss_list = []
    model.to(DEVICE)

    # Training loop
    for epoch in range(NUM_EPOCHS_CHORD):
        batch_loss = []
        for batch_idx, (data, targets) in enumerate(dataloader_train):
            if batch_idx > MAX_BATCHES_CHORD:
                break
            # Zero gradients
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        val_loss = get_validation_loss(model, dataloader_val, criterion)

        loss_list.append(np.mean(batch_loss))
        val_loss_list.append(val_loss)

        print(
            f"Epoch:  {epoch + 1} Loss: {round(loss_list[-1], 2)} Validation loss: {round(val_loss_list[-1],2)}"
        )

    with open(
        "results/data/chord/training_data" + str(NUM_EPOCHS_CHORD) + ".json", "w"
    ) as file:
        json.dump(loss_list, file)
        json.dump(val_loss_list, file)

    plot_loss(loss_list, val_loss_list)
    torch.save(model, MODEL_PATH_CHORD)


def get_validation_loss(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    model.eval()
    batch_loss = []
    for batch_idx, (data, targets) in enumerate(dataloader):
        if batch_idx > MAX_BATCHES_CHORD / 10:
            break
        # Separate note and duration targets
        output = model(data)

        # Compute loss
        loss = criterion(output, targets)

        # Backward pass and optimize
        loss.backward()
        batch_loss.append(loss.item())

    model.train()
    return np.mean(batch_loss)


def plot_loss(loss_values: list[float], val_loss_values: list[float]) -> None:
    """
    Plots the training and validation loss over batches.

    Parameters
    ----------
    loss_values : list[float]
        A list of training loss values to be plotted.
    val_loss_values : list[float]
        A list of validation loss values to be plotted.

    Returns
    -------
    None
    """

    # Plot training loss
    plt.plot(loss_values, color="blue", label="Training Loss")

    # Plot validation loss
    plt.plot(val_loss_values, color="red", label="Validation Loss")

    # Add title and labels
    plt.title("Training and Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    # Add legend
    plt.legend()

    # Optional: Add grid for better readability
    plt.grid(True)

    # Save the plot
    plt.savefig("figures/chord_training_loss.png")

    # Optional: Show the plot
    plt.show()
