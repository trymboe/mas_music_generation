import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

from config import (
    BATCH_SIZE_BASS,
    LEARNING_RATE_BASS,
    NUM_EPOCHS_BASS,
    MODEL_PATH_BASS,
    TRAIN_DATASET_PATH_BASS,
    VAL_DATASET_PATH_BASS,
    ALPHA1_BASS,
    ALPHA2_BASS,
    MAX_BATCHES_BASS,
    DEVICE,
)


def train_bass(model: nn.Module) -> None:
    """
    Trains the bass model using the provided dataset.

    Parameters
    ----------
    model : nn.Module
        The bass model to be trained.
    """
    bass_dataset_train = torch.load(TRAIN_DATASET_PATH_BASS)
    bass_dataset_val = torch.load(VAL_DATASET_PATH_BASS)

    # Create DataLoader
    dataloader_train = DataLoader(
        bass_dataset_train, batch_size=BATCH_SIZE_BASS, shuffle=True
    )
    dataloader_val = DataLoader(
        bass_dataset_val, batch_size=BATCH_SIZE_BASS, shuffle=True
    )

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_BASS)
    loss_list = []
    val_loss_list = []

    # Training loop
    for epoch in range(NUM_EPOCHS_BASS):
        batch_loss = []
        for batch_idx, (notes, durations, targets) in enumerate(dataloader_train):
            if batch_idx > MAX_BATCHES_BASS:
                break
            notes = notes.to(DEVICE)
            durations = durations.to(DEVICE)
            targets = targets.to(DEVICE)

            # Separate note and duration targets
            note_targets, duration_targets = targets[:, 0], targets[:, 1]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            note_output, duration_output = model(notes, durations)

            # Compute losses for both notes and durations
            note_loss = criterion(note_output, note_targets)
            duration_loss = criterion(duration_output, duration_targets)

            # Combine the losses
            combined_loss = note_loss * ALPHA1_BASS + duration_loss * ALPHA2_BASS

            # Backward pass and optimize
            combined_loss.backward()
            optimizer.step()
            batch_loss.append(combined_loss.item())

        val_loss = get_validation_loss(model, dataloader_val, criterion)

        loss_list.append(np.mean(batch_loss))
        val_loss_list.append(val_loss)
        print(
            f"Epoch:  {epoch + 1} Loss: {round(loss_list[-1], 2)} Validation loss: {round(val_loss_list[-1],2)}"
        )

    with open(
        "results/data/bass/training_data" + str(NUM_EPOCHS_BASS) + ".json", "w"
    ) as file:
        json.dump(loss_list, file)
        json.dump(val_loss_list, file)

    plot_loss(loss_list, val_loss_list)
    torch.save(model, MODEL_PATH_BASS)


def get_validation_loss(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    model.eval()
    batch_loss = []
    for batch_idx, (notes, durations, targets) in enumerate(dataloader):
        if batch_idx > MAX_BATCHES_BASS / 10:
            break
        # Separate note and duration targets
        note_targets, duration_targets = targets[:, 0], targets[:, 1]

        # Forward pass
        note_output, duration_output = model(notes, durations)

        # Compute losses for both notes and durations
        note_loss = criterion(note_output, note_targets)
        duration_loss = criterion(duration_output, duration_targets)

        # Combine the losses
        combined_loss = note_loss * ALPHA1_BASS + duration_loss * ALPHA2_BASS

        # Backward pass and optimize
        combined_loss.backward()
        batch_loss.append(combined_loss.item())

    model.train()
    return np.mean(batch_loss)


import matplotlib.pyplot as plt


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
    plt.savefig("figures/bass_training_loss.png")

    # Optional: Show the plot
    plt.show()
