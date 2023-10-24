import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from config import BATCH_SIZE_BASS, LEARNING_RATE_BASS, NUM_EPOCHS_BASS, MODEL_PATH_BASS


def train_bass(model: nn.Module, dataset: Dataset) -> None:
    """
    Trains the bass model using the provided dataset.

    Parameters
    ----------
    model : nn.Module
        The bass model to be trained.
    dataset : Dataset
        The dataset to train the model on.
    """

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_BASS, shuffle=True)

    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_BASS)
    loss_list = []

    # Training loop
    for epoch in range(NUM_EPOCHS_BASS):
        for batch_idx, (notes, durations, targets) in enumerate(dataloader):
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
            combined_loss = note_loss + duration_loss

            # Backward pass and optimize
            combined_loss.backward()
            optimizer.step()

            loss_list.append(combined_loss.item())

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS_BASS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {combined_loss.item():.4f}"
                )

    plot_loss(loss_list)
    torch.save(model, MODEL_PATH_BASS)


def plot_loss(loss_values: list[float]) -> None:
    """
    Plots the training loss over batches.

    Parameters
    ----------
    loss_values : List[float]
        A list of loss values to be plotted.

    Returns
    ---------
    None
    """

    plt.plot(loss_values)
    plt.title("Training Loss Bass")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("figures/bass_training_loss.png")
