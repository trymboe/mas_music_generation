import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import tensorflow as tf

from .melody_network import Melody_Network
from data_processing import Melody_Dataset


from config import (
    NUM_EPOCHS_MELODY,
    LEARNING_RATE_MELODY,
    BATCH_SIZE_MELODY,
    MODEL_PATH_MELODY,
    DEVICE,
    PITCH_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    ALPHA1,
    ALPHA2,
    DATASET_SIZE_MELODY,
)


def train_melody(
    model: Melody_Network,
    dataset: Melody_Dataset,
) -> None:
    """
    Trains the MelodyGenerator model on the provided dataset.

    Args:
        model (MelodyGenerator): The neural network model to train.
        dataset (Dataset): The dataset to train the model on.


    Returns:
        list: A list of average epoch losses for each epoch.
    """

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE_MELODY, shuffle=True, collate_fn=process_data
    )
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_MELODY)

    all_loss = []

    # Training loop
    for epoch in range(NUM_EPOCHS_MELODY):
        for idx, batch in enumerate(dataloader):
            (
                pitches,
                durations,
                current_chord,
                next_chord,
            ) = batch[0]
            gt_pitches, gt_durations = batch[1]
            accumulated_time = batch[2]
            time_left_on_chord = batch[3]

            gt_pitches.to(DEVICE)
            gt_durations.to(DEVICE)
            # Zero the parameter gradients
            optimizer.zero_grad()

            x = torch.cat(
                (
                    pitches,
                    durations,
                    current_chord,
                    next_chord,
                ),
                dim=2,
            )

            pitch_logits, duration_logits = model(
                x, accumulated_time, time_left_on_chord
            )

            pitch_loss = criterion(pitch_logits, get_gt(gt_pitches.squeeze(1)).to(DEVICE))
            duration_loss = criterion(duration_logits, get_gt(gt_durations.squeeze(1)).to(DEVICE))

            loss = pitch_loss * ALPHA1 + duration_loss * ALPHA2

            all_loss.append(loss.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            print(f"batch {idx}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model
    torch.save(model, MODEL_PATH_MELODY)
    plot_loss(all_loss)

    with open(
        "results/data/" + DATASET_SIZE_MELODY + str(NUM_EPOCHS_MELODY) + ".json", "w"
    ) as file:
        json.dump(all_loss, file)
    plt.show()


def get_gt(gt):
    if len(gt.size()) > 1 and gt.size(1) > 1:  # Check for one-hot encoding
        return torch.argmax(gt, dim=1)


def plot_loss(loss_values: list[float]) -> None:
    """
    Plots the training loss over batches.

    Parameters
    ----------
    loss_values : List[float]
        A list of loss values to be plotted.

    Returns
    -------
    None
    """

    plt.figure()
    plt.plot(loss_values)
    plt.title("Training Loss melody")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("figures/melody_training_loss.png")


def process_data(batch):
    # batch - input/label - Sequence - note - pitch/duration/chord
    # Initialize lists to store sequence data
    pitches, durations, current_chords, next_chords = [], [], [], []
    current_chord_time_lefts, accumulated_times = [], []
    ground_truth_pitches, ground_truth_durations = [], []

    # Iterate through each sequence of inputs in the batch
    for idx, sequence in enumerate(batch):
        pitches.append([])
        durations.append([])
        current_chords.append([])
        next_chords.append([])
        current_chord_time_lefts.append([])
        accumulated_times.append([])
        for note in sequence[0]:
            # Convert each item to a tensor before appending
            pitches[idx].append(torch.tensor(note[0]))
            durations[idx].append(torch.tensor(note[1]))
            current_chords[idx].append(torch.tensor(note[2]))
            next_chords[idx].append(torch.tensor(note[3]))
            current_chord_time_lefts[idx].append(torch.tensor(note[4]))
            accumulated_times[idx].append(torch.tensor(note[5]))
        pitches[idx] = torch.stack(pitches[idx])
        durations[idx] = torch.stack(durations[idx])
        current_chords[idx] = torch.stack(current_chords[idx])
        next_chords[idx] = torch.stack(next_chords[idx])
        current_chord_time_lefts[idx] = torch.stack(current_chord_time_lefts[idx])
        accumulated_times[idx] = torch.stack(accumulated_times[idx])

    # Iterate through each sequence of labels in the batch
    for idx, sequence in enumerate(batch):
        ground_truth_pitches.append([])
        ground_truth_durations.append([])
        for note in sequence[1]:
            # Convert each item to a tensor before appending
            ground_truth_pitches[idx].append(torch.tensor(note[0]))
            ground_truth_durations[idx].append(torch.tensor(note[1]))
        ground_truth_pitches[idx] = torch.stack(ground_truth_pitches[idx])
        ground_truth_durations[idx] = torch.stack(ground_truth_durations[idx])

    # Convert lists of tensors to a single tensor for each
    pitches_tensor = torch.stack(pitches)
    pitches_tensor = torch.stack(pitches)
    durations_tensor = torch.stack(durations)
    current_chord_tensor = torch.stack(current_chords)
    next_chord_tensor = torch.stack(next_chords)
    current_chord_time_left_tensor = torch.stack(current_chord_time_lefts)
    accumulated_time_tensor = torch.stack(accumulated_times)

    # Stack ground truth pitches and durations
    ground_truth_pitches_tensor = torch.stack(ground_truth_pitches)
    ground_truth_durations_tensor = torch.stack(ground_truth_durations)

    # Group the inputs and targets
    inputs = (pitches_tensor, durations_tensor, current_chord_tensor, next_chord_tensor)
    targets = (ground_truth_pitches_tensor, ground_truth_durations_tensor)

    return inputs, targets, accumulated_time_tensor, current_chord_time_left_tensor
