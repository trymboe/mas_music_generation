import random
import torch
import torch.nn.functional as F
import copy

from .chord_network import Chord_Network


def predict_next_k_notes_chords(
    model: Chord_Network, full_bass_sequence: list, dataset_primer: list, config: dict
):
    """
    Predicts the next notes of chords based on the given model, full bass sequence, and dataset primer.

    Args:
    -----
        model (Chord_Network): The chord network model used for prediction.
        full_bass_sequence (list): The full bass sequence.
        dataset_primer (list): The dataset primer.
        config (dict): The configuration dictionary.

    Returns:
    -----
        list: A list of tuples representing the predicted chords. Each tuple contains the current bass note and the predicted chord type.
    """
    if config["BAD_COMS"]:
        actuall_bass_sequence = copy.deepcopy(full_bass_sequence)
        random_durations = generate_random_durations(
            config["LENGTH"] * 4, len(full_bass_sequence)
        )
        for i in range(len(full_bass_sequence)):
            full_bass_sequence[i] = (random.randint(0, 11), random_durations[i])

    chord_primer = get_input_sequence_chords(dataset_primer, full_bass_sequence)

    predicted_chords = []

    # Add a batch dimension
    input_sequence = chord_primer.unsqueeze(0)

    model.eval()
    print("Generating chords uding ", str(model))
    with torch.no_grad():
        for i in range(len(full_bass_sequence)):
            # Predict chord type
            output = model(input_sequence)

            # Apply softmax to get probabilities
            chord_probabilities = F.softmax(
                output[0, :], dim=-1
            )  # Only consider the last prediction

            # Sample from the distribution
            next_chord_type = torch.multinomial(chord_probabilities, 1).item()
            predicted_chords.append((input_sequence[0, -1, 0].item(), next_chord_type))

            if i != len(full_bass_sequence) - 1:
                input_sequence = update_input_sequence(
                    input_sequence, next_chord_type, full_bass_sequence[i + 1][0]
                )
    if config["BAD_COMS"]:
        predicted_chords = merge_chords_and_bass(
            predicted_chords, actuall_bass_sequence
        )
        print(predicted_chords)

    return predicted_chords


def generate_random_durations(total, parts):
    """
    Generates random durations for a given total duration and number of parts.
    The durations length always sums up to the total duration.

    args:
    -----
        total (int): The total duration to split.
        parts (int): The number of parts to split the total duration into.
    """

    # Generate 'parts - 1' random split points
    split_points = sorted(random.sample(range(1, total), parts - 1))
    # Calculate durations from split points
    durations = (
        [split_points[0]]
        + [split_points[i] - split_points[i - 1] for i in range(1, parts - 1)]
        + [total - split_points[-1]]
    )
    return durations


def merge_chords_and_bass(chords, bass):
    """
    Merges the chords and bass into a single list. Uses the original bass sequence combined with the predicted chords.
    Based on the bad communications.

    Args:
    -----
        chords (list): The list of chords.
        bass (list): The list of bass notes.

    Returns:
    -----
        list: A list of tuples representing the merged chords and bass.
    """
    merged = []
    for i in range(len(chords)):
        merged.append((bass[i][0], chords[i][1]))
    return merged


def update_input_sequence(
    input_sequence: torch.tensor, next_chord_type: int, next_note: int
) -> torch.tensor:
    """
    Update the input sequence with the next chord type and note.

    Args:
    -----
        input_sequence (torch.tensor): The input sequence tensor.
        next_chord_type (int): The next chord type.
        next_note (int): The next note.

    Returns:
    -----
        torch.tensor: The updated input sequence tensor.
    """

    input_sequence_list = input_sequence.squeeze().tolist()
    input_sequence_list[-1] = [input_sequence_list[-1][0], next_chord_type]

    input_sequence_list = input_sequence_list[1:]

    input_sequence_list.append([next_note, 6])
    return torch.tensor(input_sequence_list).unsqueeze(0)


def get_input_sequence_chords(dataset_primer, full_bass_sequence):
    """
    Converts the dataset primer and full bass sequence into an input tensor for chord generation.

    Args:
    -----
        dataset_primer (list): The dataset primer containing chord events.
        full_bass_sequence (list): The full bass sequence.

    Returns:
    -----
        torch.Tensor: The input tensor for chord generation.
    """

    input_sequence = []

    for i, event in enumerate(dataset_primer):
        if i == 0:
            continue
        input_sequence.append([int(event[0]), int(event[1])])

    # use placeholder token on last value together with predicted bass note
    input_sequence.append([int(full_bass_sequence[0][0]), 6])

    # Convert the list of lists to a tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.int64)

    return input_tensor
