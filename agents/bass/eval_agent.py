import torch
import torch.nn.functional as F

from config import DEVICE
from data_processing import Bass_Dataset
from ..utils import select_with_preference


def predict_next_k_notes_bass(model, dataset_primer, config) -> list[int, int]:
    predicted_notes_durations: list[tuple[int, int]] = []

    primer, _ = get_primer_sequence_bass(dataset_primer)
    note_sequence, duration_sequence = primer
    note_sequence = note_sequence.unsqueeze(0)  # Add a batch dimension
    duration_sequence = duration_sequence.unsqueeze(0)  # Add a batch dimension

    model.eval()  # Set the model to evaluation mode
    running_length = 0

    with torch.no_grad():
        while True:
            note_output, duration_output = model(note_sequence, duration_sequence)

            # Apply softmax to get probabilities for notes and durations
            note_probabilities = F.softmax(note_output[0, :], dim=-1).view(-1)

            duration_probabilities = F.softmax(duration_output[0, :], dim=0)

            if config["DURATION_PREFERENCES_BASS"]:
                duration_probabilities = select_with_preference(
                    duration_probabilities, config["DURATION_PREFERENCES_BASS"]
                )

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)
            if next_duration.item() == 0:
                next_duration = torch.tensor([[4]])

            # Stop if the sequence length exceeds the specified length
            if (running_length + next_duration).item() >= config["LENGTH"] * 4:
                # Add the last note/duration
                predicted_notes_durations.append(
                    (0, config["LENGTH"] * 4 - running_length)
                )
                return predicted_notes_durations

            predicted_notes_durations.append((next_note.item(), next_duration.item()))
            running_length += next_duration.item()

            # Use sliding window method: drop the first note/duration, append the predicted note/duration
            note_sequence = torch.cat(
                [note_sequence[:, 1:].to(DEVICE), next_note.to(DEVICE)], dim=1
            )
            duration_sequence = torch.cat(
                [duration_sequence[:, 1:].to(DEVICE), next_duration.to(DEVICE)], dim=1
            )


def get_primer_sequence_bass(dataset_primer: int) -> tuple[int, int]:
    # primer_part: int = dataset_primer_start
    primer_sequence: tuple[int, int] = (
        dataset_primer[0],
        dataset_primer[1],
    )
    ground_truth = dataset_primer[2]

    return primer_sequence, ground_truth
