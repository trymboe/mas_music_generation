import torch
import torch.nn.functional as F

from data_processing import Bass_Dataset


def predict_next_k_notes_bass(
    model, bass_dataset_start, dataset_primer, length
) -> list[int, int]:
    predicted_notes_durations: list[tuple[int, int]] = []

    note_sequence, duration_sequence = get_primer_sequence(
        bass_dataset_start, dataset_primer
    )
    note_sequence = note_sequence.unsqueeze(0)  # Add a batch dimension
    duration_sequence = duration_sequence.unsqueeze(0)  # Add a batch dimension

    model.eval()  # Set the model to evaluation mode
    running_length = 0

    with torch.no_grad():
        while True:
            note_output, duration_output = model(note_sequence, duration_sequence)

            # Apply softmax to get probabilities for notes and durations
            note_probabilities = F.softmax(note_output[0, :], dim=-1).view(
                -1
            )  # reshape to 1D

            duration_probabilities = F.softmax(
                duration_output[0, :], dim=0
            )  # Only consider the last prediction

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)

            # Stop if the sequence length exceeds the specified length
            if running_length + next_duration > length * 2:
                # Add the last note/duration
                predicted_notes_durations.append((0, length * 2 - running_length))
                return predicted_notes_durations

            predicted_notes_durations.append((next_note.item(), next_duration.item()))
            running_length += next_duration.item()

            # Use sliding window method: drop the first note/duration, append the predicted note/duration
            note_sequence = torch.cat([note_sequence[:, 1:], next_note], dim=1)
            duration_sequence = torch.cat(
                [duration_sequence[:, 1:], next_duration], dim=1
            )


def get_primer_sequence(
    bass_dataset: Bass_Dataset, dataset_primer_start: int
) -> tuple[int, int]:
    primer_part: int = dataset_primer_start
    primer_sequence: tuple[int, int] = (
        bass_dataset[primer_part][0],
        bass_dataset[primer_part][1],
    )

    return primer_sequence
