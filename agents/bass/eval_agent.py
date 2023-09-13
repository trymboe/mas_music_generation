import torch
import torch.nn.functional as F

from config import INT_TO_NOTE


def predict_next_k_notes(model, initial_sequence, k):
    predicted_notes_durations = []

    note_sequence, duration_sequence = initial_sequence
    note_sequence = note_sequence.unsqueeze(0)  # Add a batch dimension
    duration_sequence = duration_sequence.unsqueeze(0)  # Add a batch dimension

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for _ in range(k):
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

            predicted_notes_durations.append((next_note.item(), next_duration.item()))

            # Use sliding window method: drop the first note/duration, append the predicted note/duration
            note_sequence = torch.cat([note_sequence[:, 1:], next_note], dim=1)
            duration_sequence = torch.cat(
                [duration_sequence[:, 1:], next_duration], dim=1
            )

    return predicted_notes_durations
