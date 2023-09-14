import torch
import torch.nn.functional as F

from config import INT_TO_NOTE, SEQUENCE_LENGTH_BASS, SEQUENCE_LENGTH_CHORD


def predict_next_k_notes_bass(model, initial_sequence, k):
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


def predict_next_k_notes_chords(model, input_sequence):
    predicted_chord_types = []

    # Add a batch dimension
    input_sequence = input_sequence.unsqueeze(0)

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for i in range(7, input_sequence.size(1)):  # Start predicting from the 8th note
            # Get the current sequence up to the position `i`
            current_sequence = input_sequence[:, : i + 1, :]

            # Predict chord type
            output = model(current_sequence)

            # Apply softmax to get probabilities
            chord_probabilities = F.softmax(
                output[0, :], dim=-1
            )  # Only consider the last prediction
            print(chord_probabilities[0], chord_probabilities[5])

            # Sample from the distribution
            next_chord_type = torch.multinomial(chord_probabilities, 1).item()

            # Update the placeholder for the next iteration
            input_sequence[0, i, 1] = next_chord_type

    # Prepare the output list with known chord types and predicted ones
    for i in range(input_sequence.size(1)):
        predicted_chord_types.append(
            (input_sequence[0, i, 0].item(), input_sequence[0, i, 1].item())
        )

    return predicted_chord_types
