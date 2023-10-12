import torch
import torch.nn.functional as F

from config import SEQUENCE_LENGTH_CHORD


def predict_next_k_notes_chords(model, input_sequence):
    predicted_chord_types = []

    # Add a batch dimension
    input_sequence = input_sequence.unsqueeze(0)

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for i in range(SEQUENCE_LENGTH_CHORD - 1, input_sequence.size(1)):
            # Get the current sequence up to the position `i`
            current_sequence = input_sequence[:, : i + 1, :]

            # Predict chord type
            output = model(current_sequence)

            # Apply softmax to get probabilities
            chord_probabilities = F.softmax(
                output[0, :], dim=-1
            )  # Only consider the last prediction

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
