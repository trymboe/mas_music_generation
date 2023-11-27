import torch
import torch.nn.functional as F

from config import SEQUENCE_LENGTH_CHORD


def predict_next_k_notes_chords(model, predicted_bass_sequence, dataset_primer):
    #
    chord_primer = get_input_sequence_chords(predicted_bass_sequence, dataset_primer)

    predicted_chord_types = []

    # Add a batch dimension
    input_sequence = chord_primer.unsqueeze(0)

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


def get_input_sequence_chords(full_bass_sequence, dataset_primer):
    # Extract the corresponding chord sequence from the dataset
    actual_chord_sequence = dataset_primer[:, :2]

    # Extract only the root notes from the full_bass_sequence
    bass_notes = [pair[0] for pair in full_bass_sequence]

    # Create the input sequence
    input_sequence = []

    # Iterate over the bass_notes and actual_chord_sequence to create the pairs
    for i, bass_note in enumerate(bass_notes):
        if i < len(actual_chord_sequence):  # If we have actual chords, use them
            input_sequence.append(
                [bass_note, actual_chord_sequence[i][1].item()]
            )  # Use .item() to extract scalar from tensor
        else:  # Otherwise, use the placeholder
            input_sequence.append([bass_note, 6])

    # Convert the list of lists to a tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.int64)

    return input_tensor
