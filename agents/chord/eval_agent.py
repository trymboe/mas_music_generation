import torch
import torch.nn.functional as F

from config import 


def predict_next_k_notes_chords(model, full_bass_sequence, dataset_primer):
    chord_primer = get_input_sequence_chords(dataset_primer, full_bass_sequence)

    predicted_chords = []

    # Add a batch dimension
    input_sequence = chord_primer.unsqueeze(0)

    model.eval()  # Set the model to evaluation mode

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

    return predicted_chords


def update_input_sequence(
    input_sequence: torch.tensor, next_chord_type: int, next_note: int
) -> torch.tensor:
    input_sequence_list = input_sequence.squeeze().tolist()
    input_sequence_list[-1] = [input_sequence_list[-1][0], next_chord_type]

    input_sequence_list = input_sequence_list[1:]

    input_sequence_list.append([next_note, 6])
    return torch.tensor(input_sequence_list).unsqueeze(0)


def get_input_sequence_chords(dataset_primer, full_bass_sequence):
    # Extract the corresponding chord sequence from the dataset
    actual_chord_sequence = dataset_primer[:, :2]

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
