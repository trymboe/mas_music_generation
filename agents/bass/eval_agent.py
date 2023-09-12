import torch
import torch.nn.functional as F

from config import INT_TO_NOTE


def predict_next_k_notes(model, initial_sequence, k):
    predicted_notes = []

    sequence = initial_sequence.unsqueeze(0)  # Add a batch dimension
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for _ in range(k):
            output = model(sequence)

            # Apply softmax to get probabilities
            probabilities = F.softmax(output[0], dim=0)

            # Sample from the distribution
            next_note = torch.multinomial(probabilities, 1).unsqueeze(1)

            predicted_notes.append(next_note.item())

            # Use sliding window method: drop the first note, append the predicted note
            sequence = torch.cat([sequence[:, 1:], next_note], dim=1)

    return predicted_notes
