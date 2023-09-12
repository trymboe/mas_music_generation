import math
import argparse
import torch
import random

from agents import (
    Bass_Network,
    train_bass,
    create_bass_agent,
    predict_next_k_notes,
    play_bass,
)

from data_processing import extract_chords_from_files, Notes_Dataset


if __name__ == "__main__":
    mac = True
    load = True

    if mac:
        # for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_directory: str = "data"
    number_of_chords: int = 1000
    chords, notes = extract_chords_from_files(root_directory, number_of_chords, True)

    notes_dataset: Notes_Dataset = Notes_Dataset(notes)

    bass_agent = create_bass_agent()
    primer_sequence = notes_dataset[random.randint(0, len(notes_dataset) - 1)][0]

    if load:
        bass_agent.load_state_dict(torch.load("models/bass/bass_model.pth"))
        bass_agent.eval()
    else:
        train_bass(bass_agent, notes_dataset)

    predicted_sequence = predict_next_k_notes(bass_agent, primer_sequence, 10)
    full_sequence = primer_sequence.tolist() + predicted_sequence

    play_bass(full_sequence)
