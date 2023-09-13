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
    play_agents,
)

from data_processing import extract_chords_from_files, Notes_Dataset
from utils import get_timed_notes, get_full_bass_sequence


if __name__ == "__main__":
    mac: bool = True
    load: bool = True

    if mac:
        # for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_directory: str = "data"
    number_of_chords: int = 1000
    chords, notes, beats = extract_chords_from_files(
        root_directory, number_of_chords, True
    )

    timed_notes = get_timed_notes(notes, beats)

    notes_dataset: Notes_Dataset = Notes_Dataset(timed_notes)

    bass_agent = create_bass_agent()

    primer_part = random.randint(0, len(notes_dataset) - 1)
    primer_sequence = (
        notes_dataset[primer_part][0],
        notes_dataset[primer_part][1],
    )

    if load:
        bass_agent.load_state_dict(torch.load("models/bass/bass_model.pth"))
        bass_agent.eval()
    else:
        train_bass(bass_agent, notes_dataset)

    predicted_sequence = predict_next_k_notes(bass_agent, primer_sequence, 30)
    full_bass_sequence = get_full_bass_sequence(primer_sequence, predicted_sequence)

    play_agents(full_bass_sequence, "results/bass_and_piano.mid")
