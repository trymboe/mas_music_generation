import math
import argparse
import torch
import random

from agents import (
    create_agents,
    play_agents,
    train_agents,
)

from utils import get_timed_notes, get_full_bass_sequence, get_datasets


if __name__ == "__main__":
    mac: bool = True
    load_bass: bool = True
    load_chord: bool = True

    if mac:
        # for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    notes_dataset, chords_dataset = get_datasets()

    bass_agent, chord_agent = create_agents()

    train_agents(
        bass_agent, chord_agent, notes_dataset, chords_dataset, load_bass, load_chord
    )
    primer_part = random.randint(0, len(notes_dataset) - 1)
    primer_sequence = (
        notes_dataset[primer_part][0],
        notes_dataset[primer_part][1],
    )

    play_agents(
        notes_dataset,
        chords_dataset,
        bass_agent,
        chord_agent,
        "results/bass_and_chord2.mid",
    )
