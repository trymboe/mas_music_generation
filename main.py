import matplotlib.pyplot as plt
import argparse
import torch

from agents import (
    create_agents,
    play_agents,
    train_agents,
)

from utils import get_datasets

parser = argparse.ArgumentParser(description="Choose how to run the program")

parser.add_argument(
    "-a",
    "--arpeggiate",
    action="store_true",
    help="Arpeggiate the chords",
    default=False,
)
parser.add_argument(
    "-tb",
    "--train_bass",
    action="store_true",
    help="Train the bass agent",
    default=False,
)
parser.add_argument(
    "-tc",
    "--train_chord",
    action="store_true",
    help="Train the chord agent",
    default=False,
)
parser.add_argument(
    "--non_mac", action="store_true", help="Train on non-M1 mac", default=False
)

args = parser.parse_args()

if __name__ == "__main__":
    mac = not parser.parse_args().non_mac
    train_bass = parser.parse_args().train_bass
    train_chord = parser.parse_args().train_chord
    arpeggiate = parser.parse_args().arpeggiate

    if mac:
        # for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    notes_dataset, chords_dataset = get_datasets()

    bass_agent, chord_agent = create_agents()

    train_agents(
        bass_agent, chord_agent, notes_dataset, chords_dataset, train_bass, train_chord
    )

    play_agents(
        notes_dataset,
        chords_dataset,
        bass_agent,
        chord_agent,
        arpeggiate,
        "results/bass_and_chord2.mid",
    )

    plt.show()
