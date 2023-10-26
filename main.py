print("----loading imports----")
import matplotlib.pyplot as plt
import argparse
import torch

from agents import (
    create_agents,
    play_agents,
)


from utils import get_datasets

from config import SAVE_RESULT_PATH

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
    "-td",
    "--train_drum",
    action="store_true",
    help="Train the drum agent",
    default=False,
)


if __name__ == "__main__":
    """
    Executes the main training and playing routine for music agents.

    This script is designed to be run from the command line, taking in arguments to specify
    whether training should be performed for the bass, chord, and drum agents, and whether
    arpeggiation should be applied during the playing phase.

    Parameters
    ----------
    train_bass : bool, optional
        Indicates whether the bass agent should be trained. (default is False)
    train_chord : bool, optional
        Indicates whether the chord agent should be trained. (default is False)
    train_drum : bool, optional
        Indicates whether the drum agent should be trained. (default is False)
    arpeggiate : bool, optional
        Indicates whether arpeggiation should be applied during the playing phase. (default is False)

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If invalid command line arguments are provided.
    """
    args = parser.parse_args()
    train_bass: bool = parser.parse_args().train_bass
    train_chord: bool = parser.parse_args().train_chord
    train_drum: bool = parser.parse_args().train_drum
    arpeggiate: bool = parser.parse_args().arpeggiate

    # Process the datasets
    bass_dataset, chord_dataset, drum_dataset = get_datasets()

    # Create and train the agents
    bass_agent, chord_agent, drum_agent = create_agents(
        bass_dataset,
        chord_dataset,
        drum_dataset,
        train_bass,
        train_chord,
        train_drum,
    )

    # Play the agents
    play_agents(
        bass_dataset,
        chord_dataset,
        drum_dataset,
        arpeggiate,
        SAVE_RESULT_PATH,
    )

    plt.show()
