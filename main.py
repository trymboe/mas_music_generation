print("----loading imports----")
import matplotlib.pyplot as plt
import argparse
import requests
import pretty_midi

from agents import (
    create_agents,
    play_agents,
)

from utils import get_datasets

import threading
from flask import Flask
from script.broadcaster import (
    start_broadcaster,
    add_to_queue,
)  # Import Flask app and loop

parser = argparse.ArgumentParser(description="Choose how to run the program")

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
parser.add_argument(
    "-tm",
    "--train_melody",
    action="store_true",
    help="Train the melody agent",
    default=False,
)


def main():
    """
    Executes the main training and playing routine for music agents.

    This script is designed to be run from the command line, using arguments to specify
    whether training should be performed for the bass, chord, and drum agents, and whether
    arpeggiation should be applied during the playing phase.

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    args = parser.parse_args()
    train_bass: bool = parser.parse_args().train_bass
    train_chord: bool = parser.parse_args().train_chord
    train_drum: bool = parser.parse_args().train_drum
    train_melody: bool = parser.parse_args().train_melody

    # Process the datasets
    get_datasets()

    # Create and train the agents
    create_agents(train_bass, train_chord, train_drum, train_melody)

    pm = play_agents()
    add_to_queue(pm)

    start_broadcaster()

    try:
        while True:
            pass
            # print("Generating queue")
            # pm = play_agents()
            # add_to_queue(pm)
            # print("Got queue")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Shutting down Flask server...")
        try:
            requests.post("http://localhost:5005/shutdown")
        except Exception as e:
            print(f"Error shutting down Flask server: {e}")
        plt.show()


if __name__ == "__main__":
    main()
