print("----loading imports----")
import matplotlib.pyplot as plt
import webbrowser
import argparse
import os


from broadcaster.broadcaster import start_broadcaster
from agents import create_agents
from utils import get_datasets


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

    start_broadcaster()
    # Open the web browser
    webbrowser.open("file://" + os.path.realpath("index.html"))

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Shutting down Flask server...")
        plt.show()


if __name__ == "__main__":
    main()
