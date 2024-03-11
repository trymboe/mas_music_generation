print("----loading imports----")
import matplotlib.pyplot as plt
import webbrowser
import argparse
import os


from broadcaster.midi_app import start_broadcaster
from agents import create_agents
from utils import get_datasets
from agents import eval_all_agents


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
    "-tc_nc",
    "--train_chord_noncoop",
    action="store_true",
    help="Train the chord non cooperation agent",
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
parser.add_argument(
    "-tm_nc",
    "--train_melody_noncoop",
    action="store_true",
    help="Train the melody non cooperation agent",
    default=False,
)
parser.add_argument(
    "-e",
    "--eval",
    action="store_true",
    help="evaluate the agents",
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
    train_chord_non_coop: bool = parser.parse_args().train_chord_noncoop
    train_drum: bool = parser.parse_args().train_drum
    train_melody: bool = parser.parse_args().train_melody
    train_melody_non_coop: bool = parser.parse_args().train_melody_noncoop
    eval_agents: bool = parser.parse_args().eval

    if eval_agents:
        eval_all_agents()

    # Process the datasets
    get_datasets()

    # Create and train the agents
    create_agents(
        train_bass,
        train_chord,
        train_chord_non_coop,
        train_drum,
        train_melody,
        train_melody_non_coop,
    )

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
