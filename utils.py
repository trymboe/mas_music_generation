from config import INT_TO_NOTE, NUMBER_OF_NOTES_FOR_TRAINING
from data_processing import (
    extract_chords_from_files,
    get_drum_dataset,
    Bass_Dataset,
    Chord_Dataset,
    Drum_Dataset,
    Melody_Dataset,
    get_melody_dataset,
    get_bass_dataset,
    get_chord_dataset,
)


def get_datasets() -> None:
    """
    Processes music data to create datasets for notes, chords, and drums.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    print("----Creating datasets----")
    root_directory: str = "data/POP909/transposed"

    get_melody_dataset(root_directory)
    get_drum_dataset()
    get_bass_dataset(root_directory)
    get_chord_dataset(root_directory)
