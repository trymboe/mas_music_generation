from data_processing import (
    get_drum_dataset,
    get_melody_dataset,
    get_bass_and_chord_dataset,
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
    get_bass_and_chord_dataset(root_directory)
