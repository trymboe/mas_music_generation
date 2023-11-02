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


def get_datasets() -> tuple[Bass_Dataset, Chord_Dataset, Drum_Dataset, Melody_Dataset]:
    """
    Processes music data to create datasets for notes, chords, and drums.

    Parameters
    ----------
    None

    Returns
    -------
    Bass_Dataset
        A dataset object containing timed note sequences.
    Chord_Dataset
        A dataset object containing chord progressions.
    Drum_Dataset
        A dataset object containing drum patterns.
    Melody_Dataset
        A dataset object containing melody related to chords.
    """

    print("----Creating datasets----")
    root_directory: str = "data/POP909"

    melody_dataset: Melody_Dataset = get_melody_dataset(root_directory)
    drum_dataset: Drum_Dataset = get_drum_dataset()
    bass_dataset: Bass_Dataset = get_bass_dataset(root_directory)
    chord_dataset: Chord_Dataset = get_chord_dataset(root_directory)

    return bass_dataset, chord_dataset, drum_dataset, melody_dataset
