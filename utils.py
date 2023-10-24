from config import INT_TO_NOTE, NUMBER_OF_NOTES_FOR_TRAINING
from data_processing import (
    extract_chords_from_files,
    get_drum_dataset,
    Bass_Dataset,
    Chord_Dataset,
    Drum_Dataset,
)


def get_datasets() -> tuple[Bass_Dataset, Chord_Dataset, Drum_Dataset]:
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
    """

    print("----Creating datasets----")

    root_directory: str = "data/POP909"
    chords, notes, beats = extract_chords_from_files(
        root_directory, NUMBER_OF_NOTES_FOR_TRAINING, True
    )

    timed_notes: list[list[tuple[str, int]]] = get_timed_notes(notes, beats)

    drum_dataset: Drum_Dataset = get_drum_dataset()
    bass_dataset: Bass_Dataset = Bass_Dataset(timed_notes)
    chord_dataset: Chord_Dataset = Chord_Dataset(chords)

    return bass_dataset, chord_dataset, drum_dataset


def get_timed_notes(
    notes: list[list[str]], beats: list[list[int]]
) -> list[list[tuple[str, int]]]:
    """
    Converts lists of notes and beats into a structured format of timed notes.

    Parameters
    ----------
    notes : list[list[str]]
        A list of lists where each inner list contains note representations as strings.
    beats : list[list[int]]
        A list of lists where each inner list contains beat information corresponding to the notes.

    Returns
    -------
    list[list[tuple[str, int]]]
        A list of lists where each inner list contains tuples of notes and their corresponding beats.
    """

    timed_notes: list[tuple[str, int]] = []

    for i in range(len(notes)):
        timed_notes.append([])
        for j in range(len(notes[i])):
            timed_notes[i].append((notes[i][j], beats[i][j]))

    return timed_notes
