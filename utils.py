from config import INT_TO_NOTE, NUMBER_OF_NOTES_FOR_TRAINING
from data_processing import (
    extract_chords_from_files,
    get_drum_dataset,
    Notes_Dataset,
    Chords_Dataset,
)


def get_datasets() -> (Notes_Dataset, Chords_Dataset):
    root_directory: str = "data/POP909"
    chords, notes, beats = extract_chords_from_files(
        root_directory, NUMBER_OF_NOTES_FOR_TRAINING, True
    )

    timed_notes = get_timed_notes(notes, beats)

    drum_dataset = get_drum_dataset()

    notes_dataset: Notes_Dataset = Notes_Dataset(timed_notes)
    chords_dataset: Chords_Dataset = Chords_Dataset(chords)

    return notes_dataset, chords_dataset, drum_dataset


def get_timed_notes(
    notes: list[list[str]], beats: list[list[int]]
) -> list[list[tuple[str, int]]]:
    timed_notes: list[tuple[str, int]] = []

    for i in range(len(notes)):
        timed_notes.append([])
        for j in range(len(notes[i])):
            timed_notes[i].append((notes[i][j], beats[i][j]))

    return timed_notes


def get_full_bass_sequence(primer_sequence, predicted_sequence):
    full_sequence = predicted_sequence

    note_sequence, duration_sequence = (
        primer_sequence[0].tolist(),
        primer_sequence[1].tolist(),
    )

    for i in range(len(note_sequence) - 1, -1, -1):
        full_sequence.insert(0, (note_sequence[i], duration_sequence[i]))

    # Add C as the last note for a finished sequence
    if full_sequence[-1][0] != 0:
        full_sequence.append([0, 8])

    return full_sequence
