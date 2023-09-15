NOTE_TO_INT = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
INT_TO_NOTE = {v: k for k, v in NOTE_TO_INT.items()}

CHORD_TO_INT = {
    "maj": 0,
    "min": 1,
    "dim": 2,
    "aug": 3,
    "sus2": 4,
    "sus4": 5,
}

INT_TO_TRIAD = {
    0: [0, 4, 7],
    1: [0, 3, 7],
    2: [0, 3, 6],
    3: [0, 4, 8],
    4: [0, 2, 7],
    5: [0, 5, 7],
}

NUMBER_OF_NOTES_FOR_TRAINING = 500  # 0 for all notes
K = 30  # Number of notes to predict
