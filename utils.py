from config import INT_TO_NOTE


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
    print([f"{INT_TO_NOTE[note]} - {duration}" for note, duration in full_sequence])

    return full_sequence
