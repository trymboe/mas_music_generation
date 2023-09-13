def get_timed_notes(
    notes: list[list[str]], beats: list[list[int]]
) -> list[list[tuple[str, int]]]:
    timed_notes: list[tuple[str, int]] = []

    for i in range(len(notes)):
        timed_notes.append([])
        for j in range(len(notes[i])):
            timed_notes[i].append((notes[i][j], beats[i][j]))

    return timed_notes
