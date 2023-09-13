from .bass import play_bass
from .piano import play_piano

mapping = {
    0: [0, 4, 7],
    2: [2, 5, 9],
    4: [4, 7, 11],
    5: [5, 9, 0],
    7: [7, 11, 2],
    9: [9, 0, 4],
    11: [11, 2, 5],
}


def play_agents(full_bass_sequence, filename):
    full_chord_sequence = map_bass_to_chords(full_bass_sequence)
    mid = play_bass(full_bass_sequence)

    mid = play_piano(mid, full_chord_sequence)

    mid.save(filename)


def map_bass_to_chords(full_bass_sequence):
    full_chord_sequence = []
    for idx, note in enumerate(full_bass_sequence):
        full_chord_sequence.append((mapping[note[0]], note[1]))

    return full_chord_sequence
