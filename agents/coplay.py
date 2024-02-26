from .bass import play_bass, play_known_bass
from .chord import play_chord, play_known_chord
from .drum import play_drum
from .melody import play_melody
from .harmony import play_harmony
import random
import time
import copy

import pretty_midi
import torch


from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset, Melody_Dataset


from config import (
    SEQUENCE_LENGTH_CHORD,
    SEQUENCE_LENGTH_BASS,
    SEQUENCE_LENGHT_MELODY,
    SAVE_RESULT_PATH,
    TEST_DATASET_PATH_MELODY,
    TEST_DATASET_PATH_BASS,
    TEST_DATASET_PATH_CHORD,
    TIME_LEFT_ON_CHORD_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    CHORD_SIZE_MELODY,
    PITCH_SIZE_MELODY,
    INT_TO_TRIAD,
    INT_TO_CHORD,
    INT_TO_NOTE,
)

NEW_BASS_PRIMER = None
NEW_CHORD_PRIMER = None
NEW_MELODY_PRIMER = None


def play_agents(config, kept_instruments) -> pretty_midi.PrettyMIDI:
    """
    Plays the different musical instruments (drum, bass, chord, melody, harmony) based on the given configuration and
    the previously kept instruments.

    Args:
    ----------
        config (dict): The configuration settings for the music generation.
        kept_instruments (list): The list of previously kept instruments.

    Returns:
    ----------
        pretty_midi.PrettyMIDI: The generated MIDI file.
        list: The list of instruments used in the generated MIDI file.
    """
    global NEW_BASS_PRIMER, NEW_CHORD_PRIMER, NEW_MELODY_PRIMER

    if NEW_BASS_PRIMER is None:
        chord_primer, bass_primer, melody_primer = get_primer_sequences()
    else:
        chord_primer = NEW_CHORD_PRIMER
        bass_primer = NEW_BASS_PRIMER
        melody_primer = NEW_MELODY_PRIMER

    mid = None

    # ------------------------------------------------------
    #                   playing drum
    # ------------------------------------------------------
    print("    ----playing drum----")
    start = time.time()
    if config["KEEP_DRUM"]:
        # If it is the first time, there are no drums to keep
        if not kept_instruments[0]:
            new_mid, drum_mid = play_drum(config)
        else:
            drum_mid = kept_instruments[0][0]
            new_mid = copy.deepcopy(drum_mid)
    else:
        drum_mid = play_drum(config)
        new_mid = copy.deepcopy(drum_mid)
    end = time.time()

    print("      ----drum playing time: ", end - start)
    # ------------------------------------------------------
    #                   playing bass
    # ------------------------------------------------------
    print("    ----playing bass----")
    start = time.time()
    if config["KEEP_BASS"]:
        # If it is the first time, there are no bass to keep
        if not kept_instruments[1]:
            new_mid, bass_instrument, predicted_bass_sequence = play_bass(
                new_mid, bass_primer, config
            )
        else:
            predicted_bass_sequence = kept_instruments[1][1]
            new_mid, bass_instrument, predicted_bass_sequence = play_known_bass(
                new_mid, predicted_bass_sequence, config
            )
    else:
        new_mid, bass_instrument, predicted_bass_sequence = play_bass(
            new_mid, bass_primer, config
        )
    end = time.time()
    print("      ----bass playing time: ", end - start)

    # ------------------------------------------------------
    #                   playing chord
    # ------------------------------------------------------
    print("    ----playing chord----")
    start = time.time()
    if config["KEEP_CHORD"]:
        if not kept_instruments[2]:
            new_mid, chord_instrument, predicted_chord_sequence = play_chord(
                new_mid, predicted_bass_sequence, chord_primer, config
            )
        else:
            predicted_chord_sequence = kept_instruments[2][1]
            # If we want to keep the chord, but have a new play style
            new_mid, chord_instrument = play_known_chord(
                new_mid, predicted_chord_sequence, config
            )
    else:
        new_mid, chord_instrument, predicted_chord_sequence = play_chord(
            new_mid, predicted_bass_sequence, chord_primer, config
        )
    end = time.time()
    print("      ----chord playing time: ", end - start)

    # ------------------------------------------------------
    #                   playing melody
    # ------------------------------------------------------
    print("    ----playing melody----")
    start = time.time()
    if config["KEEP_MELODY"]:
        if not kept_instruments[3]:
            new_mid, melody_instrument, predicted_melody_sequence = play_melody(
                new_mid, predicted_chord_sequence, melody_primer, config
            )
        else:
            melody_instrument = kept_instruments[3][0]
            predicted_melody_sequence = kept_instruments[3][1]
            new_mid.instruments.append(melody_instrument)
    else:
        new_mid, melody_instrument, predicted_melody_sequence = play_melody(
            new_mid, predicted_chord_sequence, melody_primer, config
        )
    end = time.time()
    print("      ----melody playing time: ", end - start)

    # ------------------------------------------------------
    #                   playing harmony
    # ------------------------------------------------------
    print("    ----playing harmony----")
    start = time.time()
    new_mid = play_harmony(new_mid, predicted_melody_sequence, config)
    end = time.time()
    print("      ----harmony playing time: ", end - start)

    if mid:
        mid = merge_pretty_midi(mid, new_mid)
    else:
        mid = new_mid

    NEW_BASS_PRIMER, NEW_CHORD_PRIMER, NEW_MELODY_PRIMER = get_new_primer_sequences(
        bass_primer,
        predicted_bass_sequence,
        chord_primer,
        predicted_chord_sequence,
        melody_primer,
        predicted_melody_sequence,
    )

    instruments = [
        [drum_mid],
        [bass_instrument, predicted_bass_sequence],
        [chord_instrument, predicted_chord_sequence],
        [melody_instrument, predicted_melody_sequence],
    ]
    mid.write(SAVE_RESULT_PATH)
    chord_progression = get_chord_progression(predicted_chord_sequence)
    return mid, instruments, chord_progression


def get_chord_progression(predicted_chord_sequence):
    """
    Converts the predicted chord sequence into a list containing the strings of the chord progression.

    Args:
    ----------
        predicted_chord_sequence (list): A list of tuples representing the predicted chord sequence.
            Each tuple contains the chord index and its duration.

    Returns:
    ----------
        list: A list of lists representing the full chord progression.
            Each inner list contains the full chord name and its duration.
    """
    full_chord_progression = []
    for chord, duration in predicted_chord_sequence:
        root, chord_type = get_chord(chord)
        root_name = INT_TO_NOTE[root]
        chord_variant = INT_TO_CHORD[chord_type]
        full_chord_name = root_name + chord_variant
        full_chord_progression.append([full_chord_name, duration])
    return full_chord_progression


def get_new_primer_sequences(
    previous_bass_primer,
    predicted_bass_sequence,
    previous_chord_primer,
    predicted_chord_sequence,
    previous_melody_primer,
    predicted_melody_sequence,
) -> tuple[list, list, list]:
    """
    Generates new primer sequences for bass, chord, and melody based on the previous primer sequences and predicted sequences.

    Args:
    ----------
        previous_bass_primer (list): The previous bass primer sequence.
        predicted_bass_sequence (list): The predicted bass sequence.
        previous_chord_primer (list): The previous chord primer sequence.
        predicted_chord_sequence (list): The predicted chord sequence.
        previous_melody_primer (list): The previous melody primer sequence.
        predicted_melody_sequence (list): The predicted melody sequence.

    Returns:
    ----------
        tuple[list, list, list]: A tuple containing the new bass primer sequence, chord primer sequence, and melody primer sequence.
    """

    # Bass and chord primer
    combined_chord_events = []
    combined_bass_events = [
        previous_bass_primer[0].tolist(),
        previous_bass_primer[1].tolist(),
    ]
    for i in range(len(previous_chord_primer)):

        combined_chord_events.append(
            [
                int(previous_chord_primer[i][0].item()),
                int(previous_chord_primer[i][1].item()),
            ]
        )

    for i in range(len(predicted_bass_sequence)):
        combined_bass_events[0].append(predicted_bass_sequence[i][0])
        combined_bass_events[1].append(predicted_bass_sequence[i][1])

        combined_chord_events.append(get_chord(predicted_chord_sequence[i][0]))

    full_bass_events = combined_bass_events
    combined_bass_events = [
        combined_bass_events[0][-SEQUENCE_LENGTH_BASS:],
        combined_bass_events[1][-SEQUENCE_LENGTH_BASS:],
    ]

    full_chord_events = combined_chord_events
    combined_chord_events = combined_chord_events[-SEQUENCE_LENGTH_BASS:]

    bass_primer = [
        torch.tensor(combined_bass_events[0]),
        torch.tensor(combined_bass_events[1]),
    ]

    chord_primer = torch.tensor(combined_chord_events)

    # Melody primer
    combined_melody_events = previous_melody_primer

    notes = get_notes(predicted_melody_sequence, full_bass_events, full_chord_events)

    for note in notes:
        combined_melody_events.append(note)

    melody_primer = combined_melody_events[-SEQUENCE_LENGHT_MELODY:]

    return bass_primer, chord_primer, melody_primer


def get_notes(predicted_melody_sequence, full_bass_events, full_chord_events):
    """
    Generates a sequence of notes based on the predicted melody sequence. It uses the bass and chord events to determine the current, next chord, and the time left on the chord.

    Args:
    ----------
        predicted_melody_sequence (list): A list of tuples representing the predicted melody sequence. Each tuple contains the pitch and duration of a note.
        full_bass_events (list): A list of lists representing the full bass events. Each inner list contains the pitch and duration of a bass note.
        full_chord_events (list): A list of lists representing the full chord events. Each inner list contains the pitches of a chord.

    Returns:
    ----------
        list: A list of lists representing the generated sequence of notes. Each inner list contains the pitch vector, duration vector, current chord, next chord, time left on chord vector, and other information about the note.
    """

    total_melody_duration = 0
    full_sequenece = []
    for idx in range(len(predicted_melody_sequence) - 1, -1, -1):
        total_melody_duration += predicted_melody_sequence[idx][1]

        # get pitch and duration vector
        pitch_vector = one_hot(
            predicted_melody_sequence[idx][0] - 60 - 1, PITCH_SIZE_MELODY
        )

        duration_vector = one_hot(
            predicted_melody_sequence[idx][1] - 1, DURATION_SIZE_MELODY
        )

        # get chord vectors
        length_bass = 0
        for j in range(len(full_bass_events[0]) - 1, -1, -1):

            length_melody = total_melody_duration / 4
            length_bass += full_bass_events[1][j]
            # if we have not arrived at the correct chord yet
            if length_bass < length_melody:
                continue

            current_chord = get_full_chord(full_chord_events[j])

            if j + 1 < len(full_chord_events):
                next_chord = get_full_chord(full_chord_events[j + 1])
            else:
                next_chord = current_chord

            tloc = length_bass - length_melody
            tloc *= 4
            tloc = min(15, tloc)
            tloc = max(0, tloc)
            tloc_vector = one_hot(int(tloc), TIME_LEFT_ON_CHORD_SIZE_MELODY)
            break

        full_sequenece.insert(
            0,
            [
                pitch_vector,
                duration_vector,
                current_chord,
                next_chord,
                tloc_vector,
                [1, 0, 0, 0],
                ["0", 0],
            ],
        )
    return full_sequenece


def get_full_chord(chord):
    """
    Converts a chord to its corresponding one-hot representation for melody generation.

    Args:
    ----------
        chord (list): A list representing the chord, where the first element is the root note and the second element is the triad.

    Returns:
    ----------
        list: The one-hot representation of the chord.

    """
    root = chord[0]
    triad = chord[1]
    # melody only deals with major or minor chords
    triad = 0 if triad > 1 else triad
    chord_index = root * 2 + triad
    return one_hot(chord_index, CHORD_SIZE_MELODY)


def get_chord(triad):
    """
    Converts a triad to a chord by finding the root note and the chord type.

    Args:
    ----------
        triad (list): A list of three notes representing a triad.

    Returns:
    ----------
        list: A list containing the root note and the chord type.

    """
    root = triad[0]
    triad = [note - root for note in triad]
    for key, value in INT_TO_TRIAD.items():
        if value == triad:
            return [root, key]


def one_hot(idx, length):
    """
    Converts an index to a one-hot encoded list of the specified length.

    Args:
    ----------
        idx (int): The index to be converted.
        length (int): The length of the one-hot encoded list.

    Returns:
    ----------
        list: A one-hot encoded list with the specified index set to 1 and all other elements set to 0.
    """
    one_hot = [0] * length
    one_hot[idx] = 1
    return one_hot


def merge_pretty_midi(pm1, pm2):
    """
    Merge two PrettyMIDI objects by shifting the start and end times of the notes in the second object
    and then merging tracks with the same program and drum status.

    Args:
    ----------
        pm1 (PrettyMIDI): The first PrettyMIDI object.
        pm2 (PrettyMIDI): The second PrettyMIDI object.

    Returns:
    ----------
        PrettyMIDI: The merged PrettyMIDI object.
    """

    # Find the end time of the last note in the first MIDI object
    end_time_pm1 = max(
        note.end for instrument in pm1.instruments for note in instrument.notes
    )

    # Shift the start and end times of all notes in the second MIDI object
    for instrument in pm2.instruments:
        for note in instrument.notes:
            note.start += end_time_pm1
            note.end += end_time_pm1

    # Merge tracks with the same program (instrument type)
    for instrument1 in pm1.instruments:
        for instrument2 in pm2.instruments:
            if (
                instrument1.program == instrument2.program
                and instrument1.is_drum == instrument2.is_drum
            ):
                instrument1.notes.extend(instrument2.notes)
                break

    return pm1


def get_primer_sequences(attempt=0) -> tuple[list, list, list]:
    """
    Retrieves primer sequences for chord, bass, and melody from the datasets.

    Parameters:
    ----------
        attempt (int): The number of attempts to retrieve the primer sequences.

    Returns:
    ----------
        tuple[list, list, list]: A tuple containing the chord primer, bass primer, and melody primer sequences.
    """

    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD)
    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY)
    bass_dataset: Bass_Dataset = torch.load(TEST_DATASET_PATH_BASS)

    primer_start = random.randint(0, len(melody_dataset) - 1)
    song_name_melody = melody_dataset[primer_start][0][0][6][0]
    last_note_timing = melody_dataset[primer_start][0][-1][6][1]

    primer_end_chord = None

    found = False
    for i in range(0, len(chord_dataset)):
        song_name = int(chord_dataset[i][0][0][2])
        # If the song name is the same
        if int(song_name) == int(song_name_melody):
            found = True
            chord_timing = chord_dataset[i][0][0][3]

            # if the chord has passed
            if last_note_timing - chord_timing < 0:
                # primer_end_chord - SEQUENCE_LENGTH_CHORD is the index that gets correct primer sequence
                primer_end_chord = i
                break

        # If we found the song, but the song name is different, we have passed the song.
        elif found:
            break

    attempt = 0
    if not primer_end_chord:
        if attempt > 30:
            print("Tried 30 times, giving up")
            exit()
        print("Could not find primer end chord, trying again")
        return get_primer_sequences(attempt + 1)

    chord_primer = chord_dataset[primer_end_chord - SEQUENCE_LENGTH_CHORD][0]
    bass_primer = bass_dataset[primer_end_chord - SEQUENCE_LENGTH_CHORD]
    melody_primer = melody_dataset[primer_start][0]

    return chord_primer, bass_primer, melody_primer
