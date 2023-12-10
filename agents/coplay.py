from .bass import play_bass
from .chord import play_chord
from .drum import play_drum
from .melody import play_melody
from .harmony import play_harmony
import random
import time

import pretty_midi
import torch

from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset, Melody_Dataset


from config import (
    SEQUENCE_LENGTH_CHORD,
    SEQUENCE_LENGTH_BASS,
    SAVE_RESULT_PATH,
    TEST_DATASET_PATH_MELODY,
    TEST_DATASET_PATH_BASS,
    TEST_DATASET_PATH_CHORD,
    TIME_LEFT_ON_CHORD_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    CHORD_SIZE_MELODY,
    PITCH_SIZE_MELODY,
    INT_TO_TRIAD,
    SEGMENTS,
)


def play_agents() -> None:
    """
    Orchestrates the playing of bass, chord, and drum agents to generate a music piece.

    This function generates a music piece by playing the bass, chord, and drum agents sequentially. The generated music
    is then written to a MIDI file. The function also handles the random selection of a primer from the dataset to
    start the generation process.

    Parameters
    ----------
    arpeggiate : bool
        Flag indicating whether to arpeggiate the chord sequences.

    Returns
    -------
    None
    """
    chord_primer, bass_primer, melody_primer = get_primer_sequences()
    mid = None
    print("----playing agents----")

    for idx, config in enumerate(SEGMENTS):
        print("  ----segment:", idx + 1, "----")
        # ------------------------------------------------------
        #                   playing drum
        # ------------------------------------------------------
        print("    ----playing drum----")
        start = time.time()
        new_mid = play_drum(config)
        end = time.time()
        print("      ----drum playing time: ", end - start)

        # ------------------------------------------------------
        #                   playing bass
        # ------------------------------------------------------
        print("    ----playing bass----")
        start = time.time()
        new_mid, predicted_bass_sequence = play_bass(new_mid, bass_primer, config)
        end = time.time()
        print("      ----bass playing time: ", end - start)

        # ------------------------------------------------------
        #                   playing chord
        # ------------------------------------------------------
        print("    ----playing chord----")
        start = time.time()
        new_mid, predicted_chord_sequence = play_chord(
            new_mid, predicted_bass_sequence, chord_primer, config
        )
        end = time.time()
        print("      ----chord playing time: ", end - start)

        # ------------------------------------------------------
        #                   playing melody
        # ------------------------------------------------------
        print("    ----playing melody----")
        start = time.time()
        new_mid, predicted_melody_sequence = play_melody(
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
        new_mid.write("results/segment_" + str(idx + 1) + ".mid")

        # bass_primer, chord_primer, melody_primer = get_new_primer_sequences(
        #     bass_primer,
        #     predicted_bass_sequence,
        #     chord_primer,
        #     predicted_chord_sequence,
        #     melody_primer,
        #     predicted_melody_sequence,
        #     config,
        # )
    mid.write(SAVE_RESULT_PATH)


def get_new_primer_sequences(
    bass_primer,
    predicted_bass_sequence,
    chord_primer,
    predicted_chord_sequence,
    melody_primer,
    predicted_melody_sequence,
    config,
) -> tuple[list, list, list]:
    if len(predicted_bass_sequence) >= SEQUENCE_LENGTH_BASS:
        bass_primer = predicted_bass_sequence[-SEQUENCE_LENGTH_BASS:]
        chord_primer = predicted_chord_sequence[-SEQUENCE_LENGTH_BASS:]
    else:
        combined_bass_events = []
        combined_chord_events = []
        combined_melody_events = []
        for idx in range(len(bass_primer[0])):
            combined_bass_events.append(
                (int(bass_primer[0][idx]), int(bass_primer[1][idx]))
            )
            combined_chord_events.append(
                (int(chord_primer[idx][0]), int(chord_primer[idx][1]))
            )
        combined_melody_events = melody_primer
        for idx in range(len(predicted_bass_sequence)):
            # Bass
            combined_bass_events.append((predicted_bass_sequence[idx]))

            # Chord
            root_note = predicted_chord_sequence[idx][0][0]
            chord_type = [
                (chord - root_note) for chord in predicted_chord_sequence[idx][0]
            ]
            combined_chord_events.append((root_note, get_key(chord_type, INT_TO_TRIAD)))

        running_time_beats: int = 0
        for idx in range(len(predicted_melody_sequence)):
            # Melody
            print("predicted_chord_sequence", predicted_chord_sequence)
            print("predicted_melody_sequence", predicted_melody_sequence)
            print("melody_primer", melody_primer[0])
            pitch_vector = one_hot(
                predicted_melody_sequence[idx][0] - 60, PITCH_SIZE_MELODY
            )
            duration_vector = one_hot(
                predicted_melody_sequence[idx][1], DURATION_SIZE_MELODY
            )
            running_time_beats += predicted_melody_sequence[idx][1] / 4
            (
                (current_root, current_chord),
                (next_root, next_chord),
                time_left_on_chord,
            ) = get_chords(running_time_beats, predicted_chord_sequence)

            print("current_root", current_root)
            print("current_chord", current_chord)
            print("next_root", next_root)
            print("next_chord", next_chord)
            print("time_left_on_chord", time_left_on_chord)
            exit()
            current_chord = max(1, current_chord)
            current_chord_idx = current_root * 2 + current_chord
            current_chord_vector = one_hot(current_chord_idx, CHORD_SIZE_MELODY)

            next_chord = max(1, next_chord)
            next_chord_idx = next_root * 2 + next_chord
            next_chord_vector = one_hot(next_chord_idx, CHORD_SIZE_MELODY)
            time_left_on_chord_vector = one_hot(
                int(time_left_on_chord), TIME_LEFT_ON_CHORD_SIZE_MELODY
            )
            # not working
            accumulated_time_vector = one_hot(1, 4)
            print(len(melody_primer[0]))

        bass_primer = combined_bass_events[-SEQUENCE_LENGTH_BASS:]
        chord_primer = combined_chord_events[-SEQUENCE_LENGTH_BASS:]

    return [bass_primer, chord_primer, None]


def get_chords(running_time_beats, predicted_chord_sequence):
    running_duration_on_chord = 0
    for idx, (chord, duration) in enumerate(predicted_chord_sequence):
        running_duration_on_chord += duration
        if running_duration_on_chord > running_time_beats:
            time_left_on_chord = float(running_duration_on_chord)
            print()
            print(duration)
            print(running_duration_on_chord)
            print(running_time_beats)
            print(time_left_on_chord)
            exit()

            root_note = chord[0]
            chord_type = [(c - root_note) for c in chord]
            current_chord = (root_note, get_key(chord_type, INT_TO_TRIAD))
            if idx + 1 < len(predicted_chord_sequence):
                root_note = predicted_chord_sequence[idx + 1][0][0]
                chord = predicted_chord_sequence[idx + 1][0]
                chord_type = [(c - root_note) for c in chord]

                next_chord = (root_note, get_key(chord_type, INT_TO_TRIAD))
            else:
                next_chord = current_chord
            return current_chord, next_chord, time_left_on_chord

    root_note = predicted_chord_sequence[-1]
    chord_type = [(c - root_note) for c in predicted_chord_sequence[-1][0]]
    current_chord = (root_note, get_key(chord_type, INT_TO_TRIAD))
    next_chord = current_chord

    return current_chord, next_chord, 0


def one_hot(idx, length):
    one_hot = [0] * length
    one_hot[idx] = 1
    return one_hot


# Function to find key from value
def get_key(val, dic):  #
    for key, value in dic.items():
        if value == val:
            return key
    return "Key not found"


def merge_pretty_midi(pm1, pm2):
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
    Gets random primer sequences for bass, chord and melody, from the test dataset.

    Parameters
    ----------
    None

    Returns
    -------

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
