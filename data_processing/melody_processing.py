import os
import pretty_midi
from .datasets import Melody_Dataset
import torch
import re

from config import (
    PITCH_VECTOR_SIZE,
    FULL_CHORD_TO_INT,
    SEQUENCE_LENGHT_MELODY,
    TRAIN_DATASET_PATH_MELODY,
    TEST_DATASET_PATH_MELODY,
    VAL_DATASET_PATH_MELODY,
)

from .utils import remove_file_from_dataset


def get_melody_dataset(root_dir: str) -> None:
    if not os.path.exists(TRAIN_DATASET_PATH_MELODY):
        all_events_list = []
        for split in ["train", "test", "val"]:
            print("Processing", split, "-split")
            all_events = process_melody(root_dir, split)
            all_events_list.append(all_events)

        melody_dataset_train: Melody_Dataset = Melody_Dataset(all_events_list[0])
        melody_dataset_test: Melody_Dataset = Melody_Dataset(all_events_list[1])
        melody_dataset_val: Melody_Dataset = Melody_Dataset(all_events_list[2])

        torch.save(melody_dataset_train, TRAIN_DATASET_PATH_MELODY)
        torch.save(melody_dataset_test, TEST_DATASET_PATH_MELODY)
        torch.save(melody_dataset_val, VAL_DATASET_PATH_MELODY)


def process_melody(root_dir: str, split) -> Melody_Dataset:
    root_dir = os.path.join(root_dir, split)
    num_files = 0
    all_events: list[list[list[int], list[int], list[list[int]], list[bool]]] = []

    for directory in os.listdir(root_dir):
        if ".DS_Store" in directory:
            continue
        for file in os.listdir(os.path.join(root_dir, directory)):
            if ".mid" in file:
                midi_file: str = os.path.join(root_dir, directory, file)
            if "chord_audio" in file:
                chord_file: str = os.path.join(root_dir, directory, file)
            else:
                continue

        list_of_events: list[
            list[list[int], list[int], list[list[int]], list[bool]]
        ] = process_melody_and_chord(midi_file, chord_file)

        if list_of_events is not None:
            all_events.append(list_of_events)
            num_files += 1
            print("Processed", num_files, "files")

        # if num_files == 10 and DATASET_SIZE_MELODY == "small":
        #     break
        # if num_files == 20 and DATASET_SIZE_MELODY == "medium":
        #     break
        # if num_files == 100 and DATASET_SIZE_MELODY == "large":
        #     break

    return all_events


def process_melody_and_chord(
    midi_file: str, chord_file: str
) -> list[list[int], list[int], list[list[int]], list[bool]]:
    pm = pretty_midi.PrettyMIDI(midi_file)

    # Iterate over the instruments in the MIDI data
    melody_track: pretty_midi.instrument = None
    for instrument in pm.instruments:
        # Check if the instrument name is 'MELODY'
        if instrument.name == "MELODY":
            melody_track = instrument
            break

    if melody_track is None:
        raise Exception("No melody track found")

    chord_list: list[list[list[int], int]] = process_chord(chord_file)

    ticks_per_beat: int = pm.resolution
    ticks_per_bar: int = ticks_per_beat * 4  # Since it's 4/4 time signature
    # Tolerance for the duration of an eighth note
    sixteenth_note_tolerance: float = ticks_per_beat / 4
    tempo: int = int(pm.get_tempo_changes()[1][0])
    placement: list[str, int] = None

    current_tick: int = 0
    list_of_events: list[list[int], list[int], list[list[int]], list[bool]] = []

    no_chord: bool = False

    file_name = midi_file.split("/")[-1].split(".")[0].split("_")[1]

    # Iterate over the notes in the melody track
    for idx, note in enumerate(melody_track.notes):
        placement = [file_name, note.start]
        # if idx > 20:
        #     print("file", midi_file)
        #     exit()
        start_tick: float = pm.time_to_tick(note.start)
        end_tick: float = pm.time_to_tick(note.end)
        note_start_seconds: float = note.start
        duration_ticks: float = end_tick - start_tick

        current_chord_vector = None
        next_chord_vector = None

        duration_vector: list[int] = get_duration_list(duration_ticks, ticks_per_bar)

        current_chord_vector, next_chord_vector, time_left_current_chord = find_chord(
            chord_list, tempo, ticks_per_beat, start_tick, note_start_seconds
        )
        # Break if there is no corresponding chord
        if not current_chord_vector:
            continue

        # # Check if the note is the start of a bar
        # is_start_of_bar: bool = (
        #     abs(start_tick % ticks_per_bar) <= sixteenth_note_tolerance
        # )
        # is_end_of_bar: bool = (
        #     abs(end_tick % ticks_per_bar) <= sixteenth_note_tolerance
        #     or abs((end_tick % ticks_per_bar) - ticks_per_bar)
        #     <= sixteenth_note_tolerance
        # )

        time_left_current_chord_vector: list[int] = one_hote_time_left(
            time_left_current_chord
        )

        accumulated_time_vector = get_accumulated_time(note.start, tempo)

        # current_chord_num = next(
        #     (i for i, value in enumerate(current_chord) if value == 1), None
        # )
        # print(
        #     "pitch",
        #     note.pitch,
        #     "note start",
        #     note.start,
        #     "note_end",
        #     note.end,
        #     "current chord",
        #     current_chord_num,
        # )
        # print("accumulated time", accumulated_time_vector)
        # print()

        if note.start > current_tick:
            # Add a rest if there is a gap between notes
            rest_duration: float = note.start - current_tick
            pitch_vector_pause: list[int] = [0] * (PITCH_VECTOR_SIZE + 1)
            pitch_vector_pause[-1] = 1
            duration_vector: list[int] = get_duration_list(rest_duration, ticks_per_bar)
            (
                current_chord_vector_pause,
                next_chord_vector_pause,
                time_left_current_chord_pause,
            ) = find_chord(
                chord_list,
                tempo,
                ticks_per_beat,
                start_tick - rest_duration,
                note_start_seconds,
            )

            if not current_chord_vector_pause:
                continue
            accumulated_time_vector_pause = get_accumulated_time(
                note.start - rest_duration, tempo
            )

            time_left_current_chord_vector_pause: list[int] = one_hote_time_left(
                time_left_current_chord_pause
            )
            list_of_events.append(
                [
                    pitch_vector_pause,
                    duration_vector,
                    current_chord_vector_pause,
                    next_chord_vector_pause,
                    time_left_current_chord_vector_pause,
                    accumulated_time_vector_pause,
                    placement,
                ]
            )

        # Add the note

        pitch_vector = add_note(note.pitch)

        current_tick = end_tick
        if not current_chord_vector:
            continue

        list_of_events.append(
            [
                pitch_vector,
                duration_vector,
                current_chord_vector,
                next_chord_vector,
                time_left_current_chord_vector,
                accumulated_time_vector,
                placement,
            ]
        )

    return list_of_events


def add_note(pitch: int):
    pitch_vector: list[int] = [0] * (PITCH_VECTOR_SIZE + 1)
    assert PITCH_VECTOR_SIZE % 12 == 0
    octaves = pitch // 12
    octaves = max(octaves, 5)

    octaves = min(octaves, 7)
    octaves -= 5

    note = pitch % 12

    index = 12 * octaves + note

    pitch_vector[index - 1] = 1
    return pitch_vector


def find_chord(chord_list, tempo, ticks_per_beat, start_tick, note_start_seconds):
    # Find the corresponding chord
    current_chord_vector = None
    next_chord_vector = None
    time_left_current_chord = None
    for j, (timing, chord) in enumerate(chord_list):
        chord_start_time = seconds_to_ticks(timing[0], tempo, ticks_per_beat)
        chord_end_time = seconds_to_ticks(timing[1], tempo, ticks_per_beat)
        if chord_start_time <= start_tick and chord_end_time >= start_tick:
            # Only save chord if there is a chord, and not the last chord
            if "N" not in chord:
                current_chord: str = chord
                chord_end = float(chord_list[j][0][1])
                time_left_current_chord: float = calculate_chord_beats(
                    note_start_seconds, chord_end, tempo
                )  # In beats

                try:
                    current_chord_vector: list[int] = get_chord_list(chord_list[j][1])
                    next_chord_vector: list[int] = get_chord_list(chord_list[j + 1][1])

                except:
                    return None, None, None

                chord = next(
                    (i for i, value in enumerate(current_chord) if value == 1), None
                )
            # If there is no chord played, or last chord
            else:
                return None, None, None

    return (
        current_chord_vector,
        next_chord_vector,
        time_left_current_chord,
    )


def get_accumulated_time(note_start: int, tempo: int) -> list[int]:
    note_start_beats = round(note_start * tempo / 60)
    relative_bar = note_start_beats % 4
    accumulated_time: list[int] = [0] * 4

    accumulated_time[relative_bar] = 1
    return accumulated_time


def get_key(val, dic):
    for key, value in dic.items():
        if value == val:
            return key
    return "Key not found"


def one_hote_time_left(time_left_current_chord: float) -> list[int]:
    list_oh: list[int] = [0] * 16
    time_left_current_chord *= 2
    index = round(time_left_current_chord) - 1
    index = max(index, 0)
    index = min(index, 15)
    list_oh[index] = 1
    return list_oh


def calculate_chord_beats(start_time: float, end_time: float, tempo: int) -> float:
    duration_seconds = end_time - start_time
    beats_per_second = tempo / 60
    number_of_beats = duration_seconds * beats_per_second
    return round(number_of_beats, 1)


def process_chord(chord_file: str) -> list[list[list[int], int]]:
    """
    Creates a list of chords with timing information.

    Args
    ----------
        chord_file (str): path to chord file

    Returns
    ----------
        list[list[list[int], int]]: list of chords with timing information
    """
    chord_timing: list[list[int]] = []
    with open(chord_file, "r") as file:
        for line in file:
            words: list[str] = line.split()

            timing: float = [float(words[0]), float(words[1])]
            chord: str = words[2]
            chord_timing.append([timing, chord])
    return chord_timing


def get_duration_list(note_duration_ticks: int, ticks_per_bar: int) -> list[int]:
    """
    Converts a note duration in ticks to a list of 16 values representing the duration of the note.

    Args:
        note_duration_ticks (int): Note duration in ticks
        ticks_per_bar (int): Ticks per bar

    Returns:
        list[int]: List of 16 values representing the duration of the note
    """

    # Calculate the duration of the note in whole notes
    duration_in_whole_notes: float = (note_duration_ticks / ticks_per_bar) * 4

    # Calculate the note type
    note_type: int = round(4 / duration_in_whole_notes)
    # Make sure the note type is within the range 1 to 16
    note_type = max(1, min(16, note_type))

    duration_vector: list[int] = [0] * 16
    duration_vector[note_type - 1] = 1

    return duration_vector


def seconds_to_ticks(seconds: float, tempo: int, resolution: int) -> int:
    """
    Calculates the number of ticks in a given number of seconds.

    Args
    ----------
        seconds (float): Number of seconds
        tempo (int): Tempo in beats per minute
        resolution (int): Resolution of the MIDI file

    Returns
    ----------
        int: Number of ticks
    """
    beats = (seconds * tempo) / 60
    ticks = beats * resolution
    return int(ticks)


def get_chord_list(input_string: str) -> list[int]:
    """
    Converts a chord string to a list of 72 values, one hot encoded to represent the chord.

    Args
    ----------
        input_string (str): Chord string

    Returns
    ----------
        list[int]: List length 72. One hot encoded to represent the correct chord
    """
    chord_list: list[int] = [0] * 72
    pattern: str = r"([A-Ga-g]#?b?:)(maj|min|dim|aug|sus2|sus4).*"
    match: re.Match = re.match(pattern, input_string)
    chord: str = match.group(1) + match.group(2)
    chord_list[FULL_CHORD_TO_INT[chord]] = 1
    return chord_list
