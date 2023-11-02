import os
import pretty_midi
from .datasets import Melody_Dataset
import torch
import re

from config import PITCH_VECTOR_SIZE, CHORD_TO_INT


def get_melody_dataset(root_dir: str) -> Melody_Dataset:
    num_files = 0
    all_events: list[list[list[int], list[int], list[list[int]], list[bool]]] = []
    if not os.path.exists("data/dataset/melody_dataset.pt"):
        for directory in os.listdir(root_dir):
            if ".DS_Store" in directory:
                continue
            for file in os.listdir(os.path.join(root_dir, directory)):
                if ".mid" in file:
                    midi_file: str = os.path.join(root_dir, directory, file)
                if "chord_midi" in file:
                    chord_file: str = os.path.join(root_dir, directory, file)
                else:
                    continue

            list_of_events: list[
                list[list[int], list[int], list[list[int]], list[bool]]
            ] = process_melody_and_chord(midi_file, chord_file)

            if list_of_events is not None:
                all_events.append(list_of_events)
                num_files += 1
                print(num_files)

        melody_dataset: Melody_Dataset = Melody_Dataset(all_events)
        torch.save(melody_dataset, "data/dataset/melody_dataset.pt")
    else:
        melody_dataset = torch.load("data/dataset/melody_dataset.pt")

    return melody_dataset


def process_melody_and_chord(
    midi_file: str, chord_file: str
) -> list[list[int], list[int], list[list[int]], list[bool]]:
    pm = pretty_midi.PrettyMIDI(midi_file)

    # Only work for time signature 4/4
    for time_signature in pm.time_signature_changes:
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            return None

    # Iterate over the instruments in the MIDI data
    melody_track: pretty_midi.instrument = None
    for instrument in pm.instruments:
        # Check if the instrument name is 'MELODY'
        if instrument.name == "MELODY":
            melody_track = instrument
            break

    if melody_track is None:
        raise Exception("No melody track found")

    print(midi_file)
    chord_list: list[list[list[int], int]] = process_chord(chord_file)

    ticks_per_beat: int = pm.resolution
    ticks_per_bar: int = ticks_per_beat * 4  # Since it's 4/4 time signature
    # Tolerance for the duration of an eighth note
    sixteenth_note_tolerance: float = ticks_per_beat / 4
    tempo: int = int(pm.get_tempo_changes()[1][0])

    current_tick: int = 0
    list_of_events: list[list[int], list[int], list[list[int]], list[bool]] = []
    chords: list[list[int]] = []

    no_chord: bool = False

    # Iterate over the notes in the melody track
    for idx, note in enumerate(melody_track.notes):
        start_tick: float = pm.time_to_tick(note.start)
        end_tick: float = pm.time_to_tick(note.end)
        duration_ticks: float = end_tick - start_tick

        duration_vector: list[int] = get_duration_list(duration_ticks, ticks_per_bar)

        for j, (timing, chord) in enumerate(chord_list):
            chord_start_time = seconds_to_ticks(timing[0], tempo, ticks_per_beat)
            chord_end_time = seconds_to_ticks(timing[1], tempo, ticks_per_beat)

            if chord_start_time <= start_tick and chord_end_time >= end_tick:
                # Only save chord if there is a chord, and not the last chord
                if "N" not in chord:
                    current_chord: str = chord
                    try:
                        current_chord: list[int] = get_chord_list(chord_list[j][1])
                        next_chord: list[int] = get_chord_list(chord_list[j + 1][1])
                    except:
                        no_chord = True
                        break
                    chords = [current_chord, next_chord]
                # If there is no chord played, or last chord
                else:
                    no_chord = True
                    break

        # Break if there is no corresponding chord
        if no_chord:
            no_chord = False
            continue

        # Check if the note is the start of a bar
        is_start_of_bar: bool = (
            abs(start_tick % ticks_per_bar) <= sixteenth_note_tolerance
        )
        is_end_of_bar: bool = (
            abs(end_tick % ticks_per_bar) <= sixteenth_note_tolerance
            or abs((end_tick % ticks_per_bar) - ticks_per_bar)
            <= sixteenth_note_tolerance
        )

        pitch_vector: list[int] = [0] * (PITCH_VECTOR_SIZE + 1)
        if note.start > current_tick:
            # Add a rest if there is a gap between notes
            rest_duration: float = note.start - current_tick
            pitch_vector[-1] = 1
            duration_vector: list[int] = get_duration_list(rest_duration, ticks_per_bar)
        else:
            pitch_vector[note.pitch - 1] = 1

        current_tick = end_tick

        list_of_events.append(
            [pitch_vector, duration_vector, chords, [is_start_of_bar, is_end_of_bar]]
        )
    return list_of_events


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
    chord_list[CHORD_TO_INT[chord]] = 1
    return chord_list
