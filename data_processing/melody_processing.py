import os
import pretty_midi
import numpy as np

from config import PITCH_VECTOR_SIZE


def extract_melody_and_chord(root_dir: str):
    num_files = 0
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

        list_of_events = process_melody_and_chord(midi_file, chord_file)
        if list_of_events:
            num_files += 1
            print("working on number", num_files)
    print(num_files)
    exit()


def process_melody_and_chord(
    midi_file: str, chord_file: str
) -> list[list[list[int]], list[list[int]], list[list[int]]]:
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

    beats_per_bar = 4  # Since it's 4/4 time signature
    ticks_per_beat = pm.resolution
    ticks_per_bar = beats_per_bar * ticks_per_beat
    # Tolerance for the duration of an eighth note
    eighth_note_tolerance = ticks_per_beat / 2

    current_tick = 0
    list_of_events = []

    # Iterate over the notes in the melody track
    for idx, note in enumerate(melody_track.notes):
        start_tick = pm.time_to_tick(note.start)
        end_tick = pm.time_to_tick(note.end)
        duration_ticks = end_tick - start_tick

        duration_vector = get_duration_list(duration_ticks, ticks_per_bar)

        # Check if the note is the start of a bar
        is_start_of_bar = abs(start_tick % ticks_per_bar) <= eighth_note_tolerance
        is_end_of_bar = (
            abs(end_tick % ticks_per_bar) <= eighth_note_tolerance
            or abs((end_tick % ticks_per_bar) - ticks_per_bar) <= eighth_note_tolerance
        )

        pitch_vector = [0] * (PITCH_VECTOR_SIZE + 1)
        if note.start > current_tick:
            # Add a rest if there is a gap between notes
            rest_duration = note.start - current_tick
            pitch_vector[-1] = 1
        else:
            pitch_vector[note.pitch - 1] = 1

        current_tick = end_tick

        list_of_events.append(
            [pitch_vector, duration_vector, [is_start_of_bar, is_end_of_bar]]
        )
    return list_of_events


def get_duration_list(note_duration_ticks: int, ticks_per_bar: int) -> list[int]:
    # Calculate the duration of the note in whole notes
    duration_in_whole_notes = (note_duration_ticks / ticks_per_bar) * 4

    # Calculate the note type
    note_type = round(4 / duration_in_whole_notes)
    # Make sure the note type is within the range 1 to 16
    note_type = max(1, min(16, note_type))

    duration_vector = [0] * 16
    duration_vector[note_type - 1] = 1

    return duration_vector
