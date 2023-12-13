from mido import MidiFile, MidiTrack, Message
import pretty_midi
import random

import torch

from .eval_agent import predict_next_k_notes_chords
from ..utils import beats_to_seconds

from config import (
    MODEL_PATH_CHORD,
    INT_TO_TRIAD,
    DEVICE,
)


def play_chord(
    mid: MidiFile, predicted_bass_sequence, dataset_primer, config: dict
) -> tuple[MidiFile, list]:
    timed_chord_sequence = get_timed_chord_sequence(
        predicted_bass_sequence, dataset_primer
    )

    if config["ARPEGIATE_CHORD"]:
        mid = play_chord_arpeggiate(mid, timed_chord_sequence, config)
    elif config["BOUNCE_CHORD"]:
        mid = play_chord_bounce(mid, timed_chord_sequence, config)
    else:
        mid = play_chord_hold(mid, timed_chord_sequence, config)

    return mid, timed_chord_sequence


def play_chord_hold(pretty_midi_obj, chord_sequence, config):
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord notes
    note_mapping = {i: 60 + i for i in range(24)}

    # Create a new Instrument instance for an Acoustic Grand Piano
    piano_instrument = pretty_midi.Instrument(program=0)  # 0: Acoustic Grand Piano

    current_time = 0.0  # Initialize a variable to keep track of time

    # Add chords to the piano_instrument
    for chord, duration in chord_sequence:
        # Create a Note instance for each note in the chord
        for note in chord:
            midi_note = note_mapping[note]
            piano_note = pretty_midi.Note(
                velocity=64,  # volume
                pitch=midi_note,  # MIDI note number
                start=beats_to_seconds(current_time, config["TEMPO"]),  # start time
                end=beats_to_seconds(
                    current_time + duration, config["TEMPO"]
                ),  # end time
            )
            # Add note to the piano_instrument
            piano_instrument.notes.append(piano_note)

        # Move the current time cursor
        current_time += duration

    # Add the piano_instrument to the PrettyMIDI object
    piano_instrument.name = "chord"
    pretty_midi_obj.instruments.append(piano_instrument)

    return pretty_midi_obj


def play_chord_arpeggiate(pm, chord_sequence, config):
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
    note_mapping = {i: 60 + i for i in range(24)}

    # Create an instrument instance for Acoustic Grand Piano
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program, is_drum=False)

    # Define the melodic pattern
    if config["ARP_STYLE"] == 0:
        melodic_pattern = [0, 0, 1, 2, 0, 2, 1, 0]
    elif config["ARP_STYLE"] == 1:
        melodic_pattern = [0, 0, 1, 2, 1, 0]
    elif config["ARP_STYLE"] == 2:
        melodic_pattern = [0, 1, 2, 1]
    elif config["ARP_STYLE"] == 3:
        melodic_pattern = [0, 2, 0, 0, 1, 2, 1, 0]
    else:
        raise ValueError("Invalid ARP_STYLE value.", config["ARP_STYLE"])

    seconds_per_beat = 60 / config["TEMPO"]
    note_duration = 4 * seconds_per_beat / len(melodic_pattern)
    start_time = 0.0
    # Add chord chords to the instrument
    for chord, duration in chord_sequence:
        num_repeats, remander = divmod(duration, (note_duration * len(melodic_pattern)))

        for _ in range(int(num_repeats)):
            for idx, pattern_note in enumerate(melodic_pattern):
                midi_note = note_mapping[chord[pattern_note]]
                if idx == 0 and config["ARP_STYLE"] == 0 or config["ARP_STYLE"] == 1:
                    midi_note -= 12  # Lower the root note by one octave
                if idx == 4 and config["ARP_STYLE"] == 0:
                    midi_note += 12  # increase the root note by one octave
                if config["ARP_STYLE"] == 3:
                    if idx == 0 or idx == 1 or idx == 2:
                        midi_note -= 24

                velocity = random.randint(40, 64)
                # Add note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=beats_to_seconds(start_time, config["TEMPO"]),
                    end=beats_to_seconds(start_time + note_duration, config["TEMPO"]),
                )
                piano.notes.append(note)
                start_time += note_duration
        if remander:
            for part_druation in [1, 2, 3, 4, 5, 6]:
                if remander == part_druation:
                    num_notes = int(remander / note_duration)
                    pattern_slice = melodic_pattern[:num_notes]
                    for idx, pattern_note in enumerate(pattern_slice):
                        midi_note = note_mapping[chord[pattern_note]]
                        if (
                            idx == 0
                            and config["ARP_STYLE"] == 0
                            or config["ARP_STYLE"] == 1
                        ):
                            midi_note -= 12  # Lower the root note by one octave
                        if idx == 4 and config["ARP_STYLE"] == 0:
                            midi_note += 12  # increase the root note by one octave

                        # Add note
                        velocity = random.randint(55, 70)
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=midi_note,
                            start=beats_to_seconds(start_time, config["TEMPO"]),
                            end=beats_to_seconds(
                                start_time + note_duration, config["TEMPO"]
                            ),
                        )
                        piano.notes.append(note)
                        start_time += note_duration

    # Append instrument to PrettyMIDI object
    piano.name = "chord"
    pm.instruments.append(piano)
    return pm


def play_chord_bounce(
    pm: pretty_midi.PrettyMIDI, chord_sequence, config: dict
) -> pretty_midi.PrettyMIDI:
    pass


def get_timed_chord_sequence(full_bass_sequence, dataset_primer):
    chord_agent = torch.load(MODEL_PATH_CHORD, DEVICE)
    chord_agent.eval()

    full_chord_sequence = predict_next_k_notes_chords(
        chord_agent, full_bass_sequence, dataset_primer
    )

    timed_chord_sequence = []
    full_chord_timed = []

    for idx, note in enumerate(full_bass_sequence):
        timed_chord_sequence.append(
            (full_chord_sequence[idx][0], full_chord_sequence[idx][1], note[1])
        )
    for root, chord, duration in timed_chord_sequence:
        try:
            full_chord = INT_TO_TRIAD[chord]
        except KeyError:
            full_chord = INT_TO_TRIAD[0]
        full_chord = [x + root for x in full_chord]
        full_chord_timed.append((full_chord, duration))

    return full_chord_timed
