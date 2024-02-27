import pretty_midi
import random

import torch

from .eval_agent import predict_next_k_notes_chords
from ..utils import beats_to_seconds, adjust_for_key

from config import (
    MODEL_PATH_CHORD,
    INT_TO_TRIAD,
    DEVICE,
)


def play_chord(
    mid: pretty_midi.PrettyMIDI, predicted_bass_sequence, dataset_primer, config: dict
) -> tuple[pretty_midi.PrettyMIDI, list, pretty_midi.Instrument]:
    """
    Plays a chord sequence based on the predicted bass sequence and dataset primer.

    Args:
    -----
        mid (pretty_midi.PrettyMIDI): The MIDI object to add the chord sequence to.
        predicted_bass_sequence: The predicted bass sequence.
        dataset_primer: The dataset primer.
        config (dict): Configuration settings.

    Returns:
    -----
        tuple[pretty_midi.PrettyMIDI, list, pretty_midi.Instrument]: A tuple containing the modified MIDI object,
        the chord instrument, and the timed chord sequence.
    """

    timed_chord_sequence = get_timed_chord_sequence(
        predicted_bass_sequence, dataset_primer
    )
    if config["ARPEGIATE_CHORD"]:
        mid, chord_instrument = play_chord_arpeggiate(mid, timed_chord_sequence, config)
    elif config["BOUNCE_CHORD"]:
        mid, chord_instrument = play_chord_bounce(mid, timed_chord_sequence, config)
    else:
        mid, chord_instrument = play_chord_hold(mid, timed_chord_sequence, config)

    return mid, chord_instrument, timed_chord_sequence


def play_known_chord(
    mid: pretty_midi.PrettyMIDI, timed_chord_sequence: list, config: dict
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    """
    Genereates a MIDI file with the given chord sequence.
    This function is used when a chord line is kept from the previous loop, but there might be
    a change in the playstyle.

    Args:
    -----
        mid (MidiFile): midi file to be modified
        timed_chord_sequence (list): list of tuples containing the chord and its duration
        config (dict): configuration dictionary

    Returns:
    -----
        tuple[MidiFile, pretty_midi.Instrument]: the modified midi file and the chord instrument
    """
    if config["ARPEGIATE_CHORD"]:
        mid, chord_instrument = play_chord_arpeggiate(mid, timed_chord_sequence, config)
    elif config["BOUNCE_CHORD"]:
        mid, chord_instrument = play_chord_bounce(mid, timed_chord_sequence, config)
    else:
        mid, chord_instrument = play_chord_hold(mid, timed_chord_sequence, config)

    return mid, chord_instrument


def play_chord_hold(
    pretty_midi_obj: pretty_midi.PrettyMIDI, chord_sequence: list, config: dict
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    """
    Adds a chord sequence to a PrettyMIDI object and returns the modified object.

    Args:
    -----
        pretty_midi_obj (pretty_midi.PrettyMIDI): The PrettyMIDI object to modify.
        chord_sequence (list): A list of tuples representing the chord sequence.
            Each tuple contains a list of notes in the chord and the duration of the chord.
        config (dict): A dictionary containing configuration parameters, including "KEY" and "TEMPO".

    Returns:
    -----
        tuple: A tuple containing the modified PrettyMIDI object and the piano instrument.

    """

    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord notes
    note_mapping = {i: 60 + i for i in range(24)}

    piano_instrument = pretty_midi.Instrument(program=0)  # 0: Acoustic Grand Piano

    current_time = 0.0

    for chord, duration in chord_sequence:
        for note in chord:
            note = adjust_for_key(note, config["KEY"])
            note = note % 12
            midi_note = note_mapping[note]
            piano_note = pretty_midi.Note(
                velocity=64,
                pitch=midi_note,
                start=beats_to_seconds(current_time, config["TEMPO"]),
                end=beats_to_seconds(current_time + duration, config["TEMPO"]),
            )
            # Add note to the piano_instrument
            piano_instrument.notes.append(piano_note)

        current_time += duration

    # Add the piano_instrument to the PrettyMIDI object
    piano_instrument.name = "chord"
    pretty_midi_obj.instruments.append(piano_instrument)

    return pretty_midi_obj, piano_instrument


def play_chord_arpeggiate(
    pm: pretty_midi.PrettyMIDI, chord_sequence: list, config: dict
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    """
    Arpeggiates and plays a chord sequence using different arpeggiation techniques.
    Using the provided PrettyMIDI object, chord sequence, and configuration.

    Args:
    -----
        pm (pretty_midi.PrettyMIDI): The PrettyMIDI object to add the arpeggiated chord to.
        chord_sequence (list): A list of tuples representing the chord sequence, where each tuple contains the chord and its duration.
        config (dict): A dictionary containing the configuration parameters for the arpeggiation.

    Returns:
    -----
        pretty_midi.PrettyMIDI: The updated PrettyMIDI object with the arpeggiated chord added.
        pretty_midi.Instrument: The instrument used to play the arpeggiated chord.
    """
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
    note_mapping = {i: 60 + i for i in range(36)}

    piano_instrument = pretty_midi.Instrument(program=0)  # 0: Acoustic Grand Piano

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

                velocity = random.randint(65, 75)
                # Add note
                midi_note = adjust_for_key(midi_note, config["KEY"])

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=beats_to_seconds(start_time, config["TEMPO"]),
                    end=beats_to_seconds(start_time + note_duration, config["TEMPO"]),
                )
                piano_instrument.notes.append(note)
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
                        velocity = random.randint(65, 75)
                        midi_note = adjust_for_key(midi_note, config["KEY"])
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=midi_note,
                            start=beats_to_seconds(start_time, config["TEMPO"]),
                            end=beats_to_seconds(
                                start_time + note_duration, config["TEMPO"]
                            ),
                        )
                        piano_instrument.notes.append(note)
                        start_time += note_duration

    # Append instrument to PrettyMIDI object
    piano_instrument.name = "chord"
    pm.instruments.append(piano_instrument)
    return pm, piano_instrument


def play_chord_bounce(
    pm: pretty_midi.PrettyMIDI, chord_sequence: list, config: dict
) -> pretty_midi.PrettyMIDI:
    """
    Plays a chord sequence with bouncing effect.

    Args:
    -----
        pm (pretty_midi.PrettyMIDI): The PrettyMIDI object to add the chord notes to.
        chord_sequence (list): A list of tuples representing the chord sequence. Each tuple contains a chord and its duration.
        config (dict): A dictionary containing configuration parameters such as key and tempo.

    Returns:
    -----
        tuple: A tuple containing the modified PrettyMIDI object and the piano instrument.

    """
    note_mapping = {i: 60 + i for i in range(24)}
    piano_instrument = pretty_midi.Instrument(program=0)  # 0: Acoustic Grand Piano
    current_time = 0.0  # Initialize a variable to keep track of time

    for chord, duration in chord_sequence:
        number_of_bounces = duration * 2
        bounce_duration = duration / number_of_bounces
        for _ in range(number_of_bounces):
            for note in chord:
                note = adjust_for_key(note, config["KEY"])
                note = note % 12
                midi_note = note_mapping[note]
                piano_note = pretty_midi.Note(
                    velocity=72,  # volume
                    pitch=midi_note,  # MIDI note number
                    start=beats_to_seconds(current_time, config["TEMPO"]),  # start time
                    end=beats_to_seconds(
                        current_time + bounce_duration, config["TEMPO"]
                    ),  # end time
                )
                # Add note to the piano_instrument
                piano_instrument.notes.append(piano_note)

            # Move the current time cursor
            current_time += bounce_duration

    # Add the piano_instrument to the PrettyMIDI object
    piano_instrument.name = "chord"
    pm.instruments.append(piano_instrument)

    return pm, piano_instrument


def get_timed_chord_sequence(full_bass_sequence: list, dataset_primer: list) -> list:
    """
    Generates a timed chord sequence based on the given full bass sequence and dataset primer.

    Args:
    -----
        full_bass_sequence (list): The full bass sequence.
        dataset_primer (list): The dataset primer.

    Returns:
    -----
        list: The generated timed chord sequence.
    """

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
