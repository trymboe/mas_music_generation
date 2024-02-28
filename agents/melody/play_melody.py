import torch
import pretty_midi


from .melody_network import Melody_Network
from .eval_agent import predict_next_notes
from ..utils import beats_to_seconds, adjust_for_key

from config import (
    MODEL_PATH_MELODY,
    MODEL_NON_COOP_PATH_MELODY,
    DEVICE,
    PITCH_SIZE_MELODY,
)


def play_melody(
    mid: pretty_midi.PrettyMIDI,
    chord_sequence: list[tuple],
    melody_primer: int,
    config: dict,
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument, list]:
    """
    Plays a melody based on the given chord sequence and configuration.

    Args:
    -----
        mid (pretty_midi.PrettyMIDI): The MIDI object to add the melody to.
        chord_sequence (list[tuple]): The sequence of chords to generate the melody from.
        melody_primer (int): The primer note for the melody generation.
        config (dict): The configuration settings for the melody generation.

    Returns:
    -----
        tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument, list]: A tuple containing the modified MIDI object,
        the melody instrument, and the generated note sequence.
    """

    if config["NON_COOPERATIVE"]:
        melody_agent: Melody_Network = torch.load(MODEL_NON_COOP_PATH_MELODY, DEVICE)
    else:
        melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    melody_agent.eval()

    note_sequence = predict_next_notes(
        chord_sequence, melody_agent, melody_primer, config
    )
    mid, melody_instrument = play_melody_notes(note_sequence, mid, config)
    return mid, melody_instrument, note_sequence


def play_known_melody(
    mid: pretty_midi.PrettyMIDI, note_sequence: list, config: dict
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    """
    Plays a known melody using the provided note sequence and configuration.
    For when the melody is "kept" and not generated.

    Args:
    -----
        mid (pretty_midi.PrettyMIDI): The PrettyMIDI object to add the melody to.
        note_sequence (list): The sequence of notes to be played.
        config (dict): The configuration settings for playing the melody.

    Returns:
    -----
        tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]: The modified PrettyMIDI object and the melody instrument.
    """
    mid, melody_instrument = play_melody_notes(note_sequence, mid, config)

    return mid, melody_instrument


def play_melody_notes(
    note_sequence, mid: pretty_midi.PrettyMIDI, config: dict
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    """
    Plays the melody notes by converting the note sequence into MIDI notes and adding them to the provided PrettyMIDI object.

    Args:
    -----
        note_sequence (list[tuple[int, float]]): A list of tuples representing the notes and their durations.
        mid (pretty_midi.PrettyMIDI): The PrettyMIDI object to which the melody notes will be added.
        config (dict): A dictionary containing configuration parameters.

    Returns:
    -----
        tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]: A tuple containing the updated PrettyMIDI object and the melody instrument.

    """
    melody_instrument = pretty_midi.Instrument(program=0)
    running_time: float = 0.0
    for note, duration in note_sequence:
        # get duration in beats, originally in quarter notes
        duration = duration * 0.25

        # If the note is a pause, skip it
        if note == 5 * 12 + PITCH_SIZE_MELODY:
            running_time += duration
            continue
        start = beats_to_seconds(running_time, config["TEMPO"])
        end = beats_to_seconds(running_time + (duration), config["TEMPO"])
        note = adjust_for_key(note, config["KEY"])
        melody_note: pretty_midi.Note = pretty_midi.Note(
            velocity=72, pitch=note, start=start, end=end
        )
        melody_instrument.notes.append(melody_note)
        running_time += duration

    melody_instrument.name = "melody"
    mid.instruments.append(melody_instrument)

    return mid, melody_instrument
