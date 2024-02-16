import pretty_midi
import torch

from .eval_agent import predict_next_k_notes_bass
from config import MODEL_PATH_BASS, DEVICE
from data_processing import Bass_Dataset
from .bass_network import Bass_Network
from ..utils import beats_to_seconds, seconds_to_beat


def play_bass(
    mid: pretty_midi.PrettyMIDI,
    primer: list,
    config: dict,
) -> tuple[pretty_midi.PrettyMIDI, list[int, int]]:
    """
    Generates and plays the bass sequence on top of the given midi file. Based on primer sequence, and configuration.

    Args:
        mid (pretty_midi.PrettyMIDI): The input MIDI file.
        primer (list): The primer sequence for generating the bass sequence.
        config (dict): The configuration settings for generating the bass sequence.

    Returns:
        tuple[pretty_midi.PrettyMIDI, list[int, int]]: A tuple containing the modified MIDI file,
        the bass instrument, and the predicted bass sequence.
    """

    bass_agent: Bass_Network = torch.load(MODEL_PATH_BASS, DEVICE)
    bass_agent.eval()

    predicted_bass_sequence: list[int, int] = predict_next_k_notes_bass(
        bass_agent, primer, config
    )

    if config["PLAYSTYLE"] == "bass_drum":
        bass_drum_times = find_bass_drum(mid, config["TEMPO"])

    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new Instrument instance for an Electric Bass
    bass_instrument = pretty_midi.Instrument(program=33)  # 33: Electric Bass

    # When playstyle is "bass_drum", synchronize the bass notes with bass drum hits
    if config["PLAYSTYLE"] == "bass_drum":
        play_bass_drum_style(
            bass_drum_times,
            bass_instrument,
            predicted_bass_sequence,
            note_mapping,
            config,
        )

    else:  # Original behavior if playstyle isn't "bass_drum"
        chord_start_time = 0.0
        for note, duration in predicted_bass_sequence:
            midi_note = note_mapping[note]
            bass_note = pretty_midi.Note(
                velocity=60,
                pitch=midi_note,
                start=chord_start_time,
                end=chord_start_time + (duration / 2),
            )
            bass_instrument.notes.append(bass_note)
            chord_start_time += duration / 2

    # Add the bass_instrument to the PrettyMIDI object
    bass_instrument.name = "bass"
    mid.instruments.append(bass_instrument)

    return mid, bass_instrument, predicted_bass_sequence


def play_bass_drum_style(
    bass_drum_times, bass_instrument, predicted_bass_sequence, note_mapping, config
) -> None:
    """
    Plays the bass drum style based on the bass drum hits.

    Args:
    ----------
        bass_drum_times (list): List of bass drum hit times.
        bass_instrument (pretty_midi.Instrument): PrettyMIDI Instrument object for the bass.
        predicted_bass_sequence (list): List of tuples representing the predicted bass sequence.
            Each tuple contains the note and its duration.
        note_mapping (dict): Dictionary mapping note names to MIDI note numbers.
        config (dict): Configuration settings.

    Returns:
    ----------
        None
    """

    running_time = start_time = end_time = duration_acc = chord_start_time = 0.0

    for note, duration in predicted_bass_sequence:
        chord_start_time = duration_acc
        duration_acc += duration
        while running_time < duration_acc:
            midi_note = note_mapping[note]

            tloc = (
                chord_start_time + duration
            ) - running_time  # Time left in the chord

            start_time = running_time
            if not bass_drum_times:
                end_time = chord_start_time + duration
            else:
                # End time is either the end of the chord or the next bass drum hit, whichever comes first
                end_time = min(running_time + tloc, bass_drum_times[0])

            play_note(
                bass_instrument,
                pitch=midi_note,
                start_time=start_time,
                end_time=end_time,
                tempo=config["TEMPO"],
            )
            if bass_drum_times and end_time == bass_drum_times[0]:
                bass_drum_times.pop(0)
            running_time = end_time


def play_note(
    bass_instrument: pretty_midi.Instrument,
    pitch: int,
    start_time: float,
    end_time: float,
    tempo: int,
    velocity: int = 70,
) -> None:
    """
    Adds a note to the bass instrument.

    Parameters:
    ----------
    - bass_instrument (pretty_midi.Instrument): The bass instrument to add the note to.
    - pitch (int): The MIDI pitch of the note.
    - start_time (float): The start time of the note in beats.
    - end_time (float): The end time of the note in beats.
    - tempo (int): The tempo of the music in beats per minute.
    - velocity (int, optional): The velocity (loudness) of the note. Defaults to 70.

    returns:
    ----------
    - None
    """

    bass_note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=beats_to_seconds(start_time, tempo),
        end=beats_to_seconds(end_time, tempo),
    )
    bass_instrument.notes.append(bass_note)


def find_bass_drum(pm: pretty_midi.PrettyMIDI, tempo: int) -> list[float]:
    """
    Find every time step where the bass drum is played in a given PrettyMIDI object.

    Args:
    - pm (pretty_midi.PrettyMIDI): A PrettyMIDI object containing one drum track.

    Returns:
    - List[float]: List of start times (in seconds) where the bass drum is played.
    """
    bass_drum_times = []

    # Assuming the drum track is the first track in the PrettyMIDI object
    drum_track = pm.instruments[0]

    # Check if the track is a drum track
    if not drum_track.is_drum:
        raise ValueError("The provided track is not a drum track.")

    # Iterate over notes in the drum track
    for note in drum_track.notes:
        if note.pitch == 36:  # 36 is the MIDI number for Acoustic Bass Drum
            bass_drum_times.append(seconds_to_beat(note.start, tempo))

    return bass_drum_times
