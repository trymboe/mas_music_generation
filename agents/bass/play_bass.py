import pretty_midi
import torch
import random

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

    if config["PLAYSTYLE"] == "bass_drum" or config["PLAYSTYLE"] == "transition":
        bass_drum_times = find_bass_drum(mid, config["TEMPO"])

    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new Instrument instance for an Electric Bass
    bass_instrument = pretty_midi.Instrument(program=33)  # 33: Electric Bass

    # When playstyle is "bass_drum", synchronize the bass notes with bass drum hits
    if config["PLAYSTYLE"] == "bass_drum":
        bass_instrument = play_bass_drum_style(
            bass_drum_times,
            bass_instrument,
            predicted_bass_sequence,
            note_mapping,
            config,
        )
    elif config["PLAYSTYLE"] == "transition":
        bass_instrument = play_transition_jam_style(
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
) -> pretty_midi.Instrument:
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
        bass_instrument (pretty_midi.Instrument): The bass instrument with the notes added.
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
    return bass_instrument


def play_transition_jam_style(
    bass_drum_times, bass_instrument, predicted_bass_sequence, note_mapping, config
) -> pretty_midi.Instrument:
    """
    Plays a transition jam style bass sequence.

    Args:
        bass_drum_times (list): List of bass drum hit times.
        bass_instrument (Instrument): The bass instrument to play.
        predicted_bass_sequence (list): List of predicted bass notes and durations.
        note_mapping (dict): Mapping of note names to MIDI values.
        config (dict): Configuration parameters.

    Returns:
        bass_instrument (pretty_midi.Instrument): The bass instrument with the notes added.
    """
    transition_style = ["octave_jump", "approach", "passing", "walking"]

    running_time = start_time = end_time = duration_acc = chord_start_time = 0.0

    for idx, note in enumerate(predicted_bass_sequence):
        note, duration = note
        if idx + 1 < len(predicted_bass_sequence):
            next_note = predicted_bass_sequence[idx + 1][0]
            shortes_distance, direction = find_shortes_distance(note, next_note)
            transition = random.choice(transition_style)
            # transition = "walking"
        else:
            next_note = predicted_bass_sequence[0][0]
            shortes_distance, direction = 0, "up"
        transition_notes, num_beat_transition = get_transition_note(
            note, next_note, shortes_distance, direction, transition
        )

        chord_start_time = duration_acc
        duration_acc += duration
        while running_time < duration_acc:
            midi_note = note_mapping[note]
            start_time = running_time

            tloc = (
                chord_start_time + duration - num_beat_transition
            ) - running_time  # Time left in the chord

            # If there are no more bass drum hits, play the rest of the chord
            if not bass_drum_times:
                end_time = chord_start_time + duration - num_beat_transition
            else:
                # End time is either the end of the chord or the next bass drum hit, whichever comes first
                end_time = min(running_time + tloc, bass_drum_times[0])

            if end_time <= running_time + tloc and duration != num_beat_transition:
                play_note(
                    bass_instrument,
                    pitch=midi_note,
                    start_time=start_time,
                    end_time=end_time,
                    tempo=config["TEMPO"],
                )

            # Check if we need to play transition notes
            # Will play if end of chord
            if end_time == running_time + tloc:
                # Play transition notes
                end_time = play_transition_notes(
                    transition_notes, bass_instrument, end_time, config
                )
                if bass_drum_times:
                    while end_time > bass_drum_times[0]:
                        bass_drum_times.pop(0)
                        if not bass_drum_times:
                            break

            # Remove the bass drum hit if it was played and exists
            if bass_drum_times and end_time == bass_drum_times[0]:
                bass_drum_times.pop(0)

            running_time = end_time

    return bass_instrument


def play_transition_notes(transition_notes, bass_instrument, end_time, config):
    """
    Plays the transition notes using the specified bass instrument.

    Args:
        transition_notes (list): A list of tuples containing the note and duration of each transition note.
        bass_instrument (str): The name of the bass instrument to be used for playing the notes.
        end_time (float): The end time of the previous note.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        float: The end time of the last played note.
    """
    for note, duration in transition_notes:
        note += 24
        play_note(
            bass_instrument,
            pitch=note,
            start_time=end_time,
            end_time=end_time + duration,
            tempo=config["TEMPO"],
        )
        end_time += duration
    return end_time


def get_transition_note(note, next_note, shortes_distance, direction, transition):
    """
    Get the transition notes and the number of beats for a given transition type.

    Parameters:
    note (int): The starting note.
    shortes_distance (float): The shortest distance between notes.
    direction (str): The direction of the transition.
    transition (str): The type of transition.

    Returns:
    transition_notes (list): A list of transition notes and their durations.
    num_beat_transition (int): The number of beats for the transition.
    """

    in_scale = [0, 2, 4, 5, 7, 9, 11]
    num_beat_transition = 0
    if transition == "octave_jump":
        num_beat_transition = 2
        transition_notes = [
            [note, 0.5],
            [note + 12, 0.5],
            [note, 0.5],
            [note + 12, 0.5],
        ]
    elif transition == "approach":
        num_beat_transition = 1
        if direction == "up":
            transition_notes = [
                [next_note - 1, 1],
            ]
        else:
            transition_notes = [
                [next_note + 1, 1],
            ]
    elif transition == "passing":
        num_beat_transition = 1
        transition_note = round(shortes_distance / 2)
        if direction == "up":
            transition_notes = [
                [note + transition_note, 1],
            ]
        else:
            transition_notes = [
                [note - transition_note, 1],
            ]
    elif transition == "walking":
        num_beat_transition = min(2, shortes_distance / 2)
        transition_notes = []
        if direction == "up":
            for i in range(int(num_beat_transition * 2)):
                note_pair = [next_note - int(num_beat_transition * 2 + 1) + i + 1, 0.5]
                transition_notes.append(note_pair)
        else:
            for i in range(int(num_beat_transition * 2)):
                note_pair = [next_note + int(num_beat_transition * 2 + 1) - i - 1, 0.5]
                transition_notes.append(note_pair)

    return transition_notes, num_beat_transition


def find_shortes_distance(a, b):
    """
    Calculates the shortest distance between two musical notes on a chromatic scale.

    Parameters:
    a (int): The first note value.
    b (int): The second note value.

    Returns:
    tuple: A tuple containing the shortest distance between the two notes and the direction of the distance.
           The distance can be either a direct distance or a circular distance on a chromatic scale.
           The direction can be either "up" or "down".
    """

    direct_distance = abs(a - b)

    direction = None
    if a < b:
        direction = "up"
    else:
        direction = "down"

    # Return the minimum of the two distances
    return direct_distance, direction


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
