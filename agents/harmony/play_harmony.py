import pretty_midi

from ..utils import beats_to_seconds, adjust_for_key
from config import PITCH_SIZE_MELODY


def play_harmony(
    mid: pretty_midi.PrettyMIDI, melody_sequence: list[list[int]], config: dict
) -> pretty_midi.PrettyMIDI:
    """
    Adds harmony to the given MIDI file based on the provided melody sequence and configuration.

    Args:
        mid (pretty_midi.PrettyMIDI): The input MIDI file.
        melody_sequence (list[list[int]]): The melody sequence to harmonize.
        config (dict): The configuration settings for harmonization.

    Returns:
        pretty_midi.PrettyMIDI: The modified MIDI file with added harmony.
    """

    harmony_instrument = pretty_midi.Instrument(program=0)

    if config["INTERVAL"] and not config["DELAY"]:
        harmony_instrument = play_interval(melody_sequence, config, harmony_instrument)
    elif config["DELAY"]:
        harmony_instrument = play_delay(melody_sequence, config, harmony_instrument)
    else:
        return mid

    harmony_instrument.name = "harmony"
    mid.instruments.append(harmony_instrument)

    return mid


def play_delay(
    melody_sequence: list[list[int]],
    config: dict,
    harmony_instrument: pretty_midi.Instrument,
) -> pretty_midi.Instrument:
    """
    Plays a delayed version of the melody sequence on the harmony instrument.
    Also adds interval harmony if specified in the configuration.

    Args:
        melody_sequence (list[list[int]]): The melody sequence to be played, where each element is a list containing the pitch and duration of a note.
        config (dict): Configuration settings for the playback, including tempo, interval, key, and length.
        harmony_instrument (pretty_midi.Instrument): The harmony instrument to play the delayed melody on.

    Returns:
        pretty_midi.Instrument: The harmony instrument with the delayed melody notes added.
    """

    running_duration = 0
    for pitch, duration in melody_sequence:
        duration *= 0.25

        tempo = config["TEMPO"]
        # delay of an 8th note
        delay = 30 / tempo

        if config["INTERVAL"]:
            play_pitch = pitch + 5
        else:
            play_pitch = pitch

        start = beats_to_seconds(running_duration, config["TEMPO"])
        end = beats_to_seconds(running_duration + duration, config["TEMPO"])

        running_duration += duration
        # If the note is a pause, skip it
        if pitch == 5 * 12 + PITCH_SIZE_MELODY:
            continue

        play_pitch = adjust_for_key(play_pitch, config["KEY"])

        max_length = (config["LENGTH"] * 4 / tempo) * 60

        # for i in range(config["DELAY_NUM"]):
        velocity = 60
        for _ in range(3):
            if start + delay > max_length:
                start = start
            else:
                start = start + delay

            if end + delay > max_length:
                end = end
            else:
                end = end + delay
            velocity = velocity - 10

            harmony_note = pretty_midi.Note(
                velocity=velocity,
                pitch=play_pitch,
                start=start,
                end=end,
            )

        harmony_instrument.notes.append(harmony_note)
    return harmony_instrument


def play_interval(
    melody_sequence: list[list[int]],
    config: dict,
    harmony_instrument: pretty_midi.Instrument,
) -> pretty_midi.Instrument:
    """
    Plays the given melody sequence as harmony notes and adds them to the harmony_instrument.

    Args:
        melody_sequence (list[list[int]]): The melody sequence to be played as harmony notes.
        config (dict): Configuration settings for the music generation.
        harmony_instrument (pretty_midi.Instrument): The instrument to which the harmony notes will be added.

    Returns:
        pretty_midi.Instrument: The harmony_instrument with the added harmony notes.
    """

    running_duration = 0
    for pitch, duration in melody_sequence:
        duration *= 0.25

        harmony_pitch = pitch + 5

        start = beats_to_seconds(running_duration, config["TEMPO"])
        end = beats_to_seconds(running_duration + duration, config["TEMPO"])

        running_duration += duration
        # If the note is a pause, skip it
        if pitch == 5 * 12 + PITCH_SIZE_MELODY:
            continue
        harmony_pitch = adjust_for_key(harmony_pitch, config["KEY"])
        harmony_note = pretty_midi.Note(
            velocity=60,
            pitch=harmony_pitch,
            start=start,
            end=end,
        )

        harmony_instrument.notes.append(harmony_note)
    return harmony_instrument
