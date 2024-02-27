import pretty_midi

from ..utils import beats_to_seconds, adjust_for_key
from config import PITCH_SIZE_MELODY


def play_harmony(
    mid: pretty_midi.PrettyMIDI, melody_sequence: list[list[int]], config: dict
) -> pretty_midi.PrettyMIDI:
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
):
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
        velocity = 72
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
):
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
