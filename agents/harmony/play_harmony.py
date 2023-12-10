import pretty_midi

from ..utils import beats_to_seconds
from config import PITCH_SIZE_MELODY


def play_harmony(
    mid: pretty_midi.PrettyMIDI, melody_sequence: list[list[int]], config: dict
) -> pretty_midi.PrettyMIDI:
    harmony_instrument = pretty_midi.Instrument(program=81)

    running_duration = 0
    for pitch, duration in melody_sequence:
        duration *= 0.25

        harmony_pitch = pitch + config["INTERVAL_HARMONY"]

        start = beats_to_seconds(running_duration, config["TEMPO"])
        end = beats_to_seconds(running_duration + duration, config["TEMPO"])

        running_duration += duration
        # If the note is a pause, skip it
        if pitch == 5 * 12 + PITCH_SIZE_MELODY:
            continue

        harmony_note = pretty_midi.Note(
            velocity=1,
            pitch=harmony_pitch,
            start=start,
            end=end,
        )

        harmony_instrument.notes.append(harmony_note)

    mid.instruments.append(harmony_instrument)

    return mid
