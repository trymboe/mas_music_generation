import torch
import pretty_midi


from .melody_network import Melody_Network
from .eval_agent import predict_next_notes

from config import *


def play_melody(
    mid: pretty_midi.PrettyMIDI, chord_sequence: list[tuple], melody_primer: int
):
    melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    melody_agent.eval()

    note_sequence = predict_next_notes(chord_sequence, melody_agent, melody_primer)
    mid = play_melody_notes(note_sequence, mid)
    return mid


def play_melody_notes(note_sequence, mid: pretty_midi.PrettyMIDI):
    melody_instrument = pretty_midi.Instrument(program=81)
    running_time: float = 0.0
    for note, duration in note_sequence:
        # get duration in beats, orignaly in quarter notes
        duration = duration * 0.25

        if note == 5 * 12 + PITCH_SIZE_MELODY:
            running_time += duration
            continue
        start = beats_to_seconds(running_time)
        end = beats_to_seconds(running_time + (duration))
        melody_note: pretty_midi.Note = pretty_midi.Note(
            velocity=5, pitch=note, start=start, end=end
        )
        melody_instrument.notes.append(melody_note)
        running_time += duration

    mid.instruments.append(melody_instrument)

    return mid


def beats_to_seconds(beats: float) -> float:
    return beats * (60 / TEMPO)


def beats_to_seconds(beats: float) -> float:
    return round(beats * (60 / TEMPO), 2)
