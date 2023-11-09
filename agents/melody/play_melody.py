import torch
import pretty_midi
import torch.nn.functional as F

from .melody_network import Melody_Network
from config import *


def play_melody(
    mid: pretty_midi.PrettyMIDI, chord_sequence: list[tuple], dataset_primer_start: int
):
    melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    melody_agent.eval()

    note_sequence = predict_next_notes(chord_sequence, melody_agent)
    mid = play_melody_notes(note_sequence, mid)
    return mid


def play_melody_notes(note_sequence, mid: pretty_midi.PrettyMIDI):
    melody_instrument = pretty_midi.Instrument(program=0)
    running_time: float = 0.0
    for note, duration in note_sequence:
        start = running_time
        end = running_time + (4 / duration)
        melody_note: pretty_midi.Note = pretty_midi.Note(
            velocity=64, pitch=note, start=start, end=end
        )
        melody_instrument.notes.append(melody_note)
        running_time += 4 / duration

    mid.instruments.append(melody_instrument)

    return mid


def predict_next_notes(chord_sequence, melody_agent):
    with torch.no_grad():
        print(chord_sequence)
        all_notes: list[list[int]] = []

        running_time_total: int = 0
        running_time_on_chord: int = 0

        chord_num: int = 0

        current_chord_duration: int = chord_sequence[0][1]
        current_chord: torch.Tensor = get_chord_tensor(chord_sequence[0][0])
        next_chord: torch.Tensor = get_chord_tensor(chord_sequence[1][0])
        pitches, durations = get_pitch_duration_tensor(chord_sequence[0][0][0] + 60, 7)
        is_start_of_bar: bool = True
        is_end_of_bar: bool = False
        bars: torch.Tensor = torch.tensor([is_start_of_bar, is_end_of_bar])

        while True:
            x = torch.cat(
                (
                    pitches,
                    durations,
                    current_chord,
                    next_chord,
                    bars.float(),
                ),
                dim=0,
            )
            # add batch dimension
            x = x.unsqueeze(0)

            note_output, duration_output = melody_agent(x)

            # Apply softmax to get probabilities for notes and durations
            note_probabilities = F.softmax(note_output[0, :], dim=-1).view(
                -1
            )  # reshape to 1D

            duration_probabilities = F.softmax(
                duration_output[0, :], dim=0
            )  # Only consider the last prediction

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)
            all_notes.append([next_note.item() + 1, (next_duration.item() + 1) / 4])

            running_time_on_chord += (1 / (next_duration.item() + 1)) / 4

            running_time_on_chord += round(running_time_on_chord, 2)

            running_time_total += running_time_on_chord

            # Change chord
            if (
                running_time_on_chord > current_chord_duration
                and chord_num + 1 == len(chord_sequence) - 1
            ):
                break

            while running_time_on_chord > current_chord_duration:
                chord_num += 1
                running_time_on_chord -= current_chord_duration
                current_chord_duration = chord_sequence[chord_num][1]

                current_chord: torch.Tensor = get_chord_tensor(
                    chord_sequence[chord_num][0]
                )
                try:
                    next_chord: torch.Tensor = get_chord_tensor(
                        chord_sequence[chord_num + 1][0]
                    )
                # If there are no more chords, current chord is set as next chord
                except:
                    next_chord: torch.Tensor = current_chord

            pitches, durations = get_pitch_duration_tensor(
                next_note.item(), (next_duration.item())
            )

            # Check if the current note is end or start of bar (With 1/8 note threshold)
            if (running_time_total / 4) % 4 < 0.125:
                is_start_of_bar: bool = True
            else:
                is_start_of_bar: bool = False
            if (running_time_total / 4) % 4 > 0.875:
                is_end_of_bar: bool = True
            else:
                is_end_of_bar: bool = False
            bars: torch.Tensor = torch.tensor([is_start_of_bar, is_end_of_bar])
    print(all_notes)
    return all_notes


def get_pitch_duration_tensor(
    pitch: int, duration: int
) -> [torch.Tensor, torch.Tensor]:
    pitch_vector = [0] * PITCH_SIZE_MELODY
    pitch_vector[pitch] = 1
    duration_vector = [0] * DURATION_SIZE_MELODY
    duration_vector[duration] = 1
    return torch.tensor(pitch_vector), torch.tensor(duration_vector)


def get_chord_tensor(chord: list[int]) -> torch.Tensor:
    """
    One hot encodes a chord into a tensor list.
    Args:
        chord (list[int]): traid chord in form [note, note, note]

    Returns:
        torch.Tensor: one hot encoded list of chord
    """
    root_note: int = chord[0]
    chord_type: list[int] = [c - root_note for c in chord]

    chord_type: int = get_key(chord_type, INT_TO_TRIAD)

    chord_index: int = root_note * 6 + chord_type

    chord_vector = [0] * CHORD_SIZE_MELODY
    chord_vector[chord_index] = 1
    return torch.tensor(chord_vector)


# Function to find key from value
def get_key(val, dic):
    for key, value in dic.items():
        if value == val:
            return key
    return "Key not found"
