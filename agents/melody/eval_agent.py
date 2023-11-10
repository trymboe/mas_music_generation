import torch

import torch.nn.functional as F

from config import (
    CHORD_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    PITCH_SIZE_MELODY,
    INT_TO_TRIAD,
    SCALE_MELODY,
    TEMPO,
)


def predict_next_notes(chord_sequence, melody_agent) -> list[list[int]]:
    with torch.no_grad():
        all_notes: list[list[int]] = []

        duration_preferences: list[int] = [0, 1, 3, 7, 15]
        if SCALE_MELODY:
            pitch_preferences: list[int] = generate_scale_preferences()

        running_time_total_beats: float = 0
        running_time_on_chord_beats: float = 0

        chord_num: int = 0
        current_chord_duration_beats: int = chord_sequence[0][1]
        current_chord: torch.Tensor = get_chord_tensor(chord_sequence[0][0])
        next_chord: torch.Tensor = get_chord_tensor(chord_sequence[0][0])
        pitches, durations = get_pitch_duration_tensor(chord_sequence[0][0][0] + 59, 7)
        is_start_of_bar: bool = False
        is_end_of_bar: bool = True
        bars: torch.Tensor = torch.tensor([is_start_of_bar, is_end_of_bar])

        sum_duration: float = 0.0
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
            note_probabilities = F.softmax(note_output[0, :], dim=-1).view(-1)
            if pitch_preferences:
                note_probabilities = select_with_preference(
                    note_probabilities, pitch_preferences
                )

            duration_probabilities = F.softmax(duration_output[0, :], dim=0)
            duration_probabilities = select_with_preference(
                duration_probabilities, duration_preferences
            )

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)
            duration_in_beats: float = round(4 / (next_duration.item() + 1), 2) * 2
            sum_duration += duration_in_beats
            current_chord_display = (
                next(i for i, value in enumerate(current_chord) if value == 1),
                None,
            )
            current_pitch = next(
                (i for i, value in enumerate(pitches) if value == 1), None
            )

            all_notes.append([next_note.item() + 1, duration_in_beats])

            running_time_on_chord_beats += duration_in_beats
            # We are done
            if (
                running_time_on_chord_beats > current_chord_duration_beats
                and chord_num >= len(chord_sequence) - 1
            ):
                break

            while running_time_on_chord_beats > current_chord_duration_beats:
                chord_num += 1

                running_time_on_chord_beats -= current_chord_duration_beats
                try:
                    current_chord_duration_beats = chord_sequence[chord_num][1]
                # If no more chord, set duration to 4 beats
                except:
                    current_chord_duration_beats = 1

                try:
                    current_chord: torch.Tensor = get_chord_tensor(
                        chord_sequence[chord_num][0]
                    )
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
            if (running_time_total_beats / 4) % 4 < 0.125:
                is_start_of_bar: bool = True
            else:
                is_start_of_bar: bool = False
            if (running_time_total_beats / 4) % 4 > 0.875:
                is_end_of_bar: bool = True
            else:
                is_end_of_bar: bool = False

            bars: torch.Tensor = torch.tensor([is_start_of_bar, is_end_of_bar])

    return all_notes


def seconds_to_beat(seconds: float) -> float:
    return round(seconds * (TEMPO / 60), 2)


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


def get_pitch_duration_tensor(
    pitch: int, duration: int
) -> [torch.Tensor, torch.Tensor]:
    pitch_vector = [0] * PITCH_SIZE_MELODY
    pitch_vector[pitch] = 1
    duration_vector = [0] * DURATION_SIZE_MELODY
    duration_vector[duration] = 1
    return torch.tensor(pitch_vector), torch.tensor(duration_vector)


def select_with_preference(probs, preferred_indices):
    # Create a mask with zeros at all positions
    mask = torch.zeros_like(probs)

    # Set the mask to 1 at preferred indices
    mask[preferred_indices] = 1

    # Apply the mask to the probabilities
    masked_probs = probs * mask

    # Check if there is at least one preferred index with non-zero probability
    if torch.sum(masked_probs) > 0:
        # Normalize the probabilities
        masked_probs /= torch.sum(masked_probs)
        # Select using the modified probabilities
        return masked_probs

    else:
        # If all preferred indices have zero probability, fall back to the original distribution
        return probs


def generate_scale_preferences() -> list[int]:
    if SCALE_MELODY == "major pentatonic":
        intervals = [0, 2, 4, 7, 9]

    full_range = []

    # Iterate through all MIDI notes
    for midi_note in range(128):  # MIDI notes range from 0 to 127
        # Check if the note is in the correct scale
        if midi_note % 12 in intervals:
            note_index = midi_note - 1
            if note_index > 0:
                full_range.append(note_index)

    return full_range
