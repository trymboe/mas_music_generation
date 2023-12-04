import torch

import torch.nn.functional as F

from config import (
    CHORD_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    PITCH_SIZE_MELODY,
    INT_TO_TRIAD,
    SCALE_MELODY,
    TEMPO,
    NOTE_TEMPERATURE_MELODY,
    DURATION_TEMPERATURE_MELODY,
    PITCH_VECTOR_SIZE,
)


def predict_next_notes(chord_sequence, melody_agent, melody_primer) -> list[list[int]]:
    with torch.no_grad():
        all_notes: list[list[int]] = []

        duration_preferences: list[int] = [0, 1, 3, 7, 15]
        if SCALE_MELODY:
            pitch_preferences: list[int] = generate_scale_preferences()

        running_time_total_beats: float = 0
        running_time_on_chord_beats: float = 0

        current_chord_duration_beats = chord_sequence[0][1]
        next_current_chord = get_chord_tensor(chord_sequence[0][0])
        next_next_chord = get_chord_tensor(chord_sequence[1][0])

        (
            pitches,
            durations,
            current_chords,
            next_chords,
            current_chord_time_lefts,
            accumulated_times,
        ) = get_tensors(melody_primer)

        chord_num: int = 0
        accumulated_time: int = 0

        sum_duration: float = 0.0
        while True:
            x = torch.cat(
                (
                    pitches,
                    durations,
                    current_chords,
                    next_chords,
                ),
                dim=1,
            )

            # add batch dimension
            x = x.unsqueeze(0)
            accumulated_times = accumulated_times.unsqueeze(0)
            current_chord_time_lefts = current_chord_time_lefts.unsqueeze(0)

            pitch_logits, duration_logits = melody_agent(
                x, accumulated_times, current_chord_time_lefts
            )

            note_logits_temperature = apply_temperature(
                pitch_logits[0, :], NOTE_TEMPERATURE_MELODY
            )

            duration_logits_temperature = apply_temperature(
                duration_logits[0, :], DURATION_TEMPERATURE_MELODY
            )

            note_probabilities = F.softmax(note_logits_temperature, dim=0).view(-1)
            duration_probabilities = F.softmax(duration_logits_temperature, dim=0).view(
                -1
            )

            if SCALE_MELODY:
                note_probabilities = select_with_preference(
                    note_probabilities, pitch_preferences
                )

            duration_probabilities = select_with_preference(
                duration_probabilities, duration_preferences
            )

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)
            duration_in_beats: float = round(4 / (next_duration.item() + 1), 2) * 2
            sum_duration += duration_in_beats

            accumulated_time += duration_in_beats

            all_notes.append([next_note.item() + 61, duration_in_beats])

            next_accumulated_time = get_accumulated_time_tensor(accumulated_time)

            running_time_on_chord_beats += duration_in_beats
            # We are done
            if (
                running_time_on_chord_beats > current_chord_duration_beats
                and chord_num >= len(chord_sequence) - 1
            ):
                break

            while running_time_on_chord_beats > current_chord_duration_beats:
                chord_num += 1
                if chord_num >= len(chord_sequence):
                    break

                running_time_on_chord_beats -= current_chord_duration_beats
                try:
                    current_chord_duration_beats = chord_sequence[chord_num][1]
                # If no more chord, set duration to 4 beats
                except:
                    current_chord_duration_beats = 1

                try:
                    next_current_chord: torch.Tensor = get_chord_tensor(
                        chord_sequence[chord_num][0]
                    )
                    next_next_chord: torch.Tensor = get_chord_tensor(
                        chord_sequence[chord_num + 1][0]
                    )
                # If there are no more chords, current chord is set as next chord
                except:
                    next_current_chord: torch.Tensor = get_chord_tensor(
                        chord_sequence[chord_num][0]
                    )

                    next_next_chord: torch.Tensor = next_current_chord

            next_pitch_vector, next_duration_vector = get_pitch_duration_tensor(
                next_note.item(), (next_duration.item())
            )

            next_current_chord_time_lefts = get_time_left_on_chord_tensor(
                current_chord_duration_beats, running_time_on_chord_beats
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

            (
                pitches,
                durations,
                current_chords,
                next_chords,
                accumulated_times,
                current_chord_time_lefts,
            ) = update_input_tensors(
                pitches,
                durations,
                current_chords,
                next_chords,
                accumulated_times,
                current_chord_time_lefts,
                next_pitch_vector,
                next_duration_vector,
                next_current_chord,
                next_next_chord,
                next_accumulated_time,
                next_current_chord_time_lefts,
            )

    return all_notes


def get_tensors(melody_primer):
    pitches = []
    durations = []
    current_chords = []
    next_chords = []
    current_chord_time_lefts = []
    accumulated_times = []

    for note in melody_primer:
        # Convert each item to a tensor before appending
        pitches.append(torch.tensor(note[0]))
        durations.append(torch.tensor(note[1]))
        current_chords.append(torch.tensor(note[2]))
        next_chords.append(torch.tensor(note[3]))
        current_chord_time_lefts.append(torch.tensor(note[4]))
        accumulated_times.append(torch.tensor(note[5]))

    pitches = torch.stack(pitches)
    durations = torch.stack(durations)
    current_chords = torch.stack(current_chords)
    next_chords = torch.stack(next_chords)
    current_chord_time_lefts = torch.stack(current_chord_time_lefts)
    accumulated_times = torch.stack(accumulated_times)

    return (
        pitches,
        durations,
        current_chords,
        next_chords,
        current_chord_time_lefts,
        accumulated_times,
    )


def update_input_tensors(
    pitches,
    durations,
    current_chords,
    next_chords,
    accumulated_times,
    current_chord_time_lefts,
    next_pitch_vector,
    next_duration_vector,
    next_current_chord,
    next_next_chord,
    next_accumulated_time,
    next_current_chord_time_lefts,
):
    pitches = torch.cat((pitches, next_pitch_vector.unsqueeze(0)), dim=0)
    durations = torch.cat((durations, next_duration_vector.unsqueeze(0)), dim=0)
    current_chords = torch.cat((current_chords, next_current_chord.unsqueeze(0)), dim=0)
    next_chords = torch.cat((next_chords, next_next_chord.unsqueeze(0)), dim=0)

    current_chord_time_lefts = torch.cat(
        (
            current_chord_time_lefts.squeeze(0),
            next_current_chord_time_lefts.unsqueeze(0),
        ),
        dim=0,
    )

    accumulated_times = torch.cat(
        (accumulated_times.squeeze(0), next_accumulated_time.unsqueeze(0)), dim=0
    )

    pitches = pitches[1:]
    durations = durations[1:]
    current_chords = current_chords[1:]
    next_chords = next_chords[1:]
    accumulated_times = accumulated_times[1:]
    current_chord_time_lefts = current_chord_time_lefts[1:]

    print("pitch", get_one_hot_index(pitches[-1]))
    print("duration", get_one_hot_index(durations[-1]))
    print("current_chord", get_one_hot_index(current_chords[-1]))
    print("next_chord", get_one_hot_index(next_chords[-1]))
    print("accumulated_times", get_one_hot_index(accumulated_times[-1]))
    print("current_chord_time_lefts", get_one_hot_index(current_chord_time_lefts[-1]))

    print("")

    return (
        pitches,
        durations,
        current_chords,
        next_chords,
        accumulated_times,
        current_chord_time_lefts,
    )


def get_one_hot_index(one_hot_list: list[int]) -> int:
    """
    Gets the index of the one hot encoded list. For debugging.

    Args:
    ----------
        one_hot_list (list[int]): one hot encoded list

    Returns:
    ----------
        int: index of the one hot encoded list
    """
    return next((i for i, value in enumerate(one_hot_list) if value == 1), None)


def get_time_left_on_chord_tensor(
    current_chord_duration_beats: int, running_time_on_chord_beats: float
) -> torch.Tensor:
    time_left_on_chord: float = (
        current_chord_duration_beats - running_time_on_chord_beats
    )
    time_left_vector: list[int] = [0] * 16

    time_left_on_chord = min(time_left_on_chord, 15)
    time_left_vector[round(time_left_on_chord * 2)] = 1

    return torch.tensor(time_left_vector)


def get_accumulated_time_tensor(
    accumulated_bars: int,
) -> torch.Tensor:
    index: int = int(accumulated_bars % 4)

    accumulated_list = [0, 0, 0, 0]
    accumulated_list[index] = 1
    return torch.tensor(accumulated_list)


def apply_temperature(logits, temperature):
    # Adjust the logits by the temperature
    return logits / temperature


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
    if SCALE_MELODY == "major scale":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    full_range = []

    # Iterate through all MIDI notes
    for midi_note in range(PITCH_VECTOR_SIZE):  # MIDI notes range from 0 to 127
        # Check if the note is in the correct scale
        if midi_note % 12 in intervals:
            note_index = midi_note - 1
            if note_index > 0:
                full_range.append(note_index)
    # for pause
    full_range.append(PITCH_VECTOR_SIZE)

    return full_range
