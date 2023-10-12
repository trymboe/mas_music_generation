from .bass import play_bass
from .chord import play_chord
import random
import torch


from agents import predict_next_k_notes_bass, predict_next_k_notes_chords
from utils import get_full_bass_sequence


from config import INT_TO_TRIAD, K, MEM_LEN, DRUM_MAPPING, TIME_STEPS_VOCAB


def play_agents(
    chord_agent_tripple,
    bass_agent_tripple,
    drum_agent_tripple,
    arpeggiate,
    filename,
    device,
):
    bass_agent, notes_dataset, train_bass_agent = (
        bass_agent_tripple[0],
        bass_agent_tripple[1],
        bass_agent_tripple[2],
    )
    chord_agent, chords_dataset, train_chord_agent = (
        chord_agent_tripple[0],
        chord_agent_tripple[1],
        chord_agent_tripple[2],
    )
    drum_agent, drum_dataset, train_drum_agent = (
        drum_agent_tripple[0],
        drum_agent_tripple[1],
        drum_agent_tripple[2],
    )
    # Generate Drums

    part_of_dataset = random.randint(0, len(notes_dataset) - 1)

    bass_primer_sequence = get_primer_sequence(notes_dataset, part_of_dataset)

    predicted_bass_sequence = predict_next_k_notes_bass(
        bass_agent, bass_primer_sequence, K
    )

    full_bass_sequence = get_full_bass_sequence(
        bass_primer_sequence, predicted_bass_sequence
    )

    chord_input_sequence = get_input_sequence_chords(
        full_bass_sequence, chords_dataset, part_of_dataset, K
    )

    full_chord_sequence = predict_next_k_notes_chords(chord_agent, chord_input_sequence)

    timed_chord_sequence = get_timed_chord_sequence(
        full_chord_sequence, full_bass_sequence
    )

    mid = play_drum(device)

    mid = play_bass(mid, full_bass_sequence)

    mid = play_chord(mid, timed_chord_sequence, arpeggiate)

    mid.write(filename)


def get_timed_chord_sequence(full_chord_sequence, full_bass_sequence):
    timed_chord_sequence = []
    full_chord_timed = []

    for idx, note in enumerate(full_bass_sequence):
        timed_chord_sequence.append(
            (full_chord_sequence[idx][0], full_chord_sequence[idx][1], note[1])
        )
    for root, chord, duration in timed_chord_sequence:
        full_chord = INT_TO_TRIAD[chord]
        full_chord = [x + root for x in full_chord]
        full_chord_timed.append((full_chord, duration))

    return full_chord_timed


def get_input_sequence_chords(full_bass_sequence, chords_dataset, part_of_dataset, k):
    # Extract the corresponding chord sequence from the dataset
    actual_chord_sequence = chords_dataset[part_of_dataset][0]
    # Extract only the root notes from the full_bass_sequence
    bass_notes = [pair[0] for pair in full_bass_sequence]

    # Create the input sequence
    input_sequence = []

    # Iterate over the bass_notes and actual_chord_sequence to create the pairs
    for i, bass_note in enumerate(bass_notes):
        if i < len(actual_chord_sequence):  # If we have actual chords, use them
            input_sequence.append(
                [bass_note, actual_chord_sequence[i][1].item()]
            )  # Use .item() to extract scalar from tensor
        else:  # Otherwise, use the placeholder
            input_sequence.append([bass_note, 6])

    # Convert the list of lists to a tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.int64)

    return input_tensor


def get_primer_sequence(notes_dataset, part_of_dataset):
    primer_part = part_of_dataset
    primer_sequence = (
        notes_dataset[primer_part][0],
        notes_dataset[primer_part][1],
    )

    return primer_sequence


def map_bass_to_chords(full_bass_sequence):
    full_chord_sequence = []
    for idx, note in enumerate(full_bass_sequence):
        full_chord_sequence.append((mapping[note[0]], note[1]))

    return full_chord_sequence
