import torch
import torch.nn.functional as F
import random

from .melody.melody_network import Melody_Network
from data_processing import Melody_Dataset, Bass_Dataset, Chord_Dataset
from .melody.eval_agent import (
    get_chord_tensor,
    get_time_left_on_chord_tensor,
    get_accumulated_time_tensor,
    get_pitch_duration_tensor,
    update_input_tensors,
    select_with_preference,
    apply_temperature,
    get_tensors,
    generate_scale_preferences,
    get_one_hot_index,
)
from .bass.bass_network import Bass_Network
from .chord.chord_network import Chord_Network


from config import (
    TEST_DATASET_PATH_MELODY,
    DEVICE,
    MODEL_PATH_MELODY,
    TEST_DATASET_PATH_BASS,
    MODEL_PATH_BASS,
    TEST_DATASET_PATH_CHORD,
    MODEL_PATH_CHORD,
)


def eval_all_agents():
    print("----Evaluating all agents")
    eval_chord()
    eval_bass()
    eval_melody()
    exit()


def eval_chord():
    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD)
    chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD, DEVICE)
    correct_predictions: int = 0
    for _ in range(1000):
        random_index = random.randint(0, len(chord_dataset))
        chord_primer = chord_dataset[random_index]
        input_sequence = chord_primer[0].unsqueeze(0)
        ground_truth = chord_primer[1]

        output = chord_network(input_sequence)

        # Apply softmax to get probabilities
        chord_probabilities = F.softmax(output[0, :], dim=-1)

        # Sample from the distribution
        next_chord_type = torch.multinomial(chord_probabilities, 1).item()
        if next_chord_type == ground_truth.item():
            correct_predictions += 1

    print(
        "Chord agent predicted", correct_predictions / 500, "% of the chords correctly."
    )


def eval_bass():
    bass_dataset: Bass_Dataset = torch.load(TEST_DATASET_PATH_BASS)
    bass_network: Bass_Network = torch.load(MODEL_PATH_BASS, DEVICE)
    correct_predictions_note: int = 0
    correct_predictions_duration: int = 0
    for _ in range(1000):
        random_index = random.randint(0, len(bass_dataset))
        bass_primer = bass_dataset[random_index]
        input_note = bass_primer[0].unsqueeze(0)
        input_duration = bass_primer[1].unsqueeze(0)

        ground_truth = bass_primer[2]

        with torch.no_grad():
            note_output, duration_output = bass_network(input_note, input_duration)

            note_probabilities = F.softmax(note_output[0, :], dim=-1).view(-1)

            duration_probabilities = F.softmax(duration_output[0, :], dim=0)

            # Sample from the distributions
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)

            gt_note, gt_duration = ground_truth[0].item(), ground_truth[1].item()

            if next_note == gt_note:
                correct_predictions_note += 1
            if next_duration == gt_duration:
                correct_predictions_duration += 1

    print(
        "Bass agent predicted",
        correct_predictions_note / 500,
        "% of the notes correctly.",
    )
    print(
        "Bass agent predicted",
        correct_predictions_duration / 500,
        "% of the durations correctly.",
    )


def eval_melody():
    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY)
    melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    correct_predictions: int = 0
    for _ in range(1000):
        random_index = random.randint(0, len(melody_dataset))
        melody_primer = melody_dataset[random_index]
        input_sequence = melody_primer[0]
        ground_truth = melody_primer[1][0]

        with torch.no_grad():
            (
                pitches,
                durations,
                current_chords,
                next_chords,
                current_chord_time_lefts,
                accumulated_times,
            ) = get_tensors(input_sequence)

            x = torch.cat(
                (
                    pitches,
                    durations,
                    current_chords,
                    next_chords,
                    current_chord_time_lefts,
                ),
                dim=1,
            )
            # add batch dimension
            accumulated_times = accumulated_times.unsqueeze(0)
            current_chord_time_lefts = current_chord_time_lefts.unsqueeze(0)
            x = x.unsqueeze(0)

            pitch_logits, duration_logits = melody_agent(
                x, accumulated_times, current_chord_time_lefts
            )

            note_probabilities = F.softmax(pitch_logits, dim=1).view(-1)
            duration_probabilities = F.softmax(duration_logits, dim=1).view(-1)

            predicted_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            predicted_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(
                1
            )

            gt_note = get_one_hot_index(ground_truth[0])

            if predicted_note == gt_note:
                correct_predictions += 1
    print(
        "Melody agent predicted",
        correct_predictions / 500,
        "% of the notes correctly.",
    )
