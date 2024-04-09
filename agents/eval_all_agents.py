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
from .chord.chord_network import Chord_Network, Chord_Network_Full


from config import (
    TEST_DATASET_PATH_MELODY,
    DEVICE,
    MODEL_PATH_MELODY,
    TEST_DATASET_PATH_BASS,
    MODEL_PATH_BASS,
    TEST_DATASET_PATH_CHORD,
    MODEL_PATH_CHORD,
    MODEL_NON_COOP_PATH_MELODY,
    MODEL_NON_COOP_PATH_CHORD,
    TEST_DATASET_PATH_CHORD_BASS,
    MODEL_CHORD_BASS_PATH,
    MODEL_PATH_CHORD_LSTM,
    MODEL_PATH_BASS_LSTM,
)

NUM_EVAL_SAMPLES = 1000


def eval_all_agents():
    print("----Evaluating all agents")
    # eval_chord()
    eval_bass()
    # eval_melody()
    # eval_chord_bass()
    # eval_chord_and_bass_separately()
    exit()


def eval_chord():
    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD, DEVICE)
    # chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD, DEVICE)
    chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD_LSTM, DEVICE)

    log_likelihood = 0.0
    correct_predictions = 0
    distribution = [0, 0, 0, 0, 0, 0, 0]

    for _ in range(NUM_EVAL_SAMPLES):
        random_index = random.randint(0, len(chord_dataset) - 1)
        chord_primer = chord_dataset[random_index]
        input_sequence, ground_truth = chord_primer[0].unsqueeze(0), chord_primer[1]

        output = chord_network(input_sequence)

        # Apply softmax to get probabilities
        chord_probabilities = F.softmax(output[0, :], dim=-1)

        # For accuracy calculation, sampling from the distribution
        next_chord_type = torch.multinomial(chord_probabilities, 1).item()
        # for i in range(len(input_sequence[:, :, 0].tolist())):
        #     print(
        #         input_sequence[:, :, 0].tolist()[i], input_sequence[:, :, 1].tolist()[i]
        #     )

        # print(next_chord_type, ground_truth.item())
        # exit()
        # print(input_sequence[:, :, 0], input_sequence[:, :, 1], ground_truth.item())
        if next_chord_type == ground_truth.item():
            correct_predictions += 1
        distribution[next_chord_type] = distribution[next_chord_type] + 1

        # For log likelihood, use the probability of the true class directly
        true_chord_probability = chord_probabilities[ground_truth.item()]
        log_likelihood += torch.log(true_chord_probability).item()

    # Convert log likelihood sum to mean log likelihood if desired
    mean_log_likelihood = log_likelihood / NUM_EVAL_SAMPLES
    print(distribution)
    print(
        "Chord agent predicted",
        correct_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the chords correctly.",
    )

    print("Mean Log Likelihood:", mean_log_likelihood)


def eval_bass():
    bass_dataset: Bass_Dataset = torch.load(TEST_DATASET_PATH_BASS, DEVICE)
    bass_network: Bass_Network = torch.load(MODEL_PATH_BASS_LSTM, DEVICE)
    log_likelihood_notes = 0.0
    log_likelihood_durations = 0.0
    correct_predictions_note = 0
    correct_predictions_duration = 0

    for _ in range(NUM_EVAL_SAMPLES):
        random_index = random.randint(0, len(bass_dataset) - 1)
        bass_primer = bass_dataset[random_index]
        input_note, input_duration, ground_truth = (
            bass_primer[0].unsqueeze(0),
            bass_primer[1].unsqueeze(0),
            bass_primer[2],
        )

        with torch.no_grad():
            note_output, duration_output = bass_network(input_note, input_duration)

            note_probabilities = F.softmax(note_output[0, :], dim=-1)
            duration_probabilities = F.softmax(
                duration_output[0, :], dim=-1
            )  # Ensure this is -1 for consistency

            # For accuracy, sampling from the distribution
            next_note = torch.multinomial(note_probabilities, 1).item()
            next_duration = torch.multinomial(duration_probabilities, 1).item()

            gt_note, gt_duration = ground_truth[0].item(), ground_truth[1].item()

            if next_note == gt_note:
                correct_predictions_note += 1
            if next_duration == gt_duration:
                correct_predictions_duration += 1

            # For log likelihood
            true_note_probability = note_probabilities[gt_note]
            true_duration_probability = duration_probabilities[gt_duration]

            log_likelihood_notes += torch.log(true_note_probability).item()
            log_likelihood_durations += torch.log(true_duration_probability).item()

    # Convert log likelihood sums to mean log likelihood
    mean_log_likelihood_notes = log_likelihood_notes / NUM_EVAL_SAMPLES
    mean_log_likelihood_durations = log_likelihood_durations / NUM_EVAL_SAMPLES

    print(
        "Bass agent predicted",
        correct_predictions_note / NUM_EVAL_SAMPLES * 100,
        "% of the notes correctly.",
    )
    print(
        "Bass agent predicted",
        correct_predictions_duration / NUM_EVAL_SAMPLES * 100,
        "% of the durations correctly.",
    )
    print("Mean Log Likelihood for Notes:", mean_log_likelihood_notes)
    print("Mean Log Likelihood for Durations:", mean_log_likelihood_durations)


def eval_melody():
    COOP = False
    ALL = True
    if ALL:
        eval_all_melody_agents()
    if COOP:
        melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    else:
        melody_agent: Melody_Network = torch.load(MODEL_NON_COOP_PATH_MELODY, DEVICE)

    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY, DEVICE)
    correct_predictions: int = 0

    for _ in range(NUM_EVAL_SAMPLES):
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

            if COOP:
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
            else:
                x = torch.cat(
                    (
                        pitches,
                        durations,
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
        correct_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the notes correctly.",
    )


def eval_chord_bass():
    chord_dataset_full: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD_BASS, DEVICE)
    chord_network_full: Chord_Network_Full = torch.load(MODEL_CHORD_BASS_PATH, DEVICE)

    correct_predictions_note: int = 0
    correct_predictions_duration: int = 0
    correct_prediction_combined: int = 0

    for _ in range(NUM_EVAL_SAMPLES):
        random_index = random.randint(0, len(chord_dataset_full) - 1)
        primer = chord_dataset_full[random_index]
        input_sequence = primer[0].unsqueeze(0)
        ground_truth = primer[1]

        root = input_sequence[:, :, 0]
        chord = input_sequence[:, :, 1]
        duration = input_sequence[:, :, 2]

        with torch.no_grad():
            root_output, chord_output, duration_output = chord_network_full(
                root, chord, duration
            )

            root_probabilities = F.softmax(root_output[0, :], dim=-1).view(-1)

            note_probabilities = F.softmax(chord_output[0, :], dim=-1).view(-1)

            duration_probabilities = F.softmax(duration_output[0, :], dim=0)

            # Sample from the distributions
            next_root = torch.multinomial(root_probabilities, 1).unsqueeze(1)
            next_note = torch.multinomial(note_probabilities, 1).unsqueeze(1)
            next_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)

            gt_root, gt_note, gt_duration = (
                ground_truth[0].item(),
                ground_truth[1].item(),
                ground_truth[2].item(),
            )

            if next_note == gt_note:
                correct_predictions_note += 1
            if next_duration == gt_duration:
                correct_predictions_duration += 1
            if (
                next_root == gt_root
                and next_note == gt_note
                and next_duration == gt_duration
            ):
                correct_prediction_combined += 1

    print(
        "chord agent predicted",
        correct_predictions_note / NUM_EVAL_SAMPLES * 100,
        "% of the notes correctly.",
    )
    print(
        "chord agent predicted",
        correct_predictions_duration / NUM_EVAL_SAMPLES * 100,
        "% of the durations correctly.",
    )
    print(
        "chord agent predicted",
        correct_prediction_combined / NUM_EVAL_SAMPLES * 100,
        "% of the combined correctly.",
    )

    print((correct_predictions_duration + correct_predictions_note) / 2)
    print()


def eval_chord_and_bass_separately():
    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD, DEVICE)
    chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD_LSTM, DEVICE)

    bass_dataset: Bass_Dataset = torch.load(TEST_DATASET_PATH_BASS, DEVICE)
    bass_network: Bass_Network = torch.load(MODEL_PATH_BASS, DEVICE)

    correct_combined_predictions: int = 0
    correct_duration_predictions: int = 0
    correct_note_predictions: int = 0
    correct_chord_predictions: int = 0
    correct_full_chord_predictions: int = 0
    for _ in range(NUM_EVAL_SAMPLES):
        random_index = random.randint(0, len(chord_dataset) - 1)
        chord_primer = chord_dataset[random_index]
        bass_primer = bass_dataset[random_index]

        input_sequence_chord = chord_primer[0].unsqueeze(0)
        input_note = bass_primer[0].unsqueeze(0)
        input_duration = bass_primer[1].unsqueeze(0)

        ground_truth_chord = chord_primer[1]
        ground_truth_bass_pitch, ground_truth_bass_duration = (
            bass_primer[2][0],
            bass_primer[2][1],
        )

        output_chord = chord_network(input_sequence_chord)
        output_bass_pitch, output_bass_duration = bass_network(
            input_note, input_duration
        )

        chord_probabilities = F.softmax(output_chord[0, :], dim=-1)
        bass_probabilities_p = F.softmax(output_bass_pitch[0, :], dim=-1)
        bass_probabilities_d = F.softmax(output_bass_duration[0, :], dim=-1)

        next_chord = torch.multinomial(chord_probabilities, 1).item()
        next_bass_p = torch.multinomial(bass_probabilities_p, 1).item()
        next_bass_d = torch.multinomial(bass_probabilities_d, 1).item()

        if (
            next_chord == ground_truth_chord.item()
            and next_bass_p == ground_truth_bass_pitch.item()
            and next_bass_d == ground_truth_bass_duration.item()
        ):
            correct_combined_predictions += 1
        if next_chord == ground_truth_chord.item():
            correct_chord_predictions += 1
        if next_bass_p == ground_truth_bass_pitch.item():
            correct_note_predictions += 1
        if next_bass_d == ground_truth_bass_duration.item():
            correct_duration_predictions += 1
        if (
            next_chord == ground_truth_chord.item()
            and next_bass_p == ground_truth_bass_pitch.item()
        ):
            correct_full_chord_predictions += 1

    print(
        "Chord and bass agents predicted",
        correct_combined_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the chords and bass correctly.",
    )
    print(
        "Chord agent predicted",
        correct_chord_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the chords correctly.",
    )
    print(
        "Bass agent predicted",
        correct_note_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the notes correctly.",
    )
    print(
        "Bass agent predicted",
        correct_duration_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the durations correctly.",
    )
    print(
        "Chord and bass agents predicted",
        correct_full_chord_predictions / NUM_EVAL_SAMPLES * 100,
        "% of the chords and bass correctly.",
    )

    print(
        (
            correct_note_predictions
            + correct_duration_predictions
            + correct_chord_predictions
        )
        / 3
    )


def eval_all_melody_agents():
    melody_agent_coop: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    melody_agent_non_coop: Melody_Network = torch.load(
        MODEL_NON_COOP_PATH_MELODY, DEVICE
    )
    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY, DEVICE)

    correct_predictions_pitch_coop: int = 0
    correct_predictions_duration_coop: int = 0
    correct_predictions_pitch_non_coop: int = 0
    correct_predictions_duration_non_coop: int = 0

    for i, input in enumerate(melody_dataset):
        if i % 100 == 0:
            print(f"Evaluated {i} of {len(melody_dataset)} samples")
        input_sequence = input[0]
        ground_truth = input[1][0]

        with torch.no_grad():
            (
                pitches,
                durations,
                current_chords,
                next_chords,
                current_chord_time_lefts,
                accumulated_times,
            ) = get_tensors(input_sequence)
            x_coop = torch.cat(
                (
                    pitches,
                    durations,
                    current_chords,
                    next_chords,
                    current_chord_time_lefts,
                ),
                dim=1,
            ).unsqueeze(0)

            x_non_coop = torch.cat(
                (
                    pitches,
                    durations,
                ),
                dim=1,
            ).unsqueeze(0)

            # add batch dimension
            accumulated_times = accumulated_times.unsqueeze(0)
            current_chord_time_lefts = current_chord_time_lefts.unsqueeze(0)

            pitch_logits_coop, duration_logits_coop = melody_agent_coop(
                x_coop, accumulated_times, current_chord_time_lefts
            )
            pitch_logits_non_coop, duration_logits_non_coop = melody_agent_non_coop(
                x_non_coop, accumulated_times, current_chord_time_lefts
            )

            note_probabilities_c = F.softmax(pitch_logits_coop, dim=1).view(-1)
            duration_probabilities_c = F.softmax(duration_logits_coop, dim=1).view(-1)

            predicted_note_c = torch.multinomial(note_probabilities_c, 1).unsqueeze(1)
            predicted_duration_c = torch.multinomial(
                duration_probabilities_c, 1
            ).unsqueeze(1)

            note_probabilities_nc = F.softmax(pitch_logits_non_coop, dim=1).view(-1)
            duration_probabilities_nc = F.softmax(duration_logits_non_coop, dim=1).view(
                -1
            )

            predicted_note_nc = torch.multinomial(note_probabilities_nc, 1).unsqueeze(1)
            predicted_duration_nc = torch.multinomial(
                duration_probabilities_nc, 1
            ).unsqueeze(1)

            gt_note = get_one_hot_index(ground_truth[0])
            gt_duration = get_one_hot_index(ground_truth[1])

            if predicted_note_c == gt_note:
                correct_predictions_pitch_coop += 1
            if predicted_duration_c == gt_duration:
                correct_predictions_duration_coop += 1
            if predicted_note_nc == gt_note:
                correct_predictions_pitch_non_coop += 1
            if predicted_duration_nc == gt_duration:
                correct_predictions_duration_non_coop += 1

    print(
        "Melody agent Coop predicted",
        correct_predictions_pitch_coop / len(melody_dataset) * 100,
        "% of the pitches correctly.",
    )
    print(
        "Melody agent Coop predicted",
        correct_predictions_duration_coop / len(melody_dataset) * 100,
        "% of the durations correctly.",
    )
    print(
        "Melody agent Non Coop predicted",
        correct_predictions_pitch_non_coop / len(melody_dataset) * 100,
        "% of the pitches correctly.",
    )
    print(
        "Melody agent Non Coop predicted",
        correct_predictions_duration_non_coop / len(melody_dataset) * 100,
        "% of the durations correctly.",
    )
