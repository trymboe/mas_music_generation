import json
import torch
import torch.nn.functional as F
import random
import scipy.stats as stats
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from scipy.stats import shapiro
import copy
from torch import nn


from .coplay import get_primer_sequences
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
from .bass import Bass_Network, get_primer_sequence_bass
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
    MODEL_PATH_BASS_LSTM_TEST,
    MODEL_PATH_CHORD_LSTM_TEST1,
    TRAIN_DATASET_PATH_CHORD,
    TRAIN_DATASET_PATH_BASS,
)

NUM_EVAL_SAMPLES = 1000


def eval_all_agents():
    print("----Evaluating all agents")
    # eval_chord()
    # eval_bass()
    # eval_melody()
    # eval_chord_bass()
    # eval_chord_and_bass_separately()
    # eval_multi_agent_vs_monolithic()
    exp3_scrambled_vs_unscrambled()
    # eval_coop_vs_non_coop(True)
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


def eval_chord_bass(verbose=True):
    chord_dataset_full: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD_BASS, DEVICE)
    chord_network_full: Chord_Network_Full = torch.load(MODEL_CHORD_BASS_PATH, DEVICE)

    correct_predictions_chord: int = 0
    correct_predictions_duration: int = 0
    correct_predictions_root: int = 0
    correct_prediction_combined: int = 0

    chord_log_likelihood: float = []
    duration_log_likelihood: float = []
    root_log_likelihood: float = []
    total_log_likelihood: float = []

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
            chord_probabilities = F.softmax(chord_output[0, :], dim=-1).view(-1)
            duration_probabilities = F.softmax(duration_output[0, :], dim=0)

            gt_root, gt_chord, gt_duration = (
                ground_truth[0].item(),
                ground_truth[1].item(),
                ground_truth[2].item(),
            )

            root_log_prob = torch.log(root_probabilities[gt_root])
            chord_log_prob = torch.log(chord_probabilities[gt_chord])
            duration_log_prob = torch.log(duration_probabilities[gt_duration])

            chord_log_likelihood.append(chord_log_prob.item())
            duration_log_likelihood.append(duration_log_prob.item())
            root_log_likelihood.append(root_log_prob.item())

            sample_log_likelihood = root_log_prob + chord_log_prob + duration_log_prob
            total_log_likelihood.append(sample_log_likelihood.item() / 3)

            if gt_chord in chord_probabilities.argmax(dim=0, keepdim=True):
                correct_predictions_chord += 1
            if gt_duration in duration_probabilities.argmax(dim=0, keepdim=True):
                correct_predictions_duration += 1
            if gt_root in root_probabilities.argmax(dim=0, keepdim=True):
                correct_predictions_root += 1
            if (
                gt_root in root_probabilities.argmax(dim=0, keepdim=True)
                and gt_chord in chord_probabilities.argmax(dim=0, keepdim=True)
                and gt_duration in duration_probabilities.argmax(dim=0, keepdim=True)
            ):
                correct_prediction_combined += 1
    if verbose:
        print("------Monolithic Evaluation------")
        print(
            "Chord agent predicted",
            correct_predictions_root / NUM_EVAL_SAMPLES * 100,
            "% of the root notes correctly.",
        )
        print(
            "Chord agent predicted",
            correct_predictions_duration / NUM_EVAL_SAMPLES * 100,
            "% of the durations correctly.",
        )
        print(
            "Chord agent predicted",
            correct_predictions_chord / NUM_EVAL_SAMPLES * 100,
            "% of the chords correctly.",
        )
        print(
            "Chord agent predicted",
            correct_prediction_combined / NUM_EVAL_SAMPLES * 100,
            "% of the combined correctly.",
        )

        print("Log-likellihood of the chord", chord_log_likelihood / NUM_EVAL_SAMPLES)
        print(
            "Log-likellihood of the duration",
            duration_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print("Log-likellihood of the root", root_log_likelihood / NUM_EVAL_SAMPLES)
        print(
            "Total log-likelihood of the model:",
            total_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print(
            "Average log-likelihood per sample:",
            total_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print()
    return (
        correct_prediction_combined,
        correct_predictions_duration,
        correct_predictions_root,
        correct_predictions_chord,
        chord_log_likelihood,
        duration_log_likelihood,
        root_log_likelihood,
        total_log_likelihood,
    )


def eval_chord_and_bass_separately(verbose=True, train_dataset=False):
    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD, DEVICE)
    chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD_LSTM_TEST1, DEVICE)

    bass_dataset: Bass_Dataset = torch.load(TEST_DATASET_PATH_BASS, DEVICE)
    bass_network: Bass_Network = torch.load(MODEL_PATH_BASS_LSTM_TEST, DEVICE)

    if train_dataset:
        chord_dataset = torch.load(TRAIN_DATASET_PATH_CHORD, DEVICE)
        bass_dataset = torch.load(TRAIN_DATASET_PATH_BASS, DEVICE)

    correct_combined_predictions: int = 0
    correct_duration_predictions: int = 0
    correct_note_predictions: int = 0
    correct_chord_predictions: int = 0

    chord_log_likelihood: list = []
    duration_log_likelihood: list = []
    root_log_likelihood: list = []
    total_log_likelihood: list = []

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

        root_log_prob = torch.log(bass_probabilities_p[ground_truth_bass_pitch])
        chord_log_prob = torch.log(chord_probabilities[ground_truth_chord])
        duration_log_prob = torch.log(bass_probabilities_d[ground_truth_bass_duration])

        chord_log_likelihood.append(chord_log_prob.item())
        duration_log_likelihood.append(duration_log_prob.item())
        root_log_likelihood.append(root_log_prob.item())

        sample_log_likelihood = root_log_prob + chord_log_prob + duration_log_prob
        total_log_likelihood.append(sample_log_likelihood.item() / 3)

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

    if verbose:
        print("------MAS Evaluation------")
        print(
            "Bass agent predicted",
            correct_note_predictions / NUM_EVAL_SAMPLES * 100,
            "% of the root notes correctly.",
        )
        print(
            "Bass agent predicted",
            correct_duration_predictions / NUM_EVAL_SAMPLES * 100,
            "% of the durations correctly.",
        )
        print(
            "Chord agent predicted",
            correct_chord_predictions / NUM_EVAL_SAMPLES * 100,
            "% of the chords correctly.",
        )

        print(
            "Chord and bass agents predicted",
            correct_combined_predictions / NUM_EVAL_SAMPLES * 100,
            "% of the chords and bass correctly.",
        )

        print("Log-likellihood of the chord", chord_log_likelihood / NUM_EVAL_SAMPLES)
        print(
            "Log-likellihood of the duration",
            duration_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print("Log-likellihood of the root", root_log_likelihood / NUM_EVAL_SAMPLES)
        print(
            "Total log-likelihood of the model:",
            total_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print(
            "Average log-likelihood per sample:",
            total_log_likelihood / NUM_EVAL_SAMPLES,
        )
        print()

    return (
        correct_combined_predictions,
        correct_duration_predictions,
        correct_note_predictions,
        correct_chord_predictions,
        chord_log_likelihood,
        duration_log_likelihood,
        root_log_likelihood,
        total_log_likelihood,
    )


def eval_all_melody_agents(verbose=False, use_cache=True):
    melody_agent_coop: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)
    melody_agent_non_coop: Melody_Network = torch.load(
        MODEL_NON_COOP_PATH_MELODY, DEVICE
    )
    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY, DEVICE)

    correct_predictions_pitch_coop: int = 0
    correct_predictions_duration_coop: int = 0
    correct_predictions_pitch_non_coop: int = 0
    correct_predictions_duration_non_coop: int = 0

    log_likelihood_pitch_coop = []
    log_likelihood_duration_coop = []
    log_likelihood_pitch_non_coop = []
    log_likelihood_duration_non_coop = []

    nllLoss_pitch_coop = []
    nllLoss_duration_coop = []
    nllLoss_pitch_non_coop = []
    nllLoss_duration_non_coop = []

    for i, input in enumerate(melody_dataset):
        if i % 100 == 0:
            print(f"Evaluated {i} of {NUM_EVAL_SAMPLES} samples")
        if i > NUM_EVAL_SAMPLES:
            break
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

            nllLoss_pitch_coop.append(
                get_nllLoss(logits=pitch_logits_coop, targets=ground_truth[0])
            )
            nllLoss_duration_coop.append(
                get_nllLoss(logits=duration_logits_coop, targets=ground_truth[1])
            )
            nllLoss_pitch_non_coop.append(
                get_nllLoss(logits=pitch_logits_non_coop, targets=ground_truth[0])
            )
            nllLoss_duration_non_coop.append(
                get_nllLoss(logits=duration_logits_non_coop, targets=ground_truth[1])
            )

            gt_note = get_one_hot_index(ground_truth[0])
            gt_duration = get_one_hot_index(ground_truth[1])

            true_coop_pitch_probability = note_probabilities_c[gt_note]
            true_coop_duration_probability = duration_probabilities_c[gt_duration]

            true_non_coop_pitch_probability = note_probabilities_nc[gt_note]
            true_non_coop_duration_probability = duration_probabilities_nc[gt_duration]

            log_likelihood_pitch_coop.append(
                torch.log(true_coop_pitch_probability).item()
            )
            log_likelihood_duration_coop.append(
                torch.log(true_coop_duration_probability).item()
            )

            log_likelihood_pitch_non_coop.append(
                torch.log(true_non_coop_pitch_probability).item()
            )
            log_likelihood_duration_non_coop.append(
                torch.log(true_non_coop_duration_probability).item()
            )

            if predicted_note_c == gt_note:
                correct_predictions_pitch_coop += 1
            if predicted_duration_c == gt_duration:
                correct_predictions_duration_coop += 1
            if predicted_note_nc == gt_note:
                correct_predictions_pitch_non_coop += 1
            if predicted_duration_nc == gt_duration:
                correct_predictions_duration_non_coop += 1
    return (
        correct_predictions_pitch_coop,
        correct_predictions_duration_coop,
        correct_predictions_pitch_non_coop,
        correct_predictions_duration_non_coop,
        log_likelihood_pitch_coop,
        log_likelihood_duration_coop,
        log_likelihood_pitch_non_coop,
        log_likelihood_duration_non_coop,
        nllLoss_pitch_coop,
        nllLoss_duration_coop,
        nllLoss_pitch_non_coop,
        nllLoss_duration_non_coop,
    )


def get_nllLoss(logits, targets):
    log_probabilities = F.log_softmax(logits, dim=1).view(-1)

    loss_function = nn.NLLLoss()
    targets = torch.tensor(get_one_hot_index(targets)).to(DEVICE)

    loss = loss_function(log_probabilities, targets)

    return loss.item()


def eval_coop_vs_non_coop(use_cache=True):
    if not use_cache:
        (
            correct_predictions_pitch_coop,
            correct_predictions_duration_coop,
            correct_predictions_pitch_non_coop,
            correct_predictions_duration_non_coop,
            log_likelihood_pitch_coop,
            log_likelihood_duration_coop,
            log_likelihood_pitch_non_coop,
            log_likelihood_duration_non_coop,
            nllLoss_pitch_coop,
            nllLoss_duration_coop,
            nllLoss_pitch_non_coop,
            nllLoss_duration_non_coop,
        ) = eval_all_melody_agents()

        accuracy_coop = {
            "Pitch": correct_predictions_pitch_coop,
            "Duration": correct_predictions_duration_coop,
        }
        accuracy_non_coop = {
            "Pitch": correct_predictions_pitch_non_coop,
            "Duration": correct_predictions_duration_non_coop,
        }
        log_likelihood_coop = {
            "Pitch": log_likelihood_pitch_coop,
            "Duration": log_likelihood_duration_coop,
        }
        log_likelihood_non_coop = {
            "Pitch": log_likelihood_pitch_non_coop,
            "Duration": log_likelihood_duration_non_coop,
        }
        nllLoss_coop = {
            "Pitch": nllLoss_pitch_coop,
            "Duration": nllLoss_duration_coop,
        }
        nllLoss_non_coop = {
            "Pitch": nllLoss_pitch_non_coop,
            "Duration": nllLoss_duration_non_coop,
        }

        save_to_json(
            nllLoss_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/nllLoss_coop.json",
        )
        save_to_json(
            nllLoss_non_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/nllLoss_non_coop.json",
        )
        save_to_json(
            log_likelihood_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/log_likelihood_coop.json",
        )
        save_to_json(
            log_likelihood_non_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/log_likelihood_non_coop.json",
        )
        save_to_json(
            accuracy_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/accuracy_coop.json",
        )
        save_to_json(
            accuracy_non_coop,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/accuracy_non_coop.json",
        )
    else:
        log_likelihood_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/log_likelihood_coop.json"
        )
        log_likelihood_non_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/log_likelihood_non_coop.json"
        )
        accuracy_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/accuracy_coop.json"
        )
        accuracy_non_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/accuracy_non_coop.json"
        )
        nllLoss_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/nllLoss_coop.json"
        )
        nllLoss_non_coop = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp2/nllLoss_non_coop.json"
        )

    print("Coop, pitch ", sum(nllLoss_coop["Pitch"]) / len(nllLoss_coop["Pitch"]))
    print(
        "Non-coop, pitch ",
        sum(nllLoss_non_coop["Pitch"]) / len(nllLoss_non_coop["Pitch"]),
    )
    print(
        "Coop, duration ", sum(nllLoss_coop["Duration"]) / len(nllLoss_coop["Duration"])
    )
    print(
        "Non-coop, duration ",
        sum(nllLoss_non_coop["Duration"]) / len(nllLoss_non_coop["Duration"]),
    )

    box_plot(
        log_likelihood_coop, log_likelihood_non_coop, name="exp2/coop_vs_non_coop_box"
    )
    exit()
    violin_plot(
        log_likelihood_coop,
        log_likelihood_non_coop,
        name="exp2/coop_vs_non_coop_violin",
    )

    print("T-test results Log-likelihood:")
    for key in log_likelihood_coop.keys():
        print(wilcoxon_test(log_likelihood_coop[key], log_likelihood_non_coop[key]))
    print()
    print("Z-test results Accuracy:")
    for key in accuracy_coop.keys():
        print(
            "Accuracy for",
            key,
            "is:",
            accuracy_non_coop[key] / NUM_EVAL_SAMPLES,
            accuracy_coop[key] / NUM_EVAL_SAMPLES,
        )
        print(
            "p_value for, key, is", z_test(accuracy_coop[key], accuracy_non_coop[key])
        )


def save_temporal_data(
    log_likelihood_chord,
    log_likelihood_melody_pitch,
    log_likelihood_melody_duration,
    accuracy_chord,
    accuracy_melody_pitch,
    accuracy_melody_duration,
):
    save_to_json(
        log_likelihood_chord,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_chord_temp.json",
    )
    save_to_json(
        log_likelihood_melody_pitch,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_pitch_temp.json",
    )
    save_to_json(
        log_likelihood_melody_duration,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_duration_temp.json",
    )
    save_to_json(
        accuracy_chord,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_chord_temp.json",
    )
    save_to_json(
        accuracy_melody_pitch,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_pitch_temp.json",
    )
    save_to_json(
        accuracy_melody_duration,
        "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_duration_temp.json",
    )


def eval_scrambled_vs_unscrambled():
    chord_dataset: Chord_Dataset = torch.load(TEST_DATASET_PATH_CHORD, DEVICE)
    chord_network: Chord_Network = torch.load(MODEL_PATH_CHORD_LSTM, DEVICE)
    melody_dataset: Melody_Dataset = torch.load(TEST_DATASET_PATH_MELODY, DEVICE)
    melody_agent: Melody_Network = torch.load(MODEL_PATH_MELODY, DEVICE)

    log_likelihood_chord: dict[str, float] = {}
    log_likelihood_melody_pitch: dict[str, float] = {}
    log_likelihood_melody_duration: dict[str, float] = {}

    accuracy_chord: dict[str, int] = {}
    accuracy_melody_pitch: dict[str, int] = {}
    accuracy_melody_duration: dict[str, int] = {}

    for i in range(NUM_EVAL_SAMPLES):
        for j, gamma in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
            if i == 0:
                log_likelihood_chord[str(gamma)] = []
                log_likelihood_melody_pitch[str(gamma)] = []
                log_likelihood_melody_duration[str(gamma)] = []
                accuracy_chord[str(gamma)] = 0
                accuracy_melody_pitch[str(gamma)] = 0
                accuracy_melody_duration[str(gamma)] = 0
            if i % 20 == 0 and gamma == 0:
                print(f"Evaluating sample {i}")

            chord_primer, _, melody_primer = get_primer_sequences()

            log_l, correct = play_chord(chord_network, chord_primer, gamma)
            log_likelihood_chord[str(gamma)].append(log_l)
            accuracy_chord[str(gamma)] += correct

            log_l_p, correct_p, log_l_d, correct_d = play_melody(
                melody_agent, melody_primer, gamma
            )
            log_likelihood_melody_pitch[str(gamma)].append(log_l_p)
            log_likelihood_melody_duration[str(gamma)].append(log_l_d)
            accuracy_melody_pitch[str(gamma)] += correct_p
            accuracy_melody_duration[str(gamma)] += correct_d
        save_temporal_data(
            log_likelihood_chord,
            log_likelihood_melody_pitch,
            log_likelihood_melody_duration,
            accuracy_chord,
            accuracy_melody_pitch,
            accuracy_melody_duration,
        )

    return (
        log_likelihood_chord,
        log_likelihood_melody_pitch,
        log_likelihood_melody_duration,
        accuracy_chord,
        accuracy_melody_pitch,
        accuracy_melody_duration,
    )


def exp3_scrambled_vs_unscrambled():
    use_cache = True
    if not use_cache:
        (
            log_likelihood_chord,
            log_likelihood_melody_pitch,
            log_likelihood_melody_duration,
            accuracy_chord,
            accuracy_melody_pitch,
            accuracy_melody_duration,
        ) = eval_scrambled_vs_unscrambled()

        save_to_json(
            log_likelihood_chord,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_chord.json",
        )
        save_to_json(
            log_likelihood_melody_pitch,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_pitch.json",
        )
        save_to_json(
            log_likelihood_melody_duration,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_duration.json",
        )
        save_to_json(
            accuracy_chord,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_chord.json",
        )
        save_to_json(
            accuracy_melody_pitch,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_pitch.json",
        )
        save_to_json(
            accuracy_melody_duration,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_duration.json",
        )
    else:
        log_likelihood_chord = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_chord.json"
        )
        log_likelihood_melody_pitch = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_pitch.json"
        )
        log_likelihood_melody_duration = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/log_likelihood_melody_duration.json"
        )
        accuracy_chord = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_chord.json"
        )
        accuracy_melody_pitch = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_pitch.json"
        )
        accuracy_melody_duration = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/exp3/accuracy_melody_duration.json"
        )
    log_likelihood_system = {}
    accuracy_system = {}

    log_likelihood_chord = {
        key: np.mean(values) for key, values in log_likelihood_chord.items()
    }
    log_likelihood_melody_pitch = {
        key: np.mean(values) for key, values in log_likelihood_melody_pitch.items()
    }
    log_likelihood_melody_duration = {
        key: np.mean(values) for key, values in log_likelihood_melody_duration.items()
    }

    for key in log_likelihood_chord.keys():
        log_likelihood_system[key] = (
            log_likelihood_chord[key]
            + log_likelihood_melody_pitch[key]
            + log_likelihood_melody_duration[key]
        ) / 3

    for key in accuracy_chord.keys():
        accuracy_chord[key] = accuracy_chord[key] / NUM_EVAL_SAMPLES
        accuracy_melody_pitch[key] = accuracy_melody_pitch[key] / NUM_EVAL_SAMPLES
        accuracy_melody_duration[key] = accuracy_melody_duration[key] / NUM_EVAL_SAMPLES
        accuracy_system[key] = (
            accuracy_chord[key]
            + accuracy_melody_pitch[key]
            + accuracy_melody_duration[key]
        ) / 3

    corrolation(np.array(list(log_likelihood_chord.values())))
    corrolation(np.array(list(log_likelihood_melody_pitch.values())))
    corrolation(np.array(list(log_likelihood_melody_duration.values())))
    corrolation(np.array(list(log_likelihood_system.values())))

    plot_mean_values(
        log_likelihood_chord,
        log_likelihood_melody_pitch,
        log_likelihood_melody_duration,
        log_likelihood_system,
        name="exp3/log_likelihood_scrambled",
    )
    # plot_mean_values(
    #     accuracy_chord,
    #     accuracy_melody_pitch,
    #     accuracy_melody_duration,
    #     accuracy_system,
    #     name="exp3/accuracy_scrambled",
    # )


def play_chord(network, primer, gamma=1.0):
    primer = get_scrambled_primer_chord(primer, gamma)
    correct = 0
    input_sequence, ground_truth = primer[0].unsqueeze(0), primer[1]

    output = network(input_sequence)

    chord_probabilities = F.softmax(output[0, :], dim=-1)

    next_chord_type = torch.multinomial(chord_probabilities, 1).item()

    if next_chord_type == ground_truth.item():
        correct = 1

    # For log likelihood, use the probability of the true class directly
    true_chord_probability = chord_probabilities[ground_truth.item()]
    log_likelihood = torch.log(true_chord_probability).item()
    return log_likelihood, correct


def play_melody(network, primer, gamma=1.0):

    primer = get_scrambled_primer_melody(primer, gamma)

    input_sequence = primer[0]
    ground_truth = primer[1][0]
    correct_pitch = 0
    correct_duration = 0
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
        ).unsqueeze(0)

        # add batch dimension
        accumulated_times = accumulated_times.unsqueeze(0)
        current_chord_time_lefts = current_chord_time_lefts.unsqueeze(0)

        pitch_logits_coop, duration_logits_coop = network(
            x, accumulated_times, current_chord_time_lefts
        )

        pitch_probabilities = F.softmax(pitch_logits_coop, dim=1).view(-1)
        duration_probabilities = F.softmax(duration_logits_coop, dim=1).view(-1)

        predicted_pitch = torch.multinomial(pitch_probabilities, 1).unsqueeze(1)
        predicted_duration = torch.multinomial(duration_probabilities, 1).unsqueeze(1)

        gt_pitch = get_one_hot_index(ground_truth[0])
        gt_duration = get_one_hot_index(ground_truth[1])

        true_pitch_probability = pitch_probabilities[gt_pitch]
        true_duration_probability = duration_probabilities[gt_duration]

        log_likelihood_pitch = torch.log(true_pitch_probability).item()
        log_likelihood_duration = torch.log(true_duration_probability).item()

        if predicted_pitch == gt_pitch:
            correct_pitch = 1
        if predicted_duration == gt_duration:
            correct_duration = 1

        return (
            log_likelihood_pitch,
            correct_pitch,
            log_likelihood_duration,
            correct_duration,
        )


def get_scrambled_primer_chord(chord_primer, gamma):
    primer = chord_primer[0].tolist()
    new_primer = []
    for i in range(8):
        if random.random() < gamma:
            root_note = random.randint(0, 11)
            chord_type = primer[i][1]
            note = [root_note, chord_type, 0, 0]
            new_primer.append(note)
        else:
            new_primer.append(primer[i])
    new_primer = (torch.tensor(new_primer, dtype=torch.float32), chord_primer[1])
    return new_primer


def get_scrambled_primer_melody(melody_primer, gamma):
    new_primer = []
    for event in melody_primer[0]:
        if random.random() < gamma:
            pitch = event[0]
            duration = event[1]
            current_chord = create_oh_vector(random.randint(0, 23), 24)
            next_chord = create_oh_vector(random.randint(0, 23), 24)
            tloc = create_oh_vector(random.randint(0, 15), 16)
            accm_time = create_oh_vector(random.randint(0, 3), 4)

            new_event = [pitch, duration, current_chord, next_chord, tloc, accm_time]
            new_primer.append(new_event)
        else:
            new_primer.append(event)
    new_primer = (new_primer, melody_primer[1])
    return new_primer


def create_oh_vector(index, length):
    return [1 if i == index else 0 for i in range(length)]


def eval_multi_agent_vs_monolithic(
    use_cache=True,
):
    if not use_cache:
        (
            correct_combined_predictions_mas,
            correct_duration_predictions_mas,
            correct_note_predictions_mas,
            correct_chord_predictions_mas,
            chord_log_likelihood_mas,
            duration_log_likelihood_mas,
            root_log_likelihood_mas,
            total_log_likelihood_mas,
        ) = eval_chord_and_bass_separately(verbose=False)

        (
            correct_combined_predictions_mono,
            correct_duration_predictions_mono,
            correct_note_predictions_mono,
            correct_chord_predictions_mono,
            chord_log_likelihood_mono,
            duration_log_likelihood_mono,
            root_log_likelihood_mono,
            total_log_likelihood_mono,
        ) = eval_chord_bass(verbose=False)

        accuracy_mono = {
            "Chord": correct_chord_predictions_mono,
            "Duration": correct_duration_predictions_mono,
            "Root": correct_note_predictions_mono,
            "Combined": correct_combined_predictions_mono,
        }
        accuracy_mas = {
            "Chord": correct_chord_predictions_mas,
            "Duration": correct_duration_predictions_mas,
            "Root": correct_note_predictions_mas,
            "Combined": correct_combined_predictions_mas,
        }
        log_likelihood_mono = {
            "Chord": chord_log_likelihood_mono,
            "Duration": duration_log_likelihood_mono,
            "Root": root_log_likelihood_mono,
            "Combined": total_log_likelihood_mono,
        }
        log_likelihood_mas = {
            "Chord": chord_log_likelihood_mas,
            "Duration": duration_log_likelihood_mas,
            "Root": root_log_likelihood_mas,
            "Combined": total_log_likelihood_mas,
        }
        save_to_json(
            log_likelihood_mono,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/log_likelihood_mono.json",
        )
        save_to_json(
            log_likelihood_mas,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/log_likelihood_mas.json",
        )
        save_to_json(
            accuracy_mono,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/accuracy_mono.json",
        )
        save_to_json(
            accuracy_mas,
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/accuracy_mas.json",
        )
    else:
        log_likelihood_mono = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/log_likelihood_mono.json"
        )
        log_likelihood_mas = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/log_likelihood_mas.json"
        )
        accuracy_mono = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/accuracy_mono.json"
        )
        accuracy_mas = read_from_json(
            "/Users/trymbo/Documents/master/mas_music_generation/data/experiments/accuracy_mas.json"
        )

    box_plot(log_likelihood_mono, log_likelihood_mas, name="exp1/box_plot")
    violin_plot(log_likelihood_mono, log_likelihood_mas)

    print("T-test results Log-likelihood:")
    for key in log_likelihood_mono.keys():
        print(wilcoxon_test(log_likelihood_mono[key], log_likelihood_mas[key]))
    print()
    print("Z-test results Accuracy:")
    for key in accuracy_mono.keys():
        print(
            accuracy_mas[key] / NUM_EVAL_SAMPLES, accuracy_mono[key] / NUM_EVAL_SAMPLES
        )
        print(z_test(accuracy_mono[key], accuracy_mas[key]))


def plot_mean_values(dict1, dict2, dict3, dict4, name=""):
    all_keys = (
        list(dict1.keys())
        + list(dict2.keys())
        + list(dict3.keys())
        + list(dict4.keys())
    )
    all_values = (
        list(dict1.values())
        + list(dict2.values())
        + list(dict3.values())
        + list(dict4.values())
    )

    # Create a DataFrame for plotting
    data = {
        "Gamma": all_keys,
        "Mean Log-Likelihood": all_values,
        "Dict": ["Chord variation"] * len(dict1)
        + ["Pitch"] * len(dict2)
        + ["Duration"] * len(dict3)
        + ["System"] * len(dict4),
    }
    df = pd.DataFrame(data)

    # Define a color palette with consistent colors
    palette = sns.color_palette("tab10", n_colors=3)

    # Plot lines separately for each dictionary
    plt.figure(figsize=(10, 6))

    # Plot lines for "Chord variation", "Pitch", and "Duration"
    sns.lineplot(
        x="Gamma",
        y="Mean Log-Likelihood",
        hue="Dict",
        data=df[df["Dict"] != "System"],
        palette=palette,
    )

    # Plot thicker line for "System"
    sns.lineplot(
        x="Gamma",
        y="Mean Log-Likelihood",
        hue="Dict",
        data=df[df["Dict"] == "System"],
        palette=["red"],
        linewidth=3,
    )

    plt.xlabel(r"$\gamma$", fontsize=18)
    plt.ylabel("Mean Log-Likelihood", fontsize=18)
    plt.title("Mean Log-Likelihood for Scrambled Communication", fontsize=20)
    plt.xticks(rotation=0)
    plt.legend(title="Metric", fontsize=14)
    sns.despine()
    plt.grid(True)
    plt.savefig(
        f"/Users/trymbo/Documents/master/mas_music_generation/figures/{name}.png"
    )


def violin_plot(group1_data, group2_data, name=""):
    for key in group1_data.keys():
        group1_data[key] = winsorize(np.array(group1_data[key]), limits=[0.05, 0.05])
        group2_data[key] = winsorize(np.array(group2_data[key]), limits=[0.05, 0.05])

    # Convert data to DataFrame
    group1_df = pd.DataFrame(group1_data)
    group2_df = pd.DataFrame(group2_data)

    # Add 'Group' column to identify the two groups
    group1_df["Group"] = "Monolitic"
    group2_df["Group"] = "Multi-Agent"

    # Concatenate dataframes
    data = pd.concat([group1_df, group2_df])

    # Melt the data
    melted_data = pd.melt(
        data, id_vars=["Group"], var_name="Metric", value_name="Log-Likelihood"
    )

    # Set palette colors
    palette_colors = sns.color_palette("pastel")

    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Metric",
        y="Log-Likelihood",
        hue="Group",
        data=melted_data,
        palette=palette_colors,
        split=True,
        inner="quartile",
    )

    plt.title("Log-Likelihood: Monolithic vs Multi-Agent", fontsize=20)

    # Optionally set the font size for labels and ticks
    plt.xlabel("Metric", fontsize=18)
    plt.ylabel("Log-Likelihood", fontsize=18)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)

    plt.legend(title="Group", fontsize=14)
    sns.despine()
    plt.savefig(
        f"/Users/trymbo/Documents/master/mas_music_generation/figures/{name}.png"
    )


def box_plot(group1_data, group2_data, name=""):
    shapiro_wilk_test(group1_data)
    shapiro_wilk_test(group2_data)

    for key in group1_data.keys():
        group1_data[key] = winsorize(np.array(group1_data[key]), limits=[0.05, 0.05])
        group2_data[key] = winsorize(np.array(group2_data[key]), limits=[0.05, 0.05])

    # Convert data to DataFrame
    group1_df = pd.DataFrame(group1_data)
    group2_df = pd.DataFrame(group2_data)

    # Add 'Group' column to identify the two groups
    group1_df["Group"] = "Coop"
    group2_df["Group"] = "Non-Coop"

    # Concatenate dataframes
    data = pd.concat([group1_df, group2_df])

    # Melt the data
    melted_data = pd.melt(
        data, id_vars=["Group"], var_name="Metric", value_name="Log-Likelihood"
    )

    # Set palette colors
    palette_colors = sns.color_palette("pastel")

    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Metric",
        y="Log-Likelihood",
        hue="Group",
        data=melted_data,
        palette=palette_colors,
    )

    plt.title("Log-Likelihood: Coop vs Non-Coop", fontsize=20)

    # Optionally set the font size for labels and ticks
    plt.xlabel("Metric", fontsize=18)
    plt.ylabel("Log-Likelihood", fontsize=18)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)

    xtick_labels = [label + "*" for label in melted_data["Metric"].unique()]
    plt.xticks(range(len(xtick_labels)), xtick_labels)

    plt.legend(title="Group", fontsize=14)
    sns.despine()
    plt.savefig(
        f"/Users/trymbo/Documents/master/mas_music_generation/figures/{name}.png"
    )


def read_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def t_test(group1, group2):
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value


def wilcoxon_test(group1, group2):
    t_stat, p_value = stats.wilcoxon(group1, group2)
    return t_stat, p_value


def z_test(group1, group2):
    count = np.array([group1, group2])
    nobs = np.array([NUM_EVAL_SAMPLES, NUM_EVAL_SAMPLES])

    # Perform the Z-test for two proportions
    z_stat, p_value = sm.stats.proportions_ztest(count, nobs)
    return p_value


def shapiro_wilk_test(data):

    for data in data.values():
        sns.kdeplot(data, fill=True)
        plt.xlabel("Data Points")
        plt.ylabel("Density")
        plt.title("Kernel Density Estimation (KDE) Plot of Data Distribution")
        plt.show()
        statistic, p_value = shapiro(data)
        print("Shapiro-Wilk Test Statistic:", statistic)
        print("P-value:", p_value)

        if p_value > 0.05:
            print("The data appear to be normally distributed.")
        else:
            print("The data do not appear to be normally distributed.")

        print()
    print()
    print()


def corrolation(performance_scores):
    parameter_values = np.linspace(0, 1, num=11)

    pearson_corr, pearson_pval = stats.pearsonr(parameter_values, performance_scores)

    spearman_corr, spearman_pval = stats.spearmanr(parameter_values, performance_scores)

    print(f"Pearson correlation coefficient: {pearson_corr}, P-value: {pearson_pval}")
    print(
        f"Spearman correlation coefficient: {spearman_corr}, P-value: {spearman_pval}"
    )
    print()
