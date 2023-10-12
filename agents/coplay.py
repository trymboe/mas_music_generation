from .bass import play_bass
from .chord import play_chord
import random
import torch
import note_seq as ns

from agents import predict_next_k_notes_bass, predict_next_k_notes_chords
from utils import get_full_bass_sequence

from .utils import TxlSimpleSampler, tokens_to_note_sequence

from bumblebeat.bumblebeat.utils.data import load_yaml
from bumblebeat.bumblebeat.output.generate import (
    load_model,
    generate_sequences,
    note_sequence_to_midi_file,
    continue_sequence,
)
from bumblebeat.bumblebeat.data import get_corpus


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
    print(drum_dataset.vocab_size)
    # Generate Drums
    drum_token_seq = generate_drum(drum_agent, drum_dataset, device)

    exit()

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

    mid = play_bass(full_bass_sequence)

    mid = play_chord(mid, timed_chord_sequence, arpeggiate)

    mid.save(filename)


def generate_drum(drum_agent, corpus, device):
    conf = load_yaml("bumblebeat/conf/train_conf.yaml")

    pitch_classes = load_yaml("bumblebeat/conf/drum_pitches.yaml")
    time_vocab = load_yaml("bumblebeat/conf/time_steps_vocab.yaml")

    model_conf = conf["model"]
    data_conf = conf["data"]

    pitch_vocab = corpus.reverse_vocab
    velocity_vocab = {v: k for k, v in corpus.vel_vocab.items()}

    path = "models/drum/drum_model.pt"
    model = load_model(path, device)

    corpus = get_corpus(
        data_conf["dataset"],
        data_conf["data_dir"],
        pitch_classes["DEFAULT_DRUM_TYPE_PITCHES"],
        time_vocab,
        conf["processing"],
    )

    USE_CUDA = False
    mem_len = MEM_LEN
    gen_len = 120
    same_len = True

    hat_prime = [
        95,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        42,
    ]
    simplified_pitches = [[36], [38], [42], [46], [45], [48], [50], [49], [51]]

    print(corpus.train)
    exit()

    random_sequence = random.choice(
        [x for x in corpus.train if x["style"]["primary"] == 7]
    )

    for i in [4, 8, 16]:
        if i:
            # To midi note sequence using magent
            dev_sequence = corpus._quantize(
                ns.midi_to_note_sequence(random_sequence["midi"]), i
            )
            quantize = True
        else:
            dev_sequence = ns.midi_to_note_sequence(random_sequence["midi"])
            quantize = False

        # note sequence -> [(pitch, vel_bucket, start timestep)]
        in_tokens = corpus._tokenize(dev_sequence, i, quantize)
        note_sequence = tokens_to_note_sequence(
            in_tokens,
            pitch_vocab,
            simplified_pitches,
            velocity_vocab,
            time_vocab,
            random_sequence["bpm"],
        )
        note_sequence_to_midi_file(
            note_sequence, f"sound_examples/experiments/original_quantize={i}.midi"
        )

    out_tokens = continue_sequence(
        model,
        seq=in_tokens[-1000:],
        prime_len=512,
        gen_len=gen_len,
        mem_len=mem_len,
        device=device,
        temp=0.95,
        topk=32,
    )

    note_sequence = tokens_to_note_sequence(
        out_tokens,
        pitch_vocab,
        simplified_pitches,
        4,
        time_vocab,
        random_sequence["bpm"],
    )
    note_sequence_to_midi_file(
        note_sequence, f"sound_examples/experiments/continued.midi"
    )


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
