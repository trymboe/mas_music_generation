from .bass import play_bass
from .chord import play_chord
import random
import torch
import note_seq as ns

from agents import predict_next_k_notes_bass, predict_next_k_notes_chords

from utils import get_full_bass_sequence

from .utils import TxlSimpleSampler, tokens_to_note_sequence

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


def generate_drum(drum_agent, drum_dataset, device):
    import random

    # Generate a list of 119 random numbers between 1 and 24
    random_numbers = [random.randint(1, 24) for _ in range(119)]

    # Add 0 as the first number
    random_numbers.insert(0, 0)
    seqs = [random_numbers]

    pitch_classes: list[list[int]] = DRUM_MAPPING["DEFAULT_DRUM_TYPE_PITCHES"]
    time_vocab = TIME_STEPS_VOCAB
    pitch_vocab = drum_dataset.reverse_vocab

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

    # seqs = generate_sequences_drum(
    #     model=drum_agent,
    #     num=1,
    #     gen_len=120,
    #     mem_len=MEM_LEN,
    #     device=device,
    #     temp=0.95,
    #     topk=5,
    # )
    print(seqs)

    for i, s in enumerate(seqs):
        note_sequence = tokens_to_note_sequence(
            s[1:],
            pitch_vocab,
            simplified_pitches,
            drum_dataset.vel_vocab,
            time_vocab,
            143.99988480009216,
        )

        note_sequence_to_midi_file(
            note_sequence, f"sound_examples/experiments/seperate_velocities_l_{i}.midi"
        )


def note_sequence_to_midi_file(note_sequence, path):
    """
    Save <note_sequence> to .midi file at <path>
    """
    ns.sequence_proto_to_midi_file(note_sequence, path)


def generate_sequences_drum(model, num, gen_len, mem_len, device, temp, topk=None):
    """
    Generate samples of len <gen_len> using pretrained transformer <model>

    Param
    =====
    model:
        Trained transformer model
    num: int
        Number of sequences to generate
    gen_len: int
        How many tokens to generate
    mem_len: int
        memory length of model
    device: torch device
        cpu or gpu
    temp: float
        Between 0 and 1.
        1 samples from model prob dist
        0 always takes most likely
    topk: n
        k for topk sampling

    Return
    ======
    Accompanying tokenised sequence (needs to be joined to original using join_sequences())
    """
    all_seqs = []
    # Generate sequences of specified length and number
    for i in range(num):
        sampler = TxlSimpleSampler(model, device, mem_len=mem_len)
        seq = [0]
        for _ in range(gen_len):
            token, _ = sampler.sample_next_token_updating_mem(
                seq[-1], temp=temp, topk=topk
            )
            seq.append(token)
        all_seqs.append(seq)

    return all_seqs


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
