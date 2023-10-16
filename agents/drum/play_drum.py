from bumblebeat.bumblebeat.output.generate import tokens_to_note_sequence

from bumblebeat.bumblebeat.utils.data import load_yaml
from bumblebeat.bumblebeat.output.generate import (
    load_model,
    generate_sequences,
    note_sequence_to_midi_file,
    continue_sequence,
)
from bumblebeat.bumblebeat.data import get_corpus

import random

import note_seq as ns

from config import MEM_LEN


def play_drum(device):
    conf = load_yaml("bumblebeat/conf/train_conf.yaml")

    pitch_classes = load_yaml("bumblebeat/conf/drum_pitches.yaml")
    time_vocab = load_yaml("bumblebeat/conf/time_steps_vocab.yaml")

    model_conf = conf["model"]
    data_conf = conf["data"]

    # path = "models/drum/drum_model.pt"
    path = "models/drum/train_step_5000/model.pt"

    model = load_model(path, device)

    corpus = get_corpus(
        data_conf["dataset"],
        data_conf["data_dir"],
        pitch_classes["DEFAULT_DRUM_TYPE_PITCHES"],
        time_vocab,
        conf["processing"],
    )

    pitch_vocab = corpus.reverse_vocab
    velocity_vocab = {v: k for k, v in corpus.vel_vocab.items()}

    USE_CUDA = False
    mem_len = model_conf["mem_len"]
    gen_len = 2000
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

    seqs = generate_sequences(
        model,
        num=1,
        gen_len=gen_len,
        mem_len=mem_len,
        device=device,
        temp=1,
        topk=5,
    )
    for i, s in enumerate(seqs):
        note_sequence = tokens_to_note_sequence(
            s[1:],
            pitch_vocab,
            simplified_pitches,
            velocity_vocab,
            time_vocab,
            120,
        )

    pm = ns.note_sequence_to_pretty_midi(note_sequence)
    note_sequence_to_midi_file(note_sequence, f"results/drum/quantisize={i}.midi")
    return pm

    """

    random_sequence = random.choice(
        [x for x in corpus.train_data if x["style"]["primary"] == 7]
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
            120,
        )

    out_tokens = continue_sequence(
        model,
        seq=in_tokens[-1000:],
        prime_len=512,
        gen_len=gen_len,
        mem_len=mem_len,
        device=device,
        temp=0.95,
        topk=None,
    )

    note_sequence = tokens_to_note_sequence(
        out_tokens,
        pitch_vocab,
        simplified_pitches,
        velocity_vocab,
        time_vocab,
        120,
    )
    """
