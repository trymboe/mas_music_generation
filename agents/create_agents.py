from .bass import Bass_Network
from .chord import Chord_Network, Chord_LSTM_Network
from .drum import Drum_Network, weights_init

import torch

from bumblebeat.bumblebeat.model import model_main
from bumblebeat.bumblebeat.utils.data import load_yaml

from config import (
    NOTE_VOCAB_SIZE_BASS,
    DURATION_VOCAB_SIZE_BASS,
    EMBED_SIZE_BASS,
    NHEAD_BASS,
    NUM_LAYERS_BASS,
    CHORD_VOCAB_SIZE_CHORD,
    ROOT_VOAB_SIZE_CHORD,
    EMBED_SIZE_CHORD,
    NHEAD_CHORD,
    NUM_LAYERS_CHORD,
    HIDDEN_SIZE_CHORD,
    NUM_TOKENS_PREDICT_DRUM,
    EXTENDED_CONTEXT_LENGTH_DRUM,
    NUM_LAYERS_DRUM,
    NHEAD_DRUM,
    D_MODEL_DRUM,
    D_HEAD_DRUM,
    D_INNER_DRUM,
    DROPOUT_DRUM,
    DROPATT_DRUM,
    SAME_LENGTH,
    ATTN_TYPE,
    CLAMP_LEN,
    SAMPLE_SOFTMAX,
    MEM_LEN,
    PRE_LNORM,
    NOT_TIED_DRUM,
    DIV_VAL_DRUM,
    N_ALL_PARAMS,
    N_NONEMB_PARAMS,
    WORK_DIR,
)


def create_agents(drum_dataset, device):
    bass_agent = create_bass_agent()
    chord_agent = create_chord_agent()
    drum_agent = create_drum_agent(drum_dataset, device)

    return bass_agent, chord_agent, drum_agent


def create_drum_agent(drum_dataset, device):
    conf = load_yaml("bumblebeat/conf/train_conf.yaml")

    pitch_classes_yaml = load_yaml("bumblebeat/conf/drum_pitches.yaml")
    pitch_classes = pitch_classes_yaml["DEFAULT_DRUM_TYPE_PITCHES"]
    time_steps_vocab = load_yaml("bumblebeat/conf/time_steps_vocab.yaml")
    model = torch.load(WORK_DIR + "/drum_model.pt")
    return model
    model = model_main(conf, pitch_classes, time_steps_vocab, device)
    return model


"""def create_drum_agent():
    drum_agent = Drum_Network(
        NUM_TOKENS_PREDICT_DRUM,
        NUM_LAYERS_DRUM,
        NHEAD_DRUM,
        D_MODEL_DRUM,
        D_HEAD_DRUM,
        D_INNER_DRUM,
        DROPOUT_DRUM,
        DROPATT_DRUM,
        tie_weight=NOT_TIED_DRUM,
        d_embed=D_MODEL_DRUM,
        div_val=DIV_VAL_DRUM,
        tie_projs=[False],
        pre_lnorm=PRE_LNORM,
        tgt_len=NUM_TOKENS_PREDICT_DRUM,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
        mem_len=MEM_LEN,
        cutoffs=[],
        same_length=SAME_LENGTH,
        attn_type=ATTN_TYPE,
        clamp_len=CLAMP_LEN,
        sample_softmax=SAMPLE_SOFTMAX,
    )

    drum_agent.apply(weights_init)
    drum_agent.word_emb.apply(
        weights_init
    )  # ensure embedding init is not overridden by out_layer in case of weight sharing

    print("Drum agent created with the following number of parameters:")
    N_ALL_PARAMS = sum([p.nelement() for p in drum_agent.parameters()])
    print("Total number of parameters: ", N_ALL_PARAMS)
    N_NONEMB_PARAMS = sum([p.nelement() for p in drum_agent.layers.parameters()])
    print("Number of non-embedding parameters: ", N_NONEMB_PARAMS, end="\n\n")

    return drum_agent"""


def create_bass_agent():
    bass_agent = Bass_Network(
        NOTE_VOCAB_SIZE_BASS,
        DURATION_VOCAB_SIZE_BASS,
        EMBED_SIZE_BASS,
        NHEAD_BASS,
        NUM_LAYERS_BASS,
    )
    return bass_agent


def create_chord_agent():
    chord_network = Chord_LSTM_Network(
        ROOT_VOAB_SIZE_CHORD,
        CHORD_VOCAB_SIZE_CHORD,
        EMBED_SIZE_CHORD,
        HIDDEN_SIZE_CHORD,
        NUM_LAYERS_CHORD,
    )

    chord_network = Chord_Network(
        ROOT_VOAB_SIZE_CHORD,
        CHORD_VOCAB_SIZE_CHORD,
        EMBED_SIZE_CHORD,
        NHEAD_CHORD,
        NUM_LAYERS_CHORD,
    )

    return chord_network


""
