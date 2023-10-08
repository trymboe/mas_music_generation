from .bass import Bass_Network
from .chord import Chord_Network, Chord_LSTM_Network
from .drum import create_exp_dir, Drum_Network, weights_init

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
    WORK_DIR,
    TRAIN_BATCH_SIZE_DRUM,
    NUM_TOKENS_PREDICT_DRUM,
    EXTENDED_CONTEXT_LENGTH_DRUM,
    EVAL_BATCH_SIZE_DRUM,
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
)


def create_agents(drum_dataset, device):
    bass_agent = create_bass_agent()
    chord_agent = create_chord_agent()
    drum_agent = create_drum_agent(drum_dataset, device)

    print(drum_agent)
    exit()

    return bass_agent, chord_agent, drum_agent


def create_drum_agent(drum_dataset, device):
    logging = create_exp_dir(WORK_DIR, scripts_to_save=None, debug=WORK_DIR)

    train_iter = drum_dataset.get_iterator(
        "train",
        TRAIN_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )
    val_iter = drum_dataset.get_iterator(
        "valid",
        EVAL_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )
    test_iter = drum_dataset.get_iterator(
        "test",
        EVAL_BATCH_SIZE_DRUM,
        NUM_TOKENS_PREDICT_DRUM,
        device=device,
        ext_len=EXTENDED_CONTEXT_LENGTH_DRUM,
    )

    drum_agent = Drum_Network(
        NUM_TOKENS_PREDICT_DRUM,
        NUM_LAYERS_DRUM,
        NHEAD_DRUM,
        D_MODEL_DRUM,
        D_HEAD_DRUM,
        D_INNER_DRUM,
        DROPOUT_DRUM,
        DROPATT_DRUM,
        NOT_TIED_DRUM,
        D_MODEL_DRUM,
        DIV_VAL_DRUM,
        [False],
        PRE_LNORM,
        NUM_TOKENS_PREDICT_DRUM,
        EXTENDED_CONTEXT_LENGTH_DRUM,
        MEM_LEN,
        [],
        SAME_LENGTH,
        ATTN_TYPE,
        CLAMP_LEN,
        SAMPLE_SOFTMAX,
    )

    drum_agent.apply(weights_init)
    drum_agent.word_emb.apply(
        weights_init
    )  # ensure embedding init is not overridden by out_layer in case of weight sharing

    N_ALL_PARAMS = sum([p.nelement() for p in drum_agent.parameters()])
    N_NONEMB_PARAMS = sum([p.nelement() for p in drum_agent.layers.parameters()])

    return drum_agent


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
