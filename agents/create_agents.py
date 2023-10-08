from .bass import Bass_Network
from .chord import Chord_Network, Chord_LSTM_Network
from .drum import create_exp_dir

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
)


def create_agents(drum_dataset, device):
    bass_agent = create_bass_agent()
    chord_agent = create_chord_agent()
    drum_agent = create_drum_agent(drum_dataset, device)

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
