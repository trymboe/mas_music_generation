from .bass import Bass_Network
from .chord import Chord_Network, Chord_LSTM_Network
from .drum import Drum_Network, weights_init
from .train_agents import train_agents

import torch

from bumblebeat.bumblebeat.model import model_main
from bumblebeat.bumblebeat.utils.data import load_yaml

from agents import train_bass, train_chord, train_drum

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
    MODEL_PATH_CHORD,
    MODEL_PATH_BASS,
)


def create_agents(
    bass_dataset,
    chord_dataset,
    drum_dataset,
    train_bass_agent,
    train_chord_agent,
    train_drum_agent,
    device,
):
    if not train_bass_agent:
        bass_agent = torch.load(MODEL_PATH_BASS)
        bass_agent.eval()
        bass_agent.to(device)
    else:
        bass_agent = create_bass_agent()
        train_bass(bass_agent, bass_dataset)

    if not train_chord_agent:
        chord_agent = torch.load(MODEL_PATH_CHORD)
        chord_agent.eval()
        chord_agent.to(device)
    else:
        chord_agent = create_chord_agent()
        train_chord(chord_agent, chord_dataset)

    drum_agent = create_drum_agent(drum_dataset, device, train_drum_agent)

    return bass_agent, chord_agent, drum_agent


def create_drum_agent(drum_dataset, device, train_drum_agent):
    conf = load_yaml("bumblebeat/conf/train_conf.yaml")

    pitch_classes_yaml = load_yaml("bumblebeat/conf/drum_pitches.yaml")
    pitch_classes = pitch_classes_yaml["DEFAULT_DRUM_TYPE_PITCHES"]
    time_steps_vocab = load_yaml("bumblebeat/conf/time_steps_vocab.yaml")
    if train_drum_agent:
        model = model_main(conf, pitch_classes, time_steps_vocab, device, drum_dataset)
    else:
        model = torch.load(WORK_DIR + "/drum_model.pt")
    return model


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
    # chord_network = Chord_LSTM_Network(
    #     ROOT_VOAB_SIZE_CHORD,
    #     CHORD_VOCAB_SIZE_CHORD,
    #     EMBED_SIZE_CHORD,
    #     HIDDEN_SIZE_CHORD,
    #     NUM_LAYERS_CHORD,
    # )

    chord_network = Chord_Network(
        ROOT_VOAB_SIZE_CHORD,
        CHORD_VOCAB_SIZE_CHORD,
        EMBED_SIZE_CHORD,
        NHEAD_CHORD,
        NUM_LAYERS_CHORD,
    )

    return chord_network


""
