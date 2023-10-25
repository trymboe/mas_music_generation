from .bass import Bass_Network
from .chord import Chord_Network, Chord_LSTM_Network
from .drum import Drum_Network

from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset

import torch

from data_processing.utils import load_yaml

from .drum import drum_network_pipeline

from agents import train_bass, train_chord

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
    MODEL_PATH_DRUM,
)


def create_agents(
    bass_dataset: Bass_Dataset,
    chord_dataset: Chord_Dataset,
    drum_dataset: Drum_Dataset,
    train_bass_agent: bool,
    train_chord_agent: bool,
    train_drum_agent: bool,
    device: torch.device,
) -> tuple[Bass_Network, Chord_Network, Drum_Network]:
    """
    Creates and optionally trains the bass, chord, and drum agents.

    This function handles the creation and training of the bass, chord, and drum agents based on the provided datasets
    and training flags. If a training flag for an agent is set to False, the function attempts to load a pre-trained
    model from the disk. If the training flag is set to True, it creates and trains a new agent.

    Parameters
    ----------
    bass_dataset : Bass_Dataset
        The dataset to be used for training the bass agent.
    chord_dataset : Chord_Dataset
        The dataset to be used for training the chord agent.
    drum_dataset : Drum_Dataset
        The dataset to be used for training the drum agent.
    train_bass_agent : bool
        Flag indicating whether to train the bass agent.
    train_chord_agent : bool
        Flag indicating whether to train the chord agent.
    train_drum_agent : bool
        Flag indicating whether to train the drum agent.
    device : torch.device
        The device to which the agents should be moved. This could be a CPU or a GPU.

    Returns
    -------
    tuple[Bass_Network, Chord_Network, Drum_Network]
        A tuple containing the bass, chord, and drum agents.
    """

    print("----Creating agents----")
    # Creating bass agent
    if not train_bass_agent:
        print("  ----Loading bass agent----")
        bass_agent: Bass_Network = torch.load(MODEL_PATH_BASS, device)
        bass_agent.eval()
        bass_agent.to(device)
    else:
        print("  ----Training bass agent----")
        bass_agent: Bass_Network = create_bass_agent()
        train_bass(bass_agent, bass_dataset)
        bass_agent.eval()
        bass_agent.to(device)

    # Creating chord agent
    if not train_chord_agent:
        print("  ----Loading chord agent----")
        chord_agent: Chord_Network = torch.load(MODEL_PATH_CHORD, device)
        chord_agent.eval()
        chord_agent.to(device)
    else:
        print("  ----Loading chord agent----")
        chord_agent: Chord_Network = create_chord_agent()
        train_chord(chord_agent, chord_dataset)
        chord_agent.eval()
        chord_agent.to(device)

    # Creating drum agent
    if not train_drum_agent:
        print("  ----Loading drum agent----")
        drum_agent: Drum_Network = torch.load(MODEL_PATH_DRUM, device)
        drum_agent.eval()
        drum_agent.to(device)
    else:
        print("  ----Loading drum agent----")
        drum_agent: Drum_Network = create_drum_agent(
            drum_dataset,
            device,
        )
        drum_agent.eval()
        drum_agent.to(device)

    return bass_agent, chord_agent, drum_agent


def create_drum_agent(drum_dataset, device):
    conf = load_yaml("config/bumblebeat/params.yaml")
    model = drum_network_pipeline(conf, drum_dataset, device)

    return model


def create_bass_agent() -> Bass_Network:
    """
    Creates and returns an instance of the Bass_Network.

    Returns
    -------
    Bass_Network
        The initialized bass agent.
    """

    bass_agent: Bass_Network = Bass_Network(
        NOTE_VOCAB_SIZE_BASS,
        DURATION_VOCAB_SIZE_BASS,
        EMBED_SIZE_BASS,
        NHEAD_BASS,
        NUM_LAYERS_BASS,
    )
    return bass_agent


def create_chord_agent(LSTM: bool = False) -> Chord_Network:
    """
    Creates and returns an instance of the Chord_Network.

    Returns
    -------
    Chord_Network
        The initialized chord agent.
    """

    if LSTM:
        chord_network = Chord_LSTM_Network(
            ROOT_VOAB_SIZE_CHORD,
            CHORD_VOCAB_SIZE_CHORD,
            EMBED_SIZE_CHORD,
            HIDDEN_SIZE_CHORD,
            NUM_LAYERS_CHORD,
        )
    else:
        chord_network: Chord_Network = Chord_Network(
            ROOT_VOAB_SIZE_CHORD,
            CHORD_VOCAB_SIZE_CHORD,
            EMBED_SIZE_CHORD,
            NHEAD_CHORD,
            NUM_LAYERS_CHORD,
        )

    return chord_network
