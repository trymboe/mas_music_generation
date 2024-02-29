from .bass import Bass_Network
from .chord import (
    Chord_Network,
    Chord_LSTM_Network,
    Chord_Network_Non_Coop,
)
from .drum import Drum_Network
from .melody import Melody_Network

from data_processing import (
    Bass_Dataset,
    Chord_Dataset,
    Drum_Dataset,
    Melody_Dataset,
    get_drum_dataset,
)

import torch

from data_processing.utils import load_yaml

from .drum import drum_network_pipeline

from agents import train_bass, train_chord, train_melody

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
    DEVICE,
    MODEL_PATH_MELODY,
    PITCH_SIZE_MELODY,
    DURATION_SIZE_MELODY,
    CHORD_SIZE_MELODY,
    TOTAL_INPUT_SIZE_MELODY,
    PITCH_VECTOR_SIZE,
)


def create_agents(
    train_bass_agent: bool,
    train_chord_agent: bool,
    train_chord_non_coop_agent: bool,
    train_drum_agent: bool,
    train_melody_agent: bool,
    train_melody_non_coop_agent: bool,
) -> None:
    """
    Creates and initializes the agents used for music generation.

    Args:
        train_bass_agent (bool): Whether to train the bass agent or load a pre-trained one.
        train_chord_agent (bool): Whether to train the chord agent or load a pre-trained one.
        train_chord_non_coop_agent (bool): Whether to train the chord non cooperation agent or load a pre-trained one.
        train_drum_agent (bool): Whether to train the drum agent or load a pre-trained one.
        train_melody_agent (bool): Whether to train the melody agent or load a pre-trained one.
        train_melody_non_coop_agent (bool): Whether to train the melody non cooperation agent or load a pre-trained one.

    Returns:
        None
    """

    print("----Creating agents----")

    if train_bass_agent:
        print("  ----Creating bass agent----")
        bass_agent: Bass_Network = create_bass_agent()
        bass_agent.to(DEVICE)
        train_bass(bass_agent)
        bass_agent.eval()

    # --- Creating chord agent ---
    if train_chord_agent or train_chord_non_coop_agent:
        print("  ----Creating chord agent----")
        chord_agent: Chord_Network = create_chord_agent(
            train_chord_agent, train_chord_non_coop_agent
        )
        chord_agent.to(DEVICE)
        train_chord(chord_agent)
        chord_agent.eval()

    # --- Creating drum agent ---
    if train_drum_agent:
        print("  ----Creating drum agent----")
        drum_agent: Drum_Network = create_drum_agent()
        drum_agent.eval()
        drum_agent.to(DEVICE)

    # --- Creating melody agent ---
    if train_melody_agent or train_melody_non_coop_agent:
        print("  ----Creating melody agent----")
        melody_agent: Melody_Network = create_melody_agent(
            train_melody_agent, train_melody_non_coop_agent
        )
        melody_agent.to(DEVICE)
        train_melody(melody_agent)
        melody_agent.eval()


def create_drum_agent():
    """
    Creates a drum agent by loading the drum dataset, loading the configuration parameters,
    and building the drum network pipeline.

    Returns:
    ----------
        The drum agent model.
    """
    drum_dataset = get_drum_dataset()
    conf = load_yaml("config/bumblebeat/params.yaml")
    model = drum_network_pipeline(conf, drum_dataset)

    return model


def create_melody_agent(
    train_melody_agent: bool, train_melody_non_coop_agent: bool
) -> Melody_Network:

    melody_agent: Melody_Network = Melody_Network()
    return melody_agent


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


def create_chord_agent(
    train_chord_agent: bool, train_chord_non_coop_agent: bool, LSTM: bool = False
) -> Chord_Network:
    """
    Creates a chord agent based on the specified parameters.

    Args:
        train_chord_agent (bool): Whether to train the chord agent.
        train_chord_non_coop_agent (bool): Whether to train the non-cooperative chord agent.
        LSTM (bool, optional): Whether to use LSTM network architecture. Defaults to False.

    Returns:
        Chord_Network: The created chord agent.
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
        if train_chord_agent:
            chord_network: Chord_Network = Chord_Network(
                ROOT_VOAB_SIZE_CHORD,
                CHORD_VOCAB_SIZE_CHORD,
                EMBED_SIZE_CHORD,
                NHEAD_CHORD,
                NUM_LAYERS_CHORD,
            )
        if train_chord_non_coop_agent:
            chord_network: Chord_Network = Chord_Network_Non_Coop(
                ROOT_VOAB_SIZE_CHORD,
                CHORD_VOCAB_SIZE_CHORD,
                EMBED_SIZE_CHORD,
                NHEAD_CHORD,
                NUM_LAYERS_CHORD,
            )

    return chord_network
