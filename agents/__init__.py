from .bass import (
    Bass_Network,
    train_bass,
    predict_next_k_notes_bass,
    play_bass,
)

from .chord import (
    Chord_Network,
    Chord_LSTM_Network,
    train_chord,
    play_chord,
    predict_next_k_notes_chords,
)

from .melody import train_melody, Melody_Network

from .drum import train_drum

from .coplay import play_agents

from .create_agents import create_agents

from .train_agents import train_agents
