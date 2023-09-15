import torch

from .bass import Bass_Network
from .chord import Chord_Network

from agents import train_bass, train_chord


def train_agents(
    bass_agent,
    chord_agent,
    notes_dataset,
    chords_dataset,
    train_bass_agent,
    train_chord_agent,
):
    if not train_bass_agent:
        bass_agent.load_state_dict(torch.load("models/bass/bass_model.pth"))
        bass_agent.eval()
    else:
        train_bass(bass_agent, notes_dataset)

    if not train_chord_agent:
        chord_agent.load_state_dict(torch.load("models/chords/chord_model.pth"))
        chord_agent.eval()
    else:
        train_chord(chord_agent, chords_dataset)
