import torch

from .bass import Bass_Network
from .chord import Chord_Network

from agents import train_bass, train_chord


def train_agents(
    bass_agent, chord_agent, notes_dataset, chords_dataset, load_bass, load_chord
):
    if load_bass:
        bass_agent.load_state_dict(torch.load("models/bass/bass_model.pth"))
        bass_agent.eval()
    else:
        train_bass(bass_agent, notes_dataset)

    if load_chord:
        chord_agent.load_state_dict(torch.load("models/chords/chord_model.pth"))
        chord_agent.eval()
    else:
        train_chord(chord_agent, chords_dataset)
