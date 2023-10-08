import torch

from .bass import Bass_Network
from .chord import Chord_Network
from .drum import Drum_Network

from agents import train_bass, train_chord, train_drum


def train_agents(bass_agent_tripple, chord_agent_tripple, drum_agent_tripple, device):
    bass_agent, notes_dataset, train_bass_agent = (
        bass_agent_tripple[0],
        bass_agent_tripple[1],
        bass_agent_tripple[2],
    )
    chord_agent, chords_dataset, train_chord_agent = (
        chord_agent_tripple[0],
        chord_agent_tripple[1],
        chord_agent_tripple[2],
    )
    drum_agent, drum_dataset, train_drum_agent = (
        drum_agent_tripple[0],
        drum_agent_tripple[1],
        drum_agent_tripple[2],
    )

    if not train_bass_agent:
        bass_agent.load_state_dict(torch.load("models/bass/bass_model.pth"))
        bass_agent.eval()
        bass_agent.to(device)
    else:
        train_bass(bass_agent, notes_dataset)

    if not train_chord_agent:
        chord_agent.load_state_dict(torch.load("models/chord/chord_model.pth"))
        chord_agent.eval()
        bass_agent.to(device)
    else:
        train_chord(chord_agent, chords_dataset)

    if not train_drum_agent:
        drum_agent.load_state_dict(torch.load("models/drum/drum_model.pth"))
        drum_agent.eval()
        bass_agent.to(device)
    else:
        train_drum(drum_agent, drum_dataset, device)

    exit()
