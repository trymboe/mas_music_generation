import torch

from .bass import Bass_Network
from .chord import Chord_Network
from .drum import Drum_Network

from agents import train_bass, train_chord, train_drum

from config import MODEL_PATH_BASS, MODEL_PATH_CHORD


def train_agents(
    bass_agent,
    bass_dataset,
    train_bass_agent,
    chord_agent,
    chord_dataset,
    train_chord_agent,
    drum_agent,
    drum_dataset,
    train_drum_agent,
    device,
):
    bass_agent.to(device)
    chord_agent.to(device)
    drum_agent.to(device)

    if not train_bass_agent:
        bass_agent = torch.load(MODEL_PATH_BASS)
        bass_agent.eval()
        bass_agent.to(device)
    else:
        train_bass(bass_agent, bass_dataset)

    if not train_chord_agent:
        chord_agent = torch.load(MODEL_PATH_CHORD)
        chord_agent.eval()
        chord_agent.to(device)
    else:
        train_chord(chord_agent, chord_dataset)

    # if not train_drum_agent:
    #     drum_agent.load_state_dict(torch.load("models/drum/drum_model.pt"))
    #     drum_agent.eval()
    #     drum_agent.to(device)
    # else:
    #     train_drum(drum_agent, drum_dataset, device)
