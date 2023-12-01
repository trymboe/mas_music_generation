import torch

NUMBER_OF_NOTES_FOR_TRAINING = 0  # 0 for all notes
LENGTH = 24  # Number of measures to be generated
LENGTH_BARS = LENGTH * 4
TEMPO = 120

SEED = 42  # Random seed

SAVE_RESULT_PATH = "results/drum_bass_chord.mid"


# DEVICE = torch.device("mps")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
