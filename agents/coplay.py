from .bass import play_bass
from .chord import play_chord
from .drum import play_drum
import random
import torch

import pretty_midi

from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset


from agents import predict_next_k_notes_bass, predict_next_k_notes_chords

from config import (
    INT_TO_TRIAD,
    LENGTH,
    LOOP_MEASURES,
    STYLE,
    MODEL_PATH_BASS,
    MODEL_PATH_CHORD,
)


def play_agents(
    bass_dataset: Bass_Dataset,
    chord_dataset: Chord_Dataset,
    drum_dataset: Drum_Dataset,
    arpeggiate: bool,
    filename: str,
    device: torch.device,
) -> None:
    print("----playing agents----")
    dataset_primer_start: int = random.randint(0, len(bass_dataset) - 1)

    print("  ----playing drum----")
    mid: pretty_midi.PrettyMIDI = play_drum(
        device,
        measures=LOOP_MEASURES,
        loops=int(LENGTH / LOOP_MEASURES),
        drum_dataset=drum_dataset,
        style=STYLE,
    )

    print("  ----playing bass----")
    mid, predicted_bass_sequence = play_bass(
        mid, bass_dataset, dataset_primer_start, device, playstyle="bass_drum"
    )

    print("  ----playing chord----")
    mid = play_chord(
        mid,
        arpeggiate,
        predicted_bass_sequence,
        chord_dataset,
        dataset_primer_start,
        device,
    )

    mid.write(filename)
