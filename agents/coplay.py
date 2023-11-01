from .bass import play_bass
from .chord import play_chord
from .drum import play_drum
import random
import time

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
    DEVICE,
)


def play_agents(
    bass_dataset: Bass_Dataset,
    chord_dataset: Chord_Dataset,
    drum_dataset: Drum_Dataset,
    arpeggiate: bool,
    filename: str,
) -> None:
    """
    Orchestrates the playing of bass, chord, and drum agents to generate a music piece.

    This function generates a music piece by playing the bass, chord, and drum agents sequentially. The generated music
    is then written to a MIDI file. The function also handles the random selection of a primer from the dataset to
    start the generation process.

    Parameters
    ----------
    bass_dataset : Bass_Dataset
        The bass dataset used for generating bass sequences.
    chord_dataset : Chord_Dataset
        The chord dataset used for generating chord sequences.
    drum_dataset : Drum_Dataset
        The drum dataset used for generating drum patterns.
    arpeggiate : bool
        Flag indicating whether to arpeggiate the chord sequences.
    filename : str
        The name of the file where the generated MIDI music will be saved.

    Returns
    -------
    None
    """

    print("----playing agents----")
    dataset_primer_start: int = random.randint(0, len(bass_dataset) - 1)

    print("  ----playing drum----")
    start = time.time()
    mid: pretty_midi.PrettyMIDI = play_drum(
        measures=LOOP_MEASURES,
        loops=int(LENGTH / LOOP_MEASURES),
        drum_dataset=drum_dataset,
        style=STYLE,
    )
    end = time.time()
    print("    ----drum playing time: ", end - start)

    print("  ----playing bass----")
    start = time.time()
    mid, predicted_bass_sequence = play_bass(
        mid, bass_dataset, dataset_primer_start, playstyle="bass_drum"
    )
    end = time.time()
    print("    ----bass playing time: ", end - start)

    print("  ----playing chord----")
    start = time.time()
    mid = play_chord(
        mid,
        arpeggiate,
        predicted_bass_sequence,
        chord_dataset,
        dataset_primer_start,
    )
    end = time.time()
    print("    ----chord playing time: ", end - start)

    mid.write(filename)
