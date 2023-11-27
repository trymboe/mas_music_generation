from .bass import play_bass
from .chord import play_chord
from .drum import play_drum
from .melody import play_melody
import random
import time

import pretty_midi
import torch

from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset, Melody_Dataset


from agents import (
    predict_next_k_notes_bass,
    predict_next_k_notes_chords,
    generate_scale_preferences,
    select_with_preference,
)

from config import (
    INT_TO_TRIAD,
    LENGTH,
    LOOP_MEASURES,
    STYLE,
    SEQUENCE_LENGTH_CHORD,
    MODEL_PATH_BASS,
    MODEL_PATH_CHORD,
    MODEL_PATH_MELODY,
    DEVICE,
    DATASET_SIZE_MELODY,
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

    chord_primer, bass_primer, melody_primer = get_primer_sequences()

    print("----playing agents----")

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
        mid, bass_dataset, bass_primer, playstyle="bass_drum"
    )
    end = time.time()
    print("    ----bass playing time: ", end - start)

    print("  ----playing chord----")
    start = time.time()
    mid, chord_sequence = play_chord(
        mid, arpeggiate, predicted_bass_sequence, chord_primer
    )
    end = time.time()
    print("    ----chord playing time: ", end - start)

    print("  ----playing melody----")
    start = time.time()
    mid = play_melody(mid, chord_sequence, melody_primer)
    end = time.time()
    print("    ----melody playing time: ", end - start)

    mid.write(filename)


def get_primer_sequences(attempt=0) -> tuple[list, list, list]:
    """
    Gets random primer sequences for bass, chord and melody, from the dataset.

    Parameters
    ----------
    None

    Returns
    -------

    """
    chord_dataset: Chord_Dataset = torch.load("data/dataset/chord_dataset.pt")
    melody_dataset: Melody_Dataset = torch.load("data/dataset/melody_dataset_small.pt")
    bass_dataset: Bass_Dataset = torch.load("data/dataset/bass_dataset.pt")

    primer_start = random.randint(0, len(melody_dataset) - 1)
    song_name_melody = melody_dataset[primer_start][0][0][6][0]
    last_note_timing = melody_dataset[primer_start][0][-1][6][1]

    primer_end_chord = None

    found = False
    for i in range(0, len(chord_dataset)):
        song_name = int(chord_dataset[i][0][0][2])
        # If the song name is the same
        if int(song_name) == int(song_name_melody):
            found = True
            chord_timing = chord_dataset[i][0][0][3]

            # if the chord has passed
            if last_note_timing - chord_timing < 0:
                # primer_end_chord - SEQUENCE_LENGTH_CHORD is the index that gets correct primer sequence
                primer_end_chord = i
                break

        # If we found the song, but the song name is different, we have passed the song.
        elif found:
            break

    if not primer_end_chord:
        if attempt > 30:
            print("Tried 30 times, giving up")
            exit()
        print("Could not find primer end chord, trying again")
        return get_primer_sequences(attempt + 1)

    chord_primer = chord_dataset[primer_end_chord - SEQUENCE_LENGTH_CHORD][0]
    bass_primer = bass_dataset[primer_end_chord - SEQUENCE_LENGTH_CHORD]
    melody_primer = melody_dataset[primer_start][0]

    return chord_primer, bass_primer, melody_primer
