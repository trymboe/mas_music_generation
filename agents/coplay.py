from .bass import play_bass
from .chord import play_chord
from .drum import play_drum
from .melody import play_melody
import random
import time

import pretty_midi
import torch

from data_processing import Bass_Dataset, Chord_Dataset, Drum_Dataset


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

    # mid = pretty_midi.PrettyMIDI()
    # melody_agent = torch.load(MODEL_PATH_MELODY, DEVICE)
    # melody_dataset = torch.load(
    #     ("data/dataset/melody_dataset_" + DATASET_SIZE_MELODY + ".pt")
    # )
    # preference = generate_scale_preferences()
    # primer = melody_dataset[0]

    # (
    #     pitches,
    #     durations,
    #     current_chord,
    #     next_chord,
    #     accumulated_time,
    #     time_left_current_chord,
    # ) = ([], [], [], [], [], [])

    # for i in primer[0]:
    #     pitches.append(i[0])
    #     durations.append(i[1])
    #     current_chord.append(i[2])
    #     next_chord.append(i[3])
    #     time_left_current_chord.append(i[4])
    #     accumulated_time.append(i[5])

    # pitches_tensors = [torch.tensor(pitch) for pitch in pitches]
    # durations_tensors = [torch.tensor(duration) for duration in durations]
    # current_chord_tensors = [
    #     torch.tensor(current_chord) for current_chord in current_chord
    # ]
    # next_chord_tensors = [torch.tensor(next_chord) for next_chord in next_chord]
    # accumulated_time = [
    #     torch.tensor(accumulated_time) for accumulated_time in accumulated_time
    # ]
    # time_left_current_chord = [
    #     torch.tensor(time_left_current_chord)
    #     for time_left_current_chord in time_left_current_chord
    # ]

    # pitches_tensor = torch.stack(pitches_tensors)
    # durations_tensor = torch.stack(durations_tensors)
    # current_chord_tensor = torch.stack(current_chord_tensors)
    # next_chord_tensor = torch.stack(next_chord_tensors)
    # accumulated_time_tensor = torch.stack(accumulated_time)
    # time_left_current_chord_tensor = torch.stack(time_left_current_chord)

    # # Concatenate tensors along the specified dimension

    # x = torch.cat(
    #     (pitches_tensor, durations_tensor, current_chord_tensor, next_chord_tensor),
    #     dim=1,
    # )
    # x = x.unsqueeze(0)
    # accumulated_time_tensor = accumulated_time_tensor.unsqueeze(0)
    # time_left_current_chord_tensor = time_left_current_chord_tensor.unsqueeze(0)

    # melody_agent.eval()
    # melody_agent.to(DEVICE)
    # pitch, duration = melody_agent(
    #     x, accumulated_time_tensor, time_left_current_chord_tensor
    # )

    # pairs = []
    # for i in range(16):
    #     pairs.append([])

    #     pitch_softmax = torch.softmax(pitch[0][i], dim=0)
    #     duration_softmax = torch.softmax(duration[0][i], dim=0)

    #     pitch_softmax = select_with_preference(pitch_softmax, preference)

    #     # duration_probabilities = F.softmax(duration_output[0, :], dim=0)
    #     duration_softmax = select_with_preference(duration_softmax, [0, 1, 3, 7, 15])

    #     next_note = torch.multinomial(pitch_softmax, 1).unsqueeze(1)
    #     next_duration = torch.multinomial(duration_softmax, 1).unsqueeze(1)
    #     duration_in_beats: float = round(4 / (next_duration.item() + 1), 1) * 2

    #     pairs[i].append(next_note.item() + 50)
    #     pairs[i].append(duration_in_beats)

    # note_sequence = pairs

    # melody_instrument = pretty_midi.Instrument(program=73)
    # running_time: float = 0.0
    # for note, duration in note_sequence:
    #     if note == 129:
    #         continue
    #     start = round(running_time * (60 / 120), 2)
    #     end = round((running_time + (duration)) * (60 / 120), 2)
    #     melody_note: pretty_midi.Note = pretty_midi.Note(
    #         velocity=64, pitch=note, start=start, end=end
    #     )
    #     melody_instrument.notes.append(melody_note)
    #     running_time += duration

    # mid.instruments.append(melody_instrument)
    # mid.write("output.mid")
    # exit()

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
    mid, chord_sequence = play_chord(
        mid,
        arpeggiate,
        predicted_bass_sequence,
        chord_dataset,
        dataset_primer_start,
    )
    end = time.time()
    print("    ----chord playing time: ", end - start)

    print("  ----playing melody----")
    start = time.time()
    mid = play_melody(mid, chord_sequence, dataset_primer_start)
    end = time.time()
    print("    ----melody playing time: ", end - start)

    mid.write(filename)
