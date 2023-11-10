import pretty_midi
import torch

from .eval_agent import predict_next_k_notes_bass
from config import MODEL_PATH_BASS, DEVICE, TEMPO
from data_processing import Bass_Dataset
from .bass_network import Bass_Network


def play_bass(
    mid: pretty_midi.PrettyMIDI,
    bass_dataset: Bass_Dataset,
    dataset_primer_start: int,
    playstyle: str = "bass_drum",
) -> tuple[pretty_midi.PrettyMIDI, list[int, int]]:
    bass_agent: Bass_Network = torch.load(MODEL_PATH_BASS, DEVICE)
    bass_agent.eval()

    predicted_bass_sequence: list[int, int] = predict_next_k_notes_bass(
        bass_agent, bass_dataset, dataset_primer_start
    )

    if playstyle == "bass_drum":
        bass_drum_times = find_bass_drum(mid)

    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new Instrument instance for an Electric Bass
    bass_instrument = pretty_midi.Instrument(program=33)  # 33: Electric Bass

    running_time = start_time = end_time = chord_start_time = 0.0
    # When playstyle is "bass_drum", synchronize the bass notes with bass drum hits
    if playstyle == "bass_drum":
        for note, duration in predicted_bass_sequence:
            midi_note = note_mapping[note]
            # Get the start time of the bass note (aligned with the bass drum hit)
            running_time = chord_start_time

            for idx, drum_beat in enumerate(bass_drum_times):
                # the chord is finished
                if bass_drum_times[idx] >= chord_start_time + duration:
                    # If no note has been played yet, play a note for the entire duration
                    if running_time == chord_start_time:
                        end_time == chord_start_time + (
                            bass_drum_times[idx + 1] - running_time
                        )
                        play_note(
                            bass_instrument,
                            pitch=midi_note,
                            start_time=running_time,
                            end_time=bass_drum_times[idx],
                        )

                    chord_start_time += duration
                    break
                # the beat is inside the current chord, play notes
                if drum_beat >= chord_start_time:
                    # If it is the first beat of the chord, and no note is played, play a bass note
                    if running_time == chord_start_time:
                        end_time = chord_start_time + duration
                        play_note(
                            bass_instrument,
                            pitch=midi_note,
                            start_time=running_time,
                            end_time=end_time,
                        )

                    start_time = bass_drum_times[idx]
                    # Song is finished if there are no more bass drum hits
                    if idx + 1 == len(bass_drum_times):
                        end_time = running_time + duration
                        play_note(
                            bass_instrument,
                            pitch=midi_note,
                            start_time=start_time,
                            end_time=end_time,
                        )
                        break

                    else:
                        end_time = bass_drum_times[idx + 1]
                        # End note if it goes past the chord
                        if end_time > chord_start_time + duration:
                            end_time = chord_start_time + duration

                    play_note(
                        bass_instrument,
                        pitch=midi_note,
                        start_time=start_time,
                        end_time=end_time,
                    )

                    running_time = end_time

    else:  # Original behavior if playstyle isn't "bass_drum"
        for note, duration in predicted_bass_sequence:
            midi_note = note_mapping[note]
            bass_note = pretty_midi.Note(
                velocity=64,
                pitch=midi_note,
                start=chord_start_time,
                end=chord_start_time + duration,
            )
            bass_instrument.notes.append(bass_note)
            chord_start_time += duration

    # Add the bass_instrument to the PrettyMIDI object
    mid.instruments.append(bass_instrument)

    return mid, predicted_bass_sequence


def play_note(
    bass_instrument: pretty_midi.Instrument,
    pitch: int,
    start_time: float,
    end_time: float,
    velocity: int = 64,
):
    bass_note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=beats_to_seconds(start_time),
        end=beats_to_seconds(end_time),
    )
    bass_instrument.notes.append(bass_note)


def find_bass_drum(pm: pretty_midi.PrettyMIDI) -> list[float]:
    """
    Find every time step where the bass drum is played in a given PrettyMIDI object.

    Args:
    - pm (pretty_midi.PrettyMIDI): A PrettyMIDI object containing one drum track.

    Returns:
    - List[float]: List of start times (in seconds) where the bass drum is played.
    """
    bass_drum_times = []

    # Assuming the drum track is the first track in the PrettyMIDI object
    drum_track = pm.instruments[0]

    # Check if the track is a drum track
    if not drum_track.is_drum:
        raise ValueError("The provided track is not a drum track.")

    # Iterate over notes in the drum track
    for note in drum_track.notes:
        if note.pitch == 36:  # 36 is the MIDI number for Acoustic Bass Drum
            bass_drum_times.append(seconds_to_beat(note.start))

    return bass_drum_times


def beats_to_seconds(beats: float) -> float:
    return round(beats * (60 / TEMPO), 2)


def seconds_to_beat(seconds: float) -> float:
    return round(seconds * (TEMPO / 60), 2)
