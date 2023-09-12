import mido
from mido import MidiFile, MidiTrack, Message

from config import INT_TO_NOTE


def play_bass(sequence):
    print([INT_TO_NOTE[note] for note in sequence])
    sequence_to_midi(sequence, "results/bass/bassline.mid")


def sequence_to_midi(sequence, filename="bassline.mid"):
    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C3 (MIDI note number 48)
    note_mapping = {i: 48 + i for i in range(12)}

    # Create a new MIDI file with a single track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Add notes to the track
    for note in sequence:
        midi_note = note_mapping[note]
        # Note on
        track.append(Message("note_on", note=midi_note, velocity=64, time=0))
        # Note off after one measure (assuming 480 ticks per beat and 4 beats per measure)
        track.append(Message("note_off", note=midi_note, velocity=64, time=1920))

    # Save the MIDI file
    mid.save(filename)
