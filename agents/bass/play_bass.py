import mido
from mido import MidiFile, MidiTrack, Message


def play_bass(full_bass_sequence):
    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new MIDI file with a single track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to Bass Guitar (Electric Bass)
    track.append(Message("program_change", program=33, time=0))

    # Add bass notes to the track
    for note, duration in full_bass_sequence:
        midi_note = note_mapping[note]
        # Note on
        track.append(Message("note_on", note=midi_note, velocity=64, time=0))
        # Note off after the specified number of beats
        # Assuming 480 ticks per beat
        ticks_per_beat = 480
        track.append(
            Message(
                "note_off", note=midi_note, velocity=64, time=ticks_per_beat * duration
            )
        )

    return mid
