import mido
from mido import MidiFile, MidiTrack, Message


# def play_bass(mid, full_bass_sequence):
#     # Mapping from sequence numbers to MIDI note numbers
#     # Starting from C1 (MIDI note number 24)
#     note_mapping = {i: 24 + i for i in range(12)}

#     # Create a new MIDI file with a single track

#     track = MidiTrack()
#     mid.tracks.append(track)

#     # Set instrument to Bass Guitar (Electric Bass)
#     track.append(Message("program_change", program=33, time=0))

#     # Add bass notes to the track
#     for note, duration in full_bass_sequence:
#         midi_note = note_mapping[note]
#         # Note on
#         track.append(Message("note_on", note=midi_note, velocity=64, time=0))
#         # Note off after the specified number of beats
#         # Assuming 480 ticks per beat
#         ticks_per_beat = 480
#         track.append(
#             Message(
#                 "note_off", note=midi_note, velocity=64, time=ticks_per_beat * duration
#             )
#         )

#     return mid

import pretty_midi


def play_bass(mid, full_bass_sequence):
    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new Instrument instance for an Electric Bass
    bass_instrument = pretty_midi.Instrument(program=33)  # 33: Electric Bass

    # Initialize a variable to keep track of time
    current_time = 0.0

    # Add bass notes to the bass_instrument
    for note, duration in full_bass_sequence:
        midi_note = note_mapping[note]
        # Create a Note instance for each note in full_bass_sequence
        bass_note = pretty_midi.Note(
            velocity=64,  # volume
            pitch=midi_note,  # MIDI note number
            start=current_time,  # start time
            end=current_time + duration,  # end time
        )
        # Add note to the bass_instrument
        bass_instrument.notes.append(bass_note)
        # Move the current time cursor
        current_time += duration

    # Add the bass_instrument to the PrettyMIDI object
    mid.instruments.append(bass_instrument)

    return mid
