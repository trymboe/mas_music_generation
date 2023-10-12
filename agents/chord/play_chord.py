from mido import MidiFile, MidiTrack, Message
import pretty_midi


def play_chord(
    mid: MidiFile, chord_sequence: list[tuple[list[int], int]], arpeggiate
) -> MidiFile:
    if arpeggiate:
        mid = play_chord_arpeggiate(mid, chord_sequence)
    else:
        mid = play_chord_hold(mid, chord_sequence)

    return mid


def play_chord_hold(pretty_midi_obj, chord_sequence):
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord notes
    note_mapping = {i: 60 + i for i in range(24)}

    # Create a new Instrument instance for an Acoustic Grand Piano
    piano_instrument = pretty_midi.Instrument(program=0)  # 0: Acoustic Grand Piano

    current_time = 0.0  # Initialize a variable to keep track of time

    # Add chords to the piano_instrument
    for chord, duration in chord_sequence:
        # Create a Note instance for each note in the chord
        for note in chord:
            midi_note = note_mapping[note]
            piano_note = pretty_midi.Note(
                velocity=64,  # volume
                pitch=midi_note,  # MIDI note number
                start=current_time,  # start time
                end=current_time + duration,  # end time
            )
            # Add note to the piano_instrument
            piano_instrument.notes.append(piano_note)

        # Move the current time cursor
        current_time += duration

    # Add the piano_instrument to the PrettyMIDI object
    pretty_midi_obj.instruments.append(piano_instrument)

    return pretty_midi_obj


# Example usage
# Create a PrettyMIDI object
pm = pretty_midi.PrettyMIDI()

# Example chord sequence: [([note_1, note_2, ...], duration), (...), ...]
example_sequence = [([0, 4, 7], 1.5), ([2, 5, 9], 1.5), ([1, 4, 8], 1.5)]

# Use the function
pm = play_chord_hold(pm, example_sequence)

# Save MIDI file
pm.write("example_chords.mid")


# def play_chord_hold(
#     mid: MidiFile, chord_sequence: list[tuple[list[int], int]]
# ) -> MidiFile:
#     # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
#     note_mapping = {i: 60 + i for i in range(24)}

#     track = MidiTrack()

#     # Set instrument to Acoustic Grand Piano on channel 1
#     track.append(Message("program_change", program=0, channel=1, time=0))

#     # Add chord chords to the track
#     for chord, duration in chord_sequence:
#         # Iterate through each note in the chord
#         for note in chord:
#             midi_note = note_mapping[note]
#             track.append(Message("note_on", note=midi_note, velocity=64, time=0))

#         # Note off after the specified number of beats
#         ticks_per_beat = 480
#         time_offset = ticks_per_beat * duration
#         for note in chord:
#             midi_note = note_mapping[note]
#             track.append(
#                 Message(
#                     "note_off",
#                     note=midi_note,
#                     velocity=64,
#                     time=time_offset,
#                 )
#             )
#             # Reset time offset for subsequent notes in the same chord
#             time_offset = 0

#     mid.tracks.append(track)  # Append track to the MIDI file after populating it
#     return mid


def play_chord_arpeggiate(mid, chord_sequence):
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
    note_mapping = {i: 60 + i for i in range(24)}

    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to Acoustic Grand Piano on channel 1
    track.append(Message("program_change", program=0, channel=1, time=0))

    # Define the melodic pattern
    melodic_pattern = [0, 0, 1, 2, 1, 0]
    ticks_per_beat = 480
    tick_duration = 4 * ticks_per_beat // len(melodic_pattern)

    # Add chord chords to the track
    for chord, duration in chord_sequence:
        num_repeats, remander = divmod(duration, 4)
        for _ in range(num_repeats):
            for idx, pattern_note in enumerate(melodic_pattern):
                midi_note = note_mapping[chord[pattern_note]]
                if idx == 0:
                    midi_note -= 12  # Lower the root note by one octave
                # Note on
                track.append(Message("note_on", note=midi_note, velocity=64, time=0))
                # Note off
                track.append(
                    Message("note_off", note=midi_note, velocity=64, time=tick_duration)
                )
        if remander == 2:
            for idx, pattern_note in enumerate(melodic_pattern[:3]):
                midi_note = note_mapping[chord[pattern_note]]
                if idx == 0:
                    midi_note -= 12  # Lower the root note by one octave
                # Note on
                track.append(Message("note_on", note=midi_note, velocity=64, time=0))
                # Note off
                track.append(
                    Message("note_off", note=midi_note, velocity=64, time=tick_duration)
                )
        if remander == 1:
            midi_note = note_mapping[chord[2]]
            # Note on
            track.append(Message("note_on", note=midi_note, velocity=64, time=0))
            # Note off
            track.append(
                Message(
                    "note_off",
                    note=midi_note,
                    velocity=64,
                    time=ticks_per_beat,
                )
            )
        if remander == 3:
            for idx, pattern_note in enumerate(melodic_pattern[:3]):
                midi_note = note_mapping[chord[pattern_note]]
                if idx == 0:
                    midi_note -= 12  # Lower the root note by one octave
                # Note on
                track.append(Message("note_on", note=midi_note, velocity=64, time=0))
                # Note off
                track.append(
                    Message("note_off", note=midi_note, velocity=64, time=tick_duration)
                )
            midi_note = note_mapping[chord[2]]
            # Note on
            track.append(Message("note_on", note=midi_note, velocity=64, time=0))
            # Note off
            track.append(
                Message("note_off", note=midi_note, velocity=64, time=ticks_per_beat)
            )

    return mid
