from mido import MidiFile, MidiTrack, Message


def play_piano(
    mid: MidiFile, piano_chord_sequence: list[tuple[list[int], int]]
) -> MidiFile:
    # Assuming the mapping starts from C4 (MIDI note number 60) for the piano chords
    note_mapping = {i: 60 + i for i in range(12)}

    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to Acoustic Grand Piano
    track.append(Message("program_change", program=0, time=0))

    # Add piano chords to the track
    for chord, duration in piano_chord_sequence:
        # Iterate through each note in the chord
        for note in chord:
            midi_note = note_mapping[note]
            track.append(Message("note_on", note=midi_note, velocity=64, time=0))

        # Note off after the specified number of beats
        ticks_per_beat = 480
        time_offset = ticks_per_beat * duration
        for note in chord:
            midi_note = note_mapping[note]
            track.append(
                Message(
                    "note_off",
                    note=midi_note,
                    velocity=64,
                    time=time_offset,
                )
            )
            # Reset time offset for subsequent notes in the same chord
            time_offset = 0

    return mid
