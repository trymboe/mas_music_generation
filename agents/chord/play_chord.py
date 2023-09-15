from mido import MidiFile, MidiTrack, Message


def play_chord(
    mid: MidiFile, chord_sequence: list[tuple[list[int], int]], arpeggiate
) -> MidiFile:
    if arpeggiate:
        mid = play_chord_arpeggiate(mid, chord_sequence)
    else:
        mid = play_chord_hold(mid, chord_sequence)

    return mid


def play_chord_hold(
    mid: MidiFile, chord_sequence: list[tuple[list[int], int]]
) -> MidiFile:
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
    note_mapping = {i: 60 + i for i in range(24)}

    track = MidiTrack()

    # Set instrument to Acoustic Grand Piano on channel 1
    track.append(Message("program_change", program=0, channel=1, time=0))

    # Add chord chords to the track
    for chord, duration in chord_sequence:
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

    mid.tracks.append(track)  # Append track to the MIDI file after populating it
    return mid


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
        print(num_repeats, remander)
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
