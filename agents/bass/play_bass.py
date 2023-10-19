import pretty_midi

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


def play_bass(mid, full_bass_sequence, playstyle="bass_drum"):
    if playstyle == "bass_drum":
        bass_drum_times = find_bass_drum(mid)
        print(bass_drum_times)

    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 24 + i for i in range(12)}

    # Create a new Instrument instance for an Electric Bass
    bass_instrument = pretty_midi.Instrument(program=33)  # 33: Electric Bass

    running_time = start_time = end_time = chord_start_time = 0.0
    # When playstyle is "bass_drum", synchronize the bass notes with bass drum hits
    if playstyle == "bass_drum":
        for note, duration in full_bass_sequence:
            # Get the start time of the bass note (aligned with the bass drum hit)
            running_time = chord_start_time

            for idx, drum_beat in enumerate(bass_drum_times):
                # the chord is finished
                if bass_drum_times[idx] >= chord_start_time + duration:
                    chord_start_time += duration
                    break
                # the beat is inside the current chord, play notes
                if drum_beat > chord_start_time:
                    # If it is the first beat of the chord, and no note is planed, play a bass note
                    if (
                        running_time == chord_start_time
                        and drum_beat != chord_start_time
                    ):
                        midi_note = note_mapping[note]
                        play_note(
                            bass_instrument,
                            pitch=midi_note,
                            start_time=running_time,
                            end_time=bass_drum_times[idx],
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

                    midi_note = note_mapping[note]
                    play_note(
                        bass_instrument,
                        pitch=midi_note,
                        start_time=start_time,
                        end_time=end_time,
                    )

                    running_time = end_time

    else:  # Original behavior if playstyle isn't "bass_drum"
        for note, duration in full_bass_sequence:
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

    return mid


def play_note(
    bass_instrument: pretty_midi.Instrument,
    pitch: int,
    start_time: float,
    end_time: float,
    velocity: int = 64,
):
    bass_note = pretty_midi.Note(
        velocity=velocity, pitch=pitch, start=start_time, end=end_time
    )
    bass_instrument.notes.append(bass_note)


def find_bass_drum(pm: pretty_midi.PrettyMIDI):
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
            bass_drum_times.append(note.start)

    return bass_drum_times
