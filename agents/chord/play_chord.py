from mido import MidiFile, MidiTrack, Message
import pretty_midi
import random

from config import TEMPO, ARP_STYLE


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


def play_chord_arpeggiate(pm, chord_sequence):
    # Assuming the mapping starts from C4 (MIDI note number 60) for the chord chords
    note_mapping = {i: 60 + i for i in range(24)}

    # Create an instrument instance for Acoustic Grand Piano
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program, is_drum=False)

    # Define the melodic pattern
    if ARP_STYLE == 0:
        melodic_pattern = [0, 0, 1, 2, 0, 2, 1, 0]
    elif ARP_STYLE == 1:
        melodic_pattern = [0, 0, 1, 2, 1, 0]
    else:
        melodic_pattern = [0, 1, 2, 1]

    seconds_per_beat = 60 / TEMPO
    note_duration = 4 * seconds_per_beat / len(melodic_pattern)
    start_time = 0.0
    # Add chord chords to the instrument
    for chord, duration in chord_sequence:
        num_repeats, remander = divmod(duration, (note_duration * len(melodic_pattern)))

        for _ in range(int(num_repeats)):
            for idx, pattern_note in enumerate(melodic_pattern):
                midi_note = note_mapping[chord[pattern_note]]
                if idx == 0 and ARP_STYLE == 0 or ARP_STYLE == 1:
                    midi_note -= 12  # Lower the root note by one octave
                if idx == 4 and ARP_STYLE == 0:
                    midi_note += 12  # increase the root note by one octave

                velocity = random.randint(40, 64)
                # Add note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=start_time,
                    end=start_time + note_duration,
                )
                piano.notes.append(note)
                start_time += note_duration
        if remander:
            for part_druation in [1, 2, 3, 4, 5, 6]:
                if remander == part_druation:
                    print(start_time)
                    print("remander", remander)
                    print("note_duration", note_duration)

                    num_notes = int(remander * note_duration * len(melodic_pattern))
                    print("num_notes", num_notes)
                    pattern_slice = melodic_pattern[:num_notes]
                    print("pattern slice", pattern_slice)
                    for idx, pattern_note in enumerate(pattern_slice):
                        midi_note = note_mapping[chord[pattern_note]]
                        if idx == 0 and ARP_STYLE == 0 or ARP_STYLE == 1:
                            midi_note -= 12  # Lower the root note by one octave
                        if idx == 4 and ARP_STYLE == 0:
                            midi_note += 12  # increase the root note by one octave

                        # Add note
                        velocity = random.randint(40, 64)
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=midi_note,
                            start=start_time,
                            end=start_time + note_duration,
                        )
                        piano.notes.append(note)
                        start_time += note_duration

    # Append instrument to PrettyMIDI object
    pm.instruments.append(piano)
    return pm
