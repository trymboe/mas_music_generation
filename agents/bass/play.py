import mido
from mido import MidiFile, MidiTrack, Message

from config import INT_TO_NOTE


def play_bass(primer_sequence, predicted_sequence):
    full_sequence = predicted_sequence

    note_sequence, duration_sequence = (
        primer_sequence[0].tolist(),
        primer_sequence[1].tolist(),
    )

    for i in range(len(note_sequence) - 1, -1, -1):
        print(i)
        full_sequence.insert(0, (note_sequence[i], duration_sequence[i]))
    print(full_sequence)
    print([f"{INT_TO_NOTE[note]} - {duration}" for note, duration in full_sequence])
    sequence_to_midi(full_sequence, "results/bass/bassline.mid")


from mido import MidiFile, MidiTrack, Message


def sequence_to_midi(sequence, filename="bassline.mid"):
    # Mapping from sequence numbers to MIDI note numbers
    # Starting from C1 (MIDI note number 24)
    note_mapping = {i: 36 + i for i in range(12)}

    # Create a new MIDI file with a single track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to Bass Guitar (Electric Bass)
    track.append(Message("program_change", program=33, time=0))

    # Add notes to the track
    for note, duration in sequence:
        midi_note = note_mapping[note]
        # Note on
        track.append(Message("note_on", note=midi_note, velocity=64, time=0))
        # Note off after the specified number of measures
        # Assuming 480 ticks per beat, 4 beats per measure, and duration is number of measures
        ticks_per_measure = 480
        track.append(
            Message(
                "note_off",
                note=midi_note,
                velocity=64,
                time=ticks_per_measure * duration,
            )
        )

    # Save the MIDI file
    mid.save(filename)
