# converts all midi files in the current folder
import music21
import mido
import os
import shutil


def transpose_to_c_major(midi_file):
    majors = dict(
        [
            ("A-", 4),
            ("A", 3),
            ("A#", 2),
            ("B-", 2),
            ("B", 1),
            ("C", 0),
            ("C#", -1),
            ("D-", -1),
            ("D", -2),
            ("D#", -3),
            ("E-", -3),
            ("E", -4),
            ("F", -5),
            ("F#", 6),
            ("G-", 6),
            ("G", 5),
            ("G#", 4),
        ]
    )
    minors = dict(
        [
            ("A-", 1),
            ("A", 0),
            ("A#", -1),
            ("B-", -1),
            ("B", -2),
            ("C", -3),
            ("C#", -4),
            ("D-", -4),
            ("D", -5),
            ("D#", 6),
            ("E-", 6),
            ("E", 5),
            ("F", 4),
            ("F#", 3),
            ("G-", 3),
            ("G", 2),
            ("G#", 1),
        ]
    )
    try:
        score = music21.converter.parse(midi_file)
    except Exception as e:
        print(e)
        return
    key = score.analyze("key")

    if key.mode == "major":
        halfSteps = majors[key.tonic.name]

    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    # Load the MIDI file
    mid = mido.MidiFile(midi_file)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "note_on" or msg.type == "note_off":
                msg.note += halfSteps

    new_dir = f"data/POP909/transposed/{midi_file.split('/')[-1]}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    mid.save(new_dir + f"/C_{midi_file.split('/')[-1]}")

    chord_midi_path = os.path.join(
        "/".join(midi_file.split("/")[:-1]), "chord_midi.txt"
    )
    new_chord_midi_path = os.path.join(new_dir, "chord_midi.txt")
    if os.path.exists(chord_midi_path):
        shutil.copy(chord_midi_path, new_chord_midi_path)


for idx, directory in enumerate(os.listdir("data/POP909/")):
    if ".DS_Store" in directory:
        continue
    for file in os.listdir(os.path.join("data/POP909/", directory)):
        if ".mid" in file:
            transpose_to_c_major(os.path.join("data/POP909/", directory, file))
            print("Working.. " + str(idx / 909 * 100) + "%", end="\r")
