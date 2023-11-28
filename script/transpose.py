# converts all midi files in the current folder
import music21
import mido
import os
import shutil
import pretty_midi

from transpose_chord import transpose_chord


def get_key(path):
    key = []
    # Check if key_audio.txt exists
    with open(path, "r") as file:
        for line in file:
            components = line.split()
            # Extract the key (string after the two numbers)
            key.append(components[2])
    return key


def check_time_signature(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)

    # Only work with time signature 4/4
    for time_signature in pm.time_signature_changes:
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            return False
    return True


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

    key_path = midi_file[:-8] + "/key_audio.txt"
    key = get_key(key_path)

    if len(key) > 1:
        return 0

    if check_time_signature(midi_file) == False:
        return 0

    key_tonic_name = key[0][0]
    key_mode = key[0][-3:]

    if key_mode == "maj":
        halfSteps = majors[key_tonic_name]

    elif key_mode == "min":
        halfSteps = minors[key_tonic_name]
    else:
        print(key_mode)
        exit()

    # Load the MIDI file
    mid = mido.MidiFile(midi_file)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "note_on" or msg.type == "note_off":
                msg.note += halfSteps

    new_dir = f"data/POP909/transposed/{midi_file.split('/')[-1]}"

    new_dir = new_dir[:-4]
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    mid.save(new_dir + f"/C_{midi_file.split('/')[-1]}")

    transpose_chord(midi_file[:-8])
    move_beat(midi_file[:-8])


def move_beat(mid_dir):
    from_path = mid_dir + "/beat_audio.txt"
    to_path = f"data/POP909/transposed/{mid_dir.split('/')[-1]}"

    shutil.copy(
        mid_dir + "/beat_audio.txt", f"data/POP909/transposed/{mid_dir.split('/')[-1]}"
    )


total = 0
for idx, directory in enumerate(os.listdir("data/POP909/")):
    if ".DS_Store" in directory:
        continue
    for file in os.listdir(os.path.join("data/POP909/", directory)):
        if ".mid" in file:
            total += 1
            print(total)
            transpose_to_c_major(os.path.join("data/POP909/", directory, file))

        # print("Working.. " + str(idx / 909 * 100) + "%", end="\r")
