import os
import shutil


def transpose_chord(directory):
    for file in os.listdir(directory):
        chords: list[tuple(str, str)] = []
        all_chords: list[list[tuple(str, str)]] = []
        chord_start_list: list[str] = []
        chord_end_list: list[str] = []
        if file == "chord_audio.txt":
            key = get_key(directory, "key_audio.txt")
            if len(key) > 1:
                # root_dir = "/".join(directory.split("/")[:2])
                # song_name = directory.split("/")[-1]
                # dir_path = os.path.join(root_dir, "transposed", song_name)
                # try:
                #     shutil.rmtree(dir_path)
                # except OSError as e:
                #     print(e)
                return

            with open(os.path.join(directory, file), "r") as file:
                for idx, line in enumerate(file):
                    components = line.split()
                    if components[2] == "N":
                        continue
                    chord_start = str(components[0])
                    chord_end = str(components[1])

                    chord_start_list.append(chord_start)
                    chord_end_list.append(chord_end)

                    root, version = components[2].split(":")

                    chords.append((root, version))

                chords = flat_to_sharp(chords)

                key = flat_to_sharp_key(key[0])

                if key[-1] == "j" and key != "C:maj":
                    chords = transpose_chord_major(chords, key)

                if key[-1] == "n" and key != "A:min":
                    chords = transpose_chord_minor(chords, key)

                all_notes = [
                    [start, end, f"{chord[0]}:{chord[1]}"]
                    for start, end, chord in zip(
                        chord_start_list, chord_end_list, chords
                    )
                ]

                # Specify the filename
                root_dir = "/".join(directory.split("/")[:2])
                song_name = directory.split("/")[-1]

                filename = os.path.join(
                    root_dir, "transposed", song_name, "chord_audio.txt"
                )
                if os.path.exists(os.path.join(root_dir, "transposed", song_name)):
                    # Write to a file
                    with open(filename, "w") as file:
                        for note in all_notes:
                            file.write(" ".join(note) + "\n")


def flat_to_sharp(chords):
    new_chords = []
    for chord in chords:
        root, version = chord
        if root == "Db":
            chord = ("C#", version)
        elif root == "Eb":
            chord = ("D#", version)
        elif root == "Gb":
            chord = ("F#", version)
        elif root == "Ab":
            chord = ("G#", version)
        elif root == "Bb":
            chord = ("A#", version)
        new_chords.append(chord)
    return new_chords


def transpose_chord_major(chords, key):
    # Define the musical notes in order
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root = key.split(":")[0]

    # Find the interval between the original key and C
    interval = notes.index("C") - notes.index(root)

    # Transpose each chord in the progression by the interval
    transposed_chords = []
    for chord in chords:
        root, version = chord
        new_root_index = (notes.index(root) + interval) % len(notes)
        new_root = notes[new_root_index]
        transposed_chords.append((new_root, version))

    return transposed_chords


def transpose_chord_minor(chords, key):
    # Define the musical notes in order
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root = key.split(":")[0]

    # Find the interval between the original key and A minor
    # Since A minor is the relative minor of C major, its root note is A
    interval = notes.index("A") - notes.index(root)

    # Transpose each chord in the progression by the interval
    transposed_chords = []
    for chord in chords:
        root, version = chord
        new_root_index = (notes.index(root) + interval) % len(notes)
        new_root = notes[new_root_index]
        transposed_chords.append((new_root, version))

    return transposed_chords


def flat_to_sharp_key(key):
    # Mapping of flat notes to their sharp equivalents
    flat_to_sharp_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

    # Split the key into the note and the mode (maj/min)
    note, mode = key.split(":")

    # Convert flat note to sharp if it's in the dictionary
    if note in flat_to_sharp_map:
        note = flat_to_sharp_map[note]

    return f"{note}:{mode}"


def get_key(dir_name, file_name):
    key = []
    # Check if key_audio.txt exists
    with open(os.path.join(dir_name, file_name), "r") as file:
        for line in file:
            components = line.split()
            # Extract the key (string after the two numbers)
            key.append(components[2])
    return key


# for idx, directory in enumerate(os.listdir("data/POP909/")):
#     if ".DS_Store" in directory:
#         continue
#     for file in os.listdir(os.path.join("data/POP909/", directory)):
#         if ".mid" in file:
#             transpose_chord(os.path.join("data/POP909/", directory))
#             # print("Working.. " + str(idx / 909 * 100) + "%", end="\r")
