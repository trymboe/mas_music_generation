import os
import re
from torch.utils.data import Dataset, DataLoader


def extract_chords_from_files(root_dir, limit, only_triads):
    all_chords: tuple(str, str) = []
    endless = True if limit == 0 else False

    # Traverse through the directory structure
    for dir_name, _, file_list in os.walk(root_dir):
        chords = []
        for file_name in file_list:
            key_file_path = os.path.join(dir_name, "key_audio.txt")
            if file_name == "chord_audio.txt":
                key = get_key(key_file_path)
                # If there is a keychange in the song, skip it
                if len(key) > 1:
                    continue
                with open(os.path.join(dir_name, file_name), "r") as file:
                    for line in file:
                        # Split the line into components
                        components = line.split()
                        if components[2] == "N":
                            continue

                        # Split the chord by ':' and save as a tuple
                        root, version = components[2].split(":")

                        if only_triads:
                            version = remove_non_triad(version)

                        chords.append((root, version))

                    chords = flat_to_sharp(chords)
                    key = flat_to_sharp_key(key[0])

                    # for major
                    if key[-1] == "j" and key != "C:maj":
                        chords = transpose_major(chords, key)
                        all_chords.append(chords)

                    # for minor
                    if key[-1] == "n" and key != "A:min":
                        pass
                        # chord = transpose_minor(chords, key)

                    if not endless and total_length(all_chords) >= limit:
                        all_notes = get_notes_from_chords(all_chords)
                        return all_chords, all_notes

    all_notes = get_notes_from_chords(all_chords)
    return all_chords, all_notes


def total_length(chords):
    count = 0
    for item in chords:
        if isinstance(item, list):
            count += total_length(item)
        else:
            count += 1
    return count


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


def flat_to_sharp_key(key):
    # Mapping of flat notes to their sharp equivalents
    flat_to_sharp_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

    # Split the key into the note and the mode (maj/min)
    note, mode = key.split(":")

    # Convert flat note to sharp if it's in the dictionary
    if note in flat_to_sharp_map:
        note = flat_to_sharp_map[note]

    return f"{note}:{mode}"


def remove_non_triad(string):
    # Replace numbers that aren't 2 or 4 with an empty string
    modified_str = re.sub(r"(?<!\d)(?:(?!2|4)\d)+(?!\d)", "", string)

    # Remove non-letter characters at the end of the string
    modified_str = re.sub(r"[^a-zA-Z]+$", "", modified_str)

    # If the resulting string is empty, replace with "maj"
    if not modified_str:
        return "maj"
    return modified_str


def get_key(file_path):
    key = []
    # Check if key_audio.txt exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                components = line.split()
                # Extract the key (string after the two numbers)
                key.append(components[2])
    return key


def transpose_major(chords, key):
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


def transpose_minor(chords, key):
    pass


def get_notes_from_chords(chords):
    all_notes = []
    for song in chords:
        song_notes = []
        for note, version in song:
            song_notes.append(note)

        all_notes.append(song_notes)

    return all_notes
