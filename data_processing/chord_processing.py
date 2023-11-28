import os
import re
import torch

from .datasets import Bass_Dataset, Chord_Dataset
from .utils import get_timed_notes

from config import NUMBER_OF_NOTES_FOR_TRAINING


def get_bass_dataset(root_directory: str) -> Bass_Dataset:
    """
    Creates a dataset object containing timed note sequences for the training
    and evaluation of the bass agent.

    Args
    ----------
        root_directory (str): The root directory of the dataset.

    Returns
    ----------
        Bass_Dataset: A dataset object containing timed note sequences.
    """
    if not os.path.exists("data/dataset/bass_dataset.pt"):
        _, notes, beats = extract_chords_from_files(
            root_directory, NUMBER_OF_NOTES_FOR_TRAINING, True
        )

        timed_notes: list[list[tuple[str, int]]] = get_timed_notes(notes, beats)
        bass_dataset: Bass_Dataset = Bass_Dataset(timed_notes)
        torch.save(bass_dataset, "data/dataset/bass_dataset.pt")
    else:
        bass_dataset: Bass_Dataset = torch.load("data/dataset/bass_dataset.pt")

    return bass_dataset


def get_chord_dataset(root_directory: str) -> Chord_Dataset:
    """
    Creates a dataset object containing chord progressions for the training
    and evaluation of the chord agent.

    Args
    ----------
        root_directory (str): The root directory of the dataset.

    Returns
    ----------
        Chord_Dataset: A dataset object containing chord progressions.
    """
    if not os.path.exists("data/dataset/chord_dataset.pt"):
        chords, _, _ = extract_chords_from_files(
            root_directory, NUMBER_OF_NOTES_FOR_TRAINING, True
        )

        chord_dataset: Chord_Dataset = Chord_Dataset(chords)
        torch.save(chord_dataset, "data/dataset/chord_dataset.pt")
    else:
        chord_dataset: Chord_Dataset = torch.load("data/dataset/chord_dataset.pt")
    return chord_dataset


def extract_chords_from_files(root_dir, limit, only_triads):
    all_chords: list[list[tuple(str, str)]] = []
    all_beats: list[list[int]] = []

    endless = True if limit == 0 else False

    # Traverse through the directory structure
    for dir_name, subdirs, file_list in os.walk(root_dir):
        if "versions" in subdirs:
            subdirs.remove("versions")
        chords = []
        for file_name in file_list:
            if file_name == ".DS_Store":
                continue
            # key: str = get_key(dir_name, "key_audio.txt")
            beat_list: list = get_beat_info(dir_name, "beat_audio.txt")
            if file_name == "chord_audio.txt":
                # # If there is a keychange in the song, skip it
                # if len(key) > 1:
                #     continue
                chords: list[tuple(str, str)] = []
                num_beats_list: list[int] = []

                placement: list[str, int] = []

                for fn in file_list:
                    if fn[0] == "C":
                        song_name = fn.split("_")[1].split(".")[0]
                print(dir_name)
                with open(os.path.join(dir_name, file_name), "r") as file:
                    for line in file:
                        # Split the line into components
                        components = line.split()
                        if components[2] == "N":
                            continue
                        # Split the chord by ':' and save as a tuple
                        chord_start = float(components[0])
                        chord_end = float(components[1])

                        placement = [song_name, chord_start]

                        num_beats = find_chord_length(chord_start, chord_end, beat_list)

                        root, version = components[2].split(":")
                        if only_triads:
                            version = remove_non_triad(version)
                        chords.append((root, version, placement))
                        num_beats_list.append(num_beats)

                    chords = flat_to_sharp(chords)
                    # key = flat_to_sharp_key(key[0])

                    # if key[-1] == "j" and key != "C:maj":
                    # chords = transpose_chord(chords, key)
                    # all_chords.append(chords)
                    # all_beats.append(num_beats_list)

                    all_chords.append(chords)
                    all_beats.append(num_beats_list)

                    if not endless and total_length(all_chords) >= limit:
                        all_notes = get_notes_from_chords(all_chords)

                        return all_chords, all_notes, all_beats

    all_notes = get_notes_from_chords(all_chords)

    return all_chords, all_notes, all_beats


def find_chord_length(chord_start, chord_end, beat_list):
    distance_to_start: float = 10000
    for index, time in enumerate(beat_list):
        current_distance = abs(float(time) - chord_start)
        if current_distance < distance_to_start:
            distance_to_start = current_distance
        else:
            chord_start = index
            break

    distance_to_end: float = 10000
    for index, time in enumerate(beat_list):
        current_distance = abs(float(time) - chord_end)
        if current_distance < distance_to_end:
            distance_to_end = current_distance
        else:
            chord_end = index
            break
    beat_time = chord_end - chord_start
    # The last note is not represented like the rest, defaults to 2 beats
    if beat_time < 0:
        beat_time = 2
    if beat_time > 8:
        beat_time = 8
    return beat_time


def get_beat_info(dir_name, file_name):
    beat_list = []
    with open(os.path.join(dir_name, file_name), "r") as file:
        for line in file:
            components = line.split()
            beat_list.append(components[0])
    return beat_list


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
        root, version, _ = chord
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
    # Remove anything after a forward slash '/'
    string = re.sub(r"/.*$", "", string)

    # Remove everything inside parentheses and after
    modified_str = re.sub(r"\(.*?\)", "", string)

    # # Replace numbers that aren't 2 or 4 with an empty string
    modified_str = re.sub(r"(?<!\d)(?:(?!2|4)\d)+(?!\d)", "", modified_str)

    # # Remove non-letter characters at the end of the string
    modified_str = re.sub(r"[^a-zA-Z24]+$", "", modified_str)

    # If the resulting string is empty, replace with "maj"
    if not modified_str:
        return "maj"
    return modified_str


def get_key(dir_name, file_name):
    key = []
    # Check if key_audio.txt exists
    with open(os.path.join(dir_name, file_name), "r") as file:
        for line in file:
            components = line.split()
            # Extract the key (string after the two numbers)
            key.append(components[2])
    return key


def transpose_chord(chords, key):
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

        for note in song[0]:
            song_notes.append(note[0])

        all_notes.append(song_notes)

    return all_notes
