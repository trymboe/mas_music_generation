import os
import re
import torch

from .datasets import Bass_Dataset, Chord_Dataset
from .utils import get_timed_notes

from config import (
    TRAIN_DATASET_PATH_BASS,
    TEST_DATASET_PATH_BASS,
    VAL_DATASET_PATH_BASS,
    TRAIN_DATASET_PATH_CHORD,
    TEST_DATASET_PATH_CHORD,
    VAL_DATASET_PATH_CHORD,
)


def get_bass_and_chord_dataset(root_directory: str) -> None:
    """
    Creates a dataset object containing timed note sequences and chords for the training
    and evaluation of the bass agent and chord agent.

    Args
    ----------
        root_directory (str): The root directory of the data.

    Returns
    ----------
        None
    """
    if not os.path.exists(TRAIN_DATASET_PATH_BASS) or not os.path.exists(
        TRAIN_DATASET_PATH_CHORD
    ):
        timed_notes_list: list[list[list[tuple[str, int]]]] = []
        chords_list: list[list[tuple[str, str]]] = []
        for split in ["train", "test", "val"]:
            chords, notes, beats = extract_chords_from_files(
                root_directory, True, split
            )
            print(split)
            print(len(chords[0]))
            print(len(notes[0]))
            chords_list.append(chords)
            timed_notes_list.append(get_timed_notes(notes, beats))

        if not os.path.exists(TRAIN_DATASET_PATH_CHORD):
            chord_dataset_train: Chord_Dataset = Chord_Dataset(chords_list[0])
            chord_dataset_test: Chord_Dataset = Chord_Dataset(chords_list[1])
            chord_dataset_val: Chord_Dataset = Chord_Dataset(chords_list[2])

            torch.save(chord_dataset_train, TRAIN_DATASET_PATH_CHORD)
            torch.save(chord_dataset_test, TEST_DATASET_PATH_CHORD)
            torch.save(chord_dataset_val, VAL_DATASET_PATH_CHORD)

        if not os.path.exists(TRAIN_DATASET_PATH_BASS):
            bass_dataset_train: Bass_Dataset = Bass_Dataset(timed_notes_list[0])
            bass_dataset_test: Bass_Dataset = Bass_Dataset(timed_notes_list[1])
            bass_dataset_val: Bass_Dataset = Bass_Dataset(timed_notes_list[2])

            torch.save(bass_dataset_train, TRAIN_DATASET_PATH_BASS)
            torch.save(bass_dataset_test, TEST_DATASET_PATH_BASS)
            torch.save(bass_dataset_val, VAL_DATASET_PATH_BASS)


def extract_chords_from_files(root_dir, only_triads, split):
    """
    Extracts chord and root note information from the chord_audio.txt files in the specified directory.
    This is the funduments behind the bass and chord datasets.

    Args:
    ----------
        root_dir (str): The root directory containing the files.
        only_triads (bool): Flag indicating whether to include only triads.
        split (str): The subdirectory within the root directory to process.

    Returns:
    ----------
        tuple: A tuple containing the extracted chords, notes, and beats.
            - all_chords (list[list[tuple(str, str)]]): A list of chords, where each chord is represented as a tuple of root and version.
            - all_notes (list): A list of notes extracted from the chords.
            - all_beats (list[list[int]]): A list of beat information for each chord.
    """

    print(root_dir, split)
    all_chords: list[list[tuple(str, str)]] = []
    all_beats: list[list[int]] = []

    # Traverse through the directory structure
    for dir_name, subdirs, file_list in os.walk(os.path.join(root_dir, split)):
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

    all_notes = get_notes_from_chords(all_chords)

    return all_chords, all_notes, all_beats


def find_chord_length(chord_start, chord_end, beat_list):
    """
    Calculates the length of a chord based on its start and end positions in a list of beats.

    Parameters:
    ----------
    chord_start (float): The start position of the chord.
    chord_end (float): The end position of the chord.
    beat_list (list): A list of beats.

    Returns:
    ----------
    int: The length of the chord in beats.
    """
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
    """
    Retrieves the beat information from a file.

    Parameters:
    ----------
    dir_name (str): The directory name where the file is located.
    file_name (str): The name of the file.

    Returns:
    ----------
    list: A list containing the beat information.
    """
    beat_list = []
    with open(os.path.join(dir_name, file_name), "r") as file:
        for line in file:
            components = line.split()
            beat_list.append(components[0])
    return beat_list


def flat_to_sharp(chords):
    """
    Converts the name of chords with flat root notes to name of chords with sharp root notes.

    Args:
    ----------
        chords (list): A list of chords, where each chord is a tuple containing the root note, version, and any additional information.

    Returns:
    ----------
        list: A new list of chords with the root notes converted from flat to sharp.
    """
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
    """
    Converts a key from flat notation to sharp notation.

    Args:
    ----------
        key (str): The key in flat notation, e.g. "Db:maj", "Eb:min".

    Returns:
    ----------
        str: The key in sharp notation, e.g. "C#:maj", "D#:min".
    """

    # Mapping of flat notes to their sharp equivalents
    flat_to_sharp_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

    # Split the key into the note and the mode (maj/min)
    note, mode = key.split(":")

    # Convert flat note to sharp if it's in the dictionary
    if note in flat_to_sharp_map:
        note = flat_to_sharp_map[note]

    return f"{note}:{mode}"


def remove_non_triad(string):
    """
    Removes non-triad elements from a chord string.
    Used to convert non-triad chords to triads.

    Args:
    ----------
        string (str): The input chord string.

    Returns:
    ----------
        str: The modified chord string with non-triad elements removed.
    """

    # Remove anything after a forward slash '/'
    string = re.sub(r"/.*$", "", string)

    # Remove everything inside parentheses and after
    modified_str = re.sub(r"\(.*?\)", "", string)

    # Replace numbers that aren't 2 or 4 with an empty string
    modified_str = re.sub(r"(?<!\d)(?:(?!2|4)\d)+(?!\d)", "", modified_str)

    # Remove non-letter characters at the end of the string
    modified_str = re.sub(r"[^a-zA-Z24]+$", "", modified_str)

    modified_str = "dim" if modified_str == "hdim" else modified_str

    modified_str = "min" if modified_str == "minmaj" else modified_str

    # If the resulting string is empty, replace with "maj"
    if not modified_str:
        return "maj"
    return modified_str


def get_key(dir_name, file_name):
    """
    Retrieves the key from a text file.

    Parameters:
    ----------
    dir_name (str): The directory name where the file is located.
    file_name (str): The name of the file.

    Returns:
    ----------
    list: A list of keys extracted from the file.
    """
    key = []
    # Check if key_audio.txt exists
    with open(os.path.join(dir_name, file_name), "r") as file:
        for line in file:
            components = line.split()
            # Extract the key (string after the two numbers)
            key.append(components[2])
    return key


def get_notes_from_chords(chords):
    """
    Extracts the root notes from a list of chords.

    Args:
    ----------
        chords (list): A list of chords, where each chord is represented as a list of notes.

    Returns:
    ----------
        list: A list of notes extracted from the chords.
    """
    all_notes = []
    for song in chords:
        song_notes = []

        for note in song:
            song_notes.append(note[0])

        all_notes.append(song_notes)

    return all_notes
