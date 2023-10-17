import os
import glob

from config import params_drum, GENRE, TRAIN_BATCH_SIZE_DRUM, VOCAB_SIZE_DRUM
from .datasets import Drum_Dataset

from bumblebeat.bumblebeat.utils.data import load_yaml
from bumblebeat.bumblebeat.data import get_corpus


def get_drum_dataset():
    conf = load_yaml("bumblebeat/conf/train_conf.yaml")
    data_conf = conf["data"]

    pitch_classes_yaml = load_yaml("bumblebeat/conf/drum_pitches.yaml")
    pitch_classes = pitch_classes_yaml["DEFAULT_DRUM_TYPE_PITCHES"]
    time_steps_vocab = load_yaml("bumblebeat/conf/time_steps_vocab.yaml")

    dataset_name = data_conf["dataset"]
    data_dir = data_conf["data_dir"]

    train_batch_size = data_conf["per_host_train_bsz"]

    drum_dataset = get_corpus(
        dataset_name, data_dir, pitch_classes, time_steps_vocab, conf["processing"]
    )

    return drum_dataset


def get_midi_files(root_dir="data/groove", genre=None, signature="4-4"):
    """
    Get a list of paths to MIDI files with a specific time signature and optional genre.

    :param root_dir: str, base directory where the files are stored.
    :param genre: str, specific genre to filter by (e.g., 'rock', 'folk'). If None, no genre filtering is applied.
    :param signature: str, time signature to filter by (default is '4-4').
    :return: list of str, paths to the MIDI files.
    """
    # Navigate through each drummer's folder
    midi_files = []
    for drummer_folder in glob.glob(os.path.join(root_dir, "drummer*")):
        # Look for MIDI files in all subfolders of the current drummer's folder
        for midi_file in glob.glob(
            os.path.join(drummer_folder, "**/*.mid"), recursive=True
        ):
            # Extract the file name without the extension
            file_name = os.path.splitext(os.path.basename(midi_file))[0]

            # Check for the genre if specified
            if genre and genre not in file_name:
                continue

            # Check for the time signature
            if signature and file_name[-3:] != signature:
                continue

            midi_files.append(midi_file)
    return midi_files
