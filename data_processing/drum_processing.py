import os
import pickle


import tensorflow as tf


from .datasets import Drum_Dataset

from .utils import load_yaml


def get_drum_dataset() -> Drum_Dataset:
    """
    Loads the drum dataset configuration, creates, and returns the drum dataset.

    Returns
    -------
    Drum_Dataset
        The loaded drum dataset based on the specified configuration.
    """

    conf: dict = load_yaml("config/bumblebeat/params.yaml")
    data_conf: dict = conf["data"]

    pitch_classes_yaml: dict[str, list[list[int]]] = load_yaml(
        "config/bumblebeat/drum_pitches.yaml"
    )
    pitch_classes: list[list[int]] = pitch_classes_yaml["DEFAULT_DRUM_TYPE_PITCHES"]

    time_steps_vocab: dict[int, int] = load_yaml(
        "config/bumblebeat/time_steps_vocab.yaml"
    )

    dataset_name: str = data_conf["dataset"]
    data_dir: str = data_conf["data_dir"]

    drum_dataset: Drum_Dataset = get_dataset(
        dataset_name, data_dir, pitch_classes, time_steps_vocab, conf["processing"]
    )

    return drum_dataset


def get_dataset(
    dataset_name: str,
    data_dir: str,
    pitch_classes: list[list[int]],
    time_steps_vocab: dict[int, int],
    processing_conf: dict,
) -> Drum_Dataset:
    """
    Load groove data into custom dataset class

    Parameters
    -------
    dataset_name: str
        Name of groove dataset to download from tensorflow datasets
    data_dir: str
        Path to store data in (dataset, tf records)
    pitch_classes: list
        list of lists indicating pitch class groupings
    time_steps_vocab: dict
        Dict of {number of ticks: token} for converting silence to tokens
    processing_conf: dict
        Dict of processing options

    Returns
    -------
    drum_dataset: Drum_Dataset

    """
    fn = os.path.join(data_dir, dataset_name, "cache.pkl")

    if tf.io.gfile.exists(fn):
        with open(fn, "rb") as fp:
            drum_dataset = pickle.load(fp)
    else:
        create_dir_if_not_exists(fn)

        print("Producing dataset...")
        drum_dataset = Drum_Dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            pitch_classes=pitch_classes,
            time_steps_vocab=time_steps_vocab,
            processing_conf=processing_conf,
        )

        print("Saving dataset...")
        with open(fn, "wb") as fp:
            pickle.dump(drum_dataset, fp, protocol=2)

    return drum_dataset


def create_dir_if_not_exists(path: str):
    """
    If the directory at <path> does not exist, create it empty

    Parameters
    ----
    path: str
        Path of where to create directory

    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ""):
        os.makedirs(directory)
