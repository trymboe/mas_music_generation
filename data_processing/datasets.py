import note_seq as ns
import numpy as np
import torch
import math
import random
from torch.utils.data import Dataset
import tensorflow_datasets as tfds

from .utils import split_range, create_vocab, get_bucket_number, LMOrderedIterator

from note_seq.sequences_lib import augment_note_sequence, quantize_note_sequence

from config import (
    SEQUENCE_LENGTH_BASS,
    NOTE_TO_INT,
    SEQUENCE_LENGTH_CHORD,
    CHORD_TO_INT,
    SEQUENCE_LENGHT_MELODY,
)


class Bass_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_BASS
        self.notes_data, self.durations_data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        notes_data, durations_data, labels = [], [], []
        for song in songs:
            for i in range(len(song) - self.sequence_length):
                seq = song[i : i + self.sequence_length]

                # Extract note and duration sequences separately
                note_seq = [NOTE_TO_INT[note_duration[0]] for note_duration in seq]
                duration_seq = [note_duration[1] for note_duration in seq]

                label_note = NOTE_TO_INT[song[i + self.sequence_length][0]]
                label_duration = song[i + self.sequence_length][1]

                notes_data.append(note_seq)
                durations_data.append(duration_seq)
                labels.append((label_note, label_duration))

        return (
            torch.tensor(notes_data, dtype=torch.int64),
            torch.tensor(durations_data, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __len__(self):
        return len(self.notes_data)

    def __getitem__(self, idx):
        return self.notes_data[idx], self.durations_data[idx], self.labels[idx]


class Chord_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_CHORD
        self.data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        data, labels = [], []
        for song in songs:
            for i in range(len(song) - (self.sequence_length + 1)):
                seq = song[i : i + self.sequence_length + 1]
                # Convert to pairs
                chord = [
                    (
                        NOTE_TO_INT[pair[0]],
                        CHORD_TO_INT[pair[1]],
                        int(pair[2][0]),
                        pair[2][1],
                    )
                    for pair in seq
                ]
                chord.pop(-1)

                data.append(chord)
                labels.append([CHORD_TO_INT[seq[-1][1]]])
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
        except TypeError as e:
            print(f"Error for index {idx}: {self.data[idx]}, {self.labels[idx]}")
            raise e


class Chord_Dataset_Bass(Chord_Dataset):
    def __init__(self, songs):
        super().__init__(songs)

    def _process_songs(self, songs):
        data, labels = [], []

        for song in songs:
            for i in range(len(song[0]) - (self.sequence_length + 1)):
                seq_chord = song[0][i : i + self.sequence_length + 1]
                seq_duration = song[1][i : i + self.sequence_length + 1]
                pairs = []
                for j in range(len(seq_chord) - 1):
                    full_chord = self.get_full_chord(seq_chord[j][0], seq_chord[j][1])
                    pairs.append([full_chord, seq_duration[j]])

                data.append(pairs)
                full_chord = self.get_full_chord(seq_chord[-1][0], seq_chord[-1][1])
                labels.append([full_chord, seq_duration[-1]])
        return data, labels

    def get_full_chord(self, root, chord):
        return len(CHORD_TO_INT) * NOTE_TO_INT[root] + CHORD_TO_INT[chord]

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class Drum_Dataset:
    """
    Dataset to handle data in pipeline

    This class together with related functions and classes are a part of the bumblebeat project
    bumblebeat https://github.com/thomasgnuttall/bumblebeat/tree/master
    """

    def __init__(
        self,
        data_dir,
        dataset_name,
        pitch_classes,
        time_steps_vocab,
        processing_conf,
        n_velocity_buckets=10,
        min_velocity=0,
        max_velocity=127,
        augment_stretch=True,
        shuffle=True,
    ):
        """
        Documentation here baby
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.pitch_classes = pitch_classes
        self.time_steps_vocab = time_steps_vocab
        self.n_velocity_buckets = n_velocity_buckets
        self.processing_conf = processing_conf
        self.augment_stretch = True
        self.shuffle = True

        self.velocity_buckets = split_range(
            min_velocity, max_velocity, n_velocity_buckets
        )
        self.pitch_class_map = self._classes_to_map(self.pitch_classes)
        self.n_instruments = len(set(self.pitch_class_map.values()))

        print(
            f"Generating vocab of {self.n_instruments} instruments and {n_velocity_buckets} velocity buckets"
        )
        self.vel_vocab = {
            i: i + len(time_steps_vocab) + 1 for i in range(n_velocity_buckets)
        }
        self.vocab, self.reverse_vocab = create_vocab(
            self.n_instruments,
            first_index=len(time_steps_vocab) + len(self.vel_vocab) + 1,
        )  # leave initial indices for time steps/velocity vocab

        self.vocab_size = (
            len(self.reverse_vocab) + len(time_steps_vocab) + len(self.vel_vocab) + 1
        )  # add 1 for <eos> token

        train_dataset = self.download_midi(dataset_name, tfds.Split.TRAIN)
        test_dataset = self.download_midi(dataset_name, tfds.Split.TEST)
        valid_dataset = self.download_midi(dataset_name, tfds.Split.VALIDATION)

        self.train_data = [x for x in train_dataset]
        self.test_data = [x for x in test_dataset]
        self.valid_data = [x for x in valid_dataset]

        self.train_data_beat = []
        self.train_data_fill = []
        self.test_data_beat = []
        self.test_data_fill = []
        self.val_data_beat = []
        self.val_data_fill = []

        for item in self.train_data:
            if item["type"] == 0:
                self.train_data_beat.append(item)
            elif item["type"] == 1:
                self.train_data_fill.append(item)

        for item in self.test_data:
            if item["type"] == 0:
                self.test_data_beat.append(item)
            elif item["type"] == 1:
                self.test_data_fill.append(item)

        for item in self.valid_data:
            if item["type"] == 0:
                self.val_data_beat.append(item)
            elif item["type"] == 1:
                self.val_data_fill.append(item)

        print("Processing dataset TRAIN...")
        self.train_all = self.process_dataset(self.train_data, conf=processing_conf)
        self.train_beat = self.process_dataset(
            self.train_data_beat, conf=processing_conf
        )
        self.train_fill = self.process_dataset(
            self.train_data_fill, conf=processing_conf
        )

        print("Processing dataset TEST...")
        self.test_all = self.process_dataset(self.test_data, conf=processing_conf)
        self.test_beat = self.process_dataset(self.test_data_beat, conf=processing_conf)
        self.test_fill = self.process_dataset(self.test_data_fill, conf=processing_conf)

        print("Processing dataset VALID...")
        self.valid_all = self.process_dataset(self.valid_data, conf=processing_conf)
        self.valid_beat = self.process_dataset(self.val_data_beat, conf=processing_conf)
        self.valid_fill = self.process_dataset(self.val_data_fill, conf=processing_conf)

    def download_midi(self, dataset_name, dataset_split):
        print(f"Downloading midi data: {dataset_name}, split: {dataset_split}")
        dataset = tfds.as_numpy(
            tfds.load(name=dataset_name, split=dataset_split, try_gcs=True)
        )
        return dataset

    def process_dataset(self, dataset, conf):
        """
        Augment, transform and tokenize each sample in <dataset>
        Return: list of tokenised sequences
        """
        # Abstract to ARGS at some point
        quantize = conf["quantize"]
        steps_per_quarter = conf["steps_per_quarter"]
        filter_4_4 = conf["filter_4_4"]  # maybe we dont want this?

        # To midi note sequence using magent
        dev_sequences = [
            ns.midi_to_note_sequence(features["midi"]) for features in dataset
        ]

        if self.augment_stretch:
            augmented = self._augment_stretch(dev_sequences)
            # Tripling the total number of sequences
            dev_sequences = dev_sequences + augmented

        if quantize:
            dev_sequences = [
                self._quantize(d, steps_per_quarter) for d in dev_sequences
            ]

        # Filter out those that are not in 4/4 and do not have any notes
        dev_sequences = [
            s
            for s in dev_sequences
            if self._is_4_4(s)
            and len(s.notes) > 0
            and s.notes[-1].quantized_end_step
            > ns.steps_per_bar_in_quantized_sequence(s)
        ]

        # note sequence -> [(pitch, vel_bucket, start timestep)]
        tokens = [self._tokenize(d, steps_per_quarter, quantize) for d in dev_sequences]

        if self.shuffle:
            np.random.shuffle(tokens)

        stream = self._join_token_list(tokens, n=1)

        return torch.tensor(stream)

    def _join_token_list(self, tokens, n=5):
        """
        Join list of lists, <tokens>
        In new list each previous list is separated by <n> instances
        of the highest token value (token assigned for placeholding)
        """
        pad = 0
        to_join = [t + n * [pad] for t in tokens]
        return [pad] + [y for x in to_join for y in x]

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            data_iter = LMOrderedIterator(self.train_beat, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid_beat if split == "valid" else self.test_beat
            data_iter = LMOrderedIterator(data, *args, **kwargs)

        return data_iter

    def _augment_stretch(self, note_sequences):
        """
        Two stretchings for each sequence in <note_sequence>:
          - faster by 1%-10%
          - slower by 1%-10%
        These are returned as new sequences
        """
        augmented_slow = [
            augment_note_sequence(x, 1.01, 1.1, 0, 0) for x in note_sequences
        ]
        augmented_fast = [
            augment_note_sequence(x, 0.9, 0.99, 0, 0) for x in note_sequences
        ]
        return augmented_fast + augmented_slow

    def _quantize(self, s, steps_per_quarter=4):
        """
        Quantize a magenta Note Sequence object
        """
        return quantize_note_sequence(s, steps_per_quarter)

    def _is_4_4(self, s):
        """
        Return True if sample, <s> is in 4/4 timing, False otherwise
        """
        ts = s.time_signatures[0]
        return ts.numerator == 4 and ts.denominator == 4

    def _tokenize(self, note_sequence, steps_per_quarter, quantize):
        """
        from magenta <note_sequence> return list of
        tokens, filling silence with time tokens in
        self.time_steps_vocab

        - if <quantized> use quantized_start_step else use start_time
        - pitch is mapped using self.pitch_class_map
        - velocities are bucketted as per self.velocity_buckets
        """
        d = [
            (
                self.pitch_class_map[n.pitch],
                get_bucket_number(n.velocity, self.velocity_buckets),
                n.quantized_start_step if quantize else n.start_time,
            )
            for n in note_sequence.notes
            if n.pitch in self.pitch_class_map
        ]

        ticks_per_quarter = note_sequence.ticks_per_quarter
        qpm = note_sequence.tempos[0].qpm  # quarters per minute
        ticks_per_second = qpm * ticks_per_quarter / 60

        filled = self._tokenize_w_ticks(
            d,
            ticks_per_second,
            ticks_per_quarter,
            steps_per_quarter,
            quantize,
            self.vocab,
            self.vel_vocab,
            self.time_steps_vocab,
        )

        return filled

    def _tokenize_w_ticks(
        self,
        triples,
        ticks_per_second,
        ticks_per_quarter,
        steps_per_quarter,
        quantize,
        pitch_vocab,
        vel_vocab,
        time_steps_vocab,
    ):
        """
        From list of <triples> in the form:
            [(pitch class, bucketed velocity, start time (seconds/timesteps)),...]
        Return list of tokens matching pitch-velocity combination to tokens in <pitch_vocab>
        and filling silence with time tokens in <time_steps_vocab>

        Returns: list
            sequence of tokens from the pitch and time_steps vocabularies
        """

        # Initalise final tokenised sequence
        w_silence = []

        # Initalise counter to keep track of consecutive pitches
        # so that we can ensure they are appended to our
        # final tokenised sequence in numerical order
        consecutive_pitches = 0

        # index, (pitch, velocity, start time)
        for i, (x, y, z) in enumerate(triples):
            if i == 0:
                silence = z
            else:
                silence = z - triples[i - 1][2]  # z of previous element

            if quantize:
                ticks = silence * ticks_per_quarter / steps_per_quarter
            else:
                ticks = int(silence * ticks_per_second)

            if ticks:
                # make sure that any consecutive pitches in sequence
                # are in numerical order so as to enforce an ordering
                # rule for pitches that are commonly hit in unison
                w_silence[-consecutive_pitches:] = sorted(
                    w_silence[-consecutive_pitches:]
                )

                # Since silences are computed using time since last pitch class,
                # every iteration in this loop is a pitch class.
                # Hence we set consecutive pitch back to one
                # (representing the pitch of this iteration, added just outside of this if-clause)
                consecutive_pitches = 1

                # Number of ticks to list of time tokens
                time_tokens = self._convert_num_to_denominations(
                    ticks, time_steps_vocab
                )

                # Add time tokens to final sequence before we add our pitch class
                w_silence += [time_tokens]
            else:
                # Remember that every iteration is a pitch.
                # If <ticks> is 0 then this pitch occurs
                # simultaneously with the previous.
                # We sort these numerically before adding the
                # next stream of time tokens
                consecutive_pitches += 1

            # Triple to tokens...
            #   Discard time since we have handled that with time tokens.
            #   Look up pitch velocity combination for corresponding token.
            pitch_tok = pitch_vocab[x]  # [pitch class]
            vel_tok = vel_vocab[y]
            w_silence.append([pitch_tok, vel_tok])

        return [x for y in w_silence for x in y]

    def _convert_num_to_denominations(self, num, time_vocab):
        """
        Convert <num> into sequence of time tokens in (<time_vocab>).
        Tokens are selected so as to return a sequence of minimum length possible

        Params
        ======
        num: int
            Number of ticks to convert
        time_vocab: d¡ct
            {num_ticks: token}

        Return
        ======
        list:
            [tokens representing number]
        """
        # Start with largest demoninations
        denominations = list(sorted(time_vocab.keys(), reverse=True))
        seq = []
        for d in denominations:
            div = num / d
            # If <num> can be divided by this number
            # Create as many tokens as possible with this denomination
            if div > 1:
                floor = math.floor(div)
                seq += floor * [time_vocab[d]]
                num -= floor * d
        return seq

    def _roundup(self, x, n):
        """
        Roundup <x> to nearest multiple of <n>
        """
        return int(math.ceil(x / n)) * n

    def _classes_to_map(self, classes):
        class_map = {}
        for cls, pitches in enumerate(classes):
            for pitch in pitches:
                class_map[pitch] = cls
        return class_map


class Melody_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sequence_length = SEQUENCE_LENGHT_MELODY
        self._get_indices()

    def _get_indices(self):
        self.indices = []
        for song in self.data:
            for i in range(len(song) - (self.sequence_length * 2) + 1):
                self.indices.append((song, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        song, start_idx = self.indices[idx]
        # Return two sequences of length sequence_length. These corresponds to the input and target sequences
        return (
            song[start_idx : start_idx + self.sequence_length],
            song[
                start_idx + self.sequence_length : start_idx + self.sequence_length + 1
            ],
        )


class Melody_Dataset_Combined(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.sequence_length = SEQUENCE_LENGHT_MELODY
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        song, start_idx = self.indices[idx]
        return (
            song[start_idx : start_idx + self.sequence_length],
            song[
                start_idx + self.sequence_length : start_idx + self.sequence_length + 1
            ],
        )
