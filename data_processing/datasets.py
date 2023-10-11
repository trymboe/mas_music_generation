from note_seq.sequences_lib import augment_note_sequence
import note_seq as ns
import numpy as np
import torch
import math
import pretty_midi as pm
from torch.utils.data import Dataset

from .utils import split_range, create_vocab, get_bucket_number

from config import (
    SEQUENCE_LENGTH_BASS,
    NOTE_TO_INT,
    SEQUENCE_LENGTH_CHORD,
    CHORD_TO_INT,
    QUANTIZE,
    STEPS_PER_QUARTER,
    TRAIN_SPLIT_DRUM,
    TEST_SPLIT_DRUM,
    VAL_SPLIT_DRUM,
)


class Notes_Dataset(Dataset):
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


class Chords_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_CHORD
        self.data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        data, labels = [], []
        for song in songs:
            for i in range(len(song) - self.sequence_length):
                seq = song[i : i + self.sequence_length]

                # Convert to pairs
                try:
                    pairs = [
                        (NOTE_TO_INT[pair[0]], CHORD_TO_INT[pair[1]]) for pair in seq
                    ]
                except KeyError:
                    continue

                # Replace the chord type of the last pair with a placeholder (6)
                pairs[-1] = (pairs[-1][0], 6)

                data.append(pairs)
                labels.append(CHORD_TO_INT[seq[-1][1]])

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return torch.tensor(self.data[idx], dtype=torch.int64), torch.tensor(
                self.labels[idx], dtype=torch.int64
            )
        except TypeError as e:
            print(f"Error for index {idx}: {self.data[idx]}, {self.labels[idx]}")
            raise e


class Drum_Dataset(Dataset):
    def __init__(
        self,
        midi_paths: list[str],
        pitch_classes: list[list[int]],
        time_steps_vocab: dict[int:int],
        n_velocity_buckets=10,
        min_velocity=0,
        max_velocity=127,
    ):
        """
        Args:
            midi_paths (list): List of paths to MIDI files.
            pitch_classes (object): Information about pitch classes.
            time_steps_vocab (object): Vocabulary for time steps.
            processing_conf (dict): Configuration for data processing.
            *args, **kwargs: Additional arguments.
        """
        self.midi_paths: list[str] = midi_paths
        self.pitch_classes: list[list[int]] = pitch_classes
        self.time_steps_vocab: dict[int:int] = time_steps_vocab
        self.n_velocity_buckets: int = n_velocity_buckets
        self.augment_stretch: bool = True
        self.shuffle: bool = True

        self.velocity_buckets = split_range(
            min_velocity, max_velocity, n_velocity_buckets
        )
        self.pitch_class_map = self._classes_to_map(self.pitch_classes)
        self.n_instruments = len(set(self.pitch_class_map.values()))

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

        train_split = math.floor(len(midi_paths) * TRAIN_SPLIT_DRUM)
        test_split = math.floor(len(midi_paths) * TEST_SPLIT_DRUM)
        # The rest of the data is used for validation

        self.train_dataset = self._get_midi_file(midi_paths=midi_paths[:train_split])
        self.test_dataset = self._get_midi_file(
            midi_paths=midi_paths[train_split : train_split + test_split]
        )
        self.val_dataset = self._get_midi_file(
            midi_paths=midi_paths[train_split + test_split :]
        )

        self.train = self.process_dataset(self.train_dataset)
        self.test = self.process_dataset(self.test_dataset)
        self.valid = self.process_dataset(self.val_dataset)

    def process_dataset(self, dataset):
        """
        Augment, transform and tokenize each sample in <dataset>
        Return: list of tokenised sequences
        """
        # Abstract to ARGS at some point
        quantize = QUANTIZE
        steps_per_quarter = STEPS_PER_QUARTER

        # To midi note sequence using magenta
        dev_sequences = [ns.midi_to_note_sequence(track) for track in dataset]

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
            if len(s.notes) > 0
            and s.notes[-1].quantized_end_step
            > ns.steps_per_bar_in_quantized_sequence(s)
        ]

        # note sequence -> [(pitch, vel_bucket, start timestep)]
        tokens = [self._tokenize(d, steps_per_quarter, quantize) for d in dev_sequences]

        if self.shuffle:
            np.random.shuffle(tokens)

        stream = self._join_token_list(tokens, n=1)

        return torch.tensor(stream)

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

    def __len__(self):
        """Return the total number of data samples"""
        return len(self.midi_paths)

    def __getitem__(self, idx):
        # Load and process a single MIDI file based on the index (idx)
        # midi_path = self.midi_paths[idx]
        # Load MIDI data
        # Process MIDI data (consider creating a separate method for processing)
        # Return processed data as a tensor

        # Example (note: actual implementation will depend on the exact processing steps)
        midi_path = self.midi_paths[idx]
        raw_data = self.load_midi(midi_path)
        processed_data = self.process_midi_data(raw_data)
        return torch.tensor(processed_data)

    def _join_token_list(self, tokens, n=5):
        """
        Join list of lists, <tokens>
        In new list each previous list is separated by <n> instances
        of the highest token value (token assigned for placeholding)
        """
        pad = 0
        to_join = [t + n * [pad] for t in tokens]
        return [pad] + [y for x in to_join for y in x]

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

    def _classes_to_map(self, classes):
        """Creates a map from all pitches to the class of the pitch.

        Args:
            classes (list): All pitches

        Returns:
            dict: Mapping from all pitch to class of pitch
        """
        class_map = {}
        for cls, pitches in enumerate(classes):
            for pitch in pitches:
                class_map[pitch] = cls
        return class_map

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            data_iter = LMOrderedIterator(self.train, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            data_iter = LMOrderedIterator(data, *args, **kwargs)

        return data_iter

    def _get_midi_file(self, midi_paths):
        """Take a list of midi file paths and return a list of PrettyMIDI objects.

        Args:
            midi_paths (list[str]): list of midi file paths

        Returns:
            list[PrettyMidi]: list of PrettyMIDI objects
        """
        midi_files = []
        for path in midi_paths:
            midi_files.append(pm.PrettyMIDI(path))
        return midi_files

    def _quantize(self, s, steps_per_quarter=4):
        """
        Quantize a magenta Note Sequence object
        """
        return ns.quantize_note_sequence(s, steps_per_quarter)


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()
