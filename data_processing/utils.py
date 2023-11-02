import numpy as np
import os
import yaml


def get_timed_notes(
    notes: list[list[str]], beats: list[list[int]]
) -> list[list[tuple[str, int]]]:
    """
    Converts lists of notes and beats into a structured format of timed notes.

    Parameters
    ----------
    notes : list[list[str]]
        A list of lists where each inner list contains note representations as strings.
    beats : list[list[int]]
        A list of lists where each inner list contains beat information corresponding to the notes.

    Returns
    -------
    list[list[tuple[str, int]]]
        A list of lists where each inner list contains tuples of notes and their corresponding beats.
    """

    timed_notes: list[tuple[str, int]] = []

    for i in range(len(notes)):
        timed_notes.append([])
        for j in range(len(notes[i])):
            timed_notes[i].append((notes[i][j], beats[i][j]))

    return timed_notes


def split_range(r1, r2, n):
    """
    Split range <r1> - <r2> into <n> equal size buckets
    """
    step = (r2 - r1) / n
    return [r1 + step * i for i in range(n + 1)]


def create_vocab(n_instruments, first_index=16):
    """
    Create vocabulary of all possible instrument-velocity combinations.

    <first_index> dictates which index to start on, default 16 to allow for
    0 to be special token indicating end and start of sequence and 1-5 to
    represent time steps in time_steps_vocab.yaml and 10 velocity buckets

    returns: 2 x dict
        {instrument_index: token}
            ...for all instruments and tokens
        {index: instrument_index}
    """
    d = {i: i + first_index for i in range(n_instruments)}
    d_reverse = {v: k for k, v in d.items()}
    return d, d_reverse


def get_bucket_number(value, srange):
    """
    Return index of bucket that <value> falls into

    srange is a list of bucket divisions from split_range()
    """
    assert srange == (sorted(srange)), "srange must be sorted list"
    assert len(set(srange)) == len(srange), "srange buckets must be unique"
    assert value <= max(srange) and value >= min(
        srange
    ), "value is not in any srange bucket"

    for i in range(len(srange) - 1):
        if value <= srange[i + 1]:
            return i


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


def load_yaml(fname):
    """
    Load yaml at <path> to dictionary, d

    returns: dict
    """
    if not os.path.isfile(fname):
        return None

    with open(fname) as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    return conf
