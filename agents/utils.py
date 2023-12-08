import numpy as np
import torch
import torch.nn.functional as F

from note_seq.protobuf import music_pb2

from config import DEVICE


class TxlSimpleSampler:
    def __init__(self, model, tgt_len=1, mem_len=896, ext_len=0):
        if tgt_len != 1:
            raise ValueError()
        if ext_len != 0:
            raise ValueError()
        self.model = model
        self.model.eval()
        self.model.reset_length(1, ext_len, mem_len)
        self.device = DEVICE
        self.reset()

    def reset(self):
        self.mems = []
        self.generated = []

    @torch.no_grad()
    def sample_next_token_updating_mem(
        self, last_token=None, temp=1.0, topk=None, exclude_eos=True
    ):
        last_token = last_token if last_token is not None else 0

        # Ensure that user is always passing 0 on first call
        if len(self.generated) == 0:
            assert len(self.mems) == 0
            if last_token != 0:
                raise Exception()

        # Ensure that user isn't passing 0 after first call
        if last_token == 0 and len(self.generated) > 0:
            raise Exception()

        # Sanitize sampling params
        if temp < 0:
            raise ValueError()
        if topk is not None and topk < 1:
            raise ValueError()

        # Append last input token because we've officially selected it
        self.generated.append(last_token)

        # Create input array
        _inp = [last_token]
        _inp = np.array(_inp, dtype=np.int64)[:, np.newaxis]
        inp = torch.from_numpy(_inp).to(self.device)

        # Evaluate the model, saving its memory.
        ret = self.model.forward_generate(inp, *self.mems)
        all_logits, self.mems = ret[0], ret[1:]

        # Select last timestep, only batch item
        logits = all_logits[-1, 0]

        if exclude_eos:
            logits = logits[1:]

        # Handle temp 0 (argmax) case
        if temp == 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            # Apply temperature spec
            if temp != 1:
                logits /= temp

            # Compute softmax
            probs = F.softmax(logits, dim=-1)

        if exclude_eos:
            probs = F.pad(probs, [1, 0])

        # Select top-k if specified
        if topk is not None:
            _, top_idx = torch.topk(probs, topk)
            mask = torch.zeros_like(probs)
            mask[top_idx] = 1.0
            probs *= mask
            probs /= probs.sum()

        # Sample from probabilities
        token = torch.multinomial(probs, 1)
        token = int(token.item())

        return token, probs


def tokens_to_note_sequence(
    tokens,
    pitch_vocab,
    pitch_classes,
    velocity_vocab,
    time_vocab,
    qpm,
    time_sig=(4, 4),
    ticks_per_quarter=480,
):
    """
    Convert sequence of tokens to note_sequence

    Param
    =====
    tokens: sequence
        Sequence of tokens to convert to note_sequence
    pitch_vocab: dict
        Dict of token:(pitch,velocity)
    pitch_classes: list of lists
        list of lists indicating grouping of similar percussion instruments
        A random candidate will be taken from each group
    velocity_vocab: int
        mapping of velocity token: velocity bucket
    time_vocab: dict
        token:number of silence ticks
    qpm: int
        quarters per minute
    time_sig: tuple
        time signature, (numerator, denominator)
    ticks_per_quarter: int
        Ticks per quarter

    Return
    ======
    music_pb2.NoteSequence
    """
    # Token to mark separation between samples

    time_tokens = list(time_vocab.values())
    reverse_time_vocab = {v: k for k, v in time_vocab.items()}

    ticks_per_second = ticks_per_quarter * qpm / 60

    these_pitches = [np.random.choice(p) for p in pitch_classes]

    n_vel_buckets = len(velocity_vocab)

    seq = music_pb2.NoteSequence()
    silence_ticks = 0
    for i, t in enumerate(tokens):
        # Aggregate periods of silent ticks
        if t in time_tokens:
            silence_ticks += reverse_time_vocab[t]
        elif t in velocity_vocab:
            # Velocities are handled with pitches
            continue
        else:
            # Token: instrument
            p = pitch_vocab[t]
            pitch = these_pitches[p]
            # velocity always follows pitch
            if i == len(tokens) - 1:
                break

            try:
                vel_bucket = velocity_vocab[tokens[i + 1]]
            except KeyError:
                vel_bucket = velocity_vocab[11]
            vel = generate_velocity_in_bucket(vel_bucket, n_vel_buckets)

            start_time = silence_ticks / ticks_per_second
            if start_time == 0:
                start_time = 0.0000001
            end_time = start_time + 0.1  # TODO make this relative to qpm
            seq.notes.add(
                pitch=pitch,
                velocity=vel,
                start_time=start_time,
                end_time=end_time,
                is_drum=True,
            )

    seq.ticks_per_quarter = ticks_per_quarter
    seq.tempos.add(qpm=qpm)
    seq.time_signatures.add(numerator=time_sig[0], denominator=time_sig[1])

    return seq


def beats_to_seconds(beats: float, tempo: int) -> float:
    """
    Converts beats to seconds based on the given tempo.

    Args
    -------
        beats (float): The number of beats to convert.
        tempo (int): The tempo in beats per minute.

    Returns
    -------
        float: The equivalent number of seconds.

    """
    return round(beats * (60 / tempo), 2)


def seconds_to_beat(seconds: float, tempo: int) -> float:
    """
    Converts the given duration in seconds to the corresponding duration in beats,
    based on the provided tempo.

    Args
    -------
        seconds (float): The duration in seconds.
        tempo: The tempo in beats per minute.

    Returns
    -------
        float: The duration in beats.
    """
    return round(seconds * (tempo / 60), 2)


def select_with_preference(probs, preferred_indices):
    # Create a mask with zeros at all positions
    mask = torch.zeros_like(probs)

    # Set the mask to 1 at preferred indices
    mask[preferred_indices] = 1

    # Apply the mask to the probabilities
    masked_probs = probs * mask

    # Check if there is at least one preferred index with non-zero probability
    if torch.sum(masked_probs) > 0:
        # Normalize the probabilities
        masked_probs /= torch.sum(masked_probs)
        # Select using the modified probabilities
        return masked_probs

    else:
        # If all preferred indices have zero probability, fall back to the original distribution
        return probs


def generate_velocity_in_bucket(bucket, n_buckets):
    """
    Generate a random velocity in <bucket> for range of <n_buckets>
        (0 -> 127 possible)
    """
    srange = split_range(1, 127, n_buckets)

    low = srange[bucket]
    high = srange[bucket + 1]

    vel = np.random.uniform(low=low, high=high)
    return int(vel)


def split_range(r1, r2, n):
    """
    Split range <r1> - <r2> into <n> equal size buckets
    """
    step = (r2 - r1) / n
    return [r1 + step * i for i in range(n + 1)]
