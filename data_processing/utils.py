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
