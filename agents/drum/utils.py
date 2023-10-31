import functools
import os, shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import note_seq as ns

import numpy as np
import note_seq.protobuf.music_pb2 as music_pb2

from config import (
    INIT_DRUM,
    INIT_STD_DRUM,
    PROJ_INIT_STD_DRUM,
    INIT_RANGE_DRUM,
)


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, "a+") as f_log:
            f_log.write(s + "\n")


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print("Debug Mode : no experiment dir created")
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, "log.txt"))


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(
                torch.zeros(self.n_clusters, self.d_embed)
            )
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)

                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False):
        """
        hidden :: [len*bsz x d_proj]
        target :: [len*bsz]
        """

        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden,
                self.out_layers[0].weight,
                self.out_layers[0].bias,
                self.out_projs[0],
            )
            nll = (
                -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            )
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(
                        hidden_i, weight_i, bias_i, proj_i
                    )
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                        1, target_i[:, None]
                    ).squeeze(1)

                if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (
                (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
            )

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(
                self.dist, n_tries, replacement=True
            ).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = (
        torch.einsum("ijk,ijk->ij", [true_w, inputs]) + true_b - true_log_probs
    )
    sample_logits = (
        torch.einsum("lk,ijk->ijl", [sample_w, inputs]) + sample_b - samp_log_probs
    )
    sample_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, PROJ_INIT_STD_DRUM)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, PROJ_INIT_STD_DRUM)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, INIT_STD_DRUM)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)


def init_weight(weight):
    if INIT_DRUM == "uniform":
        nn.init.uniform_(weight, -PROJ_INIT_STD_DRUM, INIT_RANGE_DRUM)
    elif INIT_DRUM == "normal":
        nn.init.normal_(weight, 0.0, INIT_STD_DRUM)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def create_dir_if_not_exists(path):
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ""):
        os.makedirs(directory)


def tokens_to_note_sequence(
    tokens: list[int],
    pitch_vocab: dict[int, int],
    pitch_classes: list[list[int]],
    velocity_vocab: dict[int, int],
    time_vocab: dict[int, int],
    qpm: int,
    time_sig: tuple[int, int] = (4, 4),
    ticks_per_quarter: int = 480,
) -> music_pb2.NoteSequence:
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
                velocity=int(vel * 0.8),
                start_time=start_time,
                end_time=end_time,
                is_drum=True,
            )

    seq.ticks_per_quarter = ticks_per_quarter
    seq.tempos.add(qpm=qpm)
    seq.time_signatures.add(numerator=time_sig[0], denominator=time_sig[1])

    return seq


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


def load_model(path, device):
    """
    Load pretrained Transformer model for auto-regressive prediction
    """
    # Load the best saved model
    with open(path, "rb") as f:
        model = torch.load(f, map_location=device)

    model.backward_compatible()
    model = model.to(device)

    # Make sure model uses vanilla softmax
    if model.sample_softmax > 0:
        raise NotImplementedError()
    if model.crit.n_clusters != 0:
        raise NotImplementedError()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    return model


def generate_sequences(model, num, gen_len, mem_len, device, temp, topk=32):
    """
    Generate samples of len <gen_len> using pretrained transformer <model>

    Param
    =====
    model:
        Trained transformer model
    num: int
        Number of sequences to generate
    gen_len: int
        How many tokens to generate
    mem_len: int
        memory length of model
    device: torch device
        cpu or gpu
    temp: float
        Between 0 and 1.
        1 samples from model prob dist
        0 always takes most likely
    topk: n
        k for topk sampling

    Return
    ======
    Accompanying tokenised sequence (needs to be joined to original using join_sequences())
    """
    all_seqs = []
    # Generate sequences of specified length and number
    for i in range(num):
        sampler = TxlSimpleSampler(model, device, mem_len=mem_len)
        seq = [0]
        for _ in range(gen_len):
            token, _ = sampler.sample_next_token_updating_mem(
                seq[-1], temp=temp, topk=topk
            )
            seq.append(token)
        all_seqs.append(seq)

    return all_seqs


class TxlSimpleSampler:
    def __init__(self, model, device, tgt_len=1, mem_len=896, ext_len=0):
        if tgt_len != 1:
            raise ValueError()
        if ext_len != 0:
            raise ValueError()
        self.model = model
        self.model.eval()
        self.model.reset_length(1, ext_len, mem_len)
        self.device = device
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


def note_sequence_to_midi_file(note_sequence, path):
    """
    Save <note_sequence> to .midi file at <path>
    """
    create_dir_if_not_exists(path)
    ns.sequence_proto_to_midi_file(note_sequence, path)


def continue_sequence(model, seq, prime_len, gen_len, temp, topk, mem_len, device):
    """
    Continue/accompany sequence, <seq> sampling from <model>

    Param
    =====
    model:
        Trained transformer model
    seq: list
        Tokenised sequence to continue
    prime_len: int
        How many of thje most recent tokens in <seq> to
        use to prime the model
    gen_len: int
        How many tokens to generate
    temp: float
        Between 0 and 1.
        1 samples from model prob dist
        0 always takes most likely
    topk: n
        k for topk sampling
    mem_len: int
        memory length of model
    device: torch device
        cpu or gpu

    Return
    ======
    Original tokenised sequence continued by <gen_len> tokens

    """
    assert len(seq) >= prime_len + 1, "Insufficient tokens for prime length"

    sampler = TxlSimpleSampler(model, device, mem_len=mem_len)

    inp, sampler = prime_sampler(sampler, seq, prime_len)

    nll = 0.0
    cont = seq[:]
    for i in range(gen_len):
        gen, probs = sampler.sample_next_token_updating_mem(inp, temp=temp, topk=topk)
        p = probs[gen].cpu().item()
        nll += -np.log(p)
        inp = gen
        cont.append(gen)

    return cont


def prime_sampler(sampler, seq, prime_len):
    """
    Prime TXSimpleSampler with <seq> using <prime_len>
    """
    if prime_len > len(seq) - 2:
        prime_len = len(seq) - 2
    inp = 0
    nll = 0.0
    for i in range(prime_len):
        tar = seq[i + 1]
        _, probs = sampler.sample_next_token_updating_mem(inp, exclude_eos=False)
        p = probs[tar].cpu().item()
        nll += -np.log(p)
        inp = tar

    return inp, sampler
