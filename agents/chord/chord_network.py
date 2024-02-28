import torch.nn as nn
import torch

from config import DEVICE


class Chord_Network(nn.Module):
    """
    Abstract class for chord networks.
    """

    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        self.root_vocab_size = root_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.embed_size = embed_size
        self.nhead = nhead
        self.num_layers = num_layers

    def create_network(self):
        pass  # To be implemented in subclasses

    def forward(self, src):
        pass  # To be implemented in subclasses


class Chord_Coop_Network(Chord_Network):

    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        super().__init__(
            root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
        )

    def create_network(self):
        super(Chord_Network, self).__init__()

        self.root_embedding = nn.Embedding(self.root_vocab_size, self.embed_size)
        self.chord_embedding = nn.Embedding(self.chord_vocab_size, self.embed_size)

        self.transformer = nn.Transformer(
            2 * self.embed_size,
            self.nhead,
            self.num_layers,
            batch_first=True,
        )

        # Decoder for chord types
        self.decoder = nn.Linear(2 * self.embed_size, self.chord_vocab_size)

    def forward(self, src):
        # Split the input tensor into root notes and chord types
        src = src.long()
        roots, chords = src[:, :, 0], src[:, :, 1]

        roots = roots.to(DEVICE)
        chords = chords.to(DEVICE)

        # Embed root notes and chord types
        root_embed = self.root_embedding(roots)
        chord_embed = self.chord_embedding(chords)

        # Concatenate embeddings
        x = torch.cat((root_embed, chord_embed), dim=-1)

        # Pass through transformer
        x = self.transformer(x, x)

        # Decode to get chord type predictions
        x = self.decoder(x)

        if x.dim() == 3:
            return x[:, -1, :]
        return x


class Chord_Non_Coop_Network(Chord_Network):
    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        super().__init__(
            root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
        )
        self.create_network()

    def create_network(self):
        super(Chord_Network, self).__init__()

        self.chord_embedding = nn.Embedding(self.chord_vocab_size, self.embed_size)

        self.transformer = nn.Transformer(
            self.embed_size,
            self.nhead,
            self.num_layers,
            batch_first=True,
        )

        # Decoder for chord types
        self.decoder = nn.Linear(self.embed_size, self.chord_vocab_size)

    def forward(self, src):
        src = src.long()
        _, chords = src[:, :, 0], src[:, :, 1]

        chords = chords.to(DEVICE)
        x = self.chord_embedding(chords)
        x = self.transformer(x)
        x = self.decoder(x)

        if x.dim() == 3:
            return x[:, -1, :]
        return x


class Chord_LSTM_Network(nn.Module):
    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, hidden_size, num_layers
    ):
        super(Chord_LSTM_Network, self).__init__()

        # Root note embedding
        self.root_embedding = nn.Embedding(root_vocab_size, embed_size)

        # Chord type embedding
        self.chord_embedding = nn.Embedding(chord_vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=2
            * embed_size,  # Embedding size is doubled because we're concatenating
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder for chord types
        self.decoder = nn.Linear(hidden_size, chord_vocab_size)

    def forward(self, src):
        # Split the input tensor into root notes and chord types
        roots, chords = src[:, :, 0], src[:, :, 1]

        # Embed root notes and chord types
        root_embed = self.root_embedding(roots)
        chord_embed = self.chord_embedding(chords)

        # Concatenate embeddings
        x = torch.cat((root_embed, chord_embed), dim=-1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Decode to get chord type predictions
        x = self.decoder(lstm_out)

        if x.dim() == 3:
            return x[:, -1, :]
        return x
