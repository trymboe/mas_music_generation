import torch.nn as nn
import torch

from config import (
    DEVICE,
    DURATION_VOCAB_SIZE_BASS,
    TOTAL_CHORD_INPUT_SIZE,
    ROOT_VOCAB_SIZE_CHORD,
    CHORD_VOCAB_SIZE_CHORD,
    EMBED_SIZE_CHORD,
    HIDDEN_SIZE_CHORD,
    NUM_LAYERS_CHORD,
)


class Chord_Network(nn.Module):
    """
    Chord network model. This version is cooperative, meaning that it takes the root note from the bass agent as input.
    """

    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        """
        Initializes the ChordNetwork class.

        Args:
            root_vocab_size (int): The size of the root vocabulary.
            chord_vocab_size (int): The size of the chord vocabulary.
            embed_size (int): The size of the embedding.
            nhead (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
        """
        super().__init__()
        self.root_vocab_size = root_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.embed_size = embed_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.create_chord_network()

    def __str__(self) -> str:
        return "coop"

    def create_chord_network(self):
        """
        Creates the chord network model.

        This method initializes the embedding layers for the root and chord vocabularies,
        as well as the transformer and decoder layers of the network.

        Args:
            self (ChordNetwork): The ChordNetwork instance.

        Returns:
            None
        """
        self.root_embedding = nn.Embedding(self.root_vocab_size, self.embed_size)
        self.chord_embedding = nn.Embedding(self.chord_vocab_size, self.embed_size)

        self.transformer = nn.Transformer(
            2 * self.embed_size,
            self.nhead,
            self.num_layers,
            batch_first=True,
        )

        self.decoder = nn.Linear(2 * self.embed_size, self.chord_vocab_size)

    def forward(self, src):
        """
        Forward pass of the chord network.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, num_classes).
        """
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


class Chord_Network_Non_Coop(Chord_Network):
    """
    Chord network model. This version is non cooperative, meaning that it does not take the root note from the bass agent as input.
    """

    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        super().__init__(
            root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
        )

    def __str__(self) -> str:
        return "non_coop"

    def create_chord_network(self):
        self.chord_embedding = nn.Embedding(self.chord_vocab_size, self.embed_size)

        self.transformer = nn.Transformer(
            self.embed_size,
            self.nhead,
            self.num_layers,
            batch_first=True,
        )

        self.decoder = nn.Linear(self.embed_size, self.chord_vocab_size)

    def forward(self, src):

        src = src.long()
        _, chords = src[:, :, 0], src[:, :, 1]

        chords = chords.to(DEVICE)

        x = self.chord_embedding(chords)
        x = self.transformer(x, x)
        x = self.decoder(x)

        if x.dim() == 3:
            return x[:, -1, :]
        return x


class Chord_Network_Full(Chord_Network):
    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        super().__init__(
            root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
        )

    def __str__(self) -> str:
        return "full"

    def create_chord_network(self):
        self.duration_embedding = nn.Embedding(
            DURATION_VOCAB_SIZE_BASS, self.embed_size
        )
        self.root_embedding = nn.Embedding(ROOT_VOCAB_SIZE_CHORD, self.embed_size)
        self.chord_embedding = nn.Embedding(EMBED_SIZE_CHORD, self.embed_size)
        self.duration_embedding = nn.Embedding(EMBED_SIZE_CHORD, self.embed_size)

        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE_CHORD * 3,
            hidden_size=HIDDEN_SIZE_CHORD,
            num_layers=NUM_LAYERS_CHORD * 2,
            batch_first=True,
            bidirectional=True,
        )

        self.FC_root = nn.Linear(HIDDEN_SIZE_CHORD * 2, ROOT_VOCAB_SIZE_CHORD)
        self.FC_chord = nn.Linear(HIDDEN_SIZE_CHORD * 2, CHORD_VOCAB_SIZE_CHORD)
        self.FC_duration = nn.Linear(HIDDEN_SIZE_CHORD * 2, DURATION_VOCAB_SIZE_BASS)

    def forward(self, root, duration, chord):
        duration = duration.to(DEVICE)
        chord = chord.to(DEVICE)
        root = root.to(DEVICE)

        # Embed root notes and chord types
        duration_embed = self.duration_embedding(duration)
        root_embed = self.root_embedding(root)
        chord_embed = self.chord_embedding(chord)

        # Concatenate embeddings
        x = torch.cat((duration_embed, root_embed, chord_embed), dim=-1)

        # Pass through lstm
        x, _ = self.lstm(x)

        # Decode to get chord type predictions
        x_root = self.FC_root(x)
        x_chord = self.FC_chord(x)
        x_duration = self.FC_duration(x)

        return x_root[:, 1, :], x_chord[:, -1, :], x_duration[:, -1, :]


#################################################################
# LSTM Model
#################################################################


class Chord_LSTM_Network(nn.Module):
    def __init__(self):
        super(Chord_LSTM_Network, self).__init__()

        self.embed_size = EMBED_SIZE_CHORD

        self.root_embedding = nn.Embedding(ROOT_VOCAB_SIZE_CHORD, EMBED_SIZE_CHORD)
        self.chord_embedding = nn.Embedding(CHORD_VOCAB_SIZE_CHORD, EMBED_SIZE_CHORD)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE_CHORD * 2,
            hidden_size=HIDDEN_SIZE_CHORD,
            num_layers=NUM_LAYERS_CHORD,
            batch_first=True,
            bidirectional=True,
        )

        self.FC = nn.Linear(HIDDEN_SIZE_CHORD * 2, CHORD_VOCAB_SIZE_CHORD)

    def __str__(self) -> str:
        return "lstm"

    def forward(self, src):

        roots, chords = src[:, :, 0], src[:, :, 1]

        root_embeds = self.root_embedding(roots.long())  # Embedding expects LongTensor
        chord_embeds = self.chord_embedding(chords.long())

        x = torch.cat(
            (root_embeds, chord_embeds), dim=-1
        )  # Concat along the embedding dimension

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        x = self.FC(lstm_out)

        if x.dim() == 3:
            x = x[:, -1, :]
        return x
