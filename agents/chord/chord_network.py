import torch.nn as nn
import torch

from config import DEVICE, TOTAL_CHORD_INPUT_SIZE, DURATION_VOCAB_SIZE_BASS


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
        self.chord_embedding = nn.Embedding(TOTAL_CHORD_INPUT_SIZE, self.embed_size)

        self.transformer = nn.Transformer(
            2 * self.embed_size,
            self.nhead,
            self.num_layers,
            batch_first=True,
        )

        self.decoder_duration = nn.Linear(2 * self.embed_size, DURATION_VOCAB_SIZE_BASS)
        self.decoder_chord = nn.Linear(2 * self.embed_size, TOTAL_CHORD_INPUT_SIZE)

    def forward(self, src):
        src = src.long()
        chords, duration = src[:, :, 0], src[:, :, 1]

        duration = duration.to(DEVICE)
        chords = chords.to(DEVICE)

        # Embed root notes and chord types
        duration_embed = self.duration_embedding(duration)
        chord_embed = self.chord_embedding(chords)

        # Concatenate embeddings
        x = torch.cat((duration_embed, chord_embed), dim=-1)

        # Pass through transformer
        x = self.transformer(x, x)

        # Decode to get chord type predictions
        x_chord = self.decoder_chord(x)
        x_duration = self.decoder_duration(x)

        # if x.dim() == 3:
        #     return x[:, -1, :]
        return x_chord[:, -1, :], x_duration[:, -1, :]


#################################################################
# LSTM Model
#################################################################


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
        super().create_chord_network()
        self.decoder

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
