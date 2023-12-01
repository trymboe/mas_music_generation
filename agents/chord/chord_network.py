import torch.nn as nn
import torch

from config import DEVICE


class Chord_Network(nn.Module):
    def __init__(
        self, root_vocab_size, chord_vocab_size, embed_size, nhead, num_layers
    ):
        super(Chord_Network, self).__init__()

        # Root note embedding
        self.root_embedding = nn.Embedding(root_vocab_size, embed_size)

        # Chord type embedding
        self.chord_embedding = nn.Embedding(chord_vocab_size, embed_size)

        # Transformer block
        self.transformer = nn.Transformer(
            2 * embed_size,  # Embedding size is doubled because we're concatenating
            nhead,
            num_layers,
            batch_first=True,
        )

        # Decoder for chord types
        self.decoder = nn.Linear(2 * embed_size, chord_vocab_size)

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
