import torch
import torch.nn as nn

from config import (
    DEVICE,
    SEQUENCE_LENGTH_BASS,
    NOTE_VOCAB_SIZE_BASS,
    DURATION_VOCAB_SIZE_BASS,
    EMBED_SIZE_BASS,
    NUM_LAYERS_BASS,
    BATCH_SIZE_BASS,
    LEARNING_RATE_BASS,
    HIDDEN_SIZE_BASS,
)


class Bass_Network(nn.Module):
    def __init__(
        self, note_vocab_size, duration_vocab_size, embed_size, nhead, num_layers
    ):
        super(Bass_Network, self).__init__()

        # Separate embedding layers for notes and durations
        self.note_embedding = nn.Embedding(note_vocab_size, embed_size)
        self.duration_embedding = nn.Embedding(duration_vocab_size, embed_size)

        # Transformer block
        self.transformer = nn.Transformer(
            embed_size * 2,  # We will concatenate note and duration embeddings
            nhead,
            num_layers,
            batch_first=True,
        )

        # Separate decoders for notes and durations
        self.note_decoder = nn.Linear(embed_size * 2, note_vocab_size)
        self.duration_decoder = nn.Linear(embed_size * 2, duration_vocab_size)

    def __str__(self) -> str:
        return "transformer"

    def forward(self, notes, durations):
        # Embed notes and durations
        notes = notes.to(DEVICE)
        durations = durations.to(DEVICE)

        note_embed = self.note_embedding(notes)
        duration_embed = self.duration_embedding(durations)

        # Combine note and duration embeddings (you can also try adding them)
        combined_embed = torch.cat((note_embed, duration_embed), dim=-1)

        # Pass through the transformer
        x = self.transformer(combined_embed, combined_embed)

        # Decode the transformer's output into separate predictions for the next note and its duration
        note_output = self.note_decoder(x)
        duration_output = self.duration_decoder(x)

        if x.dim() == 3:
            note_output = note_output[:, -1, :]
            duration_output = duration_output[:, -1, :]

        return note_output, duration_output


class Bass_Network_LSTM(nn.Module):
    def __init__(self):
        super(Bass_Network_LSTM, self).__init__()
        self.embed_size = EMBED_SIZE_BASS

        self.note_embedding = nn.Embedding(NOTE_VOCAB_SIZE_BASS, EMBED_SIZE_BASS)
        self.duration_embedding = nn.Embedding(
            DURATION_VOCAB_SIZE_BASS, EMBED_SIZE_BASS
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE_BASS * 2,
            hidden_size=HIDDEN_SIZE_BASS,
            num_layers=NUM_LAYERS_BASS,
            batch_first=True,
            bidirectional=True,
        )

        self.FC_note = nn.Linear(HIDDEN_SIZE_BASS * 2, NOTE_VOCAB_SIZE_BASS)
        self.FC_duration = nn.Linear(HIDDEN_SIZE_BASS * 2, DURATION_VOCAB_SIZE_BASS)

    def __str__(self) -> str:
        return "lstm"

    def forward(self, notes, durations):

        note_embeds = self.note_embedding(notes.long())  # Embedding expects LongTensor
        durations_embeds = self.duration_embedding(durations.long())

        x = torch.cat(
            (note_embeds, durations_embeds), dim=-1
        )  # Concat along the embedding dimension

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        x_note = self.FC_note(lstm_out)
        x_duration = self.FC_duration(lstm_out)

        if x.dim() == 3:
            x_note = x_note[:, -1, :]
            x_duration = x_duration[:, -1, :]
        return x_note, x_duration
