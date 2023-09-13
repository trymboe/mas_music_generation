import torch
import torch.nn as nn


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

    def forward(self, notes, durations):
        # Embed notes and durations
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
