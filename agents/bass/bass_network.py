import torch
import torch.nn as nn


class Bass_Network(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers):
        super(Bass_Network, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Transformer block
        self.transformer = nn.Transformer(
            embed_size, nhead, num_layers, batch_first=True
        )

        # Decoder
        self.decoder = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer(x, x)
        x = self.decoder(x)
        if x.dim() == 3:
            return x[:, -1, :]
        return x
