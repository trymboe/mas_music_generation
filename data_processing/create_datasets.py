import torch
from torch.utils.data import Dataset, DataLoader

from config import SEQUENCE_LENGTH_BASS, NOTE_TO_INT


class Notes_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_BASS
        self.data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        data, labels = [], []
        for song in songs:
            for i in range(len(song) - self.sequence_length):
                seq = song[i : i + self.sequence_length]
                label = song[i + self.sequence_length]
                data.append([NOTE_TO_INT[note] for note in seq])
                labels.append(NOTE_TO_INT[label])
        return torch.tensor(data), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
