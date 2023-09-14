import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    SEQUENCE_LENGTH_BASS,
    NOTE_TO_INT,
    SEQUENCE_LENGTH_CHORD,
    CHORD_TO_INT,
)


class Notes_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_BASS
        self.notes_data, self.durations_data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        notes_data, durations_data, labels = [], [], []
        for song in songs:
            for i in range(len(song) - self.sequence_length):
                seq = song[i : i + self.sequence_length]

                # Extract note and duration sequences separately
                note_seq = [NOTE_TO_INT[note_duration[0]] for note_duration in seq]
                duration_seq = [note_duration[1] for note_duration in seq]

                label_note = NOTE_TO_INT[song[i + self.sequence_length][0]]
                label_duration = song[i + self.sequence_length][1]

                notes_data.append(note_seq)
                durations_data.append(duration_seq)
                labels.append((label_note, label_duration))

        return (
            torch.tensor(notes_data, dtype=torch.int64),
            torch.tensor(durations_data, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __len__(self):
        return len(self.notes_data)

    def __getitem__(self, idx):
        return self.notes_data[idx], self.durations_data[idx], self.labels[idx]


class Chords_Dataset(Dataset):
    def __init__(self, songs):
        self.sequence_length = SEQUENCE_LENGTH_CHORD
        self.data, self.labels = self._process_songs(songs)

    def _process_songs(self, songs):
        data, labels = [], []
        for song in songs:
            for i in range(len(song) - self.sequence_length):
                seq = song[i : i + self.sequence_length]

                # Convert to pairs
                try:
                    pairs = [
                        (NOTE_TO_INT[pair[0]], CHORD_TO_INT[pair[1]]) for pair in seq
                    ]
                except KeyError:
                    continue

                # Replace the chord type of the last pair with a placeholder (6)
                pairs[-1] = (pairs[-1][0], 6)

                data.append(pairs)
                labels.append(CHORD_TO_INT[seq[-1][1]])

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return torch.tensor(self.data[idx], dtype=torch.int64), torch.tensor(
                self.labels[idx], dtype=torch.int64
            )
        except TypeError as e:
            print(f"Error for index {idx}: {self.data[idx]}, {self.labels[idx]}")
            raise e
