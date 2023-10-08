# Hyperparameters
SEQUENCE_LENGTH_BASS = 8
NOTE_VOCAB_SIZE_BASS = 12
DURATION_VOCAB_SIZE_BASS = 9
EMBED_SIZE_BASS = 64  # Embedding size
NHEAD_BASS = 2  # Number of self-attention heads
NUM_LAYERS_BASS = 2  # Number of transformer layers
NUM_EPOCHS_BASS = 2  # Number of epochs
BATCH_SIZE_BASS = 8
LEARNING_RATE_BASS = 0.001


# Mappings
NOTE_TO_INT = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
INT_TO_NOTE = {v: k for k, v in NOTE_TO_INT.items()}
