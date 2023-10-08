# Hyperparameters
SEQUENCE_LENGTH_CHORD = 4
ROOT_VOAB_SIZE_CHORD = 12  # Number of root notes
CHORD_VOCAB_SIZE_CHORD = 7  # Number of chords variations
EMBED_SIZE_CHORD = 64  # Embedding size
NHEAD_CHORD = 2  # Number of self-attention heads
NUM_LAYERS_CHORD = 2  # Number of transformer layers
NUM_EPOCHS_CHORD = 2  # Number of epochs
BATCH_SIZE_CHORD = 8
LEARNING_RATE_CHORD = 0.001
HIDDEN_SIZE_CHORD = 64

# Mappings
CHORD_TO_INT = {
    "maj": 0,
    "min": 1,
    "dim": 2,
    "aug": 3,
    "sus2": 4,
    "sus4": 5,
}

INT_TO_TRIAD = {
    0: [0, 4, 7],
    1: [0, 3, 7],
    2: [0, 3, 6],
    3: [0, 4, 8],
    4: [0, 2, 7],
    5: [0, 5, 7],
}