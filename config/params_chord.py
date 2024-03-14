# Hyperparameters
SEQUENCE_LENGTH_CHORD = 8
ROOT_VOAB_SIZE_CHORD = 12  # Number of root notes
CHORD_VOCAB_SIZE_CHORD = 7  # Number of chords variations
TOTAL_CHORD_INPUT_SIZE = ROOT_VOAB_SIZE_CHORD * (CHORD_VOCAB_SIZE_CHORD - 1) + 1
EMBED_SIZE_CHORD = 128  # Embedding size
NHEAD_CHORD = 4  # Number of self-attention heads
NUM_LAYERS_CHORD = 4  # Number of transformer layers
BATCH_SIZE_CHORD = 8
LEARNING_RATE_CHORD = 0.00001
HIDDEN_SIZE_CHORD = 64
WEIGHT_DECAY_CHORD = 0.001

NUM_EPOCHS_CHORD = 50  # Number of epochs
MAX_BATCHES_CHORD = (
    10000  # Max number of batches to train on per Epoch, for shorter training
)

# Playing parameters


MODEL_PATH_CHORD = "models/chord/chord_model_2.pt"
MODEL_NON_COOP_PATH_CHORD = "models/chord/chord_model_non_coop_2.pt"
MODEL_CHORD_BASS_PATH = "models/chord/chord_bass_model_2.pt"

TRAIN_DATASET_PATH_CHORD = "data/dataset/chord_dataset_train.pt"
TEST_DATASET_PATH_CHORD = "data/dataset/chord_dataset_test.pt"
VAL_DATASET_PATH_CHORD = "data/dataset/chord_dataset_val.pt"

TRAIN_DATASET_PATH_CHORD_BASS = "data/dataset/chord_bass_dataset_train.pt"
TEST_DATASET_PATH_CHORD_BASS = "data/dataset/chord_bass_dataset_test.pt"
VAL_DATASET_PATH_CHORD_BASS = "data/dataset/chord_bass_dataset_val.pt"

ARP_STYLE = 2  # Style of the arpegiator 0 for 16th notes, 1 for 12th note, 2 for 8th notes, 3 for full range 16th notes

# Mappings
CHORD_TO_INT = {
    "maj": 0,
    "min": 1,
    "dim": 2,
    "aug": 3,
    "sus2": 4,
    "sus4": 5,
}

INT_TO_CHORD = {v: k for k, v in CHORD_TO_INT.items()}

INT_TO_TRIAD = {
    0: [0, 4, 7],
    1: [0, 3, 7],
    2: [0, 3, 6],
    3: [0, 4, 8],
    4: [0, 2, 7],
    5: [0, 5, 7],
}
