# Hyperparameters
SEQUENCE_LENGTH_BASS = 8
NOTE_VOCAB_SIZE_BASS = 12
DURATION_VOCAB_SIZE_BASS = 9
EMBED_SIZE_BASS = 128  # Embedding size
NUM_LAYERS_BASS = 4  # Number of layers
BATCH_SIZE_BASS = 8
HIDDEN_SIZE_BASS = 128
LEARNING_RATE_BASS = 0.0001


NHEAD_BASS = 4  # Number of self-attention heads for transformer

NUM_EPOCHS_BASS = 100  # Number of epochs
MAX_BATCHES_BASS = (
    1000  # Max number of batches to train on per Epoch, for shorter training
)

MODEL_PATH_BASS = "models/bass/bass_model.pt"
MODEL_PATH_BASS_LSTM = "models/bass/bass_model_lstm.pt"
MODEL_PATH_BASS_LSTM_TEST = "models/bass/bass_model_lstm_test.pt"
TRAIN_DATASET_PATH_BASS = "data/dataset/bass_dataset_train.pt"
TEST_DATASET_PATH_BASS = "data/dataset/bass_dataset_test.pt"
VAL_DATASET_PATH_BASS = "data/dataset/bass_dataset_val.pt"

ALPHA1_BASS = 0.7
ALPHA2_BASS = 0.3

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
