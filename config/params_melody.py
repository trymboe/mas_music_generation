# Dataset parameters
PITCH_VECTOR_SIZE = 36  # Number of pitches that can be represented
SEQUENCE_LENGHT_MELODY = 16
CHORD_SIZE_MELODY = 72
DURATION_SIZE_MELODY = 16
PITCH_SIZE_MELODY = PITCH_VECTOR_SIZE + 1

# Training parameters
NUM_EPOCHS_MELODY = 20
HIDDEN_SIZE_LSTM_MELODY = 256
ALPHA1_MELODY = 0.6
ALPHA2_MELODY = 0.4
LEARNING_RATE_MELODY = 0.0001
BATCH_SIZE_MELODY = 64
MAX_BATCHES_MELODY = 50  # float("inf")  # Max batches per epoch
WEIGHT_DECAY_MELODY = 0  # 0.0001

# Generation parameters
NOTE_TEMPERATURE_MELODY = 1
DURATION_TEMPERATURE_MELODY = 0.8
SCALE_MELODY = "major pentatonic"  # "major pentatonic",

TOTAL_INPUT_SIZE_MELODY = (
    PITCH_VECTOR_SIZE + 1 + DURATION_SIZE_MELODY + CHORD_SIZE_MELODY * 2 + 2
)  # sum of the sizes of all inputs

COMMENT_MELODY = "test"
MODEL_PATH_MELODY = (
    "models/melody/melody_model_"
    + str(NUM_EPOCHS_MELODY)
    + "_"
    + COMMENT_MELODY
    + ".pt"
)
TRAIN_DATASET_PATH_MELODY = "data/dataset/melody_dataset_train.pt"
TEST_DATASET_PATH_MELODY = "data/dataset/melody_dataset_test.pt"
VAL_DATASET_PATH_MELODY = "data/dataset/melody_dataset_val.pt"


FULL_CHORD_TO_INT = {
    "C:maj": 0,
    "C:min": 1,
    "C:dim": 2,
    "C:aug": 3,
    "C:sus2": 4,
    "C:sus4": 5,
    "C#:maj": 6,
    "C#:min": 7,
    "C#:dim": 8,
    "C#:aug": 9,
    "C#:sus2": 10,
    "C#:sus4": 11,
    "D:maj": 12,
    "D:min": 13,
    "D:dim": 14,
    "D:aug": 15,
    "D:sus2": 16,
    "D:sus4": 17,
    "D#:maj": 18,
    "D#:min": 19,
    "D#:dim": 20,
    "D#:aug": 21,
    "D#:sus2": 22,
    "D#:sus4": 23,
    "E:maj": 24,
    "E:min": 25,
    "E:dim": 26,
    "E:aug": 27,
    "E:sus2": 28,
    "E:sus4": 29,
    "F:maj": 30,
    "F:min": 31,
    "F:dim": 32,
    "F:aug": 33,
    "F:sus2": 34,
    "F:sus4": 35,
    "F#:maj": 36,
    "F#:min": 37,
    "F#:dim": 38,
    "F#:aug": 39,
    "F#:sus2": 40,
    "F#:sus4": 41,
    "G:maj": 42,
    "G:min": 43,
    "G:dim": 44,
    "G:aug": 45,
    "G:sus2": 46,
    "G:sus4": 47,
    "G#:maj": 48,
    "G#:min": 49,
    "G#:dim": 50,
    "G#:aug": 51,
    "G#:sus2": 52,
    "G#:sus4": 53,
    "A:maj": 54,
    "A:min": 55,
    "A:dim": 56,
    "A:aug": 57,
    "A:sus2": 58,
    "A:sus4": 59,
    "A#:maj": 60,
    "A#:min": 61,
    "A#:dim": 62,
    "A#:aug": 63,
    "A#:sus2": 64,
    "A#:sus4": 65,
    "B:maj": 66,
    "B:min": 67,
    "B:dim": 68,
    "B:aug": 69,
    "B:sus2": 70,
    "B:sus4": 71,
}
