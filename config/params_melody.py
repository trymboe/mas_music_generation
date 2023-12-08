# Dataset parameters
PITCH_VECTOR_SIZE = 36  # Number of pitches that can be represented
SEQUENCE_LENGHT_MELODY = 16
CHORD_SIZE_MELODY = 24
DURATION_SIZE_MELODY = 16
PITCH_SIZE_MELODY = PITCH_VECTOR_SIZE + 1
TIME_LEFT_ON_CHORD_SIZE_MELODY = 16
INPUT_SIZE_MELODY = (
    PITCH_SIZE_MELODY
    + DURATION_SIZE_MELODY
    + TIME_LEFT_ON_CHORD_SIZE_MELODY
    + CHORD_SIZE_MELODY * 2
)

# Training parameters
NUM_EPOCHS_MELODY = 150
HIDDEN_SIZE_LSTM_MELODY = 256
ALPHA1_MELODY = 0.6
ALPHA2_MELODY = 0.4
LEARNING_RATE_MELODY = 0.0005
BATCH_SIZE_MELODY = 64
MAX_BATCHES_MELODY = 20  # float("inf")  # Max batches per epoch
WEIGHT_DECAY_MELODY = 0.001
DROPOUT_MELODY = 0.5
NUM_LAYERS_LSTM_MELODY = 2

# Generation parameters
NOTE_TEMPERATURE_MELODY = 0.8
DURATION_TEMPERATURE_MELODY = 0.8
NO_PAUSE = False
SCALE_MELODY = "major pentatonic"  # "major pentatonic"  # "major scale"",
DURATION_PREFERENCES = [1, 3, 5, 7, 9, 11, 13, 15]  # False, [7, 11, 15]


TOTAL_INPUT_SIZE_MELODY = (
    PITCH_VECTOR_SIZE + 1 + DURATION_SIZE_MELODY + CHORD_SIZE_MELODY * 2 + 2
)  # sum of the sizes of all inputs

COMMENT_MELODY = "new2"
MODEL_PATH_MELODY = (
    "models/melody/melody_model_"
    + str(NUM_EPOCHS_MELODY)
    + "_"
    + COMMENT_MELODY
    + ".pt"
)

# MODEL_PATH_MELODY = "models/melody/checkpoint_no_reg2_15.pt"
MODEL_PATH_MELODY = "models/melody/melody_model_150_no_reg2.pt"

TRAIN_DATASET_PATH_MELODY = "data/dataset/melody_dataset_train.pt"
TEST_DATASET_PATH_MELODY = "data/dataset/melody_dataset_test.pt"
VAL_DATASET_PATH_MELODY = "data/dataset/melody_dataset_val.pt"

# Uses a dataset where train and valiadtion data comes from the same songs. There is no leakage
COMBINED = False
TRAIN_DATASET_COMBINED_PATH_MELODY = "data/dataset/melody_dataset_combined_train.pt"
VAL_DATASET_COMBINED_PATH_MELODY = "data/dataset/melody_dataset_combined_val.pt"


FULL_CHORD_TO_INT = {
    "C:maj": 0,
    "C:min": 1,
    "C#:maj": 2,
    "C#:min": 3,
    "D:maj": 4,
    "D:min": 5,
    "D#:maj": 6,
    "D#:min": 7,
    "E:maj": 8,
    "E:min": 9,
    "F:maj": 10,
    "F:min": 11,
    "F#:maj": 12,
    "F#:min": 13,
    "G:maj": 14,
    "G:min": 15,
    "G#:maj": 16,
    "G#:min": 17,
    "A:maj": 18,
    "A:min": 19,
    "A#:maj": 20,
    "A#:min": 21,
    "B:maj": 22,
    "B:min": 23,
}
