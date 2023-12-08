# Mappings
DRUM_MAPPING = {
    "DEFAULT_DRUM_TYPE_PITCHES": [
        # kick drum
        [36, 35],
        # snare drum
        [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],
        # closed hi-hat
        [42, 44, 54, 68, 69, 70, 71, 73, 78, 80, 22],
        # open hi-hat
        [46, 67, 72, 74, 79, 81, 26],
        # low tom
        [45, 29, 41, 43, 61, 64, 84],
        # mid tom
        [48, 47, 60, 63, 77, 86, 87],
        # high tom
        [50, 30, 62, 76, 83],
        # crash cymbal
        [49, 52, 55, 57, 58],
        # ride cymbal
        [51, 53, 59, 82],
    ],
    "SIMPLIFIED_PITCHES": [[36], [38], [42], [46], [45], [48], [50], [49], [51]],
}

TIME_STEPS_VOCAB = {1: 1, 10: 2, 100: 3, 1000: 4, 10000: 5}

DRUM_STYLES = {
    "afrobeat": 0,
    "afrocuban": 1,
    "blues": 2,
    "country": 3,
    "dance": 4,
    "funk": 5,
    "gospel": 6,
    "highlife": 7,
    "hiphop": 8,
    "jazz": 9,
    "latin": 10,
    "middleeastern": 11,
    "neworleans": 12,
    "pop": 13,
    "punk": 14,
    "reggae": 15,
    "rock": 16,
    "soul": 17,
}


# Data processing
GENRE = "rock"
QUANTIZE = True
STEPS_PER_QUARTER = 4
WORK_DIR = "models/drum"

MODEL_PATH_DRUM = "models/drum/drum_model.pt"


# Hyperparameters
D_EMBED_DRUM = 16  # 512   # Dimention of embeded layer
D_MODEL_DRUM = 16  # 512  # Dimention of model
D_HEAD_DRUM = 2  # 64  # Dimention of each attention head
D_INNER_DRUM = 64  # 2048  # Dimension of inner hidden size in positionwise feed-forward
DROPOUT_DRUM = 0.2  # Dropout rate
DROPATT_DRUM = 0.1  # Attention dropout rate
NUM_LAYERS_DRUM = 2  # 12  # Number of layers
TRAIN_BATCH_SIZE_DRUM = 6  # Batch size for training
VOCAB_SIZE_DRUM = 25  # Number of tokens in the vocabulary
NUM_TOKENS_PREDICT_DRUM = 120  # Number of tokens to predict
EXTENDED_CONTEXT_LENGTH_DRUM = 0  # Number of tokens to extend the context by
TRAIN_SPLIT_DRUM = 0.8  # Training set size
TEST_SPLIT_DRUM = 0.1  # Test set size
VAL_SPLIT_DRUM = 0.1  # Validation set size
EVAL_BATCH_SIZE_DRUM = 10  # Batch size for evaluation
NHEAD_DRUM = 8  # Number of attention heads
NOT_TIED_DRUM = True  # untie r_w_bias and r_r_bias?
DIV_VAL_DRUM = 1  # Divide the embedding size by this val for each bin
PRE_LNORM = True  # Apply LayerNorm to the input instead of the output
MEM_LEN = 16  # 512  # Number of steps to cache
SAME_LENGTH = True  # Same length attention
ATTN_TYPE = 0  # Attention type. 0 for Bumblebeat, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
CLAMP_LEN = -1  # Clamp length
SAMPLE_SOFTMAX = -1  # Number of samples in sampled softmax
VARLEN = False  # Use varable length

N_ALL_PARAMS = None  # Total number of parameters
N_NONEMB_PARAMS = None  # Number of non-embedding parameters

# Optimizers
ETA_MIN_DRUM = 0  # Minimum learning rate
LEARNING_RATE_DRUM = 0.00001  # Maximum learning rate
CLIP_DRUM = 0.25  # Gradient clipping value
WARMUP_STEPS_DRUM = 0  # Number of steps for linear warmup

## Parameter Initialization
INIT_DRUM = "normal"  # ["normal", "uniform"],
INIT_STD_DRUM = 0.02  # Initialization std when init is normal.
PROJ_INIT_STD_DRUM = 0.01  # Initialization std for embedding projection.
INIT_RANGE_DRUM = 0.1  # Initialization std when init is uniform.
EMB_INIT_DRUM = "normal"
EMB_INIT_RANGE_DRUM = 0.01  # Initialization std when init is uniform.


## Training ##
MAX_STEP_DRUM = 10000  # Upper epoch limit
ITERATIONS_DRUM = 200  # Number of iterations per repeat loop
SAVE_STEPS_DRUM = 4000  # Number of steps for model checkpointing

## Evaluation config ##
DO_TEST_DRUM = False  # Run on the test set
MAX_EVAL_STEPS_DRUM = -1  # Set -1 to turn off. Only used in test mode
DO_EVAL_ONLY_DRUM = False  # Run evaluation only
START_EVAL_STEPS_DRUM = 10000  # Which checkpoint to start with in `do_eval_only` mode
EVAL_SPLIT_DRUM = "valid"  # Which data split to evaluate


# Logging
LOG_INTERVAL_DRUM = 200  # Report interval
EVAL_INTERVAL_DRUM = 1000  # 4000  # Evaluation interval

DEBUG = False  # Debug mode
