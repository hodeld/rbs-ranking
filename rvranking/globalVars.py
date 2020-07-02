from rvranking.dataPrep import base_store_path

_FAKE = False
#_FAKE = True

#RV specific variables
_RV_FEATURE = 'rv_tokens'
_EVENT_FEATURE = 'event_tokens'
_EMBEDDING_DIMENSION = 20
RELEVANCE = 1

# needed for pre tests example

# The document relevance label.
_LABEL_FEATURE = "relevance"

_PADDING_LABEL = -1
#_BATCH_SIZE = 10

# The maximum number of rv's per event in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 10  #of rvs


# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

# Parameters to the scoring function. part 1
_BATCH_SIZE = 32

#following same as in example:

# Store the paths to files containing training and test instances.
#needed for write and read -> adapt on localhost
_BASE_TF_DATA_PATH = base_store_path + "/tfdata"
_TRAIN_DATA_PATH = _BASE_TF_DATA_PATH + "/train.tfrecords"
_TEST_DATA_PATH = _BASE_TF_DATA_PATH + "/test.tfrecords"

# Store the vocabulary path for query and document tokens. -> not needed
#_VOCAB_PATH = "/tmp/vocab.txt"

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function. part 2
#hidden layers: between input and output
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
#dropout rate:fraction of zeroed out values in the output layer
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1  # Pointwise scoring.

# Location of model directory and number of training steps.
_MODEL_DIR = base_store_path + "/tmp/ranking_model_dir"
# max train steps defines nr of epochs (if steps == data size -> 1 epoch)
#_NUM_TRAIN_STEPS = 15 * 1000
_NUM_TRAIN_STEPS = 5 * 1000

#_SAVE_CP_STEPS
#_SAVE_CHECKPOINT_STEPS = 1000
_SAVE_CHECKPOINT_STEPS = 1000

_NDGC_TOP_NRS = [1, 2, 5]