from secrets import base_path
import os

#BASE PATH
#base_path = ''


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
_LIST_SIZE = 5  #of rvs


# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

# Parameters to the scoring function. part 1
_BATCH_SIZE = 10 # 32

#following same as in example:

# Store the paths to files containing training and test instances.
#needed for write and read -> adapt on localhost
_BASE_TF_DATA_PATH = base_path + "/tmp/tfdata"
_TRAIN_DATA_PATH = _BASE_TF_DATA_PATH + "/train.tfrecords"
_TEST_DATA_PATH = _BASE_TF_DATA_PATH + "/test.tfrecords"

# Store the vocabulary path for query and document tokens. -> not needed
#_VOCAB_PATH = "/tmp/vocab.txt"

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function. part 2
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1  # Pointwise scoring.

# Location of model directory and number of training steps.
_MODEL_DIR = base_path + "/tmp/ranking_model_dir"
_NUM_TRAIN_STEPS = 15 * 1000