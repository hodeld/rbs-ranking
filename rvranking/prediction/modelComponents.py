import tensorflow as tf
import tensorflow_ranking as tfr

from rvranking.globalVars import _BATCH_SIZE, _LIST_SIZE
from rvranking.rankingComponents import context_feature_columns, example_feature_columns


def predict_input_fn(path):
    """
    similar to the input_fn used for training and evaluation,
    predict_input_fn reads in data in ELWC format and stored
    as TFRecords to generate features.
    We set number of epochs to be 1, so that the generator stops iterating
    when it reaches the end of the dataset.
    Also the datapoints are not shuffled while reading, so that the
    behavior of the predict() function is deterministic.
    """
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()))
    dataset = tfr.data.build_ranking_dataset(
        file_pattern=path,
        data_format=tfr.data.ELWC,
        batch_size=_BATCH_SIZE,
        list_size=_LIST_SIZE,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=False,
        num_epochs=1)
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return features


