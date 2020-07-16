import random

import tensorflow as tf
from tensorflow_serving.apis import input_pb2
from rvranking.globalVars import _FAKE_ELWC

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_context_examples(path, samples):
    def serialize_example_fake(relevance, rvfeatures):
        """
        fake -> same number in context and relevant example
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'rv_tokens': _int64_list_feature(rvfeatures),  # _RV_FEATURE
            'relevance': _int64_feature(relevance),  # _LABEL_FEATURE
        }
        # Create a Features message using tf.train.Example.

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example  # .SerializeToString()

    def serialize_context_fake(contfeatures):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'event_tokens': _int64_list_feature(contfeatures),
        }
        # Create a Features message using tf.train.Example.

        context = tf.train.Example(features=tf.train.Features(feature=feature))
        return context  # .SerializeToString()

    def serialize_example(rv):
        """
        Creates a tf.Example message ready to be written to a file.
        concententate: 'rv.feat + rv.tline'
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        if _FAKE_ELWC:
            feature = {
                'rv_tokens': _int64_list_feature(rv.features_fake()),  # _RV_FEATURE
                'relevance': _int64_feature(rv.relevance),  # _LABEL_FEATURE
            }
        else:
            feature = {
                'rv_tokens': _int64_list_feature(rv.features()),  # _RV_FEATURE
                'relevance': _int64_feature(rv.relevance),  # _LABEL_FEATURE
            }
        # Create a Features message using tf.train.Example.

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example  # .SerializeToString()

    def serialize_context(contfeatures):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'event_tokens': _int64_list_feature(contfeatures),
        }
        # Create a Features message using tf.train.Example.

        context = tf.train.Example(features=tf.train.Features(feature=feature))
        return context  # .SerializeToString()

    elwc_list = []

    for s in samples:

        rvli = s.rvli
        example_list = []
        if _FAKE_ELWC:
            #1 is relevant rest is not
            #example = serialize_example_fake(1, [1])
            #example_list.append(example)
            for i in range(1, 2):
                #relev = random.randint(0, 1)
                if i == 1:
                    relev = 1
                else:
                    relev = 0
                example = serialize_example_fake(relev, [i])
                example_list.append(example)
            cont_feature = random.randint(1, 2)
            #cont_feature = 1
            context = serialize_context_fake([cont_feature])

        else:
            for rv in rvli:
                example = serialize_example(rv)
                example_list.append(example)
            # context = serialize_context(s.features)
            context = serialize_context(s.features())

        ELWC = input_pb2.ExampleListWithContext()
        ELWC.context.CopyFrom(context)
        for example in example_list:
            example_features = ELWC.examples.add()
            example_features.CopyFrom(example)
        elwc_list.append(ELWC)

    file_path = path
    with tf.io.TFRecordWriter(file_path) as writer:

        for elwc in elwc_list:  # [:2]:
            # print(elwc)
            writer.write(elwc.SerializeToString())