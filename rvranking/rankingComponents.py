import tensorflow as tf
import tensorflow_ranking as tfr
from rvranking.dataPrep import RV_TLINE_LEN
from rvranking.globalVars import (_EMBEDDING_DIMENSION, _RV_FEATURE, _LABEL_FEATURE,
                                  _PADDING_LABEL, _BATCH_SIZE, _LIST_SIZE, _DROPOUT_RATE, _HIDDEN_LAYER_DIMS,
                                  _GROUP_SIZE,
                                  _RANK_TOP_NRS, _SHUFFLE_DATASET, _EVENT_FEATURES, _RV_FEATURES)


#GET_FEATURE
# input: sparse tensor


def get_feature_columns(keyname, num_buckets=1000, emb_dim=_EMBEDDING_DIMENSION):

    sparse_column = tf.feature_column.categorical_column_with_identity(
        key=keyname, num_buckets=num_buckets)
    # inputs and outputs should be in range [0, num_buckets]

    # indicator_column OR embedding_column but embedding gives floats
    ##dense_column = tf.feature_column.indicator_column(sparse_column)

    dense_column = tf.feature_column.embedding_column(
        sparse_column, emb_dim)

    return dense_column

_nr_evtypes = 30
def feat_evtype(kname):
    dense_col = get_feature_columns(kname, num_buckets=_nr_evtypes, )  # 30 evtype emb_dim=2
    return dense_col


def feat_rvff(kname):
    dense_col = get_feature_columns(kname, num_buckets=100, )  # 100 rvs emb_dim=2
    return dense_col


def feat_bool(kname):
    dense_col = get_feature_columns(kname, num_buckets=2, )  # 100 rvs emb_dim=2
    return dense_col


def feat_sex(kname):
    dense_col = get_feature_columns(kname, num_buckets=3, )  # 100 rvs emb_dim=2
    return dense_col


def feat_tline(kname):
    dense_col = get_feature_columns(kname, num_buckets=_nr_evtypes, emb_dim=20)
    return dense_col


get_feat_fn = {
    'evtype': feat_evtype,  # get_feature_columns, #
    #'rv_ff': feat_rvff,
    'gespever': feat_sex,
    'hwx': feat_bool,
    'uma': feat_bool,

    'rv_ff': feat_bool,
    'sex': feat_sex,
    'id': feat_rvff,
    'tline': feat_tline,
    'tline_binary': feat_tline
}


def context_feature_columns():
    """Returns context feature names to column definitions."""
    feature_dict = {}
    for fname in _EVENT_FEATURES:
        dense_column = get_feat_fn[fname](fname)
        feature_dict[fname] = dense_column
    return feature_dict


def example_feature_columns():
    """Returns example feature names to column definitions."""
    feature_dict = {}
    for fname in _RV_FEATURES:
        dense_column = get_feat_fn[fname](fname)
        feature_dict[fname] = dense_column
    return feature_dict


# INPUT_FN

#shuffle: true -> shuffle BEFORE Padding
def input_fn(path, num_epochs=None):  # num_epochs was: none
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
    label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)

    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])
    dataset = tfr.data.build_ranking_dataset(
        file_pattern=path,
        data_format=tfr.data.ELWC,
        batch_size=_BATCH_SIZE,
        list_size=_LIST_SIZE,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=_SHUFFLE_DATASET,
        num_epochs=num_epochs)
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)

    return features, label

#TRANSFORM_FN

#from here same as in example
def make_transform_fn():
    def _transform_fn(features, mode):
        """Defines transform_fn."""
        context_features, example_features = tfr.feature.encode_listwise_features(
            features=features,
            context_feature_columns=context_feature_columns(),
            example_feature_columns=example_feature_columns(),
            mode=mode,
            scope="transform_layer")

        return context_features, example_features

    return _transform_fn

#SCORE_FN

def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""
    with tf.compat.v1.name_scope("input_layer"):
      context_input = [
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(context_feature_columns())
      ]
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns())
      ]
      input_layer = tf.concat(context_input + group_input, 1)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = input_layer
    cur_layer = tf.compat.v1.layers.batch_normalization(
      cur_layer,
      training=is_training,
      momentum=0.99)

    for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer,
        training=is_training,
        momentum=0.99)
      cur_layer = tf.nn.relu(cur_layer)
      cur_layer = tf.compat.v1.layers.dropout(
          inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)
    logits = tf.compat.v1.layers.dense(cur_layer, units=_GROUP_SIZE)
    return logits

  return _score_fn


#EVAL


def eval_metric_fns():
    """Returns a dict from name to metric functions.

    This can be customized as follows. Care must be taken when handling padded
    lists. (only takes labels >= 0.

    def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    metric_fns["auc"] = _auc

    Returns:
    A dict mapping from metric name to a metric function with above signature.
    """
    metric_fns = {}
    metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in _RANK_TOP_NRS
    })

    # metric_fns.update(precision_d)
    # The reciprocal rank of a query response is the multiplicative inverse of the rank
    # of the first correct answer:
    # 1 for first place, ​1⁄2 for second place, ​1⁄3 for third place and so on

    # with topn: -> checks if there is a relevant item up to this rank
    # so mrr@1 -> checks only first ranked item if this is relevant otherwise result is zero
    # see  test_mean_reciprocal_rank
    # MRR all makes more sense (checks all ranks for first correct answer)

    mrr_d_all = {
        "metric/MRR@ALL": tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.MRR, topn=None)
    }
    mrr_d = {
      "metric/MRR@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.MRR, topn=topn)
      for topn in _RANK_TOP_NRS
    }
    metric_fns.update(mrr_d_all)
    metric_fns.update(mrr_d)

    return metric_fns


def mrr_custom(labels, predictions, features):
    is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid))
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid))

    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    #metric_fns["mrr_custom"] = mrr_custom


