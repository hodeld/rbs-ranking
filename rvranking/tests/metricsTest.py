# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ranking metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow_ranking.python import metrics as metrics_lib
from tensorflow_ranking.python import metrics_impl


def _dcg(label,
         rank,
         weight=1.0,
         gain_fn=lambda l: math.pow(2.0, l) - 1.0,
         rank_discount_fn=lambda r: 1. / math.log(r + 1.0, 2.0)):
    """Returns a single dcg addend.
    Args:
      label: The document label.
      rank: The document rank starting from 1.
      weight: The document weight.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
    Returns:
      A single dcg addend. e.g. weight*(2^relevance-1)/log2(rank+1).
    """
    return weight * gain_fn(label) * rank_discount_fn(rank)


def _ap(relevances, scores, topn=None):
    """Returns the average precision (AP) of a single ranked list.
    The implementation here is copied from Equation (1.7) in
    Liu, T-Y "Learning to Rank for Information Retrieval" found at
    https://www.nowpublishers.com/article/DownloadSummary/INR-016
    Args:
      relevances: A `list` of document relevances, which are binary.
      scores: A `list` of document scores.
      topn: An `integer` specifying the number of items to be considered in the
        average precision computation.
    Returns:
      The MAP of the list as a float computed using the formula
      sum([P@k * rel for k, rel in enumerate(relevance)]) / sum(relevance)
      where P@k is the precision of the list at the cut off k.
    """

    def argsort(arr, reverse=True):
        arr_ind = sorted([(a, i) for i, a in enumerate(arr)], reverse=reverse)
        return list(zip(*arr_ind))[1]

    num_docs = len(relevances)
    if isinstance(topn, int) and topn > 0:
        num_docs = min(num_docs, topn)
    indices = argsort(scores)[:num_docs]
    ranked_relevances = [1. * relevances[i] for i in indices]
    precision = {}
    for k in range(1, num_docs + 1):
        precision[k] = sum(ranked_relevances[:k]) / k
    num_rel = sum(ranked_relevances[:num_docs])
    average_precision = sum(precision[k] * ranked_relevances[k - 1]
                            for k in precision) / num_rel if num_rel else 0
    return average_precision


def _label_boost(boost_form, label):
    """Returns the label boost.
    Args:
      boost_form: Either NDCG or PRECISION.
      label: The example label.
    Returns:
      A list of per list weight.
    """
    boost = {
        'NDCG': math.pow(2.0, label) - 1.0,
        'PRECISION': 1.0 if label >= 1.0 else 0.0,
        'MAP': 1.0 if label >= 1.0 else 0.0,
    }
    return boost[boost_form]


def _example_weights_to_list_weights(weights, relevances, boost_form):
    """Returns list with per list weights derived from the per example weights.
    Args:
      weights: List of lists with per example weight.
      relevances:  List of lists with per example relevance score.
      boost_form: Either NDCG or PRECISION.
    Returns:
      A list of per list weight.
    """
    list_weights = []
    nonzero_relevance = 0.0
    for example_weights, labels in zip(weights, relevances):
        boosted_labels = [_label_boost(boost_form, label) for label in labels]
        numerator = sum((weight * boosted_labels[i])
                        for i, weight in enumerate(example_weights))
        denominator = sum(boosted_labels)
        if denominator == 0.0:
            list_weights.append(0.0)
        else:
            list_weights.append(numerator / denominator)
            nonzero_relevance += 1.0
    list_weights_sum = sum(list_weights)
    if list_weights_sum > 0.0:
        list_weights = [
            list_weights_sum / nonzero_relevance if w == 0.0 else w
            for w in list_weights
        ]

    return list_weights


class MetricsTest(tf.test.TestCase):

    def setUp(self):
        super(MetricsTest, self).setUp()
        tf.compat.v1.reset_default_graph()

    def _check_metrics(self, metrics_and_values):
        """Checks metrics against values."""
        with self.test_session() as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            for (metric_op, update_op), value in metrics_and_values:
                sess.run(update_op)
                self.assertAlmostEqual(sess.run(metric_op), value, places=5)

    def test_mean_recicprocical_rank_custom(self):
        with tf.Graph().as_default():
            tf.compat.v1.disable_eager_execution()  # needed for mrr function
            scores = [[1., 3., 2., 0.5], [1., 2., 3., 4.]]
            labels = [[1., 0, 0, 1.], [1., 1., 1., 0]]  # new with to same label
            # Note that the definition of MRR only uses the highest ranked
            # relevant item, where an item is relevant if its label is > 0.
            # ranks = [[3, 1, 2, 4], [4, 3, 2, 1]]
            rel_rank = [3, 2]

            # Note that the definition of MRR only uses the highest ra

            m = metrics_lib.mean_reciprocal_rank

            self._check_metrics([  # list of should
                (m([labels[0]], [scores[0]]), 1. / rel_rank[0]),
            ])

    def test_reset_invalid_labels(self):
        with tf.Graph().as_default():
            scores = [[1., 3., 2.]]
            labels = [[0., -1., 1.]]
            labels, predictions, _, _ = metrics_impl._prepare_and_validate_params(
                labels, scores)
            self.assertAllClose(labels, [[0., 0., 1.]])
            self.assertAllClose(predictions, [[1., 1. - 1e-6, 2]])

    def test_mean_reciprocal_rank(self):
        with tf.Graph().as_default():
            scores = [[1., 3., 2.], [1., 2., 3.], [3., 1., 2.]]
            # Note that scores are ranked in descending order.
            # ranks = [[3, 1, 2], [3, 2, 1], [1, 3, 2]]
            labels = [[0., 0., 1.], [0., 1., 2.], [0., 1., 0.]]
            # Note that the definition of MRR only uses the highest ranked
            # relevant item, where an item is relevant if its label is > 0.
            rel_rank = [2, 1, 3]
            weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
            mean_relevant_weights = [
                weights[0][2], sum(weights[1][1:]) / 2, weights[2][1]
            ]
            num_queries = len(scores)
            self.assertAlmostEqual(num_queries, 3)
            m = metrics_lib.mean_reciprocal_rank
            self._check_metrics([
                (m([labels[0]], [scores[0]]), 1. / rel_rank[0]),
                (m([labels[0]], [scores[0]], topn=1), 0.),
                (m([labels[0]], [scores[0]], topn=2), 1. / rel_rank[0]),
                (m([labels[1]], [scores[1]]), 1. / rel_rank[1]),
                (m([labels[1]], [scores[1]], topn=1), 1. / rel_rank[1]),
                (m([labels[1]], [scores[1]], topn=6), 1. / rel_rank[1]),
                (m([labels[2]], [scores[2]]), 1. / rel_rank[2]),
                (m([labels[2]], [scores[2]], topn=1), 0.),
                (m([labels[2]], [scores[2]], topn=2), 0.),
                (m([labels[2]], [scores[2]], topn=3), 1. / rel_rank[2]),
                (m(labels[:2], scores[:2]), (0.5 + 1.0) / 2),
                (m(labels[:2], scores[:2], weights[:2]),
                 (3. * 0.5 + (6. + 5.) / 2. * 1.) / (3. + (6. + 5) / 2.)),
                (m(labels,
                   scores), sum([1. / rel_rank[ind] for ind in range(num_queries)]) /
                 num_queries),
                (m(labels, scores,
                   topn=1), sum([0., 1. / rel_rank[1], 0.]) / num_queries),
                (m(labels, scores, topn=2),
                 sum([1. / rel_rank[0], 1. / rel_rank[1], 0.]) / num_queries),
                (m(labels, scores, weights),
                 sum([
                     mean_relevant_weights[ind] / rel_rank[ind]
                     for ind in range(num_queries)
                 ]) / sum(mean_relevant_weights)),
                (m(labels, scores, weights,
                   topn=1), sum([0., mean_relevant_weights[1] / rel_rank[1], 0.]) /
                 sum(mean_relevant_weights)),
            ])

    # more functions -> https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/metrics_test.py

    def test_eval_metric(self):
        with tf.Graph().as_default():
            scores = [[1., 3., 2.], [1., 2., 3.], [3., 1., 2.]]
            labels = [[0., 0., 1.], [0., 1., 2.], [0., 1., 0.]]
            weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
            gain_fn = lambda rel: rel
            rank_discount_fn = lambda rank: 1. / rank
            self._check_metrics([
                (metrics_lib.mean_reciprocal_rank(labels, scores),
                 metrics_lib.eval_metric(
                     metric_fn=metrics_lib.mean_reciprocal_rank,
                     labels=labels,
                     predictions=scores)),
                (metrics_lib.mean_reciprocal_rank(labels, scores, topn=1),
                 metrics_lib.eval_metric(
                     metric_fn=metrics_lib.mean_reciprocal_rank,
                     labels=labels,
                     predictions=scores,
                     topn=1)),
                (metrics_lib.mean_reciprocal_rank(labels, scores, weights),
                 metrics_lib.eval_metric(
                     metric_fn=metrics_lib.mean_reciprocal_rank,
                     labels=labels,
                     predictions=scores,
                     weights=weights)),
                (metrics_lib.discounted_cumulative_gain(
                    labels,
                    scores,
                    gain_fn=gain_fn,
                    rank_discount_fn=rank_discount_fn),
                 metrics_lib.eval_metric(
                     metric_fn=metrics_lib.discounted_cumulative_gain,
                     labels=labels,
                     predictions=scores,
                     gain_fn=gain_fn,
                     rank_discount_fn=rank_discount_fn)),
            ])

    def test_compute_mean(self):
        with tf.Graph().as_default():
            scores = [[1., 3., 2.], [1., 2., 3.]]
            labels = [[0., 0., 1.], [0., 1., 2.]]
            weights = [[1., 2., 3.], [4., 5., 6.]]
            with self.test_session() as sess:
                for key in [
                    'mrr',
                    'arp',
                    'ndcg',
                    'dcg',
                    'precision',
                    'map',
                    'ordered_pair_accuracy',
                ]:
                    value = sess.run(
                        metrics_lib.compute_mean(
                            key, labels, scores, weights, 2, name=key))
                    self.assertGreater(value, 0.)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
