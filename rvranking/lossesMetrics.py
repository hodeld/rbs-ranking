import tensorflow as tf
import tensorflow_ranking as tfr
from rvranking.globalVars import _LEARNING_RATE
from rvranking.rankingComponents import eval_metric_fns

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.

_LOSS = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
loss_fn = tfr.losses.make_loss_fn(_LOSS)

#RANKING HEAD

optimizer = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=_LEARNING_RATE)

def _train_op_fn(loss):
  """Defines train op used in ranking head."""
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  minimize_op = optimizer.minimize(
      loss=loss, global_step=tf.compat.v1.train.get_global_step())
  train_op = tf.group([update_ops, minimize_op])
  return train_op

ranking_head = tfr.head.create_ranking_head(
      loss_fn=loss_fn,
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=_train_op_fn)