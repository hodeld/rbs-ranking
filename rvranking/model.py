import tensorflow as tf
import tensorflow_ranking as tfr
from rvranking.globalVars import (_GROUP_SIZE, _MODEL_DIR, _TRAIN_DATA_PATH, _TEST_DATA_PATH,
                                  _NUM_TRAIN_STEPS)
from rvranking.rankingComponents import make_score_fn, make_transform_fn, input_fn
from rvranking.lossesMetrics import ranking_head


model_fn = tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          transform_fn=make_transform_fn(),
          group_size=_GROUP_SIZE,
          ranking_head=ranking_head)


def train_and_eval_fn():
  """Train and eval function used by `tf.estimator.train_and_evaluate`."""
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=1000)
  ranker = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=_MODEL_DIR,
      config=run_config)

  train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
  eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)
  eval_spec =  tf.estimator.EvalSpec(
          name="eval",
          input_fn=eval_input_fn,
          throttle_secs=15)
  return (ranker, train_spec, eval_spec)


