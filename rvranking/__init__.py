import tensorflow as tf
from rvranking.elwcWrite import write_elwc
from rvranking.model import train_and_eval_fn
from rvranking.globalVars import _MODEL_DIR
import shutil


if __name__ == '__main__':
    write_elwc()
    print('finished writing')
    ranker, train_spec, eval_spec = train_and_eval_fn()
    #tf.compat.v1.summary.FileWriterCache.clear()  # clear cache for issues with tensorboard
    shutil.rmtree(_MODEL_DIR, ignore_errors=True)
    result = tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
    print(result)
    tb_cmd = 'tensorboard --logdir="%s"' % _MODEL_DIR
    print('show tensorbord on localhost:6006 -> run in terminal:', tb_cmd)


