import tensorflow as tf
from rvranking.elwcWrite import write_elwc
from rvranking.logs import logging_basic, hplogger
from rvranking.model import train_and_eval_fn
from rvranking.globalVars import (_MODEL_DIR, _FAKE, _LIST_SIZE,
                                  _BATCH_SIZE, _LEARNING_RATE, _DROPOUT_RATE,
                                  _HIDDEN_LAYER_DIMS, _GROUP_SIZE,
                                  _NUM_TRAIN_STEPS,
                                  _EMBEDDING_DIMENSION,
                                  RELEVANCE,
                                  _SAVE_CHECKPOINT_STEPS)
from rvranking.dataPrep import base_store_path
from rvranking.baseline.rankNaive import rank_rvs
from datetime import datetime


# COLAB
import shutil


def main_routine():
    write_elwc()
    print('finished writing')
    ranker, train_spec, eval_spec = train_and_eval_fn()
    #tf.compat.v1.summary.FileWriterCache.clear()  # clear cache for issues with tensorboard
    shutil.rmtree(_MODEL_DIR, ignore_errors=True)
    result = tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
    print(result)
    tb_cmd = 'tensorboard --logdir="%s"' % _MODEL_DIR
    print('show tensorbord on localhost:6006 -> run in terminal:', tb_cmd)

    #file

    res_d = result[0]
    comment = input('comment on run: ')
    hyparams = {'comment': comment,
                'fake': _FAKE,
                'embedding_dimension': _EMBEDDING_DIMENSION,
                'relevance': RELEVANCE,
                'list_size': _LIST_SIZE,
                'batch_size': _BATCH_SIZE,
                'learning_rate': _LEARNING_RATE,
                'dropout_rate': _DROPOUT_RATE,
                'group_size': _GROUP_SIZE,
                'num_train_steps': _NUM_TRAIN_STEPS,
                'hidden_layers_dims': _HIDDEN_LAYER_DIMS,
                'save_checkpoint_steps': _SAVE_CHECKPOINT_STEPS,
                }
    hyparams.update(res_d)
    write_file(hyparams)


def write_file(hyparams):
    hplogger.info('date: ' + datetime.now().strftime('%c'))
    for k, v in hyparams.items():
        hplogger.info(k + ': ' + str(v))
    hplogger.info('new run -----------------------------------------')


def baseline():
    ndcg1_mean = rank_rvs()
    print('ndcg1_mean', ndcg1_mean)
    comment = input('comment on run: ')
    if not comment == 'n':
        hyparams = {'baseline': True,
                    'comment': comment,
                    'ndcg1_mean': ndcg1_mean}
        write_file(hyparams)


if __name__ == '__main__':
    logging_basic()
    main_routine()
    #baseline()



