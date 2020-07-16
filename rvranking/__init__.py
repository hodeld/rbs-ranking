import tensorflow as tf
from rvranking.sampling.elwcWrite import write_elwc
from rvranking.logs import logging_basic, hplogger, logs_to_csv
from rvranking.model import train_and_eval_fn
from rvranking.globalVars import (_MODEL_DIR, _FAKE, _LIST_SIZE,
                                  _BATCH_SIZE, _LEARNING_RATE, _DROPOUT_RATE,
                                  _HIDDEN_LAYER_DIMS, _GROUP_SIZE,
                                  _NUM_TRAIN_STEPS,
                                  _EMBEDDING_DIMENSION,
                                  RELEVANCE,
                                  _SAVE_CHECKPOINT_STEPS,
                                  _SAMPLING,
                                  _LOSS,
                                  change_var)
from rvranking.dataPrep import base_store_path, IN_COLAB, WEEKS_B
from rvranking.baseline.rankNaive import rank_rvs
from rvranking.prediction import make_predictions, get_next_prediction


# COLAB
import shutil


def iterate_samples_train():
    sample_ids = [13502, 13097]
    for sid in sample_ids:
        change_var['sample_id'] = sid
        print(change_var['sample_nr'])
        write_elwc()
        print('finished writing')
        ranker, train_spec, eval_spec = train_and_eval_fn()
        shutil.rmtree(_MODEL_DIR, ignore_errors=True)
        result = tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
        mrr_all = result[0]['metric/MRR@ALL']
        #if mrr_all < 1:
        print('sample_id, mrr_all', sid, mrr_all)
        hplogger.info('sample_id ' + str(sid))
        hplogger.info('mrr_all: ' + str(mrr_all))
        hplogger.info('mini-run -------------')

    comment = input('comment on run: ')
    hyparams = {'iterate_samples_train': True,
                'sampling method': _SAMPLING,
                'weeks_b': WEEKS_B,
                'comment': comment,

                }
    write_file(hyparams)


def main_routine(include_comment=True):
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
    if not IN_COLAB and include_comment:
        comment = input('comment on run: ')
    else:
        comment = 'in colab or prediction'
    hyparams = {'comment': comment,
                'fake': _FAKE,
                'sampling method': _SAMPLING,
                'embedding_dimension': _EMBEDDING_DIMENSION,
                'loss_function': _LOSS,
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
    return ranker


def write_file(hyparams):
    for k, v in hyparams.items():
        hplogger.info(k + ': ' + str(v))


def baseline():
    ndcg1_mean, mrr_mean = rank_rvs()
    print('ndcg1_mean:', ndcg1_mean, 'mrr_mean:', mrr_mean)
    comment = input('comment on run: ')
    if not comment == 'n':
        hyparams = {'baseline': True,
                    'comment': comment,
                    'sampling method': _SAMPLING,
                    'ndcg1_mean': ndcg1_mean,
                    'mrr_mean:': mrr_mean,
                    }
        write_file(hyparams)


def predictions():
    ranker = 'd'

    ranker = main_routine(False)
    #ranker.export_saved_model('path', 'serving_inout_receiver_fn')

    predicts = make_predictions(ranker)
    p1 = get_next_prediction(predicts)
    print(p1)
    comment = input('comment on prediction: ')
    hplogger.info('comment_predict: ' + comment)
    hplogger.info('predictions: ' + str(p1))


if __name__ == '__main__':
    logging_basic()

    dispatch_fn = {
        1: main_routine,
        2: baseline,
        3: predictions,  # including train
        4: iterate_samples_train,  # including train
    }
    dispatch_fn[4]()





