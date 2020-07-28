import tensorflow as tf
import numpy as np
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
                                  change_var, _SHUFFLE_DATASET)
from rvranking.dataPrep import base_store_path, IN_COLAB, WEEKS_B, TD_PERWK
from rvranking.baseline.rankNaive import rank_rvs
from rvranking.prediction import make_predictions


# COLAB
import shutil


def iterate_samples_train():
    #sample_ids = [15588,]
    sid = None
    i = None
    for i in range(36, 100):  # for sid in sample_ids: OR for i in range(1, 100):
        if sid:
            change_var['sample_id'] = sid
            s_id_nr = sid
        else:
            change_var['sample_nr'] = i
            s_id_nr = i
        write_elwc()
        print('finished writing')
        ranker, train_spec, eval_spec = train_and_eval_fn()
        shutil.rmtree(_MODEL_DIR, ignore_errors=True)
        result = tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
        mrr_all = result[0]['metric/MRR@ALL']
        print('s_id_nr, mrr_all', s_id_nr, mrr_all)

        if sid:
            hplogger.info('sample_id ' + str(sid))
        else:
            hplogger.info('sample_nr ' + str(i))
        hplogger.info('mrr_all: ' + str(mrr_all))
        if mrr_all < 1:
            print('<1, s_id_nr, mrr_all', s_id_nr, mrr_all)
            break
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
                'shuffle_dataset': _SHUFFLE_DATASET,
                'td_perweek': TD_PERWK,
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

    mrrs = make_predictions(ranker)
    mrrs_arr = np.array(mrrs)
    mrr_av = mrrs_arr.sum() / len(mrrs_arr)
    print('mrr_av', mrr_av)
    comment = input('comment on prediction: ')
    hplogger.info('comment_predict: ' + comment)
    hplogger.info('mrr_predictions: ' + str(mrrs))
    hplogger.info('mrr_predictions_av: ' + str(mrr_av))


if __name__ == '__main__':
    logging_basic()

    dispatch_fn = {
        1: main_routine,
        2: baseline,
        3: predictions,  # including train
        4: iterate_samples_train,  # including train
    }
    dispatch_fn[3]()





