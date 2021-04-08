from rvranking.logs import hplogger
from rvranking.globalVars import _BASE_TF_DATA_PATH, _LIST_SIZE_PREDICT, _EVENT_FEATURE, \
    _RV_FEATURE
from rvranking.prediction.modelComponents import predict_input_fn
from rvranking.prediction.sampling import write_testsamples
import tensorflow as tf
import numpy as np


def make_predictions(ranker):
    test_path = _BASE_TF_DATA_PATH + "/test_samples.tfrecords"
    s_order = write_testsamples(test_path)
    s_sorted = sorted(s_order)
    indices = [s_order.index(s) for s in s_sorted]
    high_ranked = read_samples(test_path, s_order)
    weights(ranker)
    predictions = ranker.predict(input_fn=lambda: predict_input_fn(test_path))  # = iterator
    mrrs = eval_prediction(high_ranked, predictions, s_order)
    assert (len(high_ranked) == len(mrrs))
    for i in indices:
        print('sample ', s_order[i], 'mrr', mrrs[i])
    return mrrs


def eval_prediction(high_ranked, predictions, s_order):
    mrrs = []
    predicts = []
    for i, high_ranks in enumerate(high_ranked):
        try:
            x = next(predictions)
        except StopIteration:
            break
        assert (len(x) == _LIST_SIZE_PREDICT)  ## Note that this includes padding.
        pre = list(x)
        print('sample / prediction', s_order[i], pre)
        predicts.append(pre)
        pre_sorted = sorted(pre, reverse=True)  # descending
        ranks = [pre_sorted.index(p) + 1 for p in pre]
        rel_ranks = []
        for hr in high_ranks:
            rel_rank = ranks[hr]
            rel_ranks.append(rel_rank)
        rel_ranks.sort()  # ascending
        mrr = 1/rel_ranks[0]
        mrrs.append(mrr)
    hplogger.info('predictions: ' + str(predicts))
    return mrrs


def read_samples(test_path, s_order):
    feat, labs = predict_input_fn(test_path, include_labels=True)  # so each sample once
    print('label', labs.shape)
    for k, item in feat.items():
        print('feat', k, item.shape)
    #event_t = tf.sparse.to_dense(feat[_EVENT_FEATURE])  # spare tensor to dense
    #rv_t = tf.sparse.to_dense(feat[_RV_FEATURE])
    # print ('indices', query_st.indices[0][0]) #which indix has first value
    ind_highest_ranks = []
    for i, si in enumerate(s_order):
        label = labs[i, :5].numpy()
        ind_arr = np.where(label == 1)[0]
        ind_highest_ranks.append(ind_arr)
        #print('sample id; first 5 labels', si, label)  # [0. 0. 0. 0. 1.]
        #print('event values', event_t[i].numpy(),)
        #print('rv values of first 5 rvs', rv_t[i, :5, :3].numpy())  # sample 1, first 5 rvs, first 10 features
    return ind_highest_ranks


def weights(ranker):
    # wt_names = ranker.get_variable_names()
    #event_wt_n = 'encoding_layer/event_tokens_embedding/embedding_weights'
    rv_wt_n = 'encoding_layer/rv_tokens_embedding/embedding_weights'
    evtype_n = 'encoding_layer/evtype_embedding/embedding_weights'
    rvff_n = 'encoding_layer/rv_ff_embedding/embedding_weights'

    rvid_n = 'encoding_layer/id_embedding/embedding_weights'
    sex_n = 'encoding_layer/sex_embedding/embedding_weights'
    tline_n = 'encoding_layer/tline_embedding/embedding_weights'

    f_names = [evtype_n, rvff_n, rvid_n, sex_n, tline_n]
    print('mean / min / max of all num_buckets:')
    for f_n in f_names:
        try:
            wts = ranker.get_variable_value(f_n)
            pr_n = f_n.split('/')[1]
            print(pr_n, np.mean(wts, axis=0), np.min(wts, axis=0),  np.max(wts, axis=0))
        except tf.errors.NotFoundError: #Exception as exp:
            pass
