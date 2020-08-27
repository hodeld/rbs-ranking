from rvranking.logs import hplogger
from rvranking.sampling.main import prep_samples_list, get_train_test
import pandas as pd
from rvranking.globalVars import _FAKE_ELWC, _EVENT_FEATURES, _RV_FEATURES


def get_data():
    sample_list_train, sample_list_test = get_train_test()

    x_train, y_train = toTensor(sample_list_train)
    x_test, y_test = toTensor(sample_list_test)

    return x_train, y_train, x_test, y_test


def toTensor(s_list):
    """
    :param s_list: list of samples with rvs
    :return:
    X = [[ 1,  2,  3],  # 2 samples, 3 features
    [11, 12, 13]]
    y = [0, 1]  # classes of each sample
    :param s_list:
    """

    def conc_features(obj):
        feat_arr = pd.Series(dtype='int')
        feat_list = obj.features()
        for f in feat_list:
            try:
                f_arr = pd.Series(f, dtype='int')
            except ValueError:
                f_arr = pd.Series([f], dtype='int')
            feat_arr = pd.concat([feat_arr, f_arr])
        return feat_arr

    labels = []
    feat_matrix = []
    for s in s_list:
        rvli = s.rvli
        s_feat_arr = conc_features(s)

        for rv in rvli:
            rv_feat_arr = conc_features(rv)
            tot_feat_arr = pd.concat([rv_feat_arr, s_feat_arr])
            labels.append(rv.relevance)
            feat_matrix.append(tot_feat_arr)

    return feat_matrix, labels


def toDFwColumsn(s_list):
    """
    :param s_list: list of samples with rvs
    :return:
    problem: df should only of nr not series (as tline)
    """

    ev_features = _EVENT_FEATURES
    rv_features = _RV_FEATURES
    all_feat_names = ev_features + rv_features

    labels = []
    tot_feat_list = []
    for s in s_list:
        rvli = s.rvli
        s_feat_li = s.features()

        for rv in rvli:
            rv_feat_li = rv.features()
            rv_feat_li.extend(s_feat_li) #  len == nr of tot features
            tot_feat_list.append(rv_feat_li)
            labels.append(rv.relevance)
    feat_matrix = pd.DataFrame(tot_feat_list, columns=all_feat_names)

    return feat_matrix, labels


