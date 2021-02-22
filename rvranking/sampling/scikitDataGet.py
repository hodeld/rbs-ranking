from rvranking.logs import hplogger
from rvranking.sampling.main import prep_samples_list, get_train_test
import pandas as pd
from rvranking.globalVars import _FAKE_ELWC, _EVENT_FEATURES, _RV_FEATURES


def get_data():
    sample_list_train, sample_list_test = get_train_test()

    x_train, y_train, f_per_s_train = x_y_data(sample_list_train)
    x_test, y_test = x_y_data(sample_list_test)

    return x_train, y_train, x_test, y_test


def x_y_data(sample_list):
    # x_train, y_train = toTensor(sample_list)
    x, y = to_data_frame(sample_list)
    return x, y


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
            feat_arr = pd.concat([feat_arr, f_arr], ignore_index=True)
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


def to_data_frame(s_list):
    """
    :param s_list: list of samples with rvs
    :return:
    problem: df should only of nr not series (as tline)
    """

    def ext_features(obj):
        """
        :param obj:
        :return: list of features -> for dataframe
        """
        feat_arr = []
        feat_list = obj.features()
        for f in feat_list:
            try:
                f_arr = list(f)
            except TypeError:
                f_arr = [f]
            feat_arr.extend(f_arr)
        return feat_arr

    def get_rv_feat_cols(obj, feat_names):
        feat_list = obj.features()
        col_names = []
        for f, n in zip(feat_list, feat_names) :
            try:
                f_arr = pd.Series(f, dtype='int')
                li = list(range(f_arr.size))
                f_cols = [n + str(i) for i in li]
            except ValueError:
                f_cols = [n]
            col_names.extend(f_cols)
        return col_names

    ev_features = _EVENT_FEATURES
    rv_features = _RV_FEATURES
    rv0 = s_list[0].rvli[0]
    rv_cols = get_rv_feat_cols(rv0, rv_features)

    all_feat_names = rv_cols + ev_features

    labels = []
    tot_feat_list = []
    for s in s_list:
        rvli = s.rvli
        s_feat_arr = ext_features(s)

        for rv in rvli:
            rv_feat_arr = ext_features(rv)
            rv_feat_arr.extend(s_feat_arr)
            tot_feat_list.append(rv_feat_arr)
            labels.append(rv.relevance)
    feat_matrix = pd.DataFrame(tot_feat_list, columns=all_feat_names, dtype='int')

    return feat_matrix, labels


