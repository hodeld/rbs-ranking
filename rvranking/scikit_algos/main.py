import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

from rvranking.prediction.sampling import get_test_samples
from rvranking.sampling.scikitDataGet import get_data, x_y_data
from rvranking.dataPrep import MIN_ID, MAX_ID
from rvranking.logs import hplogger
from rvranking.globalVars import _LIST_SIZE, _LIST_SIZE_PREDICT


def scale_evtypes(x):
    z = (x-MIN_ID)/MAX_ID
    return z


fct_transformer = FunctionTransformer(func=scale_evtypes)
std_scaler = StandardScaler()
cat_encoder = OneHotEncoder(handle_unknown='ignore')  # ignore unknown categories in test (if not seen in train)
evtype_scaler = fct_transformer  # fct_transformer or  (cat_encoder) -> not working yet

transf_dict = {
# columns with this regex pattern get transformered,
    # remainder according minmax -> come as last columns -> order of columns changes
    'column': make_column_transformer(
        (evtype_scaler,
         make_column_selector('evtype')),
        (evtype_scaler,
         make_column_selector('tline[0-9]+')),
        remainder=MinMaxScaler()),

    'std_scaler': StandardScaler()
}


def fit_predict():
    x_train, y_train, xy_train, x_test, y_test, xy_test = get_data()  # uma and hwx are objects
    clf = RandomForestClassifier(random_state=0)

    tfr_name = 'column'
    tfr = transf_dict[tfr_name]

    hplogger.info('transformer: ' + tfr_name)

    pipe = make_pipeline(tfr, clf)  # memory='scikitmodel' cache works only once program is running or prob can be pickle.load(x)
    pipe.fit(x_train, y_train)
    acc_sc = accuracy_score(y_test, pipe.predict(x_test))
    print('acc_sc', acc_sc)
    print('classes', pipe.named_steps['randomforestclassifier'].classes_)  # todo better: regression?!
    hplogger.info('named_steps: ' + str(list(pipe.named_steps.keys())))
    hplogger.info('acc_sc: ' + str(acc_sc))

    features_importance(pipe, x_train)
    features_plot(x_train, ['tline0', 'tline7'])

    mrr_mean, mrrs, li_probs = score_per_event(pipe, x_test, xy_test)
    hplogger.info('mrr_mean: ' + str(mrr_mean))

    sample_list_pred, s_order = get_test_samples()
    x_pred, y_pred, xy_pred = x_y_data(sample_list_pred)
    mrr_mean, mrrs, li_probs = score_per_event(pipe, x_pred, xy_pred)

    s_sorted = sorted(s_order)
    indices = [s_order.index(s) for s in s_sorted]
    pred_in_order = []
    rv_data = get_rv_data(sample_list_pred)
    for i in indices:
        print('sample ', s_order[i], 'mrr', mrrs[i], 'labels', xy_pred[i], 'probs',
              [round(p, 2) for p in li_probs[i]], 'rvid', rv_data[i])
        pred_in_order.append(mrrs[i])

    hplogger.info('mrr_pred_sorted: ' + str(pred_in_order))
    hplogger.info('mrr_predictions_av: ' + str(mrr_mean))


def analyze_transform(x_train, pipe):

    s1 = pd.DataFrame([x_train.loc[0]])
    pipe.named_steps['columntransformer'].transform(s1)
    pipe.named_steps['columntransformer'].transformers_


def score_per_event(pipe, x, xy):
    prob_arr = pipe.predict_proba(x)
    classes = list(pipe.named_steps['randomforestclassifier'].classes_)
    rel_rv_ind = classes.index(1)
    proba_rel = list(prob_arr[:, rel_rv_ind])
    kst = 0 #iterate start number
    mrrs = []
    list_rv_probs = []
    for label_i in xy:
        rvli_size = len(label_i)
        ket = kst + rvli_size
        rv_sort_list = []
        indeces_rv = [i for i, n in enumerate(label_i) if n == 1] #positions where label_i == 1
        rv_probs = proba_rel[kst:ket]
        list_rv_probs.append(rv_probs)
        for ind, prob in enumerate(rv_probs):
            rv_sort_list.append((ind, prob))
        rv_sort_list.sort(key=lambda rv_p: rv_p[1], reverse=True)

        for i, rv_p in enumerate(rv_sort_list):
            ind = rv_p[0]
            rank = i + 1
            if ind in indeces_rv:
                mrr = 1 / rank
                mrrs.append(mrr)
                break
        kst = ket
    mrr_mean = sum(mrrs) / len(mrrs)

    return mrr_mean, mrrs, list_rv_probs


def get_rv_data(sample_li):
    rv_data = []
    for s in sample_li:
        rv_data_i = [(r.id, r.relevance) for r in s.rvli]
        rv_data.append(rv_data_i)
    return rv_data


def features_importance(pipe, x):
    forest = pipe.named_steps['randomforestclassifier']
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    max_range = x.shape[1]
    range_nr = min(max_range, 10)
    for f in range(range_nr):
        print("%d. f %s (%f)" % (f + 1, x.columns[indices[f]], importances[indices[f]]))


def features_importance_perm(pipe, x, y):
    rf = pipe.named_steps['randomforestclassifier']
    result = permutation_importance(rf, x, y, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    sorted_idx = importances.argsort()[::-1]
    max_range = x.shape[1]
    range_nr = min(max_range, 10)
    for f in range(range_nr):
        print("%d. f %s (%f)" % (f + 1, x.columns[sorted_idx[f]], importances[sorted_idx[f]]))


def features_plot(feat_matrix, feat_ns):
    for n in feat_ns:
        feat_matrix[n].value_counts().plot.bar(title=n)
        plt.show()


# to plot features:


if __name__ == '__main__':
    fit_predict(True)