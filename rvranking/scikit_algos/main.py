import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import onnxruntime as rt
import pathlib

from rvranking.prediction.sampling import get_test_samples
from rvranking.sampling.scikitDataGet import get_data, x_y_data
from rvranking.dataPrep import MIN_ID, MAX_ID
from rvranking.logs import hplogger
from rvranking.globalVars import _LIST_SIZE, _LIST_SIZE_PREDICT, _RESTORE, _SAVE_MODEL


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

    tfr_name = 'std_scaler'  # make_column_selector not working with onnx, needs ColumnTranformer wout named columns
    tfr = transf_dict[tfr_name]

    hplogger.info('transformer: ' + tfr_name)

    parent = pathlib.Path(__file__).resolve().parents[2]
    fpath = pathlib.Path(parent, 'output', 'saved_models', 'scikit_mod', 'pipe.onnx')
    if not _RESTORE:
        pipe = make_pipeline(tfr, clf)
        pipe.fit(x_train, y_train)
        if _SAVE_MODEL:
            model_onnx = to_onnx(pipe, np.array(x_train[:1].astype(np.float32)))  #  training set, can be None, it is
                                                                        # used to infered the input types

            with open(fpath, "wb") as f:
                f.write(model_onnx.SerializeToString())

    else:
        sess = rt.InferenceSession(fpath.absolute())
        pred_onx = sess.run(None, {'X': np.array(x_test.astype(np.float32))})[0]
        acc_sc = accuracy_score(y_test, pred_onx)

    #  analysis for prediction needs to be rewritten


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


def features_importance(pipe, x, max_nr=10, nr_return=4):
    forest = pipe.named_steps['randomforestclassifier']
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    max_range = x.shape[1]
    range_nr = min(max_range, max_nr)
    imp_features = []
    for f in range(range_nr):
        print("%d. f %s (%f)" % (f + 1, x.columns[indices[f]], importances[indices[f]]))
        if f < nr_return:
            imp_features.append(x.columns[indices[f]])
    return imp_features


def features_importance_perm(pipe, x, y, max_nr=10):
    rf = pipe.named_steps['randomforestclassifier']
    result = permutation_importance(rf, x, y, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    sorted_idx = importances.argsort()[::-1]
    max_range = x.shape[1]
    range_nr = min(max_range, max_nr)
    for f in range(range_nr):
        print("%d. f %s (%f)" % (f + 1, x.columns[sorted_idx[f]], importances[sorted_idx[f]]))


def features_plot(feat_matrix, feat_ns):
    for n in feat_ns:
        feat_matrix[n].value_counts().plot.bar(title=n)
        plt.show()


# to plot features:


if __name__ == '__main__':
    fit_predict(True)