from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from rvranking.prediction.sampling import get_test_samples
from rvranking.sampling.scikitDataGet import get_data, x_y_data
from rvranking.dataPrep import MIN_ID, MAX_ID
from rvranking.logs import hplogger
from rvranking.globalVars import _LIST_SIZE, _LIST_SIZE_PREDICT


def scale_evtypes(x):
    z = (x-MIN_ID)/MAX_ID
    return z


evtype_scaler = FunctionTransformer(func=scale_evtypes)
std_scaler = StandardScaler()

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
    x_train, y_train, x_test, y_test = get_data() # uma and hwx are objects
    clf = RandomForestClassifier(random_state=0)

    tfr_name = 'column'
    tfr = transf_dict[tfr_name]

    hplogger.info('transformer: ' + tfr_name)

    pipe = make_pipeline(tfr, clf)  # memory='scikitmodel' cache works only once program is running or prob can be pickle.load(x)
    pipe.fit(x_train, y_train)
    acc_sc = accuracy_score(y_test, pipe.predict(x_test))
    print('acc_sc', acc_sc)
    print('classes', pipe.named_steps['randomforestclassifier'].classes_)
    hplogger.info('named_steps: ' + str(list(pipe.named_steps.keys())))
    hplogger.info('acc_sc: ' + str(acc_sc))

    mrr_mean, mrrs= score_per_event(pipe, x_test, y_test, _LIST_SIZE)
    hplogger.info('mrr_mean: ' + str(mrr_mean))

    sample_list_pred, s_order = get_test_samples()
    x_pred, y_pred = x_y_data(sample_list_pred)
    mrr_mean, mrrs = score_per_event(pipe, x_pred, y_pred, _LIST_SIZE)

    s_sorted = sorted(s_order)
    indices = [s_order.index(s) for s in s_sorted]
    pred_in_order = []
    for i in indices:
        print('sample ', s_order[i], 'mrr', mrrs[i])
        pred_in_order.append(mrrs[i])

    hplogger.info('mrr_pred_sorted: ' + str(pred_in_order))
    hplogger.info('mrr_predictions_av: ' + str(mrr_mean))


def analyze_transform(x_train, pipe):
    import pandas as pd
    s1 = pd.DataFrame([x_train.loc[0]])
    pipe.named_steps['columntransformer'].transform(s1)
    pipe.named_steps['columntransformer'].transformers_


def score_per_event(pipe, x, y, rvli_size):
    prob_arr = pipe.predict_proba(x)
    classes = list(pipe.named_steps['randomforestclassifier'].classes_)
    rel_rv_ind = classes.index(1)
    proba_rel = list(prob_arr[:, rel_rv_ind])
    kst = 0 #iterate start number
    mrrs = []
    evs = int(len(proba_rel)/rvli_size)
    for ev in range(evs):
        ket = kst + rvli_size
        rv_sort_list = []
        label_i = y[kst:ket]
        indeces_rv = [i for i, n in enumerate(label_i) if n == 1] #positions where label_i == 1
        rv_probs = proba_rel[kst:ket]
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

    return mrr_mean, mrrs


if __name__ == '__main__':
    fit_predict(True)