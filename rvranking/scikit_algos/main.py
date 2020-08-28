from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from rvranking.sampling.scikitDataGet import get_data
from rvranking.dataPrep import MIN_ID, MAX_ID
from rvranking.logs import hplogger


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

    pipe = make_pipeline(tfr, clf)

    pipe.fit(x_train, y_train)
    acc_sc = accuracy_score(y_test, pipe.predict(x_test))
    print('acc_sc', acc_sc)
    print('classes', pipe.named_steps['randomforestclassifier'].classes_)
    #pipe.predict_proba(x_predict)
    hplogger.info('named_steps: ' + str(list(pipe.named_steps.keys())))
    hplogger.info('acc_sc: ' + str(acc_sc))

    return pipe


def analyze_transform(x_train, pipe):
    import pandas as pd
    s1 = pd.DataFrame([x_train.loc[0]])
    pipe.named_steps['columntransformer'].transform(s1)
    pipe.named_steps['columntransformer'].transformers_


if __name__ == '__main__':
    fit_predict()