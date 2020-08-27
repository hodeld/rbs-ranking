from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from rvranking.sampling.scikitDataGet import get_data


def fit_predict():
    x_train, y_train, x_test, y_test = get_data() # uma and hwx are objects
    clf = RandomForestClassifier(random_state=0)

    trf_mc = make_column_transformer(
        (StandardScaler(), ['evtype']),
    remainder = StandardScaler())

    trf = StandardScaler()
    pipe = make_pipeline(trf, clf)

    pipe.fit(x_train, y_train)
    acc_sc = accuracy_score(pipe.predict(x_test), y_test)
    print('acc_sc', acc_sc)


if __name__ == '__main__':
    fit_predict()