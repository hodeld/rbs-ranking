from rvranking import train_and_eval_fn, _LIST_SIZE
from rvranking.globalVars import _BASE_TF_DATA_PATH
from rvranking.prediction.modelComponents import predict_input_fn
from rvranking.prediction.sampling import write_testsamples


def make_predictions(ranker):
    test_path = _BASE_TF_DATA_PATH + "/test_samples.tfrecords"
    write_testsamples(test_path)
    predictions = ranker.predict(input_fn=lambda: predict_input_fn(test_path))  # = iterator
    return predictions


def get_next_prediction(predictions):
    x = next(predictions)
    assert(len(x) == _LIST_SIZE)  ## Note that this includes padding.
    return x