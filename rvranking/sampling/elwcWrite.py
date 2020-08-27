from rvranking.logs import hplogger
from rvranking.sampling.main import prep_samples_list, get_train_test
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
from rvranking.globalVars import _TRAIN_DATA_PATH, _TEST_DATA_PATH, _BASE_TF_DATA_PATH, _EVENT_FEATURES, _RV_FEATURES


# START JUPYPTER
from pathlib import Path


def write_elwc():
    sample_list_train, sample_list_test = get_train_test()

    Path(_BASE_TF_DATA_PATH).mkdir(parents=True, exist_ok=True)
    # write train
    write_context_examples(_TRAIN_DATA_PATH, sample_list_train)
    # write test
    write_context_examples(_TEST_DATA_PATH, sample_list_test)



