from rvranking.logs import hplogger
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
from rvranking.globalVars import _TRAIN_DATA_PATH, _TEST_DATA_PATH, _BASE_TF_DATA_PATH, _EVENT_FEATURES
from rvranking.dataPrep import samples, rvs, timelines, rvfirstev, allevents

# START JUPYPTER
from pathlib import Path


def write_elwc():
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]  # samples.iloc[:5].iterrows()])

    # needed for get_example_features
    rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])

    hplogger.info('event_tokens: ' + str(_EVENT_FEATURES))
    r0 = rvlist_all[0]
    r0.log_features()

    sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                            rvlist_all,
                                                            train_ratio=0.7,
                                                            timelines_spec=timelines,
                                                            rvfirstev_spec=rvfirstev,
                                                            allevents_spec=allevents
                                                            )

    #! rm - rf "/tmp/tfdata"  # Clean up the tmp directory.

    Path(_BASE_TF_DATA_PATH).mkdir(parents=True, exist_ok=True)
    # write train
    write_context_examples(_TRAIN_DATA_PATH, sample_list_train)
    # write test
    write_context_examples(_TEST_DATA_PATH, sample_list_test)



