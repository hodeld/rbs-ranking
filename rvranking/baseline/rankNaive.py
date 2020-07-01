from rvranking.baseline.priorizeRV import predict_rv
from rvranking.elwcClasses import Sample, RV, RVList, prep_samples_list
from rvranking.elwcFunctions import write_context_examples
from rvranking.globalVars import _TRAIN_DATA_PATH, _TEST_DATA_PATH, _BASE_TF_DATA_PATH
from rvranking.dataPrep import samples, rvs


def rank_rvs():
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]
    rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])

    train_ratio = 0  # all test
    # get rvs, check availability, check evtype, check sex; UMA and HWX not checked yet
    sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                            rvlist_all,
                                                      train_ratio=train_ratio)
    ncdg1_li = []
    for s in sample_list_test:
        ncdg1 = predict_rv(s)  # sets rv.prediction$
        ncdg1_li.append(ncdg1)

    # simplified as just look at 1st -> if correct ngcd = 1, else = 0
    ndgc1_mean = sum(ncdg1_li)/len(ncdg1_li)
    return ndgc1_mean

