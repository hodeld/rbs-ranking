from rvranking.baseline.priorizeRV import predict_rv
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.dataPrep import samples, rvs, timelines, rvfirstev, allevents


def rank_rvs():
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]
    rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])

    train_ratio = 0  # all test
    # get rvs, check availability, check evtype, check sex; UMA and HWX not checked yet
    sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                            rvlist_all,
                                                            train_ratio=train_ratio,
                                                            timelines_spec=timelines,
                                                            rvfirstev_spec=rvfirstev,
                                                            allevents_spec=allevents
                                                            )
    ncdg1_li = []
    mrr_li = []
    for s in sample_list_test:
        ncdg1, mrr = predict_rv(s)  # sets rv.prediction$
        ncdg1_li.append(ncdg1)
        mrr_li.append(mrr)

    ndgc1_mean = sum(ncdg1_li)/len(ncdg1_li)  # simplified as just look at 1st -> if correct ngcd = 1, else = 0
    mrr_mean = sum(mrr_li) / len(mrr_li)
    return ndgc1_mean, mrr_mean

