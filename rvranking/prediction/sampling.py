from rvranking.dataPrep import main_path
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
import pandas as pd


def write_testsamples(test_path):
    timelines_raw = pd.read_csv(main_path + 'timelines_test.csv', index_col=0)  # , header=0)
    timelines = timelines_raw.drop(index='dt_col')
    timelines = timelines.apply(pd.to_numeric)

    samples_raw = pd.read_csv(main_path + 'samples_test.csv')
    samples = samples_raw.iloc[:, 0:16]  # additional columns
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]

    rvfirstev_raw = pd.read_csv(main_path + 'rvfirstev_test.csv', index_col=0)
    rvfirstev = rvfirstev_raw.copy()
    rvfirstev[rvfirstev_raw == 0] = 1

    rvs = pd.read_csv(main_path + 'RVs_test.csv')
    rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])

    allevents = pd.read_csv(main_path + 'allevents_test.csv', index_col=0)

    train_ratio = 0  # all test
    # get rvs, check availability, check evtype, check sex; UMA and HWX not checked yet
    sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                            rvlist_all,
                                                            train_ratio=train_ratio,
                                                            timelines_spec=timelines,
                                                            rvfirstev_spec=rvfirstev,
                                                            allevents_spec=allevents
                                                            )

    # write test
    write_context_examples(test_path, sample_list_test)