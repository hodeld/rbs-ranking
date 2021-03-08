from rvranking.dataPrep import main_path, prep_samples, prep_allevents, prep_timelines, get_timelines_raw, \
    prep_rv_first_ev, prep_rv, get_test_files
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
import pandas as pd
from rvranking.logs import hplogger


def get_test_samples():
    samples, timelines, allevents, rvs, rvfirstev = get_test_files()
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

    sample_order = []
    for s in sample_list_test:
        rele = [r.relevance for r in s.rvli]
        sample_order.append(s.id)
    print('sample_order:', sample_order)

    return sample_list_test, sample_order


def write_testsamples(test_path):
    sample_list_test, sample_order = get_test_samples()

    # write test
    write_context_examples(test_path, sample_list_test)
    return sample_order


def replace_delimiter():
    # todo test it -> now different delimiter for timelines
    def replace(fp, fpn):
        with open(fp) as f:
            lines = f.readlines()
        for l in lines:
            l.replace(';', ',')
        with open(fpn) as f:
            f.writelines(lines)

    fp = main_path + 'timelines_test.csv'
    new_fp = main_path + 'timelines_test2.csv'
    replace(fp, new_fp)