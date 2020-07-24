from rvranking.dataPrep import main_path, prep_samples, prep_allevents, prep_timelines, get_timelines_raw, \
    prep_rv_first_ev
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
import pandas as pd
from rvranking.logs import hplogger


def write_testsamples(test_path):
    timelines_raw = get_timelines_raw('timelines_test.csv', ';')
    timelines = prep_timelines(timelines_raw)
    samples = prep_samples(file_n='samples_test.csv', sep=';')
    allevents = prep_allevents('allevents_test.csv')
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]

    rvfirstev = prep_rv_first_ev('rvfirstev_test.csv', sep=',')

    rvs = pd.read_csv(main_path + 'RVs_test.csv')
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