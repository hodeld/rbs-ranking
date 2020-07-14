import random
import unittest
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.dataPrep import samples, rvs, timelines, rvfirstev, allevents, prep_samples, get_timelines_raw, \
    prep_timelines, prep_allevents


def sample_test(cls, s, tlines, allevs):
    rv = s.rv
    tline = tlines.loc[str(rv)]
    ev_tline_val = tline.loc[str(s.start):str(s.end)].values
    ev = allevs.loc[s.id]
    print('id', s.id)
    if s.teams:
        pass
    else:
        cls.assertEqual(ev['End'], s.end)
    cls.assertEqual(ev['Start'], s.start)
    cls.assertEqual(ev['Rv'], s.rv)
    cls.assertEqual(ev['Type'], s.evtype)
    evtype = s.evtype
    print('id, tline', s.id, ev_tline_val)
    cls.assertEqual((evtype == ev_tline_val).all(), True)
    print(s.id)


class TestSampling(unittest.TestCase):

    def test_samples(self):
        sample_list_all = [Sample(s) for i, s in samples.iterrows()]
        random.shuffle(sample_list_all)
        tlines = timelines
        allevs = allevents
        for s in sample_list_all:  # test 10 samples
            sample_test(self, s, tlines, allevs)

    def test_prediction_samples(self):
        samples_pred = prep_samples(file_n='samples_test.csv', sep=';')
        timelines_raw = get_timelines_raw('timelines_test.csv', ';')
        tlines = prep_timelines(timelines_raw)
        allevs = prep_allevents('allevents_test.csv')
        sample_list_all = [Sample(s) for i, s in samples_pred.iterrows()]
        s = sample_list_all[0]
        sample_test(self, s, tlines, allevs)


if __name__ == '__main__':
    unittest.main()
    print('sampling tests finished')
