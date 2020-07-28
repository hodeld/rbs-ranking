import random
import unittest

from rvranking.rankingComponents import input_fn
from rvranking.sampling.elwcWrite import write_elwc
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RVList, RV
from rvranking.globalVars import _TRAIN_DATA_PATH, _RV_FEATURE, _EVENT_FEATURE
from rvranking.dataPrep import samples, timelines,  allevents, prep_samples, get_timelines_raw, \
    prep_timelines, prep_allevents, TD_PERWK, WEEKS_B, WEEKS_A, KMAX, rvs, rvfirstev
import tensorflow as tf


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
        for s in sample_list_all:
            sample_test(self, s, tlines, allevs)

    def test_prediction_samples(self):
        samples_pred = prep_samples(file_n='samples_test.csv', sep=';')
        timelines_raw = get_timelines_raw('timelines_test.csv', ';')
        tlines = prep_timelines(timelines_raw)
        allevs = prep_allevents('allevents_test.csv', ';')
        sample_list_all = [Sample(s) for i, s in samples_pred.iterrows()]
        s = sample_list_all[0]
        ist = int(s.start - (TD_PERWK * WEEKS_B))
        iet = int(s.start + TD_PERWK * WEEKS_A)
        self.assertGreaterEqual(ist, 0)  # ist >= 0
        self.assertLessEqual(iet, KMAX)

        sample_test(self, s, tlines, allevs)

    def _sampling(self):
        sample_list_all = [Sample(s) for i, s in samples.iloc[:5].iterrows()]
        rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])
        train_ratio = 0.7

        sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                                rvlist_all,
                                                                train_ratio=train_ratio,
                                                                timelines_spec=timelines,
                                                                rvfirstev_spec=rvfirstev,
                                                                allevents_spec=allevents
                                                                )

    def _test_write_and_input(self):
        # _sampling
        write_elwc()

        feat, labs = input_fn(_TRAIN_DATA_PATH)
        print('label', labs.shape)
        for k, item in feat.items():
            print('feat', k, item.shape)
        print('first 5 labels', labs[0, :5].numpy())  # [0. 0. 0. 0. 1.]
        event_t = tf.sparse.to_dense(feat[_EVENT_FEATURE])  # spare tensor to dense
        rv_t = tf.sparse.to_dense(feat[_RV_FEATURE])
        # print ('indices', query_st.indices[0][0]) #which indix has first value
        print('event values', event_t[0])
        # check slicing notification!
        print('rv values', rv_t[0, :5, :10].numpy()) # sample 1, first 5 rvs, first 10 features


if __name__ == '__main__':
    unittest.main()
    print('sampling tests finished')
