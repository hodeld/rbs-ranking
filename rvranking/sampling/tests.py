import random
import unittest
import numpy as np

from rvranking.rankingComponents import input_fn
from rvranking.sampling.elwcWrite import write_elwc
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RVList, RV
from rvranking.globalVars import _TRAIN_DATA_PATH, _RV_FEATURE, _EVENT_FEATURE, _EVENT_FEATURES, _RV_FEATURES
from rvranking.dataPrep import samples, timelines, allevents, prep_samples, get_timelines_raw, \
    prep_timelines, prep_allevents, TD_PERWK, WEEKS_B, WEEKS_A, KMAX, rvs, rvfirstev, get_test_files, PPH, RV_TLINE_LEN
import tensorflow as tf

from rvranking.sampling.scikitDataGet import x_y_data


def sample_test(cls, s, tlines, allevs):
    rv = s.rv
    tline = tlines.loc[str(rv)]
    ev_tline_val = tline.loc[str(s.start):str(s.end)].values
    ev = allevs.loc[s.id]
    if s.teams:
        pass
    else:
        cls.assertEqual(ev['End'], s.end)
    cls.assertEqual(ev['Start'], s.start)
    cls.assertEqual(ev['Rv'], s.rv)
    cls.assertEqual(ev['Type'], s.evtype)
    cls.assertEqual(ev['Rv added'], s.rv_added)
    evtype = s.evtype
    cls.assertEqual((evtype == ev_tline_val).all(), True)


def sampling_test(cls, s, allevs_all=None):
    if allevs_all is not None:
        oneday = PPH * 24
        range_start = int(s.rangestart - oneday)
        if range_start < 0:
            range_start = 0
        range_end = int(s.rangeend + oneday)
        if range_end > KMAX:
            range_end = KMAX

        idx = np.where((allevs_all['Start'] >= range_start) & (allevs_all['End'] <= range_end))
        evs_range = allevs_all.iloc[idx]

    for r in s.rvli:
        cls.assertEqual(r.tline.size, RV_TLINE_LEN)
        cls.assertEqual(r.tline.iloc[WEEKS_B * TD_PERWK], 0)  # WEEKS_B  * TD_PERWK
        cls.assertEqual(r.tline.loc[str(s.start):str(s.end)].any(), False)  # all zero
        if allevs_all is not None:
            index_vals = r.tline.index.values
            min_t = int(index_vals[0])
            max_t = int(index_vals[-1])
            rv_added = s.rv_added
            allevs = evs_range[evs_range['Rv'] == r.id]
            allevs = allevs[allevs['Rv added'] >= rv_added]
            for eid, row in allevs.iterrows():
                if eid == s.id:
                    continue
                st = row['Start']
                et = row['End']
                if st < min_t or et > max_t:
                    continue
                cls.assertEqual(st, s.start)
                r_tline_ev = r.tline.loc[str(st):str(et)].values
                cls.assertEqual((0 == r_tline_ev).all(), True)
        if 'rv_ff' in _EVENT_FEATURES or 'rv_ff' in _RV_FEATURES:
            if s.rv_ff == r.id:
                cls.assertEqual(r.rv_ff, 1)


class TestSampling(unittest.TestCase):

    def test_samples(self):
        sample_list_all = [Sample(s) for i, s in samples.iterrows()]
        random.shuffle(sample_list_all)
        tlines = timelines
        allevs = allevents
        for s in sample_list_all:
            sample_test(self, s, tlines, allevs)

    def test_prediction_samples(self):
        samples_pred, tlines, allevs, rvs, rvfirstev = get_test_files()
        sample_list_all = [Sample(s) for i, s in samples_pred.iterrows()]
        s = sample_list_all[0]
        ist = int(s.start - (TD_PERWK * WEEKS_B))
        iet = int(s.start + TD_PERWK * WEEKS_A)
        self.assertGreaterEqual(ist, 0)  # ist >= 0
        self.assertLessEqual(iet, KMAX)
        for s in sample_list_all:
            sample_test(self, s, tlines, allevs)

    def test_sampling(self):
        less_samples = samples.sample(n=50) # random rows
        sample_list_all = [Sample(s) for i, s in less_samples.iterrows()]
        rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])
        train_ratio = 0.7

        sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                                rvlist_all,
                                                                train_ratio=train_ratio,
                                                                timelines_spec=timelines,
                                                                rvfirstev_spec=rvfirstev,
                                                                allevents_spec=allevents
                                                                )
        s_list_tot = sample_list_train + sample_list_test
        assert len(s_list_tot) > 0
        for s in s_list_tot:
            sampling_test(self, s)

        x_train, y_train, xy_train = x_y_data(s_list_tot)

    def test_prediction_sampling(self):
        samples_pred, tlines_pred, allevs_pred, rvs_pred, rvfirstev_pred = get_test_files()
        sample_list_all = [Sample(s) for i, s in samples_pred.iterrows()]
        rvlist_all = RVList([RV(r) for i, r in rvs_pred.iterrows()])
        train_ratio = 0.7

        sample_list_train, sample_list_test = prep_samples_list(sample_list_all,
                                                                rvlist_all,
                                                                train_ratio=train_ratio,
                                                                timelines_spec=tlines_pred,
                                                                rvfirstev_spec=rvfirstev_pred,
                                                                allevents_spec=allevs_pred
                                                                )
        s_list_tot = sample_list_train + sample_list_test
        for s in s_list_tot:
            sampling_test(self, s, allevs_pred)
            self.assertEqual(len(s.rvli), 5)
        x_train, y_train, xy_train = x_y_data(s_list_tot)

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
