import random
import unittest
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.sampling.elwcFunctions import write_context_examples
from rvranking.globalVars import _TRAIN_DATA_PATH, _TEST_DATA_PATH, _BASE_TF_DATA_PATH
from rvranking.dataPrep import samples, rvs, timelines, rvfirstev, allevents

# START JUPYPTER
from pathlib import Path

from rvranking.sampling.setTimeLines import get_rvlist_fresh


class TestSampling(unittest.TestCase):

    def test_samples(self):
        sample_list_all = [Sample(s) for i, s in samples.iterrows()]  # samples.iloc[:5].iterrows()])
        random.shuffle(sample_list_all)

        for s in sample_list_all[:10]:  # test 10 samples
            rv = s.rv
            tline = timelines.loc[str(rv)]
            ev_tline_val = tline.loc[str(s.start):str(s.end-1)].values
            ev = allevents.loc[s.id]
            print('id', s.id)
            if s.teams:
                pass
            else:
                self.assertEqual(ev['End'], s.end)
            self.assertEqual(ev['Start'], s.start)
            self.assertEqual(ev['Rv'], s.rv)
            self.assertEqual(ev['Type'], s.evtype)
            evtype = s.evtype
            print('id, tline', s.id, ev_tline_val)
            self.assertEqual((evtype == ev_tline_val).all(), True)
            print(s.id)




if __name__ == '__main__':
    unittest.main()
    print('sampling tests finished)')