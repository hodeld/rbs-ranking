from rvranking.globalVars import change_var
from rvranking.logs import hplogger
import pandas as pd

from rvranking.sampling.samplingClasses import RV


def iterate_samples(sample_list_prep):
    sid = change_var['sample_id']
    for s in sample_list_prep:
        if s.id == sid:
            s0 = s
            break
    # s0 = sample_list_prep[change_var['sample_id']]
    print(s0.id)
    tline0_li = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    tline0 = pd.Series(tline0_li)
    # hplogger.info('tline0: ' + str(tline0_li))
    print('rvlist len', len(s0.rvli))
    rvli = []
    s0.rv = 1
    for i in range(1, 6):
        rvvals = (i, 1, i)
        r = RV(rvvals)
        if r.id == s0.rv:
            r.relevance = 1
        r.tline = tline0
        rvli.append(r)

    s0.rvli = s0.rvli[:5]
    tlinex = s0.rvli[0].tline
    ridx = s0.rvli[0].id
    for r in s0.rvli:
        r.relevance = 0
        if r.tline.equals(tlinex) and r.id != ridx:
            hplogger.info('same tline as r_relevant : ' + str(r.id))

    s0.rvli[0].relevance = 1

    # hplogger.info('rvvals: (i, 1, i)')
    # hplogger.info('s0.rv: 1, ')
    hplogger.info('s0.rvli[0].relevance: 1')
    hplogger.info('s0.rvli[1:] .relevance: 0')
    rele = [r.relevance for r in s0.rvli]
    print('rv-relevance', rele)
    hplogger.info('relevance_list: ' + str(rele))
    # hplogger.info('s0.rv: ' + '1')

    # hplogger.info('sample_id: ' + str(s0.id))
    s_list_train = []
    for i in range(15):
        s_list_train.append(s0)
    s_list_test = s_list_train