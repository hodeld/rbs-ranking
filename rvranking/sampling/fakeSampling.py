from rvranking.globalVars import change_var
from rvranking.logs import hplogger
import pandas as pd

from rvranking.sampling.samplingClasses import RV


def iterate_samples_by_id(sample_list_prep):
    sid = change_var['sample_id']
    for s in sample_list_prep:
        if s.id == sid:
            s0 = s
            break
    s_list_train, s_list_test = get_train_test_fake_sample(s0)
    return s_list_train, s_list_test


def iterate_samples_by_nr(sample_list_prep):
    s0 = sample_list_prep[change_var['sample_nr']]
    hplogger.info('s0.id: ' + str(s0.id))
    print(s0.id)
    s_list_train, s_list_test = get_train_test_real_sample(s0)
    return s_list_train, s_list_test


def get_train_test_fake_sample(s0):
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
    for r in s0.rvli:
        r.relevance = 0

    s0.rvli[0].relevance = 1
    compare_tlines(s0)

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
    return s_list_train, s_list_test


def get_train_test_real_sample(s0):
    rele = [r.relevance for r in s0.rvli]
    compare_tlines(s0)
    print('rv-relevance', rele)
    hplogger.info('relevance_list: ' + str(rele))
    s_list_train = []
    for i in range(15):
        s_list_train.append(s0)
    s_list_test = s_list_train
    return s_list_train, s_list_test


def compare_tlines(s):
    for r in s.rvli:
        if r.relevance == 1:
            tlinex = r.tline
            ridx = r.id
    for r in s.rvli:
        if r.tline.equals(tlinex) and r.id != ridx:
            hplogger.info('same tline as r_relevant : ' + str(r.id))
            print('same tline as r_relevant s, rv', str(s.id), str(r.id))