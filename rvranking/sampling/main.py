from rvranking.globalVars import RELEVANCE, _SAMPLING
from rvranking.dataPrep import *
from rvranking.logs import hplogger

# in colab
import random

from rvranking.sampling.samplingClasses import SampleList
from rvranking.sampling.setTimeLines import cut_timelines, tline_fn_d


def get_example_features(s, rvli_d, rvlist_all, sample_list,
                         timelines_spec, rvfirstev_spec, allevents_spec):
    if s.rvli is None:  # all day events have already rvli
        tline_vars = [s, rvlist_all, sample_list,
                      timelines_spec, allevents_spec,
                      rvli_d]
        tline_fn_d[_SAMPLING](tline_vars)
    cut_timelines(s)
    get_pot_rvs(s, rvfirstev_spec)


def get_timerange(s):
    weeks_before = WEEKS_B
    weeks_after = WEEKS_A
    td_perwk = PPH * 24 * 7
    ist = int(s.start - (td_perwk * weeks_before))
    if ist < 0:
        return False
        # ist = 0
    iet = int(s.start + td_perwk * weeks_after)
    if iet > KMAX:
        # iet = KMAX
        return False
    s.rangestart = ist
    s.rangeend = iet
    return True


def get_pot_rvs(s, rvfirstev_spec):
    rvlist = s.rvli.copy() # needs to be seperate list with same items to remove items and iterate over!
    for rv in rvlist:
        if check_availability(rv, s) == False:
            s.rvli.remove(rv)
        elif check_evtype(rv, s, rvfirstev_spec) == False:
            s.rvli.remove(rv)
    check_feat(rv, s)  # probably better leave features


def check_availability(rv, s):
    try:
        rv.tline.loc[str(s.start):str(s.end)].any()
    except(KeyError, ValueError) as e:
        print('event created outside timerange: err, start, end', e, s.start, s.end)
        return False
    if rv.tline.loc[str(s.start):str(s.end)].any():  # not all are 0
        return False
    else:
        return True


def check_evtype(rv, s, rvfirstev_spec):
    firstev = rvfirstev_spec.loc[rv.id, str(s.evtype)]  # 1, None or date_int
    if firstev == None:
        return False
    if firstev <= s.start:
        return True
    else:
        return False


def check_feat(rv, s):
    if s.gespever > 0:
        rvlist = s.rvli
        rvlist = rvlist.filter(s.gespever, 'sex')
        s.rvli = rvlist
        # UMA and HWX just left with features


def assign_relevance(s):
    rvs = [s.rv] + list(s.rv_eq)  # correct answer
    cnt_relevant_rvs = 0
    for rv in s.rvli:
        if rv.id in rvs:
            rv.relevance = RELEVANCE
            cnt_relevant_rvs += 1
    if cnt_relevant_rvs == 0:
        return False
    return True


def prep_samples_list(sample_list_all, rvlist_all, train_ratio,
                      timelines_spec, rvfirstev_spec, allevents_spec):
    def get_list(sample_list):
        k_er, k_tr, k_rr,  = 0, 0, 0
        rvli_d = {}
        i = 0
        sample_list_prep = []
        nr_rvs_li = []
        for s in sample_list:
            i += 1
            if not get_timerange(s):
                k_tr += 1
                continue
            get_example_features(s, rvli_d, rvlist_all, sample_list,
                                 timelines_spec=timelines_spec,
                                 rvfirstev_spec=rvfirstev_spec,
                                 allevents_spec=allevents_spec)  # rvs, SHOULD BE ALWAYS SAME # OF RVS

            if not assign_relevance(s):
                k_rr += 1
                continue

            nr_rvs = len(s.rvli)
            if nr_rvs == 0:
                k_er += 1
                continue
            sample_list_prep.append(s)
            nr_rvs_li.append(nr_rvs)
        if len(sample_list_prep) > 50:
            hplogger.info('_'.join([mode_name, 'empty rv list: ' + str(k_er)]))
            hplogger.info('_'.join([mode_name, 'timerange too short: ' + str(k_tr)]))
            hplogger.info('_'.join([mode_name, 'no relevant rv in rvlist: ' + str(k_rr)]))
            hplogger.info('_'.join([mode_name, 'mean_rvs: ' + str(sum(nr_rvs_li)/len(nr_rvs_li))]))
            print('mean_rvs: ', mode_name, str(sum(nr_rvs_li)/len(nr_rvs_li)))

        return sample_list_prep

    orig_list_len = len(sample_list_all)
    slice_int = int(orig_list_len * train_ratio)
    random.shuffle(sample_list_all)
    sample_list_train = SampleList(sample_list_all[:slice_int])
    sample_list_test = SampleList(sample_list_all[slice_int:])

    mode_name = 'train'
    s_list_train = get_list(sample_list_train)
    mode_name = 'test'
    s_list_test = get_list(sample_list_test)
    msg = ' '.join(['orig_list_len, train_len, test_list:',
                    str(orig_list_len),
                    str(len(s_list_train)),
                    str(len(s_list_test))
                    ])
    print(msg)
    hplogger.info(msg)
    return s_list_train, s_list_test

