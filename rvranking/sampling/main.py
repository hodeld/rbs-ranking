from rvranking.globalVars import RELEVANCE, _SAMPLING, _LIST_SIZE, _SAME_TEST_TRAIN, _EVENT_FEATURES, _RV_FEATURES
from rvranking.dataPrep import TD_PERWK, WEEKS_A, WEEKS_B, KMAX
from rvranking.logs import hplogger
from rvranking.sampling.fakeSampling import iterate_samples

from rvranking.sampling.samplingClasses import SampleList
from rvranking.sampling.setTimeLines import cut_timelines, tline_fn_d

# in colab
import random


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
    ist = int(s.start - (TD_PERWK * weeks_before))
    if ist < 0:
        return False
        # ist = 0
    iet = int(s.start + TD_PERWK * weeks_after)
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
    if rv.tline.loc[str(s.start):str(s.end)].any():  # if not all are 0
        if rv.id == s.rv:  # that should not be the case:
            print('sample where rv is not available', s.id, s.rv)
        return False
    else:
        return True


def check_evtype(rv, s, rvfirstev_spec):
    try:
        firstev = rvfirstev_spec.loc[rv.id, str(s.evtype)]  # 1, None or date_int
    except KeyError:
        return False
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
    rvs = [s.rv] + list(s.rv_eq)  # correct answer; rv_eq without s.rv
    cnt_relevant_rvs = 0
    relevant_rvs = []
    rvli = s.rvli.copy()  # needs to be seperate list with same items to remove items and iterate over!
    for rv in rvli:
        if rv.id in rvs:
            rv.relevance = RELEVANCE
            cnt_relevant_rvs += 1
            relevant_rvs.append(rv)
            s.rvli.remove(rv)  # will be added later
    if cnt_relevant_rvs == 0:
        return False
    int_slice = _LIST_SIZE - cnt_relevant_rvs
    random.shuffle(s.rvli)
    s.rvli = s.rvli[:int_slice] + relevant_rvs
    return True


def normalize_features(s):
    # todo randomize tline -> evtype
    def rvs_random():
        randint = random.randint(1, 10)  # 1...10
        s.rv += randint
        for r in s.rvli:
            r.id += randint

    def rvs():
        for i, r in enumerate(s.rvli, 1):
            if r.id == s.rv:
                s.rv = i
            if r.id == s.rv_ff:
                r.rv_ff = 1
                #s.rv_ff = i
            r.id = i

    if 'rv_ff' in _EVENT_FEATURES or 'id' in _RV_FEATURES:  # s.features_attrs
        random.shuffle(s.rvli)
        rvs()


def prep_samples_list(sample_list_all, rvlist_all, train_ratio,
                      timelines_spec, rvfirstev_spec, allevents_spec):

    def get_list(sample_list):
        i = 0
        nonlocal k_er, k_tr, k_rr, k_ls

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
            normalize_features(s)

            nr_rvs = len(s.rvli)
            if nr_rvs == 0:
                k_er += 1
                continue
            if nr_rvs < _LIST_SIZE:
                k_ls += 1
            sample_list_prep.append(s)
            nr_rvs_li.append(nr_rvs)

    sample_list_s = SampleList(sample_list_all)
    nr_rvs_li = []
    rvli_d = {}
    sample_list_prep = []
    k_er, k_tr, k_rr, k_ls = 0, 0, 0, 0

    get_list(sample_list_s)

    if len(sample_list_prep) > 50:
        hplogger.info('empty rv list: ' + str(k_er))
        hplogger.info('timerange too short: ' + str(k_tr))
        hplogger.info('0relevantRV: ' + str(k_rr))
        hplogger.info('mean_rvs: ' + str(sum(nr_rvs_li) / len(nr_rvs_li)))
        hplogger.info('rvs_tooshort: ' + str(k_ls))

    print('empty rv list: ' + str(k_er))
    print('timerange too short: ' + str(k_tr))
    print('0relevantRV: ' + str(k_rr))
    print('mean_rvs: ', str((sum(nr_rvs_li)+1) / (len(nr_rvs_li)+1)))
    print('s with nr_rvs<_LIST_SIZE: ', str(k_ls))

    orig_list_len = len(sample_list_all)
    prep_list_len = len(sample_list_prep)

    if _SAME_TEST_TRAIN:
        s_list_train, s_list_test = iterate_samples(sample_list_prep)
    else:
        slice_int = int(prep_list_len * train_ratio)
        random.shuffle(sample_list_prep)
        s_list_train = sample_list_prep[:slice_int]
        s_list_test = sample_list_prep[slice_int:]
    msg = ' '.join(['length orig, prep, train, test:',
                    str(orig_list_len),
                    str(prep_list_len),
                    str(len(s_list_train)),
                    str(len(s_list_test)),
                    ])
    print(msg)
    hplogger.info(msg)
    return s_list_train, s_list_test