from rvranking.globalVars import RELEVANCE, _SAMPLING
from rvranking.dataPrep import *
from rvranking.logs import hplogger

# in colab
import copy
import random

from rvranking.sampling.samplingClasses import SampleList


def get_example_features(s, rvli_d, rvlist_all, sample_list,
                         timelines_spec, rvfirstev_spec, allevents_spec):
    if s.rvli is None:  # all day events have already rvli
        if _SAMPLING == 'filling_up':
            s.rvli = get_rvlist_fresh(s, rvlist_all, timelines_spec)
            set_tlines_fillingup(s, sample_list, allevents_spec)
        else:
            s.rvli = get_rvlist_from_dict(s, rvli_d, rvlist_all, timelines_spec)
            set_tlines_allzeroes(s, sample_list, allevents_spec)
    cut_timelines(s)
    get_pot_rvs(s, rvfirstev_spec)


def get_rvlist_fresh(s, rvlist_all, timelines_spec):
    rvlist = rvlist_all.filter(s.location, 'location')  # [rv for rv in rvs if rv.location == loc_id]
    rvlist = get_rv_timelines(timelines_spec, rvlist)  # same for all same day events
    return rvlist


def get_rvlist_from_dict(s, rvli_d, rvlist_all, timelines_spec):
    rvlist = rvli_d.get(s.locday, None)
    if rvlist:
        return rvlist

    rvlist = rvlist_all.filter(s.location, 'location')  # [rv for rv in rvs if rv.location == loc_id]
    rvlist = get_rv_timelines(timelines_spec, rvlist)  # same for all same day events
    rvli_d[s.locday] = rvlist
    return rvlist


def get_rv_timelines(timelines_spec, rvlist):
    for rv in rvlist:
        tline = timelines_spec.loc[str(rv.id)]
        rv.tline = tline
    return rvlist


def cut_timelines(s):
    ist = s.rangestart
    iet = s.rangeend
    for rv in s.rvli:
        try:
            rv.tline = rv.tline.loc[str(ist):str(iet)]
        except:
            print('shit happens here')


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


def remove_evs(evs, rvli, allevs_spec):
    for eid in evs:
        ev = allevs_spec.loc[eid]
        rvid = ev['Rv']
        if type(rvid) == pd.core.series.Series:
            print('series err, rvid, eid, ev -> due to 2 gs in one event', rvid, eid, ev)
            continue
        rv = rvli.get(rvid)
        if rv is None:
            continue  # only as day_evs for all locations
        try:
            rv.tline.loc[str(ev['Start']):str(ev['End'])] = 0
        except(KeyError, ValueError) as e:
            print('event created outside timerange: err, start, end', e, ev['Start'], ev['End'])


def set_tlines_fillingup(s, sample_list, allevents_spec):
    """
    - day_evs are all events CREATED on the same day (or later) including sample itself
    - iterate through day events (samples):
        - delete ev and all connected sevs for assigned rv for this sample
        - copy rvlist and assign to next sample
    - last sample most empty timelines in rvlist
    - cut to the exact length is after that (in other function)
    """

    day_evs = s.day_evs  # rendomized only part of it to zero

    rvli = s.rvli
    for eid in day_evs:
        si = sample_list.get(eid)
        if si is None:  # when samples are sliced
            # print('None ev in day evs')
            continue
        evs = [eid]
        evs += si.sevs  # can be empty list
        remove_evs(evs, rvli, allevents_spec)
        rvli = copy.deepcopy(rvli)
        si.rvli = rvli


def set_tlines_allzeroes(s, sample_list, allevents_spec):
    """
    - day_evs are all events CREATED on the same day (or later
    - delete for each event all connected events for assigned rv
    - rvlist gets more and more zeroes
    - rvlist can be copied for all day events and then cut to the exact length

    """
    rvli = s.rvli
    evs = []
    evs += s.sevs  # can be empty list
    day_evs = s.day_evs  # rendomized only part of it to zero
    evs += day_evs
    sample_li = []

    for eid in day_evs:
        ev = sample_list.get(eid)
        if ev == None:  # when samples are sliced
            continue

        evs += ev.sevs
        sample_li.append(ev)

    remove_evs(evs, rvli, allevents_spec)

    # day_evs have same rvli -> exact cutting is in next step
    for s in sample_li:
        s.rvli = copy.deepcopy(rvli)


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

            if len(s.rvli) == 0:
                k_er += 1
                continue
            sample_list_prep.append(s)
        hplogger.info(' - '.join([mode_name, 'empty rv list: ' + str(k_er)]))
        hplogger.info(' - '.join([mode_name, 'timerange too short: ' + str(k_tr)]))
        hplogger.info(' - '.join([mode_name, 'no relevant rv in rvlist: ' + str(k_rr)]))

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

