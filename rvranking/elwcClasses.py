from rvranking.globalVars import *
from rvranking.dataPrep import *


import copy
import operator
import random


class Sample():
    """class for events for ranking problem"""

    def __init__(self, sample_li):
        (s_id, location, dbid, day_evs, sevs, rv_eq,
         start, end, rv, group, cat,
         evtype, rv_ff, gespever, hwx, uma) = sample_li

        day = int(start // (24 * PPH) * (24 * PPH))
        locday = str(location) + '-' + str(day)

        def get_li(li_str):
            if isinstance(li_str, str):
                li = [int(s) for s in li_str.split(';')]
            else:
                li = []
            return li

        day_evs = get_li(day_evs)
        rv_eq = get_li(rv_eq)
        sevs = get_li(sevs)

        self.location = location
        self.day = (start // 24 * PPH) * (24 * PPH)
        self.locday = locday
        self.start = start
        self.end = end
        self.rangestart = 0
        self.rangeend = 0
        self.rv = rv
        self.rv_eq = rv_eq
        self.id = s_id
        self.evtype = evtype
        self.group = group
        self.day_evs = day_evs
        self.sevs = sevs
        self.rv_ff = rv_ff
        self.gespever = gespever
        self.hwx = hwx
        self.uma = uma
        self.rvli = None

    def features(self):
        flist = [
            self.evtype,
            self.rv_ff,
            self.gespever,
            self.hwx,
            self.uma,
        ]
        return flist


class SampleList(list):
    '''base class for list of samples'''

    def get(self, variable_value, item_attr='id'):
        vv = variable_value
        ra = item_attr
        f = operator.attrgetter(ra)
        for s in self:
            if f(s) == vv:
                return s
        return None


class RV():
    '''base class for rvs'''

    def __init__(self, rvvals
                 ):
        (rvid, location,
         sex) = rvvals
        self.id = rvid
        self.location = location
        self.sex = sex
        self.relevance = 0
        self.tline = None
        self.prediction = 0

    def features(self):
        """concententate
        sex: 1 or 2
        tline: [1 … 20]"""
        flist = [
            self.sex,
            *self.tline,
        ]
        return flist

    def features_fake(self):
        if self.relevance == RELEVANCE:  # if it is correct rv
            return [1]
        else:
            return [0]



class RVList(list):
    '''base class for list of rvs'''

    def filter(self, variable_value, rv_attr):
        vv = variable_value
        ra = rv_attr
        f = operator.attrgetter(ra)
        newli = RVList(filter(lambda x: f(x) == vv, self))  # calls x.ev
        return newli

    def get(self, variable_value, rv_attr='id'):
        vv = variable_value
        ra = rv_attr
        f = operator.attrgetter(ra)
        for rv in self:
            if f(rv) == vv:
                return rv

        return None


def get_example_features(s, rvli_d, rvlist_all, sample_list):
    if s.rvli == None:
        rvli = get_rvlist(s, rvli_d, rvlist_all)
        s.rvli = rvli
        set_tlines(s, sample_list)
    cut_timelines(s)
    get_pot_rvs(s)


def get_rvlist(s, rvli_d, rvlist_all):
    rvlist = rvli_d.get(s.locday, None)
    if rvlist:
        return rvlist

    rvlist = rvlist_all.filter(s.location, 'location')  # [rv for rv in rvs if rv.location == loc_id]
    rvlist = get_rv_timelines(rvlist, s)  # same for all same day events
    rvli_d[s.locday] = rvlist
    return rvlist


def get_rv_timelines(rvlist, s):
    for rv in rvlist:
        tline = timelines.loc[str(rv.id)]
        rv.tline = tline
    return rvlist


def cut_timelines(s):
    ist = s.rangestart
    iet = s.rangeend
    for rv in s.rvli:
        rv.tline = rv.tline.loc[str(ist):str(iet)]


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


def set_tlines(s, sample_list):
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
            print('None ev in day evs')
            continue

        evs += ev.sevs
        sample_li.append(ev)

    for eid in evs:
        ev = allevents.loc[eid]
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
    # day_evs have same rvli -> exact cutting is in next step
    for s in sample_li:
        s.rvli = copy.deepcopy(rvli)


def get_pot_rvs(s):
    rvlist = s.rvli
    for rv in rvlist:
        if check_availability(rv, s) == False:
            rvlist.remove(rv)
        elif check_evtype(rv, s) == False:
            rvlist.remove(rv)
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


def check_evtype(rv, s):
    firstev = rvfirstev.loc[rv.id, str(s.evtype)]  # 1, None or date_int
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



def prep_samples_list(sample_list_all, rvlist_all, train_ratio):
    def get_list(sample_list):
        k_emptyrvli = 0
        rvli_d = {}
        i = 0
        sample_list_prep = []
        for s in sample_list:
            i += 1
            if not get_timerange(s):
                print(i, s.id, 'timerange too short')
                continue
            get_example_features(s, rvli_d, rvlist_all, sample_list)  # rvs, SHOULD BE ALWAYS SAME # OF RVS

            if not assign_relevance(s):
                print(i, s.id, 'no relevant rv list')
                continue

            if len(s.rvli) == 0:
                k_emptyrvli += 1
                print(i, s.id, 'no rvs in list')
                continue
            sample_list_prep.append(s)
        print('k_emptyrvli', k_emptyrvli)
        return sample_list_prep

    orig_list_len = len(sample_list_all)
    slice_int = int(orig_list_len * train_ratio)
    random.shuffle(sample_list_all)
    sample_list_train = SampleList(sample_list_all[:slice_int])
    sample_list_test = SampleList(sample_list_all[slice_int:])

    s_list_train = get_list(sample_list_train)
    s_list_test = get_list(sample_list_test)
    print('orig_list_len, train_len, test_list:', orig_list_len, len(s_list_train), len(s_list_test))
    return s_list_train, s_list_test

