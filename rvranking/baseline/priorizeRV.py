import re
from rvranking.dataPrep import tstart, tend, TD_PERWK, MO_START, KMAX, ev_types
from rvranking.globalVars import RELEVANCE, _RV_FEATURES
from datetime import timedelta, datetime


def get_week_boundaries(s):
    weeks = (s.start - MO_START) // TD_PERWK
    wk_start_int = int(MO_START + weeks * TD_PERWK)  #
    wk_end_int = int(wk_start_int + TD_PERWK) - 1  # last element of week, pandas slice includes last element
    if wk_start_int < 0:
        wk_start_int = 0
    if wk_end_int > KMAX:
        wk_end_int = KMAX
    return wk_start_int, wk_end_int


def get_evtypes(s):
    def get_ids(names):
        ids = ev_types.loc[ev_types['Name'].isin(names)]['Id'].values
        return ids

    dgs_n = ['DG', 'DG_Block']
    anhs_n = ['B1', 'B2']
    dgs = get_ids(dgs_n)
    anhs = get_ids(anhs_n)
    if s.evtype in dgs:
        return dgs
    else:
        return anhs


def predict_rv(s):
    """the higher prio value the lower prio of rv"""

    def prio_rv_ff():
        nonlocal prio
        if 'rv_ff' in _RV_FEATURES:
            if rv.rv_ff == 1:
                p = -1
                prio = p
                return True
        return False

    def prio_time():
        nonlocal prio
        r = re.compile('.*tline.*')
        le_f = len(list(filter(r.match, _RV_FEATURES)))
        if le_f > 0:
            p = prio_masterev(rv, wk_start, wk_end, evs_same_mev)
            p += prio_freetime(rv, st_range, et_range)
            prio = p

    def prio_hwx_uma():
        nonlocal prio
        if ('hwx' in _RV_FEATURES and rv.hwx == 1) or \
        ('uma' in _RV_FEATURES and rv.uma == 1):
            prio = -2

    wk_start, wk_end = get_week_boundaries(s)
    st_range = s.rangestart
    et_range = s.start

    evs_same_mev = get_evtypes(s)
    rvli = s.rvli
    rv_sort_list = []
    for rv in rvli:
        prio = 0
        if prio_rv_ff() is False:
            prio_time()
        prio_hwx_uma()

        rv_sort_list.append((rv, prio))
    rv_sort_list.sort(key=lambda rv_p: rv_p[1], reverse=False)

    # simplified as just look at 1st -> if correct ngcd = 1, else = 0
    ndcg1, mrr = 0, 0
    for i, rv_p in enumerate(rv_sort_list):
        rv = rv_p[0]
        rank = i + 1
        if s.rv == rv.id:
            if rank == 1:
                ndcg1 = 1
            mrr = 1 / rank
            break
    if mrr == 0:
        print('rv not found')
    return ndcg1, mrr


def prio_masterev(rv, wk_start, wk_end, evs_same_mev):
    """get sum of events with same mastertype"""
    # pandas slice includes last element
    # in eventordering its not sum of time but # of events
    prio = rv.tline.loc[str(wk_start):str(wk_end)].isin(evs_same_mev).sum()

    wk_b_st = wk_start - TD_PERWK
    wk_b_et = wk_end - TD_PERWK
    try:
        prio += .5 * rv.tline.loc[str(wk_b_st):str(wk_b_et)].isin(evs_same_mev).sum()
    except KeyError:  # if week before is not included
        pass
    return prio


def prio_freetime(rv, st, et):
    """get reciprocical value of sum of # of zeros"""
    try:
        sum0 = rv.tline.loc[str(st):str(et)].isin([0]).sum()  # counts zero values
    except KeyError:
        sum0 = 0

    prio = round(1 / (sum0 + 1), 3)
    return prio
