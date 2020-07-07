from rvranking.dataPrep import tstart, tend, td_perwk, TD_SEQ
from rvranking.globalVars import RELEVANCE
from datetime import timedelta


def get_week_boundaries(s):
    st = tstart + timedelta(minutes=s.start * TD_SEQ)
    wd = st.weekday()
    wk_start_td = timedelta(days=wd, hours=st.hour, minutes=st.minute)

    wk_start_td_min = (wk_start_td.seconds/60 + wk_start_td.days * 24 * 60)
    wk_start = int(s.start - (wk_start_td_min / TD_SEQ))
    wk_end = int(wk_start + td_perwk) - 1  # last element of week, pandas slice includes last element
    return wk_start, wk_end


def get_evtypes(s):
    # evtypes id from excel
    # in future: from evtypes list with same master type
    dgs = [15, 16, 17]
    anhs = [20, 21]
    if s.evtype in dgs:
        return dgs
    else:
        return anhs


def predict_rv(s):
    """the higher prio value the lower prio of rv"""
    wk_start, wk_end = get_week_boundaries(s)
    st_range = s.rangestart
    et_range = s.start

    evs_same_mev = get_evtypes(s)
    rvli = s.rvli
    rv_sort_list = []
    for rv in rvli:
        if rv.id == s.rv_ff:
            prio = -1
        else:
            prio = prio_masterev(rv, wk_start, wk_end, evs_same_mev)
            prio += prio_freetime(rv, st_range, et_range)
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
            mrr = 1/rank
            break
    if mrr == 0:
        print('rv not found')
    return ndcg1, mrr


def prio_masterev(rv, wk_start, wk_end, evs_same_mev):
    """get sum of events with same mastertype"""
    # pandas slice includes last element
    # in eventordering its not sum of time but # of events
    prio = rv.tline.loc[str(wk_start):str(wk_end)].isin(evs_same_mev).sum()
    wk_b_st = wk_start - td_perwk
    wk_b_et = wk_end - td_perwk
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

    prio = round(1/(sum0+1), 3)
    return prio