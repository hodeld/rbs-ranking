import copy

import pandas as pd


def filling_up(tline_vars):
    (s, rvlist_all, sample_list,
     timelines_spec, allevents_spec) = tline_vars[:-1]

    s.rvli = get_rvlist_fresh(s, rvlist_all, timelines_spec)
    set_tlines_fillingup(s, sample_list, allevents_spec)


def filling_opposite(tline_vars):
    (s, rvlist_all, sample_list,
     timelines_spec, allevents_spec) = tline_vars[:-1]

    s.rvli = get_rvlist_fresh(s, rvlist_all, timelines_spec)
    set_tlines_fill_oposite(s, sample_list, allevents_spec)


def all_zero(tline_vars):
    (s, rvlist_all, sample_list,
     timelines_spec, allevents_spec,
     rvli_d) = tline_vars

    s.rvli = get_rvlist_from_dict(s, rvli_d, rvlist_all, timelines_spec)
    set_tlines_allzeroes(s, sample_list, allevents_spec)


tline_fn_d = {'filling_up': filling_up,
              'filling_opposite': filling_opposite,
              'all_zero': all_zero,
              }


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
        except (KeyError, ValueError):
            print('event outside timerange: rv, ist, iet', rv.id, ist, iet)


def remove_ev_rv(rv, ev):
    if rv is None:
        return False
    try:
        rv.tline.loc[str(ev['Start']):str(ev['End'])] = 0
    except(KeyError, ValueError) as e:
        print('event created outside timerange: rv.id, start, end', rv.id, ev['Start'], ev['End'])
        return False
    return True


def remove_evs(evs, rvli, allevs_spec):
    for eid in evs:
        ev = allevs_spec.loc[eid]
        rvid = ev['Rv']
        if type(rvid) == pd.core.series.Series:
            print('series err, rvid, eid, ev -> due to 2 gs in one event', rvid, eid, ev)
            continue
        rv = rvli.get(rvid)
        if remove_ev_rv(rv, ev) is False:
            rvli.remove(rv)


def rm_evs_all_rv(evs, rvli, allevs_spec):
    for eid in evs:
        ev = allevs_spec.loc[eid]
        rvli_copy = rvli.copy()
        for rv in rvli_copy:
            if remove_ev_rv(rv, ev) is False:
                rvli.remove(rv)


def set_tlines_fill_oposite(s, sample_list, allevents_spec):
    """
    - day_evs are all events CREATED on the same day (or later) including sample itself
    - iterate through day events (samples) 1st time:
        - delete ev and all connected sevs for all rvs for this sample
        - gives rvlist
    - iterate through day events (samples) 2nd time:
        - assign rvlist
        - add event + sevs for assigned rv
        - copy rvlist and assign to next sample
    - last sample most full timelines in rvlist
    - cut to the exact length is after that (in other function)
    """

    day_evs = s.day_evs  # rendomized only part of it to zero

    rvli = s.rvli
    day_samples = []
    for eid in day_evs:
        si = sample_list.get(eid)
        if si is None:  # when samples are sliced in train and test
            # print('None ev in day evs')
            continue
        evs = [eid]
        evs += si.sevs  # can be empty list
        rm_evs_all_rv(evs, rvli, allevents_spec)
        day_samples.append(si)

    rvli = copy.deepcopy(rvli)  # as s is not always first sample in day_evs ->
    for i, si in enumerate(day_samples):
        si.rvli = rvli
        rvli = copy.deepcopy(rvli)
        evs = [si.id]
        evs += si.sevs  # can be empty list
        rvid = si.rv
        rv = rvli.get(rvid)
        for eid in evs:
            ev = allevents_spec.loc[eid]
            rv.tline.loc[str(ev['Start']):str(ev['End'])] = ev['Type']


def set_tlines_fillingup(s, sample_list, allevents_spec):
    """
    - day_evs are all events CREATED on the same day (or later) including sample itself
    - iterate through day events (samples):
        - delete ev and all connected sevs for all rvs for this sample
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
        rm_evs_all_rv(evs, rvli, allevents_spec)
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
    #evs += s.sevs  # can be empty list
    day_evs = s.day_evs  # rendomized only part of it to zero
    #evs += day_evs
    sample_li = []

    for eid in day_evs:
        ev = sample_list.get(eid)
        if ev == None:  # when samples are sliced
            continue

        evs += [eid]
        evs += ev.sevs
        sample_li.append(ev)

    remove_evs(evs, rvli, allevents_spec)

    # day_evs have same rvli -> exact cutting is in next step
    for s in sample_li:
        s.rvli = copy.deepcopy(rvli)