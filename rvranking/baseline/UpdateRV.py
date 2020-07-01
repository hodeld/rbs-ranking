'''
Created on 03.01.2019

@author: daim
'''
from .Event import NormEvent
from datetime import timedelta
from .RV import RV_Display


def get_rv_mt(mastertype):
    rv_mt = 'anh'  # if mastertype == 0

    if mastertype == 1:
        rv_mt = 'dgs'
    elif mastertype == 2:
        rv_mt = 'anh'
    elif mastertype == 3:
        rv_mt = 'ent'

    return rv_mt


def set_prio_rv(list_with_rv_time):
    for event in list_with_rv_time:
        if isinstance(event, NormEvent):
            update_prio_rv(event)


def update_prio_rv(event, add=True):
    i = 1
    if add == False:
        i = -1

    if event.mastertype:
        mastertype = event.mastertype
        rv_mt = get_rv_mt(mastertype)
        rvobj = event.rv
        kw = event.start.strftime('%W')
        priodict = rvobj.priodict

        def set_priodict(key):
            prio = priodict.get(key, 0) + i
            if prio < 0:
                prio = 0
            priodict[key] = prio

        kwsend = ['00', '52', '53']
        if kw in kwsend:
            kws = kwsend
        else:
            kws = [kw]

        for kw in kws:
            if rv_mt == 'anh':
                key = rv_mt + kw
                set_priodict(key)

            elif rv_mt == 'dgs':
                key = rv_mt + kw
                set_priodict(key)


def sort_rv_list(rv_list, now):
    wkdelta = timedelta(days=7)

    kw_last = now - wkdelta  # now - timedelta (days = 7) #int(now.strftime('%W')) - 1

    names = ['dgs', 'anh', 'ent']

    rv_sort_dict = {}
    rv_order_dict = {}
    i = 0
    for i in range(6):  # if 4 weeks plan range
        kw = (kw_last + i * wkdelta).strftime('%W')  # '{0:02d}'.format(kw_last + i) 
        kw_before = (kw_last + (i - 1) * wkdelta).strftime('%W')  # '{0:02d}'.format(kw_last + i - 1)

        for name in names:
            mt_kw = name + kw
            mt_kw_before = name + kw_before

            rv_p_list = []
            for rv_obj in rv_list:
                priodict = rv_obj.priodict
                prio = priodict.get(mt_kw, 0)
                prio = 0.5 * priodict.get(mt_kw_before, 0) + prio
                rv_p_list.append([rv_obj, prio])

            rv_av_new = []
            rv_order = []


            rv_sort_dict[mt_kw] = rv_p_list
            #rv_order_dict[mt_kw] = rv_order
            kwsend = ['00', '52', '53']  # all needed as in the last week of year can be all 3 numbers
            if kw in kwsend:
                for kw in kwsend:
                    mt_kw = name + kw
                    rv_sort_dict[mt_kw] = rv_p_list
                    #rv_order_dict[mt_kw] = rv_order

        i += 1
        #rv_order = (rv_sort_dict, rv_order_dict)
    return rv_sort_dict


def sort_rv_pot(event, rv_sort_dict, list_evs, now):

    kw = event.start.strftime('%W')
    rv_mt = get_rv_mt(event.mastertype)
    kw_name = rv_mt + kw
    wkdelta = timedelta(days=7)
    d_kw_before = event.start - wkdelta
    kw_before = d_kw_before.strftime('%W')
    kw_bef_name = rv_mt + kw_before

    rv_list_sort = rv_sort_dict.get(kw_name, 0)

    if rv_list_sort == 0:
        rv_mt = 'anh'  # if no specific -> BX-type
        kw_name = rv_mt + kw
        rv_list_sort = rv_sort_dict[kw_name]
    rv_list_scopy = rv_list_sort.copy()

    rv_list_prio = []
    for rv_p in rv_list_scopy:
        if rv_p[0] in event.rv_pot:
            rv = rv_p[0]
            days = list_evs.spare_time(rv, now, event.start)
            rv_p[1] += 1/(days*10 + 1)
            if days < 0:
                print ('days < 0')
            if rv == event.rv_ff:
                rv_p[1] = -1 #first place
            rv_disp = RV_Display(rv, days, kw_name, kw_bef_name)
            rv_p.append(rv_disp)
            rv_list_prio.append(rv_p)

    rv_list_prio.sort(key=lambda rv_p: rv_p[1], reverse=False)
    #rv_av_sort =  list(sorted(rv_list_sort, key=lambda tup: tup[1], reverse=False)
    event.rv_pot = [rv_p[0] for rv_p in rv_list_prio]
    event.rv_pot_display = [rv_p[2] for rv_p in rv_list_prio]
    if event.rv_pot is None:
        print('non')


if __name__ == '__main__':
    pass
