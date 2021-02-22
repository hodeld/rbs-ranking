try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    import pandas as pd
    from datetime import datetime

    drive.mount("/content/drive", force_remount=False)
    base_get_path = '/content/drive/My Drive/Colab Notebooks/rbs-data'
    base_store_path = ''
    log_path = base_get_path + '/output'
else:
    from secrets import base_path
    import pandas as pd
    from datetime import datetime
    import os

    prjct_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    base_store_path = prjct_root_path + '/output'
    base_get_path = base_path
    log_path = base_store_path

    INT_N = 'Int64'
    STR_N = 'object'

_TD_PERWK = 2 * 5 # nr or None


def end_c(val):
    val_int = int(val) - 1
    return val_int


CONV = {'End': end_c, }  # instead of dtype!


def prep_samples(file_n='Samples.csv', sep=','):
    dtype_dict = {
        'Id': INT_N,
        'Location': INT_N,
        'Start': INT_N,
        'Rv added': INT_N,
        'Day evs': STR_N,
        'Teams': STR_N,
        'Sevs': STR_N,
        'Rv eq': STR_N,
    }

    samples_raw = pd.read_csv(main_path + file_n,
                              delimiter=sep,
                              dtype=dtype_dict,
                              converters=CONV)
    samples = samples_raw.iloc[:, 0:18]  # additional columns
    return samples


def prep_allevents(file_n='AllEvents.csv', sep=','):
    allevents_p = pd.read_csv(main_path + file_n, index_col=0,
                              delimiter=sep,
                              converters=CONV)
    return allevents_p


def get_timelines_raw(file_n='timelines.csv', sep=','):
    timelines_r = pd.read_csv(main_path + file_n, index_col=0,
                              delimiter=sep)  # , header=0)

    return timelines_r


def prep_timelines(timelines_r):
    timelines_p = timelines_r.drop(index='dt_col')
    timelines_p = timelines_p.apply(pd.to_numeric)
    return timelines_p


def get_time_vars(timelines_raw):
    dtform = '%Y-%m-%d %H:%M:%S'
    tstart = datetime.strptime(timelines_raw.loc['dt_col', '0'], dtform)
    tend = datetime.strptime(timelines_raw.loc['dt_col', '1'], dtform)
    td_seq = (tend - tstart).seconds / 60
    if td_seq <= 60:
        pph = 60 // td_seq
    else:
        pph = 60 / td_seq
    weeks_b = 1  # 4 #for cutting timelines
    weeks_a = weeks_b
    kmax = int(timelines_raw.columns[-1])  # last column name as int
    td_perwk = int(pph * 24 * 7)
    tot_size_tline = int((weeks_b + weeks_a) * td_perwk)
    rv_tline_len = tot_size_tline + 1 # pandas slice includes 1st value
    print(td_seq, pph, tot_size_tline)
    time_vars = (tstart, tend, td_seq, td_perwk,
                 pph, weeks_b, weeks_a, kmax,
                 rv_tline_len)
    return time_vars


def prep_rv_first_ev(file_n='rvfirstev.csv', sep=','):
    rvfirstev_r = pd.read_csv(main_path + file_n, index_col=0,
                                delimiter=sep)
    rvfirstev_p = rvfirstev_r.copy()
    rvfirstev_p[rvfirstev_r == 0] = 1
    return rvfirstev_p


def prep_rv(file_n='RVs.csv', sep=','):
    rvs_p = pd.read_csv(main_path + file_n, delimiter=sep)
    return rvs_p


def get_evtypes(file_n='EventType.csv', sep=','):
    evtypes_raw = pd.read_csv(main_path + file_n,
                              delimiter=sep)

    evtypes = evtypes_raw.iloc[:, 0:1]  # additional columns
    ev_min = evtypes.min()[0]
    ev_max = evtypes.max()[0]
    return ev_min, ev_max


def get_test_files():
    samples_pred = prep_samples(file_n='samples_test.csv', sep=',')
    timelines_raw = get_timelines_raw('timelines_test.csv', ',')
    tlines = prep_timelines(timelines_raw)
    allevs = prep_allevents('allevents_test.csv', ',')
    return samples_pred, tlines, allevs


main_path = base_get_path + '/alopt_files/'

samples = prep_samples(file_n='Samples.csv', sep=',')

rvs = prep_rv()

allevents = prep_allevents()
# TIMELINES
MIN_ID, MAX_ID = get_evtypes()
timelines_raw = get_timelines_raw()
timelines = prep_timelines(timelines_raw)

(tstart, tend, TD_SEQ, TD_PERWK,
 PPH, WEEKS_B, WEEKS_A,
 KMAX, RV_TLINE_LEN) = get_time_vars(timelines_raw)

if _TD_PERWK:
    TD_PERWK = _TD_PERWK

rvfirstev = prep_rv_first_ev(file_n='rvfirstev.csv', sep=',')


