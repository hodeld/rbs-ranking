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


def end_c(val):
    val_int = int(val) - 1
    return val_int


CONV = {'End': end_c, }  # instead of dtype!


def prep_samples(file_n='Samples.csv', sep=','):
    dtype_dict = {
        'Id': INT_N,
        'Location': INT_N,
        'Start': INT_N,
        # 'End': INT_N,
        'Day evs': STR_N,
        'Teams': STR_N,
        'Sevs': STR_N,
        'Rv eq': STR_N,
    }

    samples_raw = pd.read_csv(main_path + file_n,
                              delimiter=sep,
                              dtype=dtype_dict,
                              converters=CONV)
    samples = samples_raw.iloc[:, 0:17]  # additional columns
    return samples


def prep_allevents(file_n='AllEvents.csv', sep=','):
    allevents_p = pd.read_csv(main_path + file_n, index_col=0,
                              delimiter=sep,
                              converters=CONV)
    return allevents_p


def simplify_tlines_notused(timelines_r):
    dtform = '%Y-%m-%d %H:%M:%S'
    tst = datetime.strptime(timelines_r.loc['dt_col', '0'], dtform)
    tet = datetime.strptime(timelines_r.loc['dt_col', '1'], dtform)
    td_seq = (tst - tet).seconds / 60
    pph = 60 // td_seq
    ppd = 24 * pph
    kmax = int(timelines_r.columns[-1])
    time_point = 0
    if tst.weekday() > 4:
        tdays = 6 - tst.weekday()
        time_point += tdays * ppd  # 24 or 48h
    morning = time_point + 8 * pph
    lunch = morning + 5
    st = morning
    et = lunch
    while et <= kmax:
        tline = timelines_r.loc[:, str(st):str(et)]


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
    pph = 60 // td_seq
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


main_path = base_get_path + '/alopt_files/'

samples = prep_samples(file_n='Samples.csv', sep=',')

rvs = prep_rv()

allevents = prep_allevents()
# TIMELINES

timelines_raw = get_timelines_raw()
timelines = prep_timelines(timelines_raw)

(tstart, tend, TD_SEQ, td_perwk,
 PPH, WEEKS_B, WEEKS_A,
 KMAX, RV_TLINE_LEN) = get_time_vars(timelines_raw)


rvfirstev = prep_rv_first_ev(file_n='rvfirstev.csv', sep=',')


