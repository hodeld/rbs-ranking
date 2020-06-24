import pandas as pd
from datetime import datetime
from secrets import base_path

#drive.mount("/content/drive", force_remount=False)
#base_path = '/content/drive/My Drive/Colab Notebooks/rbs-data'

main_path = base_path + '/alopt_files/'
timelines_raw = pd.read_csv(main_path + 'timelines.csv', index_col=0)  # , header=0)
samples = pd.read_csv(main_path + 'Samples.csv')
rvs = pd.read_csv(main_path + 'RVs.csv')
allevents = pd.read_csv(main_path + 'AllEvents.csv', index_col=0)
rvfirstev_raw = pd.read_csv(main_path + 'rvfirstev.csv', index_col=0)

# TIMELINES

#from datetime import datetime

# help(timelines_raw.loc)
dtform = '%Y-%m-%d %H:%M:%S'
tstart = datetime.strptime(timelines_raw.loc['dt_col', '0'], dtform)
tend = datetime.strptime(timelines_raw.loc['dt_col', '1'], dtform)
TD_SEQ = (tend - tstart).seconds / 60
PPH = 60 // TD_SEQ
WEEKS_B = 1  # 4 #for cutting timelines
WEEKS_A = WEEKS_B
KMAX = int(timelines_raw.columns[-1])  # last column name as int

td_perwk = PPH * 24 * 7
tot_size_tline = int((WEEKS_B + WEEKS_A) * td_perwk)
rv_feat_len = 1
RV_TOKEN_LEN = rv_feat_len + tot_size_tline + 1  # pandas slice includes 1st value
print(TD_SEQ, PPH, tot_size_tline)

timelines = timelines_raw.drop(index='dt_col')

# now get rv with timelines.loc[str(52)]
timelines.head(3)
# to numeric of values (not columns
timelines = timelines.apply(pd.to_numeric)
# now get rv with timelines.loc[str(52)]

# RVS first ev
rvfirstev = rvfirstev_raw.copy()
rvfirstev[rvfirstev_raw == 0] = 1

