import os
from rvranking.dataPrep import log_path
import logging
from datetime import datetime

from rvranking.logs.key_names import LOG_KEYS

hplogger = logging.getLogger('hyperlogger')
DATE_FORMAT = '%c'


def logging_basic():
    """defines logger handling for hyperlogger"""
    # should be before any logging calls
    filename = 'hyparams_log.txt'
    filepath = log_path + '/' + filename
    hyperhandler = logging.FileHandler(filepath)
    # hyperhandler.setLevel(logging.INFO) # not needed as only one handler for this logger

    formatter = logging.Formatter('%(levelname)s:%(message)s')
    hyperhandler.setFormatter(formatter)

    hplogger.addHandler(hyperhandler)
    hplogger.setLevel(logging.INFO)

    run_nr_file = log_path + '/' + 'run_nr.txt'
    if os.path.exists(run_nr_file):
        with open(run_nr_file) as f:
            run_nr = int(f.readline())
        os.remove(run_nr_file)  # this deletes the file
    else:
        run_nr = 0
    run_nr += 1
    run_nr_str = str(run_nr)
    with open(run_nr_file, 'w') as f:
        f.write(run_nr_str)
    print('run nr:', run_nr)
    hplogger.info('new run -----------------------------------------')
    hplogger.info('nr: ' + run_nr_str)
    hplogger.info('date: ' + datetime.now().strftime(DATE_FORMAT))


def logs_to_csv():
    filename = 'hyparams_log.txt'
    filepath = log_path + '/' + filename
    with open(filepath, 'r') as f:
        lines = f.readlines()
    ignore_strs = ['INFO', 'HYPER']
    keys_notused = set()
    runs = []
    run_d = {}
    k_run = 0
    for l in lines:
        if 'new run' in l:
            k_run += 1
            run_d['nr'] = k_run
            runs.append(run_d)
            run_d = {}
            continue
        if '\n' in l:
            l = l.replace('\n', '')
        li = l.split(':')
        if len(li) < 2:
            print(li)
            continue
        k = 0
        for item in ignore_strs:
            if item in l:
                k += 1
        key = li[k]
        if not (key in LOG_KEYS):
            if not (key in list(keys_notused)):
                print(key)
                keys_notused.add(key)
            continue

        if 'date' in l:
            date_str = ':'.join(li[k + 1:])
            try:
                val = datetime.strptime(date_str, ' ' + DATE_FORMAT).strftime('%d.%m.%Y %X')
            except ValueError:
                print('wrong date format', date_str)
                continue
        else:
            val = li[k + 1]
        run_d[key] = val

    runs.append(run_d)

    csv_file = 'hyparams_log.csv'
    wfile_path = log_path + '/' + csv_file

    if os.path.exists(wfile_path):
        os.remove(wfile_path)  # this deletes the file
    else:
        print("The file does not exist")  # add this to prevent errors

    with open(wfile_path, 'w') as f:
        f.write('\t'.join(LOG_KEYS))
        f.write('\n')
        for run_d in runs:
            for k in LOG_KEYS:
                val = run_d.get(k, '')
                f.write(str(val))
                f.write('\t')
            f.write('\n')


if __name__ == '__main__':
    logs_to_csv()
    print('csv written')
