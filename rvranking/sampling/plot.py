import matplotlib.pyplot as plt
import numpy as np
from rvranking.sampling.main import prep_samples_list
from rvranking.sampling.samplingClasses import Sample, RV, RVList
from rvranking.dataPrep import samples, rvs, timelines, rvfirstev, allevents


def plot_samples():
    sample_list_all = [Sample(s) for i, s in samples.iterrows()]
    rvlist_ids = [r[0] for i, r in rvs.iterrows()]
    analyze_rv_ff(sample_list_all)
    #samples_rv = [s.rv for s in sample_list_all]
    samples_evtype = [s.evtype for s in sample_list_all]
    evtypes = list(set(samples_evtype))
    evtypes.sort()
    rvlist_ids.sort()
    ev_li = [0] * len(rvlist_ids)
    ev_rv = {}
    for e in evtypes:
        li = ev_li.copy()
        ev_rv[e] = li
    for s in sample_list_all:
        li = ev_rv[s.evtype]
        index = rvlist_ids.index(s.rv)
        li[index] += 1

    pi = None
    plots = []
    legends = []
    width = .3
    bar_opts = {'align': 'edge',
                'width': width}
    plt.figure(figsize=(15, 8))  # width, height
    for e, nrevs in ev_rv.items():
        if pi is None:
            nrevs_sum = np.array(nrevs)
            pi = plt.bar(rvlist_ids, nrevs, **bar_opts)
        else:
            pi = plt.bar(rvlist_ids, nrevs,
                         bottom=nrevs_sum,
                         **bar_opts)
            nrevs_sum += np.array(nrevs)
        plots.append(pi[0])
        legends.append(str(e))

    plt.xlabel('RVs')
    plt.ylabel('# Samples')
    plt.xticks(rvlist_ids, rvlist_ids)
    plt.legend(plots, legends)
    plt.show()


def analyze_rv_ff(smples):
    s_rv_ff = [s for s in smples if s.rv_ff == s.rv]
    ratio_rv_ff = len(s_rv_ff)/len(smples)
    print('ratio rv = rv_ff', ratio_rv_ff)


if __name__ == '__main__':
    plot_samples()