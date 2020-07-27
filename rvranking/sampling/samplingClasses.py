import operator
import random

from rvranking.logs import hplogger
from rvranking.globalVars import RELEVANCE
from rvranking.dataPrep import PPH


class Sample():
    """class for events for ranking problem"""

    def __init__(self, sample_li):
        (s_id, location, dbid, day_evs, sevs, rv_eq,
         start, end, rv, group, cat,
         evtype, rv_ff, gespever, hwx, uma, teams) = sample_li

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
        teams = get_li(teams)

        self.location = location
        self.day = day
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
        self.teams = teams
        self.features_attrs = ['evtype', 'rv_ff', 'gespever', 'hwx', 'uma']  # ['evtype', 'rv_ff', 'gespever', 'hwx', 'uma']

    def features(self):
        f = operator.attrgetter(*self.features_attrs)
        res = f(self)
        if type(res) == tuple:
            li = list(res)
        else:
            li = [res]
        return li

    def features_fake_random(self):
        flist = [
            random.randint(1, 30),
        ]
        return flist

    def log_features(self):
        hplogger.info('event_tokens: ' + str(self.features_attrs))


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

        self.features_attrs = ['id', 'sex', 'tline']  #  # self.sex, self.id, tline 'tline'

    def features(self):
        """concatenate
        sex: 1 or 2
        tline: [1 … 20]"""
        feat_attrs = self.features_attrs
        if 'tline' in feat_attrs:
            feat_attrs.remove('tline')
            tline = list(self.tline)  # todo both as series
        else:
            tline = []
        f = operator.attrgetter(*feat_attrs)
        res = f(self)
        if type(res) == tuple:
            li = list(res)
        else:
            li = [res]
        flist = li + tline
        return flist

    def features_fake(self):
        if self.relevance == RELEVANCE:  # if it is correct rv
            return [1]
        else:
            return [0]

    def features_fake_random(self):
        flist = [
            random.randint(1, 30),
            random.randint(1, 30),
            random.randint(1, 30),
        ]
        return flist

    def log_features(self):
        hplogger.info('rv_tokens: ' + str(self.features_attrs))


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
