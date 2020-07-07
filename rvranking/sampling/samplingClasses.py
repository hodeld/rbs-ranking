import operator

from rvranking import RELEVANCE
from rvranking.dataPrep import PPH


class Sample():
    """class for events for ranking problem"""

    def __init__(self, sample_li):
        (s_id, location, dbid, day_evs, sevs, rv_eq,
         start, end, rv, group, cat,
         evtype, rv_ff, gespever, hwx, uma) = sample_li

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

        self.location = location
        self.day = (start // 24 * PPH) * (24 * PPH)
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

    def features(self):
        flist = [
            self.evtype,
            self.rv_ff,
            self.gespever,
            self.hwx,
            self.uma,
        ]
        return flist


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

    def features(self):
        """concententate
        sex: 1 or 2
        tline: [1 … 20]"""
        flist = [
            self.sex,
            *self.tline,
        ]
        return flist

    def features_fake(self):
        if self.relevance == RELEVANCE:  # if it is correct rv
            return [1]
        else:
            return [0]


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