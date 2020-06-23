'''
Created on 24 Mar 2020

@author: daim
'''
'''
Created on 24 Mar 2020

@author: daim
'''

import sys
#choose in runtime type 

import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2


from datetime import datetime
import copy
import operator
    
class Sample():

    """class for events for ranking problem"""
    def __init__(self,  sample_li):
        (s_id, location, dbid, day_evs, sevs, rv_eq, 
        start, end, rv, group, cat, 
        evtype, rv_ff, gespever, hwx, uma) = sample_li
    
        day = (start//24*PPH)*(24*PPH)
        locday =str(location)+'-' + str(day)

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
        self.day = (start//24*PPH)*(24*PPH)
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
    def get(self, variable_value, item_attr  = 'id'): 
        vv = variable_value
        ra = item_attr
        f = operator.attrgetter(ra)
        for s in self:
            if f(s) == vv: 
                return s                
        return None

class RV():
    '''base class for rvs'''
    def __init__(self,  rvvals
                  ):
        (rvid, location,
         sex) = rvvals
        self.id = rvid
        self.location = location
        self.sex = sex
        self.relevance = 0
        self.tline = None
    
    def features(self):
        flist = [
                self.sex,
                 *self.tline,
                 ]
        return flist

class RVList(list):
    '''base class for list of rvs'''
    def filter(self, variable_value, rv_attr): 
        vv = variable_value
        ra = rv_attr
        f = operator.attrgetter(ra)
        newli = RVList(filter(lambda x: f(x) == vv, self)) #calls x.ev
        return newli
    
    def get(self, variable_value, rv_attr  = 'id'): 
        vv = variable_value
        ra = rv_attr
        f = operator.attrgetter(ra)
        for rv in self:
            if f(rv) == vv: 
                return rv                
        return None

  
def get_example_features(s):  
    if s.rvli == None:
        rvli = get_rvlist(s)
        s.rvli = rvli
        set_tlines(s)
    get_pot_rvs(s)

def get_rvlist(s):
    rvlist = rvli_d.get(s.locday, None)
    if rvlist:
        return rvlist
    
    rvlist = rvlist_all.filter(s.location, 'location') #[rv for rv in rvs if rv.location == loc_id]
    rvlist = cut_timelines(rvlist, s) #same for all same day events
    rvli_d[s.locday] = rvlist
    return rvlist


def cut_timelines(rvlist, s):
    ist = s.rangestart 
    iet = s.rangeend 
    
    for rv in rvlist:
        tline = timelines.loc[str(rv.id)]
        rv.tline = tline.loc[str(ist):str(iet)] 
    return rvlist

def get_timerange(s):
    weeks_before = WEEKS_B
    weeks_after = WEEKS_A
    td_perwk = PPH*24*7
    ist = int(s.start - (td_perwk*weeks_before))
    if ist < 0:
        return False
        #ist = 0
    iet = int(s.start + td_perwk*weeks_after)
    if iet > KMAX:
        #iet = KMAX
        return False
    s.rangestart = ist
    s.rangeend = iet
        
    
def set_tlines(s):
    rvli = s.rvli
    evs = [] 
    evs  += s.sevs #can be empty list
    day_evs = s.day_evs #rendomized only part of it to zero
    evs += day_evs
    sample_li = []
    '''
    -> delete ev and sev for 1 rv 
    -> assign rvlist to sample
    -> copy rvlist
    -> delete ev and sev for next ev in 1 rv
    -> rvlist gets more and more zeros'''
    for eid in day_evs:
        ev = sample_list.get(eid)
        if ev == None: #when samples are sliced
            print('None ev in day evs')
            continue
            
        evs += ev.sevs
        sample_li.append(ev)
    
    for eid in evs:
        ev = allevents.loc[eid]
        rvid = ev['Rv']
        rv = rvli.get(rvid)
        if rv == None:
            continue #only as day_evs for all locations
        try:
            rv.tline.loc[str(ev['Start']):str(ev['End'])] = 0 
        except KeyError:
            print('event created outside timerange')
    #all day_evs have same rvli
    for s in sample_li:
        s.rvli = copy.deepcopy(rvli)


def get_pot_rvs(s):
    rvlist = s.rvli
    for rv in rvlist:
        if check_availability(rv, s) == False:
            rvlist.remove(rv)
        if check_evtype(rv, s) == False:
            rvlist.remove(rv)
    check_feat(rv, s) #probably better leave features
    
def check_availability(rv, s):
    if rv.tline.loc[str(s.start):str(s.end)].any() == True: #not all are 0
        return False
    else:
        return True
     
def check_evtype(rv, s):
    firstev = rvfirstev.loc[rv.id, str(s.evtype)] #1, None or date_int
    if firstev == None:
        return False
    if firstev <= s.start:
        return True
    else:
        return False

def check_feat(rv, s):
    if s.gespever > 0: 
        rvlist = s.rvli
        rvlist = rvlist.filter(s.gespever, 'sex')
        s.rvli = rvlist
        #UMA and HWX just left with features

RELEVANCE = 1 
def assign_relevance(s):
    rvs = [s.rv ] + list(s.rv_eq) #correct answer
    for rv in s.rvli:
        if rv.id in rvs:
            relev = RELEVANCE
        else:
            relev = 0
        rv.relevance = relev
        
_RV_FEATURE = 'rv_tokens'
_EVENT_FEATURE = 'event_tokens'
_EMBEDDING_DIMENSION = 20

#input: sparse tensor
def get_feature_columns(token_len, keyname, default_value = 0):
    
    default_tensor = [default_value]*token_len
    
    sparse_column = tf.feature_column.categorical_column_with_identity(
        key= keyname, num_buckets = 1000)
    #indicator_column OR embedding_column but embedding gives floats
    ##dense_column = tf.feature_column.indicator_column(sparse_column)
    
    dense_column = tf.feature_column.embedding_column(
                sparse_column, _EMBEDDING_DIMENSION)
    
    return dense_column
   # return tf.feature_column.numeric_column(key= keyname, 
       #                                     shape=(token_len,),
      #                                      dtype=tf.dtypes.int64,
      #                                      default_value=default_tensor)
def context_feature_columns():
    """Returns context feature names to column definitions."""
    event_token_len = 5 #evtype, rf_ff, gespever, hwx, uma
    dense_column = get_feature_columns(event_token_len, 'event_tokens')
    return {"event_tokens": dense_column}

def example_feature_columns():
    """Returns context feature names to column definitions."""
    rv_token_len = RV_TOKEN_LEN
    dense_column = get_feature_columns(rv_token_len, _RV_FEATURE )
    
    return {"rv_tokens": dense_column}

#as in example
_LABEL_FEATURE = 'relevance'
_PADDING_LABEL = -1
_BATCH_SIZE = 10
_LIST_SIZE = 5 # #of rvs
def input_fn(path, num_epochs= 1): #none
    context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
    label_column = tf.feature_column.numeric_column(
          _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
    
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])
    dataset = tfr.data.build_ranking_dataset(
          file_pattern=path,
          data_format=tfr.data.ELWC,
          batch_size=_BATCH_SIZE,
          list_size=_LIST_SIZE,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          reader=tf.data.TFRecordDataset,
          shuffle=False,
          num_epochs=num_epochs)
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)
    
    return features, label

def make_transform_fn():
    def _transform_fn(features, mode):
        """Defines transform_fn."""
        context_features, example_features = tfr.feature.encode_listwise_features(
            features=features,
            context_feature_columns=context_feature_columns(),
            example_feature_columns=example_feature_columns(),
            mode=mode,
            scope="transform_layer")
        
        return context_features, example_features
    return _transform_fn
  
  
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  
def write_context_examples(path, samples):  
    def serialize_example(rv):
        """
        Creates a tf.Example message ready to be written to a file.
        'rv.feat + rv.tline'
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'rv_tokens': _int64_list_feature(rv.features()), #_RV_FEATURE
            'relevance': _int64_feature(rv.relevance), #_LABEL_FEATURE
        }       
        # Create a Features message using tf.train.Example.
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        return example #.SerializeToString()
    
    def serialize_context(contfeatures):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'event_tokens': _int64_list_feature(contfeatures),
        }        
        # Create a Features message using tf.train.Example.
        
        context = tf.train.Example(features=tf.train.Features(feature=feature))        
        return context #.SerializeToString()
    
    elwc_list = []   
  
    for s in samples:
        
        rvli = s.rvli
        example_list = []
        for rv in rvli:
            example = serialize_example(rv)
            example_list.append(example)
        #context = serialize_context(s.features)
        context = serialize_context(s.features())
        
        ELWC = input_pb2.ExampleListWithContext()
        ELWC.context.CopyFrom(context)
        for example in example_list:
            example_features = ELWC.examples.add()
            example_features.CopyFrom(example)
        elwc_list.append(ELWC)

    file_path = path 
    with tf.io.TFRecordWriter(file_path) as writer:
        
        for elwc in elwc_list:#[:2]:
            #print(elwc)
            writer.write(elwc.SerializeToString())


if __name__ == '__main__':

    
    #IMPORT

    base_path = '/Users/daim/2 Projekte/RBS-Test/Wetlomat/deeplearning/'
    main_path = base_path + 'alopt_files/'
    timelines_raw = pd.read_csv(main_path+'timelines.csv', index_col =0) #, header=0)
    samples = pd.read_csv(main_path+'Samples.csv')
    rvs = pd.read_csv(main_path+'RVs.csv')
    allevents = pd.read_csv(main_path+'AllEvents.csv', index_col =0)
    rvfirstev_raw = pd.read_csv(main_path+'rvfirstev.csv', index_col =0)
  
    
    #TIMELINES
    #help(timelines_raw.loc)
    dtform = '%Y-%m-%d %H:%M:%S'
    tstart = datetime.strptime(timelines_raw.loc['dt_col', '0'], dtform)
    tend = datetime.strptime(timelines_raw.loc['dt_col', '1'], dtform)  
    TD_SEQ = (tend - tstart).seconds/60
    PPH = 60//TD_SEQ
    WEEKS_B = 1 # 4 #for cutting timelines
    WEEKS_A = WEEKS_B
    KMAX = int(timelines_raw.columns[-1]) #last column name as int

    
    td_perwk = PPH*24*7
    tot_size_tline = int((WEEKS_B + WEEKS_A)* td_perwk)
    rv_feat_len = 1
    RV_TOKEN_LEN = rv_feat_len + tot_size_tline + 1 #pandas slice includes 1st value
    print(TD_SEQ, PPH, tot_size_tline)
    
    timelines = timelines_raw.drop(index='dt_col')
    
    #now get rv with timelines.loc[str(52)]  
    timelines.head(3)
    #to numeric of values (not columns
    timelines = timelines.apply(pd.to_numeric) 
    #now get rv with timelines.loc[str(52)] 
    
    
    #RVS
    rvfirstev = rvfirstev_raw.copy()
    rvfirstev[rvfirstev_raw == 0] = 1
    
    sample_list = SampleList([Sample(s) for i, s in samples.iterrows()]) #samples.iloc[:5].iterrows()])
    rvlist_all = RVList([RV(r) for i, r in rvs.iterrows()])
    
    rvli_d = {}
    i = 0
    
    sample_list_prep = []
    for s in sample_list:
        i += 1
        if get_timerange(s) == False:
            print(i, 'timerange too short')
            continue
        get_example_features(s) #rvs, SHOULD BE ALWAYS SAME # OF RVS
        assign_relevance(s)
        sample_list_prep.append(s)
        
    file_path = base_path + 'data.tfrecords'
    
    write_context_examples(file_path, sample_list_prep) 
    
    #get data back
    feat, labs = input_fn(file_path)
    
    for k, item in feat.items():
        print ('feat', k, item.shape)
    print ('first 5 labels', labs[0,:5].numpy())
    event_t = feat['event_tokens'] #spare tensor to dense
    rv_t = feat['rv_tokens']
    event_t = tf.sparse.to_dense(event_t)
    rv_t = tf.sparse.to_dense(rv_t)
    print ('event values', event_t[0])
    #check slicing notification!
    print ('rv values', rv_t[0,:5,:10].numpy())
    
    transfunc = make_transform_fn()
    cont_feats, ex_feats = transfunc(feat, 'test')
    cont_feat_t = cont_feats['event_tokens']
    examp_feat_t = ex_feats['rv_tokens']
    print ('cont_feat_t', cont_feat_t.shape)
    print ('examp_feat_t', examp_feat_t.shape)
    
    print ('first 5 cont features', cont_feat_t[0,:5].numpy())
    print ('first 5 features von 1 doc', examp_feat_t[0,0,:5].numpy())


    print('finished')