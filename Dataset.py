import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow_datasets as tfds

class Dataset():
  def __init__(self,full_data):
    self.full_data=full_data

  def compute_rul(self,feature):
        self.full_data['rul']=0
        rlu=0
        last_index=0
        #compute the value
        for index,row in self.full_data.iterrows():
          if self.full_data[feature][index]==1.0:
            for i in range(last_index,index,1):
              self.full_data['rul'][i]=index-i
            last_index=index+1
        #finging the max value of rul
        max=self.full_data['rul'][0]
        for index,row in self.full_data.iterrows():
          if self.full_data['rul'][index]>max:
            max=self.full_data['rul'][index]
        #noralize rul
        self.full_data['rul']=self.full_data['rul']*1.0
        for index,row in self.full_data.iterrows():
          self.full_data['rul'][index]=self.full_data['rul'][index]*1.0/max


  def split_by_feature(self,feature,ratio):
    features=[]
    for feat, gdata in self.full_data.groupby(feature):
      features.append(feat)
    np.random.shuffle(features)
    sep = int(ratio * len(features))
    tr_feat = set(features[:sep])
    tr_list, ts_list = [], []
    for feat, gdata in self.full_data.groupby(feature):
      if feat in tr_feat:
        tr_list.append(gdata)
      else:
        ts_list.append(gdata)
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


  def random_split(self,ratio):
    tr_len=int(len(self.full_data) * ratio)
    data=self.full_data.sample(frac=1)
    train_set=data.head(tr_len)
    test_set=data.tail(len(data)-tr_len) 
    return train_set, test_set