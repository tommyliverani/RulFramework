import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow_datasets as tfds

class Model:
  def __init__(self, input_features, output_feature):
      self.net=None
      self.input_features=input_features
      self.output_feature=output_feature
  
  def build_net(self,hidden,optimizer='Adam',loss='mae'):
    model_in = keras.Input(shape=len(self.input_features), dtype='float32')
    x = model_in
    for h,activation in hidden:
      x = layers.Dense(h, activation=activation)(x)
    model_out = layers.Dense(1)(x)
    model = keras.Model(model_in, model_out)
    model.compile(optimizer=optimizer, loss=loss)     
    self.net=model

  def import_model(self,net):
    self.net=net
  
  def train(self,train_set, batch_size=10, validation_split=0.2, epochs=20, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]):
    return self.net.fit(train_set[self.input_features], train_set[self.output_feature],batch_size=batch_size, validation_split=validation_split,epochs=epochs, verbose=verbose, callbacks=callbacks)
  
  def predict(self,set):
    return self.net.predict(set[self.input_features]).ravel()