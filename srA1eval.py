# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:48:18 2018

@author: js
"""

import numpy as np
import tensorflow as tf
import sklearn
from sklearn.metrics import confusion_matrix
from DataGenTimit import DataGenerator

num_timesteps = 50

p = np.load('srA1p.npy')
logp = np.load('srA1logp.npy')
y_pred = np.load('Phonemes39pred.npy')
y_true = np.load('Phonemes39true.npy')
Y = np.load('Phonemes39consecutive.npy')

idx_after16ms = np.load('Phonemes39_position_index.npy')
idx_after16ms = idx_after16ms[idx_after16ms>num_timesteps]-num_timesteps
idx_after16ms = idx_after16ms[idx_after16ms<len(p)]
idx = np.arange(len(y_pred))
idx = idx.reshape(idx.shape[0],)
data_split = int( 0.9 * len(idx))
idx_train  = idx[:data_split]
idx_val    = idx[data_split:]

class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(Y),Y)
class_p      = 1 / class_weight

model=tf.keras.models.load_model('srA1_h_0.h5')

y_both = np.c_[y_pred,y_true]

cf_matrix = confusion_matrix(y_true[idx_val], y_pred[idx_val])

a = np.sum(cf_matrix,axis=0)
b = np.sum(cf_matrix,axis=1)