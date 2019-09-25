# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 01:07:00 2019

@author: js2251
"""

import numpy as np
import tensorflow as tf
from DataGenTimitTri import DataGeneratorTri

def predictPhonemeProbabilitiesNonCausalNN( filename_X = 'srA1_logp_combined_all.npy', filename_Y = 'Phonemes39consecutive.npy', model_name = 'srA2_d_14', data_split_factor = 0.9):
    ''' use data_split_factor = 0 for running the model on all data or evaluating a test set, and 0.9for evaluating a train+dev set '''  

    X = np.load(filename_X)
    Y = np.load(filename_Y)
    Y = Y[50:]   # 50 timesteps in first causal NN
    idx = np.arange(len(X))   # predict all including first after phoneme border
    
    idx = idx.reshape(idx.shape[0],)
    
    batch_size   = 1024
    num_features = X.shape[1]
    num_timesteps = 305
    dim          = ( batch_size, num_timesteps, num_features )
    
    data_split = int( data_split_factor * len(idx))
    idx_val   = idx[data_split:]
    
    model=tf.keras.models.load_model( model_name + '.h5')
    
    predict_generator = DataGeneratorTri(idx_val, X, Y, dim, reduce_factor = 1,non_causal_steps = int(num_timesteps/2), shuffle = False)
    
    evaluation = model.evaluate_generator( predict_generator, verbose = 1 )
    
    p = model.predict_generator( predict_generator, verbose = 1 )
    
    p_prev = p[0]
    p_now = p[1]
    p_next = p[2]
    np.save( model_name + '_p_prev.npy',p_prev)
    np.save( model_name + '_p_now.npy',p_now)
    np.save( model_name + '_p_next.npy',p_next)
    
    idx_valed = np.array(idx_val)[np.array(range(len(p_now)))] + num_timesteps - int(num_timesteps/2)
    
    y_pred = np.argmax(p_now,axis=1)
    y_true = Y[idx_valed]
    pred_correct = sum( ( np.logical_or( y_pred[:-8] == y_true[:-8], y_pred[8:] == y_true[:-8]  ) ) ) / (len(y_pred)-8)
    np.save('Phonemes39pred_' + model_name + '.npy',y_pred)
    np.save('Phonemes39true_' + model_name + '.npy',y_true) # difference to causal NN: first 50 were discarded
    
    print(evaluation[4:])
    print(pred_correct)