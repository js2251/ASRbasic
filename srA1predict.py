# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:29:18 2019

@author: js2251
"""

import numpy as np
import tensorflow as tf
import sklearn
from DataGenTimit import DataGenerator

def predictPhonemeProbabilitiesCausalNN( filename_X = 'L_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', model_name = 'srA1_k_1', data_split_factor = 0.9):
    ''' data_split_factor = 0 to predict and store for all timesteps. set to 0.9 for evaluation of development set (and use X of train+dev set)
        see also srA1eval script for evaluation purposes '''
    X = np.load( filename_X )
    Y = np.load( filename_Y )
    idx = np.arange(len(X))   # predict all including first after phoneme boundary
    idx = idx.reshape(idx.shape[0],)
    
    num_timesteps = 50
    batch_size    = 1024
    num_features  = X.shape[1]
    dim           = ( batch_size, num_timesteps, num_features )
    
    data_split = int( data_split_factor * len(idx))
    idx_val   = idx[data_split:]
    
    predict_generator = DataGenerator(idx_val, X, Y, dim, reduce_factor = 1, shuffle = False)
    
    model=tf.keras.models.load_model( model_name +'.h5')
    
    evaluation = model.evaluate_generator( predict_generator, verbose = 1 )
    
    p = model.predict_generator( predict_generator, verbose = 1 )
    idx_valed = np.array(idx_val)[np.array(range(len(p)))] + num_timesteps
    
    logp = np.log(p)
    np.save( model_name+'_logp.npy',logp)
    
    y_pred = np.argmax(p,axis=1)
    y_true = Y[idx_valed]
    pred_correct = sum( ( np.logical_or( y_pred[:-8] == y_true[:-8], y_pred[8:] == y_true[:-8]  ) ) ) / (len(y_pred)-8)
    np.save('Phonemes39pred_' + model_name + '.npy',y_pred)
    np.save('Phonemes39true.npy',y_true)
    
    print(evaluation[1])
    print(pred_correct)

######### combine
    
def combineModelCausalNN( model_name_1 = 'srA1_k_1', model_name_2 = 'srA1_h_0', model_name_out = 'srA1', filename_X1 = 'L_scaled.npy', filename_X2 = 'CC_scaled.npy', filename_Y = 'Phonemes39consecutive.npy' ):
    ''' combine models based on level and cepstral coefficients (or any two models) by adding their log probabilities '''

    predictPhonemeProbabilitiesCausalNN( filename_X = filename_X1, filename_Y = filename_Y, model_name = model_name_1, data_split_factor = 0)
    logp_1  = np.load( model_name_1 + '_logp.npy')
    predictPhonemeProbabilitiesCausalNN( filename_X = filename_X2, filename_Y = filename_Y, model_name = model_name_2, data_split_factor = 0)
    logp_2  = np.load( model_name_2 + '_logp.npy')
    
    logp = logp_1 + logp_2
    np.save( model_name_out + '_logp_combined_all.npy',logp)
    np.save( model_name_out + '_p_combined_all.npy',np.exp(logp))