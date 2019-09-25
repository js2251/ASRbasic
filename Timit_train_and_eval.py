# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:24:24 2019

@author: js2251
"""

import numpy as np
#import multiprocessing as mp
import pickle
from TimitData import savePhonemeInformation, getPhonemeDict, savePhonemeInformation39, savePhoneme39Dict, getPhonemeDict39,\
    getIndexNonSa, saveStartIndices, saveAuditoryInformation, scaleAuditoryVariables, removeStartOfPhonemes # , saveMelInformation
from srA1 import trainCausalNN
from srA1predict import combineModelCausalNN
from srA2 import trainNonCausalNN
from srA2predict import predictPhonemeProbabilitiesNonCausalNN
from TimitLanguage import getPhonemeTransitionP, getBiphoneme2PhonemeTransitionP, getTriphonemeStateTransitionP, getTriphonemeStartP, \
    getFullTriphonemeStartP, getPreviousTriphonemeStatesAndTransitionP, \
    ViterbiTriphonemeFromNN, ViterbiOutput2Phonemes, PhonemePerTimestep2Sequence, minimumEditDistance
#import time
    
# run everything

def main():
    extractBasicFeatures()
    runPhonemeInfo()
    saveIndices()
    
    trainNeuralNets()
    
    savePhonemeTransitionProbabilities()
    setupViterbi()
    runViterbi()
    calculatePhonemeErrorRate()

# calculate band levels and cepstral coefficients - run once

def extractBasicFeatures():
    saveAuditoryInformation( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000) # not Mel scale in our case, performance similar to Mel
    scaleAuditoryVariables( filename_CC = 'CC.npy', filename_L = 'L.npy' )
    # saveMelInformation( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000 ) # Mel scale, finer than usual

#phoneme information - run once

def runPhonemeInfo():
    savePhonemeInformation( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000 )
    getPhonemeDict( dir_data = 'D:\TIMIT\TRAIN' )
    savePhonemeInformation39()
    
    dict_phonemes_num61 = eval( open('PhonemeDict61.txt', 'r').read() )
    getPhonemeDict39(dict_phonemes_num61)
    savePhoneme39Dict()

#language training - run once

def saveIndices():
    getIndexNonSa(ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000, split = 0.9) # prepare indices
    saveStartIndices( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000, split = 0.9 )
    removeStartOfPhonemes('Phonemes39',8) # indices to ignore first 16 ms of each phoneme -> use for training only!

#train neural nets - run once
    
def trainNeuralNets():
    trainCausalNN( filename_X = 'L_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_a', epochs_to_save = 1, epochs_total = 12,  batch_size = 128 )
    trainCausalNN( filename_X = 'L_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_b', epochs_to_save = 1, epochs_total = 1, batch_size = 1024, load_model='srA1_a_11.h5' )
    trainCausalNN( filename_X = 'L_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_c', epochs_to_save = 1, epochs_total = 1, batch_size = 4096, load_model='srA1_b_0.h5' )
    
    trainCausalNN( filename_X = 'CC_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_d', epochs_to_save = 1, epochs_total = 12,  batch_size = 128 )
    trainCausalNN( filename_X = 'CC_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_e', epochs_to_save = 1, epochs_total = 1, batch_size = 1024, load_model='srA1_d_11.h5' )
    trainCausalNN( filename_X = 'CC_scaled.npy', filename_Y = 'Phonemes39consecutive.npy', filename_idx = 'Phonemes39_position_index.npy', file_identifier_out = 'srA1_f', epochs_to_save = 1, epochs_total = 1, batch_size = 4096, load_model='srA1_e_0.h5' )
    
    combineModelCausalNN( model_name_1 = 'srA1_c_0', model_name_2 = 'srA1_f_0', model_name_out = 'srA1', filename_X1 = 'L_scaled.npy', filename_X2 = 'CC_scaled.npy', filename_Y = 'Phonemes39consecutive.npy' )
    
    trainNonCausalNN( filename_X = 'srA1_logp_combined_all.npy', filename_Y = 'Phonemes39consecutive.npy', file_identifier_out = 'srA2_a', epochs_to_save = 1, epochs_total = 10, batch_size = 1024, reduce_factor = 10, load_model = None )
    trainNonCausalNN( filename_X = 'srA1_logp_combined_all.npy', filename_Y = 'Phonemes39consecutive.npy', file_identifier_out = 'srA2_b', epochs_to_save = 1, epochs_total = 10, batch_size = 1024, reduce_factor = 1, load_model = 'srA2_a_9.h5' )
    
    predictPhonemeProbabilitiesNonCausalNN( filename_X = 'srA1_logp_combined_all.npy', filename_Y = 'Phonemes39consecutive.npy', model_name = 'srA2_b_9', data_split_factor = 0)
    
    
# transition probabilities for Viterbi - run once

def savePhonemeTransitionProbabilities():
    Y = np.load('Phonemes39consecutive.npy')
    Y = Y.astype('int')
    getPhonemeTransitionP(Y)
    getBiphoneme2PhonemeTransitionP(Y)
    getTriphonemeStateTransitionP(Y)
    getTriphonemeStartP(Y)

# load model parameters and data for Viterbi. Change loaded models and timesteps for the output of your trained NN

def setupViterbi():
    Pi_small    = np.load('Start_Probabilities.npy')
    Pi          = getFullTriphonemeStartP( Pi_small )
    A_before    = np.load('Transitions_TriphonemeState.npy')
    A, Ind      = getPreviousTriphonemeStatesAndTransitionP( A_before )
    A_phoneme   = np.load('Transitions_Phoneme2Phoneme.npy')
    A_biphoneme = np.load('Transitions_Biphoneme2Phoneme.npy')
    Y           = np.load('Phonemes39consecutive.npy')
    
    ### replace with computation of test set
    num_timesteps_NN_Causal    = 50           # replace with your settings
    num_timesteps_NN_NonCausal = 305          # replace with your settings
    X = np.load('srA1_logp_combined_all.npy') # replace with your trained model
    idx = np.arange(len(X))   
    idx = idx.reshape(idx.shape[0],)
    data_split = int( 0.9 * len(idx)) # replace with your settings, current setup with data split on training set
#    idx_train = idx[:data_split]
    idx_val   = idx[data_split:]
    
    Y_prev = np.load('srA2_d_14_p_prev.npy')
    Y_now  = np.load('srA2_d_14_p_now.npy')
    Y_next = np.load('srA2_d_14_p_next.npy')
    
    idx_valed = np.array(idx_val)[np.array(range(len(Y_now )))] + num_timesteps_NN_NonCausal - int(num_timesteps_NN_NonCausal/2)
    start_idx = np.load('StartIndices_dev.npy')
    start_idx = np.append(start_idx,len(idx)-1) # add the end -> not a start id, but the end of the last segment
    return Pi, A, Ind, A_phoneme, A_biphoneme, num_timesteps_NN_Causal, num_timesteps_NN_NonCausal, Y, Y_prev, Y_now, Y_next, idx_valed, start_idx

# run Viterbi. Add parallel processing here or in ViterbiTriphonemeFromNN to speed up

def runViterbi():
    Pi, A, Ind, A_phoneme, A_biphoneme, num_timesteps_NN_Causal, num_timesteps_NN_NonCausal, Y, Y_prev, Y_now, Y_next, idx_valed, start_idx = setupViterbi()
    Q = []
    with open("Q_dev.txt", "rb") as fp:   # Unpickling
        Q = pickle.load(fp)
    
    for i in range(len(Q),len(start_idx)-1):
        print(i)
        start = np.where( idx_valed == start_idx[i] )[0][0]
        end   = np.where( idx_valed == start_idx[i+1] )[0][0]
        
        Y_prev_this = Y_prev[start:end,:]
        Y_now_this  = Y_now[start:end,:]
        Y_next_this = Y_next[start:end,:]
        
        Q_this = ViterbiTriphonemeFromNN(Y_prev_this,Y_now_this,Y_next_this,A,Ind,Pi, A_phoneme, A_biphoneme)
        Q.append( Q_this )
                
    with open("Q_dev.txt", "wb") as fp:   #Pickling
        pickle.dump(Q, fp)

# calculate PER for the output of Viterbi runs

def calculatePhonemeErrorRate():
    Pi, A, Ind, A_phoneme, A_biphoneme, num_timesteps_NN_Causal, num_timesteps_NN_NonCausal, Y, Y_prev, Y_now, Y_next, idx_valed, start_idx = setupViterbi()
    with open("Q_dev.txt", "rb") as fp:   # Unpickling
        Q = pickle.load(fp)
    
    errors = 0
    total_len = 0
    
    for i in range(len(Q)):
        print(i)
        Q_this = Q[i]
        
        Y_true    = Y[start_idx[i]+num_timesteps_NN_Causal:start_idx[i+1]+num_timesteps_NN_Causal]
        Y_now_est = ViterbiOutput2Phonemes(Q_this)
        seq_true   = PhonemePerTimestep2Sequence(Y_true,min_timesteps=2)
        seq_est    = PhonemePerTimestep2Sequence(Y_now_est,min_timesteps=2)
        
        errors += minimumEditDistance(seq_true,seq_est)
        total_len += len(seq_true)
        
        print( minimumEditDistance(seq_true,seq_est) / len(seq_true) )
        
    print( errors/total_len )