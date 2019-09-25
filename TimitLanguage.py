# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:13:04 2019

@author: js2251
"""

import numpy as np
from sklearn.preprocessing import normalize
from TimitData import getNextLabel, getPreviousLabel, getIndexNonSa
import multiprocessing as mp
import pickle

def getPhonemeTransitionP(Y,time_steps=1,max_offset = 100, min_p = 0.001):
    idx = np.load('NonSaIdx_train.npy') # getIndexNonSa() # do not use SA sentences for the language model
    Y_next = getNextLabel(Y,max_offset)
    idx    = idx[::time_steps]
    A      = np.zeros((40,40))
    for i in range(len(idx)):
        if idx[i] > len(Y):
            break; # if Y was truncated to yield train and dev/test set
        A[Y[idx[i]],Y_next[idx[i]]] += 1
    A = normalize(A, axis=1, norm='l1')
    A += min_p / ( 1 - 40 * min_p )
    A = normalize(A, axis=1, norm='l1')
    np.save('Transitions_Phoneme2Phoneme.npy',A)
    return A

def getBiphoneme2PhonemeTransitionP(Y,time_steps=1,max_offset = 100, min_p = 0.001):
    idx = np.load('NonSaIdx_train.npy') # getIndexNonSa() # do not use SA sentences for the language model
    Y_next = getNextLabel(Y,max_offset)
    Y_prev = getPreviousLabel(Y,max_offset)
    idx    = idx[::time_steps]
    A      = np.zeros((40*40,40))     # 40 * prev + current -> next
    for i in range(len(idx)):
        if idx[i] > len(Y):
            break; # if Y was truncated to yield train and dev/test set
        A[Y_prev[idx[i]]*40+Y[idx[i]],Y_next[idx[i]]] += 1
    A = normalize(A, axis=1, norm='l1')
    A += min_p / ( 1 - 40 * min_p )
    A = normalize(A, axis=1, norm='l1')
    np.save('Transitions_Biphoneme2Phoneme.npy',A)
    return A

def getTriphonemeObservationFromNN(Y_prev,Y_now,Y_next, A_phoneme, A_biphoneme):
    ''' return probability for each of the 40**3 triphonemes based on the 40*3 outputs of the NN '''
    if (Y_prev.ndim == 1):
        num_cols = 1
    else:
        num_cols = Y_prev.shape[0] # timesteps in the input matrices
    B = np.zeros((40*40*40,num_cols))
    for i in range(num_cols):
        for iPrev in range(40):
            for iNow in range(40):
                p_Biphoneme = Y_prev[iPrev] * A_phoneme[iPrev,iNow] * Y_now[iNow]
                for iNext in range(40):
                    p = p_Biphoneme * A_biphoneme[iPrev*40+iNow,iNext] * Y_next[iNext]
                    B[40*40*iPrev+40*iNow+iNext,i] = p
    return B

#def getPhonemeObservationFromNN(Y_prev,Y_now,Y_next, A_phoneme, A_biphoneme):
#    ''' return probability for each of the 40**3 triphonemes based on the 40*3 outputs of the NN '''
#    if (Y_now.ndim == 1):
#        num_cols = 1
#    else:
#        num_cols = Y_now.shape[0] # timesteps in the input matrices
#    B = np.zeros((40*40*40,num_cols))
#    for i in range(num_cols):
#        for iPrev in range(40):
#            for iNow in range(40):
#                for iNext in range(40):
#                    B[40*40*iPrev+40*iNow+iNext,i] = 0.1 #Y_now[iNow]
#    return B

def getPhonemeObservationFromNN(Y_now):
    ''' return probability for each of the 40**3 triphonemes based on the 40*3 outputs of the NN '''
    if (Y_now.ndim == 1):
        num_cols = 1
    else:
        num_cols = Y_now.shape[0] # timesteps in the input matrices
    B = np.zeros((40*40*40,num_cols))
    for i in range(num_cols):
        for iPrev in range(40):
            for iNow in range(40):
                for iNext in range(40):
                    B[40*40*iPrev+40*iNow+iNext,i] = Y_now[iNow]
    return B
                

def getTriphonemeStateTransitionP(Y,time_steps=5,max_offset = 100, min_p = 0.001):
    idx = np.load('NonSaIdx_train.npy') # getIndexNonSa() # do not use SA sentences for the language model
    Y_next = getNextLabel(Y,max_offset)
    Y_prev = getPreviousLabel(Y,max_offset)
    idx    = idx[::time_steps]
    A      = np.zeros((40*40*40,41))     # 40*40 * prev + 40 * current + next -> next state, 41st is 'same'
    for i in range(len(idx)-1):
        if idx[i] > len(Y):
            break; # if Y was truncated to yield train and dev/test set
        if idx[i] + time_steps != idx[i+1]:
            continue; # over sentences, don't use
        if ( Y_prev[idx[i]] == Y_prev[idx[i+1]] and Y[idx[i]] == Y [idx[i+1]] and Y_next[idx[i]] == Y_next[idx[i+1]] ): # for the few occassion where the triphone stays in one phone, this is considered here, and the ,idx[i] erroneously gets min_p, not 0
            A[Y_prev[idx[i]]*40*40+Y[idx[i]]*40+Y_next[idx[i]],40] += 1
        else:
            A[Y_prev[idx[i]]*40*40+Y[idx[i]]*40+Y_next[idx[i]],Y_next[idx[i+1]]] += 1
    A = normalize(A, axis=1, norm='l1')
    A += min_p / ( 1 - 41 * min_p )
    A = normalize(A, axis=1, norm='l1')
    np.save('Transitions_TriphonemeState.npy',A)
    return A

def getTriphonemeStartP(Y,max_offset = 100, min_p = 0.001):
    ''' 1 and 2 are silence, only third one (next) is relevant -> 40 entries'''
    idx_start = np.load('StartIndices_train.npy')
    s_type = np.load('SentenceType_train.npy')
    idx = idx_start[s_type != 'SA']
    Y_next = getNextLabel(Y,max_offset)
    Pi = np.zeros((40,1))
    for i in range(len(idx)):
        Pi[Y_next[idx[i]],0] += 1
    Pi = normalize(Pi, axis=0, norm='l1')
    Pi += min_p / ( 1 - 40 * min_p )
    Pi = normalize(Pi, axis=0, norm='l1')
    np.save('Start_Probabilities.npy',Pi)
    return Pi

def getFullTriphonemeStartP( Pi_small ):
    ''' first and second phoneme are silence for start, expand 40-element array to 40^3 array'''
    Pi = np.zeros((40*40*40,1))
    Pi[40*40*15+40*15:40*40*15+40*16,0] = Pi_small.reshape((40,))
    return Pi

def getPreviousTriphonemeStatesAndTransitionP( A_mono ):
    ''' return matrices with possible previous states and their transition probabilities to current state
        new matrices are not sorted by i, but by j'''
    A   = np.full((40*40*40,41),np.nan)
    Ind = np.full((40*40*40,41),np.nan,dtype='int64')
    for i1 in range(40):
        for i2 in range(40):
            for i3 in range(40):
                i_mono = i1 * 40*40 + i2 * 40 + i3
                for iNext in range(41):                   
                    if ( iNext == 40 ):
                        A[i_mono,40]   = A_mono[i_mono,40]
                        Ind[i_mono,40] = i_mono
                    else:
                        i_after         = i2 * 40 * 40 + i3 * 40 + iNext
                        A[i_after,i1]   = A_mono[i_mono,iNext]
                        Ind[i_after,i1] = i_mono                        
    return A, Ind

def ViterbiTriphonemeFromNN(Y_prev,Y_now,Y_next,A,Ind,Pi, A_phoneme, A_biphoneme):
    num_timesteps = Y_prev.shape[0]
    Delta = np.zeros((40*40*40,num_timesteps))
    Psi   = np.zeros((40*40*40,num_timesteps),dtype='int64')
    Q     = np.zeros((num_timesteps,),dtype='int64')
    
    B          = getTriphonemeObservationFromNN(Y_prev[0,:],Y_now[0,:],Y_next[0,:], A_phoneme, A_biphoneme)
    Delta[:,0] = np.multiply(B.reshape((B.shape[0],)),Pi.reshape((Pi.shape[0],)))
    
    for t in range(1,num_timesteps):
        print( '\r'+str(t)+' of '+str(num_timesteps)+'\r',end='', flush=True)
        B = getTriphonemeObservationFromNN(Y_prev[t,:],Y_now[t,:],Y_next[t,:], A_phoneme, A_biphoneme)
        limit = max(B) / 10000000
        for j in range(40*40*40):
            if ( B[j] > limit ):
                Delta_Candidates = np.multiply( A[j,:], Delta[Ind[j,:],t-1] ) * B[j]
                Delta[j,t] = np.max( Delta_Candidates )
                Psi[j,t]   = Ind[ j, np.argmax( Delta_Candidates ) ]
        c = np.sum( Delta[:,t] )
        Delta[:,t] = Delta[:,t] / c
    Q[-1] = np.argmax( Delta[:,-1] )
    for t in range(num_timesteps-1,0,-1):
        Q[t-1] = Psi[ Q[t],t ]
    return Q

def getFlatIndicesForViterbiStep( A, Ind ):
    idx_A = np.zeros(40*40*40*41,dtype='int64')
    idx_Ind = np.zeros(40*40*40*41,dtype='int64') 
    for j in range(40*40*40):
        for i in range(41):
            idx_A[j*41+i]   = j*41 + i
            idx_Ind[j*41+i] = Ind[j,i]
    return idx_A,idx_Ind

def ViterbiFromNN(Y_now_this,A,Ind,Pi):
    num_timesteps = Y_now_this.shape[0]
    Delta = np.zeros((40*40*40,num_timesteps))
    Psi   = np.zeros((40*40*40,num_timesteps),dtype='int64')
    Q     = np.zeros((num_timesteps,),dtype='int64')
    idx_A,idx_Ind = getFlatIndicesForViterbiStep( A, Ind )
    
    B          = getPhonemeObservationFromNN(Y_now_this[0,:])  
    Delta[:,0] = np.multiply(B.reshape((B.shape[0],)),Pi.reshape((Pi.shape[0],)))
    
    for t in range(1,num_timesteps):
        print( '\r'+str(t)+' of '+str(num_timesteps)+'\r',end='', flush=True)
        B = getPhonemeObservationFromNN(Y_now_this[t,:])        
        limit = max(B) / 1000
        idx_row_B_use = B > limit
        idx_row_B_use = idx_row_B_use .reshape((idx_row_B_use .shape[0],))
        B = np.repeat(B,41,axis=0)
        idx_flat_B_use = B > limit
        idx_flat_B_use = idx_flat_B_use.reshape((idx_flat_B_use.shape[0],))
        
        Delta_Candidates = np.log( A.flatten()[idx_A[idx_flat_B_use]] ) + np.log( Delta[idx_Ind[idx_flat_B_use],t-1] ) 
        Delta_Candidates = np.log( Delta_Candidates ) + np.log( B[idx_flat_B_use] )
        Delta_Candidates = Delta_Candidates.reshape((int(Delta_Candidates.shape[0]/41),41))
        Delta[ idx_row_B_use, t ] = np.exp( np.max( Delta_Candidates, axis = 1 ) )
        Psi[ idx_row_B_use, t ]   = Ind[ idx_row_B_use, np.argmax( Delta_Candidates, axis = 1 ) ]
        
#        c = np.sum( Delta[:,t] )
#        Delta[:,t] = Delta[:,t] / c
    Q[-1] = np.argmax( Delta[:,-1] )
    for t in range(num_timesteps-1,0,-1):
        Q[t-1] = Psi[ Q[t],t ]
    return Q

def ViterbiFromNN2(Y_now,A,Ind,Pi,i_sequence = 0):
    ''' simpler code as above but slower '''
    print(i_sequence)
    num_timesteps = Y_now.shape[0]
    Delta = np.zeros((40*40*40,num_timesteps))
    Psi   = np.zeros((40*40*40,num_timesteps),dtype='int64')
    Q     = np.zeros((num_timesteps,),dtype='int64')
    
    B          = getPhonemeObservationFromNN(Y_now[0,:])  
    Delta[:,0] = np.multiply(B.reshape((B.shape[0],)),Pi.reshape((Pi.shape[0],)))
    
    for t in range(1,num_timesteps):
        print( '\r'+str(t)+' of '+str(num_timesteps)+'\r',end='', flush=True)
        B = getPhonemeObservationFromNN(Y_now[t,:])
        limit = max(B) / 1000
        for j in range(40*40*40):
#            B = Y_now[t, int( j / 40 ) % 40]
            if ( B[j] > limit ):
                Delta_Candidates = np.multiply( A[j,:], Delta[Ind[j,:],t-1] ) * B[j]
                Delta[j,t] = np.max( Delta_Candidates )
                Psi[j,t]   = Ind[ j, np.argmax( Delta_Candidates ) ]
        c = np.sum( Delta[:,t] )
        Delta[:,t] = Delta[:,t] / c
    Q[-1] = np.argmax( Delta[:,-1] )
    for t in range(num_timesteps-1,0,-1):
        Q[t-1] = Psi[ Q[t],t ]
    return Q

def PreviousStates( Delta_now, Psi_now, Delta_prev, A, B, idx, limit, Ind, t ):
    pass
    for j in range(idx):
        if ( B[idx[j]] > limit ):
            Delta_Candidates = np.multiply( A[idx[j],:], Delta_prev[Ind[idx[j],:]] ) * B[idx[j]]
            Delta_now[idx[j]] = np.max( Delta_Candidates )
            Psi_now[idx[j]]   = Ind[ idx[j], np.argmax( Delta_Candidates ) ]

def ViterbiFromNNParallel(Y_now,A,Ind,Pi,i_sequence = 0):
    print(i_sequence)
    num_timesteps = Y_now.shape[0]
    Delta = np.zeros((40*40*40,num_timesteps))
    Psi   = np.zeros((40*40*40,num_timesteps),dtype='int64')
    Q     = np.zeros((num_timesteps,),dtype='int64')
    
    B          = getPhonemeObservationFromNN(Y_now[0,:])  
    Delta[:,0] = np.multiply(B.reshape((B.shape[0],)),Pi.reshape((Pi.shape[0],)))
    
    for t in range(1,num_timesteps):
        print( '\r'+str(t)+' of '+str(num_timesteps)+'\r',end='', flush=True)
        B = getPhonemeObservationFromNN(Y_now[t,:])
        limit = max(B) / 1000
        
        print(" " + str(t))
        
        Delta_now = mp.Array('d', 40*40*40)
        Psi_now   = mp.Array('d', 40*40*40)
        
        p0 = mp.Process(target=easy_func, args=(1)) 
#        p0 = mp.Process(target=PreviousStates, args=(Delta_now, Psi_now, Delta[:,t-1], A, B, list(range(0,40*40*40,4)), limit, Ind, t)) 
#        p1 = mp.Process(target=PreviousStates, args=(Delta_now, Psi_now, Delta[:,t-1], A, B, list(range(1,40*40*40,4)), limit, Ind, t)) 
#        p2 = mp.Process(target=PreviousStates, args=(Delta_now, Psi_now, Delta[:,t-1], A, B, list(range(2,40*40*40,4)), limit, Ind, t)) 
#        p3 = mp.Process(target=PreviousStates, args=(Delta_now, Psi_now, Delta[:,t-1], A, B, list(range(3,40*40*40,4)), limit, Ind, t)) 
        
        p0.start()
#        p1.start()
#        p2.start()
#        p3.start()
        
        p0.join()
#        p1.join()
#        p2.join()
#        p3.join()
        
        
        
        Delta[:,t] = Delta_now
        Psi[:,t] = Psi_now
        c = np.sum( Delta[:,t] )
        Delta[:,t] = Delta[:,t] / c
    Q[-1] = np.argmax( Delta[:,-1] )
    for t in range(num_timesteps-1,0,-1):
        Q[t-1] = Psi[ Q[t],t ]
    return i_sequence, Q

def collect_result(result):
    global results
    results.append(result)

def ViterbiMultiple( Y_now,A,Ind,Pi,idx,start_idx ):
    pool = mp.Pool(mp.cpu_count()-1) # omit the -1 if you don't browse, read, etc, in the meantime
    global results
    results = []
    
    for i in range(len(start_idx-1)):
        print(i)
        start = np.where( idx == start_idx[i] )[0][0]
        end   = np.where( idx == start_idx[i+1] )[0][0]
        Y_now_this  = Y_now[start:end,:]
        pool.apply_async(ViterbiFromNN, args=(Y_now_this,A,Ind,Pi,i), callback=collect_result)
    
    pool.close()
    pool.join()
    
    with open("Qresults_dev.txt", "wb") as fp:   #Pickling
        pickle.dump(results, fp)
    
    results.sort(key=lambda x: x[0])
    Q = [r for i, r in results]
    
    with open("Q3_dev.txt", "wb") as fp:   #Pickling
        pickle.dump(Q, fp)
    

def ViterbiOutput2Phonemes(Q):
    Y_now_est = np.floor( Q / 40 ) % 40
    
    return Y_now_est

def PhonemePerTimestep2Sequence( Y, min_timesteps = 1, discard_q = True ):
    sequence =[]
    for i in range( len(Y) - min_timesteps):
#        print(str(i)+ " " + str(len(Y)) + " " + str(len(sequence)))
        if ( ( len(sequence) == 0 or Y[i] != sequence[-1] ) and Y[i] == Y[i+min_timesteps] ):
            if ( discard_q == False or Y[i] != 28 ):
                sequence.append(Y[i])
    return sequence
    
def minimumEditDistance(s1,s2):
    ''' from https://rosettacode.org/wiki/Levenshtein_distance#Iterative
        not the most efficient, but easy to understand '''
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

def PhonemeErrorRate(seq_true,seq_est):
    return minimumEditDistance(seq_true,seq_est) / len(seq_true)
    
    
    
    
    