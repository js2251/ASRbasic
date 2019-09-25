# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:56:53 2019

@author: js2251
"""

#'D:\matlab\TIMIT\TRAIN' short
# 'D:\TIMIT\TRAIN' full

import numpy as np
import pandas as pd
import os
import os.path
import librosa
from scipy.io import wavfile
import sklearn

import sys
sys.path.insert(0, 'D:\\python\\')
from myfft.FourierTransforms import AuditoryFeatures


def getPhonemeDict( dir_data = 'D:\TIMIT\TRAIN' ):
    phonemes = set()
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.PHN')]:
            X = pd.read_csv(dirpath+'/'+file, sep=' ', header=None)
            for i in range(len(X)):
                phonemes.add(X[2][i])
        if len(phonemes) >= 61:
            break
    phonemes_sorted = sorted(phonemes)
    dict_phonemes_num = dict()
    i=0
    for val in phonemes_sorted:
        dict_phonemes_num[ val ] = i
        i+=1
     
    f = open('PhonemeDict61.txt','w')
    f.write( str(dict_phonemes_num) )
    f.close()
    return dict_phonemes_num
    

def getPhonemeDict39(dict_phonemes_num61):
    dict_phonemes_num39   = dict_phonemes_num61
    dict_phonemes_num39['ao'] = dict_phonemes_num61['aa']
    dict_phonemes_num39['ax'] = dict_phonemes_num61['ah']
    dict_phonemes_num39['ax-h'] = dict_phonemes_num61['ah']
    dict_phonemes_num39['axr'] = dict_phonemes_num61['er']
    dict_phonemes_num39['hv'] = dict_phonemes_num61['hh']
    dict_phonemes_num39['ix'] = dict_phonemes_num61['ih']
    dict_phonemes_num39['el'] = dict_phonemes_num61['l']
    dict_phonemes_num39['em'] = dict_phonemes_num61['m']
    dict_phonemes_num39['en'] = dict_phonemes_num61['n']
    dict_phonemes_num39['nx'] = dict_phonemes_num61['n']
    dict_phonemes_num39['eng'] = dict_phonemes_num61['ng']
    dict_phonemes_num39['zh'] = dict_phonemes_num61['sh']
    dict_phonemes_num39['ux'] = dict_phonemes_num61['uw']
    dict_phonemes_num39['pcl'] = dict_phonemes_num61['h#']   # actually a new phoneme, phonetic silence
    dict_phonemes_num39['tcl'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['kcl'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['bcl'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['dcl'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['gcl'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['pau'] = dict_phonemes_num61['h#']
    dict_phonemes_num39['epi'] = dict_phonemes_num61['h#']
                       
    f = open('PhonemeDict39bignumbers.txt','w')
    f.write( str(dict_phonemes_num39) )
    f.close()

    return dict_phonemes_num39

def savePhoneme39Dict():
    ''' after savePhonemeInformation39, to save the dict as a txt file '''
    dict_phonemes_num39big = eval( open('PhonemeDict39bignumbers.txt', 'r').read() )
    dict_num_phonemes39big = {v: k for k, v in dict_phonemes_num39big.items()}
    f = open('PhonemeDict39bignumbers_num_key.txt','w')
    f.write( str(dict_num_phonemes39big) )
    f.close()
    
    # corresponding to Phoneme39ToConsecutive(Y)
    dict_num_phonemes39 = dict()
    dict_num_phonemes39[0] = 'aa'
    dict_num_phonemes39[1] = 'ae'
    dict_num_phonemes39[2] = 'ah'
    dict_num_phonemes39[3] = 'aw'
    dict_num_phonemes39[4] = 'ay'
    dict_num_phonemes39[5] = 'b'
    dict_num_phonemes39[6] = 'ch'
    dict_num_phonemes39[7] = 'd'
    dict_num_phonemes39[8] = 'dh'
    dict_num_phonemes39[9] = 'dx'
    dict_num_phonemes39[10] = 'eh'
    dict_num_phonemes39[11] = 'er'
    dict_num_phonemes39[12] = 'ey'
    dict_num_phonemes39[13] = 'f'
    dict_num_phonemes39[14] = 'g'
    dict_num_phonemes39[15] = 'h#'
    dict_num_phonemes39[16] = 'hh'
    dict_num_phonemes39[17] = 'ih'
    dict_num_phonemes39[18] = 'iy'
    dict_num_phonemes39[19] = 'jh'
    dict_num_phonemes39[20] = 'k'
    dict_num_phonemes39[21] = 'l'
    dict_num_phonemes39[22] = 'm'
    dict_num_phonemes39[23] = 'n'
    dict_num_phonemes39[24] = 'ng'
    dict_num_phonemes39[25] = 'ow'
    dict_num_phonemes39[26] = 'oy'
    dict_num_phonemes39[27] = 'p'
    dict_num_phonemes39[28] = 'q'
    dict_num_phonemes39[29] = 'r'
    dict_num_phonemes39[30] = 's'
    dict_num_phonemes39[31] = 'sh'
    dict_num_phonemes39[32] = 't'
    dict_num_phonemes39[33] = 'th'
    dict_num_phonemes39[34] = 'uh'
    dict_num_phonemes39[35] = 'uw'
    dict_num_phonemes39[36] = 'v'
    dict_num_phonemes39[37] = 'w'
    dict_num_phonemes39[38] = 'y'
    dict_num_phonemes39[39] = 'z'
    
    f = open('PhonemeDict39_num_key.txt','w')
    f.write( str(dict_num_phonemes39) )
    f.close()
    
    dict_phonemes_num39 = {v: k for k, v in dict_num_phonemes39.items()}
    f = open('PhonemeDict39.txt','w')
    f.write( str(dict_phonemes_num39) )
    f.close()

def checkDicts(Y39nc,Y39):
    ''' testing if conversions from big to small numbers were correct '''
    dict_phonemes_num39big = eval( open('PhonemeDict39bignumbers.txt', 'r').read() )
    dict_phonemes_num39    = eval( open('PhonemeDict39.txt', 'r').read() )
    for k, v in dict_phonemes_num39.items():
        a = np.argmax(Y39==v)
        b = np.argmax(Y39nc==dict_phonemes_num39big[k])
        print(a==b)

def getDictNumToPhoneme61():
    dict_phonemes_num61 = eval( open('PhonemeDict61.txt', 'r').read() )
    dict_num_phonemes61 = dict()
    for key in dict_phonemes_num61:
        dict_num_phonemes61[ dict_phonemes_num61[key] ] = key
    return dict_num_phonemes61
        
def savePhonemeInformation( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000 ):
    dict_phonemes_num = getPhonemeDict( dir_data )
    Y        = np.empty((20000000,1), dtype='float32')
    num_data = 0
    seg_size = round( ms_per_seg * fs / 1000 )
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.PHN')]:
            print(file)
            fs, s = wavfile.read(dirpath+'/'+file[:-4]+'.WAV')
            end   = len(s)
            M     = pd.read_csv(dirpath+'/'+file, sep=' ', header=None)
            vec_end     = M[1].values
            vec_end[-1] = end
            Y_len_this  = int( np.floor(end/seg_size) )
            for i in range( 1,Y_len_this+1 ):
                id_in_stream = sum( vec_end < i*seg_size )
                Y[num_data]     = dict_phonemes_num[M[2][id_in_stream]]
                num_data         += 1
    Y = Y[:num_data]
    np.save('Phonemes',Y)

def Phoneme39ToConsecutive(Y): # from 39 numbers between 0 and 60 to between 0 and 39
    Y = Y.reshape( Y.shape[0], )
    labels_big = list(set(Y))   # 'big' numbers to 61, length 40
    labels_small = -np.ones(61,) # small numbers to 40, list with length 61
    for i in range(len(labels_big)):
        labels_small[ int( labels_big[i] ) ] = i
    for i in range(len(Y)):
        Y[i] = labels_small[ int( Y[i] ) ]
    return Y
    
def savePhonemeInformation39():
    ''' as above, but 39 phonemes'''
    dict_phonemes_num61 = eval( open('PhonemeDict61.txt', 'r').read() )
    dict_phonemes_num39 = getPhonemeDict39(dict_phonemes_num61)
    dict_num_phonemes61 = getDictNumToPhoneme61()
    
    Y61 = np.load('Phonemes.npy')
    Y = Y61    
    for i in range(len(Y61)):
        Y[i] = dict_phonemes_num39[ dict_num_phonemes61[ Y61[i][0] ] ]
    np.save('Phonemes39',Y)
    Y = Phoneme39ToConsecutive(Y)
    np.save('Phonemes39consecutive',Y)
    
def saveAuditoryInformation( ms_per_seg, dir_data = 'D:\TIMIT\TRAIN', fs = 16000): # 2 ms, D:\matlab\TIMIT\TRAIN
    win_length = 2 ** np.arange(6,12) # 64 to 2048
    L  = np.empty([12000000, int(len(win_length)*(win_length[0]/2+1))],dtype = 'float32' )
    CC = np.empty([12000000, int(len(win_length)*win_length[0]/2)],dtype = 'float32' )
    num_data = 0
    seg_size = round( ms_per_seg * fs / 1000 )                # step size: 2 * 16 = 32 samples
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.WAV')]:
            print(file + ' ' + str(num_data))
            s, fs = librosa.load(dirpath+'/'+file[:-4]+'.WAV',sr=None)
            end   = len(s)
            X_len_this  = int( np.floor(end/seg_size) )
            s = np.r_[ np.zeros([win_length[-1]-seg_size,],dtype = 'float32'),s ]
            for i in range( X_len_this ):
                s_this         = s[i*seg_size:i*seg_size+win_length[-1]] # 2048 samples segment
                L_this,CC_this = AuditoryFeatures( s_this, int(np.log2(win_length[-1])-np.log2(win_length[0])+1), fs ) # s_this, 6, 16000
                L[num_data,]   = L_this
                CC[num_data,]  = CC_this
                num_data       += 1
    L  = L[:num_data,]
    CC = CC[:num_data,]
    np.save('CC',CC)
    np.save('L',L)
    
def AuditoryToFrame( name_in, name_out, frame_length=10 ):
    X  = np.load(name_in)
    width,height = X.shape
    X2 = np.empty([int(np.floor(width/frame_length)), frame_length,height],dtype = 'float32' )
    for i in range( int(np.floor(width/frame_length)) ):
        if (i % 10000 == 0):
            print(i)
        X_this = X[i*frame_length:(i+1)*frame_length,].reshape(1,frame_length,height)
        X2[i,:,:] = X_this
    np.save(name_out,X2)
                
def saveMelInformation( ms_per_seg, dir_data = 'D:\TIMIT\TRAIN', fs = 16000 ):
    Smel = np.empty([12000000, 128],dtype = 'float32' )
    MFCC = np.empty([12000000, 20],dtype = 'float32' )
    win_length = 512
    num_data = 0
    seg_size = round( ms_per_seg * fs / 1000 )
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.WAV')]:
            print(file)
            s, fs = librosa.load(dirpath+'/'+file[:-4]+'.WAV',sr=None)
            end   = len(s)
            X_len_this  = int( np.floor(end/seg_size) )
            s = np.r_[ np.zeros([win_length-seg_size,],dtype = 'float32'),s ]
            for i in range( X_len_this ):
                s_this          = s[i*seg_size:i*seg_size+win_length]
                Smel_this       = librosa.feature.melspectrogram(s_this, sr=fs, n_fft=win_length, hop_length=win_length*2, power=2.0, n_mels=128)
                Smel_this       = Smel_this.reshape(128,)
                MFCC_this       = librosa.feature.mfcc( sr=fs, S=librosa.power_to_db(Smel), n_mfcc=20, dct_type=2, norm='ortho' )
                MFCC_this       = MFCC_this.reshape(20,)
                Smel[num_data,] = Smel_this
                MFCC[num_data,] = MFCC_this
                num_data       += 1
    Smel = Smel[:num_data,]
    MFCC = MFCC[:num_data,]
    np.save('Smel',Smel)
    np.save('MFCC',MFCC)
    
def TotalDurationTimit( dir_data = 'D:\TIMIT\TRAIN', fs = 16000):
    num_ms = 0
    seg_size = round(  fs / 1000 )
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.WAV')]:
            print(file)
            s, fs = librosa.load(dirpath+'/'+file[:-4]+'.WAV',sr=None)
            end   = len(s)
            X_len_this  = int( np.floor(end/seg_size) )
            num_ms += X_len_this
            
    return num_ms

def saveStartIndices( ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000, split = 0.9 ):
    ''' vector with indices that indicate the start of a sentence, and type of sentence (SA,SI,SX) '''
    
    idx_start    = np.empty((6300,), dtype='int')
    idx_start[0] = 0
    idx_type     = []
    
    num_data    = 0
    Y_len_total = 0
    seg_size = round( ms_per_seg * fs / 1000 )
    for dirpath, dirnames, filenames in os.walk( dir_data ):
        for file in [f for f in filenames if f.endswith('.WAV')]:
            print(file)
            fs, s                 = wavfile.read(dirpath+'/'+file)
            end                   = len(s)
            Y_len_this            = int( np.floor(end/seg_size) )
            idx_start[num_data+1] = idx_start[num_data] + Y_len_this
            idx_type.append( file[:2] )
            Y_len_total           += Y_len_this
            num_data              += 1
            
    idx_start = idx_start[:num_data]
    print(num_data)
    np.save('StartIndices',idx_start)
    np.save('SentenceType',idx_type)
    np.save('StartIndices_train',idx_start[:int(split*len(idx_start))])
    np.save('SentenceType_train',idx_type[:int(split*len(idx_start))])
    np.save('StartIndices_dev',idx_start[int(split*len(idx_start)):])
    np.save('SentenceType_dev',idx_type[int(split*len(idx_start)):])
    return num_data, Y_len_total

def getIndexNonSa(ms_per_seg = 2, dir_data = 'D:\TIMIT\TRAIN', fs = 16000, split = 0.9):
    num_data, Y_len_total = saveStartIndices( ms_per_seg, dir_data, fs )
    idx_start_all         = np.load('StartIndices.npy')
    idx_type              = np.load('SentenceType.npy')
    idx_start_all         = np.append(idx_start_all,Y_len_total)
    idx_train             = np.empty((0,),dtype = 'int')
    idx_test              = np.empty((0,),dtype = 'int')
    for i in range(int( split * len(idx_type)) ):
        if idx_type[i] != 'SA':
            idx_train = np.append( idx_train,np.arange(idx_start_all[i],idx_start_all[i+1] ) )
    for i in range(int( split * len(idx_type)), len(idx_type) ):
        if idx_type[i] != 'SA':
            idx_test = np.append( idx_test,np.arange(idx_start_all[i],idx_start_all[i+1] ) )
    np.save('NonSaIdx_train.npy',idx_train)
    np.save('NonSaIdx_dev.npy',idx_test)
    return idx_train, idx_test
    
    
def removeStartOfPhonemes(filename_Y,n):
    Y = np.load(filename_Y+'.npy')
    Y_bool  = np.zeros(Y.shape,dtype='int8')
    Y_index = -np.ones(Y.shape,dtype='int32')
    Y_pos   = -np.ones(Y.shape,dtype='int32')     # index to position in Y
    m = 0
    for i in range(n,len(Y)):
        if ( Y[i] == Y[i-n] ):
            Y_bool[i] = 1
            Y_index[i] = m
            m += 1
    np.save(filename_Y+'_include.npy',Y_bool)
    np.save(filename_Y+'_index.npy',Y_index)
    for i in range(len(Y)):
        if (Y_index[i] >= 0):
            Y_pos[ int(Y_index[i]) ] = int(i)
    Y_pos = Y_pos[:int(max(Y_index))]
    np.save(filename_Y+'_position_index.npy',Y_pos)
    
def scaleAuditoryVariables( filename_CC = 'CC.npy', filename_L = 'L.npy' ):
    CC = np.load( filename_CC )
    CC = sklearn.preprocessing.scale(CC)
    np.save('CC_scaled',CC)
    del CC
    L = np.load( filename_L)
    L = sklearn.preprocessing.scale(L)
    np.save('L_scaled',L)
    
def getTimeOffsetLabel(Y,offset=0):
    if offset == 0:
        return Y
    default = np.ones(offset,) * 15 # default end/start to silence
    idx = np.arange( len(Y) )
    if offset >= 0:
        Y[:-offset] = Y[idx[offset:]]
        Y[-offset:] = default
    else:
        Y[offset:] = Y[idx[:-offset]]
        Y[:offset] = default
    return Y

def getNextLabel(Y_orig,max_offset = 100):
    Y_next = np.empty(len(Y_orig),dtype='int')
    next_label = Y_orig[0]
    j = 0
    while (next_label == Y_orig[0] and j < max_offset):
        j+= 1
        next_label = Y_orig[j]       
    Y_next[0] = next_label
    for i in range(1,len(Y_orig)):
        while( next_label == Y_orig[i] and j < i + max_offset and j < len(Y_orig) - 1 ):
            j+= 1
            next_label = Y_orig[j] 
        Y_next[i] = next_label
    return Y_next

def getPreviousLabel(Y_orig,max_offset = 100):
    Y_prev = np.empty(len(Y_orig),dtype='int')
    prev_label = Y_orig[0]
    j = 0
    while (prev_label == Y_orig[j] and j < max_offset):
        j+=1
        Y_prev[j] = prev_label
    for i in range(j,len(Y_orig)):
        if (Y_orig[i] != Y_orig[i-1]):
            prev_label = Y_orig[i-1]
            j = i-1
        if ( i > j + max_offset ):
            j = i - max_offset
            prev_label = Y_orig[j]
        Y_prev[i] = prev_label
    return Y_prev            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        