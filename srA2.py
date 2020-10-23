# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:48:18 2018

@author: js
"""

import numpy as np
import tensorflow as tf
import sklearn
from DataGenTimitTri import DataGeneratorTri

def trainNonCausalNN( filename_X = 'srA1_logp_combined_all.npy', filename_Y = 'Phonemes39consecutive.npy', file_identifier_out = 'srA2_a', epochs_to_save = 1, epochs_total = 100, batch_size = 1024, reduce_factor = 1, load_model = None ): 
    X = np.load( filename_X )
    Y = np.load( filename_Y )
    Y = Y[50:]   # 50 timesteps in first causal NN
    idx = np.arange(len(X))   # predict all including first after phoneme border
    
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(Y),Y)
    
    idx = idx.reshape(idx.shape[0],)
    
    num_features = X.shape[1]
    num_timesteps = 305
    dim          = ( batch_size, num_timesteps, num_features )
    
    data_split = int( 0.9 * len(idx))
    idx_train = idx[:data_split]
    idx_val   = idx[data_split:]
    
    #model=tf.keras.models.load_model('srA2_c_9.h5')
    #weights = model.get_weights()
    
    inp = tf.keras.Input( shape = ( num_timesteps, num_features ) )
    
    x = tf.keras.layers.AveragePooling1D( pool_size = 5, name = "pool_10ms")(inp)
    
    x = tf.keras.layers.Bidirectional( tf.keras.layers.GRU( 128, return_sequences=True, trainable = True ), name = "gru1" )(x) # , kernel_regularizer=tf.keras.regularizers.l2(0.01)
    x = tf.keras.layers.Bidirectional( tf.keras.layers.GRU( 128  ), name = "gru2", trainable = True )(x) # , recurrent_dropout=0.2
    
    out1 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_prev2"  )(x)
    out2 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_now2"  )(x)
    out3 = tf.keras.layers.Dense( 40, activation='softmax', name = "out_next2"  )(x)
    
    model = tf.keras.Model(inputs=inp, outputs=[out1,out2,out3])
    
    if load_model is not None:
            model=tf.keras.models.load_model( load_model ) # start from a previously saved model, discard model that was just compiled
        
    tf.keras.utils.plot_model(model,show_shapes=1, to_file='model_bidirectional.png')
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['categorical_accuracy']) # reporting the accuracy
    
    training_generator = DataGeneratorTri(idx_train, X, Y, dim, reduce_factor = reduce_factor,non_causal_steps = int(num_timesteps/2))
    validation_generator = DataGeneratorTri(idx_val, X, Y, dim, reduce_factor = reduce_factor,non_causal_steps = int(num_timesteps/2))
    
    # Train model on dataset
    for i in range( int(epochs_total / epochs_to_save) ):
        model.fit_generator(generator = training_generator,
                            validation_data = validation_generator,
                            class_weight = [class_weight,class_weight,class_weight],
                            epochs = epochs_to_save)
        model.save(file_identifier_out+'_'+str(i)+'.h5')
    
    


