#!/usr/bin/env python3
#Author: Caspar v. Lengerke <caspar.lengerke@rwth-aachen.de>
#
#Script to train an autoencoder to transmit 8 bits using 8 complex channel samples.
#Between each set of data channel samples, a synchronization symbol of 8 complex 
#channel samples is utilized. 
#For training, the channel is modeled by Keras layers, some of which are custom
#made.
import numpy as np
import tensorflow as tf
from keras.layers import (Input, Dense, TimeDistributed, MaxPooling1D, Conv1D,
                          Reshape, Flatten, Concatenate, Multiply, Embedding)
from keras.models import Model

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from custom_layers import (GaussianNoiseCustomComplex, Complex2Real,
                           Real2Complex, ComplexRotate, Normalize, Serialize, 
                           MergeRandomTimeShift, ArgMax)
from custom_functions import (generate_data_sparse_fair, make_window_data_sparse,
                              generate_phase, generate_shift, generate_attenuation)
import os
#prevent warning about not using all available CPU instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#training parameters
epochs = 100
batch_size = 300
# Training dataset size
N = 5400000
#number of source symbols, correpsonds to k=log2(M) bits
M = 256 #8 bits
#number of complex samples n per symbol
symbol_length = 8

#number of data symbols per preamble symbol
no_data_encoders=1
#number of complex samples simultaneously looked at by the decoder
window_size = 3*symbol_length-1
#number of encoders needed during training to generate samples for all timeshifts
no_encoder = 5
#output width of synchronization feature detector_layer sub-network
paramter_detector_layer_output_width = 10

#=================Channel settings==================================
#phase 
random_phase = True
#timeshift
random_shift = True
#attenuation
random_attenuation = True
#Gaussian noise
EsN0_dB = 5
EsN0 = np.power(10,EsN0_dB/10)
EbN0 = EsN0 * symbol_length / int(np.log2(M))

#generate the necessary data and channel parameters for training
train_input_symbol = generate_data_sparse_fair(N,M)
train_phase = generate_phase(N, random_phase = random_phase)
train_shift = generate_shift(N, window_size,symbol_length,no_encoder,
                             random_shift = random_shift)
train_attenuation = generate_attenuation(N, window_size,
                                         random_attenuation = random_attenuation)
#create arrays of 5 symbols to give to the 5 encoders (needed to model time shift)
window_data = make_window_data_sparse(train_input_symbol, no_encoder)

#initialization for TensorBoard visualization
dir = os.path.dirname(os.path.realpath(__file__))
tbCallBack = TensorBoard(log_dir=dir+'/tf_graphs/train_autoencoder',
                         histogram_freq=0, write_graph=True, write_images=True)
#==============================================================================
#Encoder
#==============================================================================
input_symbol = Input(shape=(1,),name='input', dtype = 'int32')
with tf.name_scope('encoder'):
    encoder_layer = Embedding(M,M,input_length=1,
                              name='Embedding')(input_symbol)
    encoder_layer = Dense(M, activation='relu',
                          name='Encoder_Dense_1')(encoder_layer)
    encoder_layer = Dense(2*symbol_length, activation='relu',
                          name='Encoder_Dense_2')(encoder_layer)
    encoder_layer = Dense(2*symbol_length, activation='linear',
                          name='Encoder_Dense_3')(encoder_layer)
    #normalize pairs of real values to not exceed the unit circle of the 
    #complex plane, before casting each normalized pair of real numbers to complex
    encoder_layer = Normalize(norm_max_value = 1, sparse=True,
                              name='Encoder_Normalization') (encoder_layer)
    encoder_layer = Real2Complex()(encoder_layer)

encoder = Model(input_symbol,
                encoder_layer)

input_sequences = Input(shape=(no_encoder, 1),name = 'window_inputs')
#create no_encoder parallel copies of encoder (needed to model time shift)
processed_sequences = TimeDistributed(encoder, name = 'parallel_encoders')(input_sequences)                         
#==============================================================================
#Channel
#==============================================================================
input_shift = Input(shape=(1,),name='Shift_Input',dtype='int32') 
input_phase = Input(shape=(1,),name='Phase_Input')   
input_attenuation = Input(shape=(window_size,),
                          name='Attenuation_Input',dtype='complex64')  
with tf.name_scope('channel'):
    channel_layer = Serialize()(processed_sequences)    
    channel_layer = MergeRandomTimeShift(no_encoder=no_encoder,
                                         window_size=window_size)([channel_layer,
                                                                    input_shift])
    channel_layer = ComplexRotate(name='Phase_Noise') ([channel_layer,
                                                          input_phase])
    channel_layer = GaussianNoiseCustomComplex(1.0/(np.sqrt(2*EsN0)),
                                         name='Channel_Noise')(channel_layer)
    channel_layer = Multiply(name='Attenuation')([channel_layer,
                                                  input_attenuation])
#==============================================================================
#Decoder
#==============================================================================
deco_comp2real = Complex2Real()(channel_layer)
with tf.name_scope('parameterdetector_layer'):
    detector_layer = Reshape((window_size ,2,))(deco_comp2real)
    detector_layer = Conv1D(M, kernel_size=3,
                            strides=1, activation='relu')(detector_layer)
    detector_layer = MaxPooling1D(pool_size=1, strides=1)(detector_layer)
    detector_layer = Conv1D(128, kernel_size=2,
                            activation='relu')(detector_layer)
    detector_layer = MaxPooling1D(pool_size=2)(detector_layer)
    detector_layer = Flatten()(detector_layer)
    detector_layer = Dense(M*4, activation='relu' )(detector_layer)
    detector_layer = Dense(paramter_detector_layer_output_width,
                           activation='relu')(detector_layer)

with tf.name_scope('decoder'):
    decoder_layer = Concatenate(axis=-1)([detector_layer,
                                          deco_comp2real])
    decoder_layer = Dense(512, activation='relu')(decoder_layer)
    decoder_layer = Dense(512, activation='relu')(decoder_layer)   
    decoder_layer = Dense(256, activation='relu')(decoder_layer)
    decoder_layer = Dense(256, activation='relu')(decoder_layer)
    decoder_layer = Dense(M, activation='relu')(decoder_layer)
    decoder_layer = Dense(M, activation='softmax',
                          name='Decoder_Softmax')(decoder_layer)

autoencoder = Model(inputs = [input_sequences,
                              input_shift,
                              input_phase,
                              input_attenuation], 
                    outputs = decoder_layer)
autoencoder.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy')

print (autoencoder.summary())
#==============================================================================
#End of Model Definition, Start of Training
#==============================================================================
history = autoencoder.fit([window_data,
                           train_shift,
                           train_phase,
                           train_attenuation],
                            train_input_symbol,
                            epochs=epochs,
                            batch_size=batch_size,                
                            callbacks=[tbCallBack])
#plt.plot(history.history['loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Training', 'Validation'], loc='upper left')
#plt.show()

#==============================================================================
#Extraction of trained Decoder
#==============================================================================
encoded_input = Input(shape=(window_size,),dtype='complex64')
deco0 = autoencoder.layers[-16](encoded_input)
deco = autoencoder.layers[-15](deco0)
for i in range(-14,-7):
    deco = autoencoder.layers[i](deco)
deco = autoencoder.layers[-7]([deco,
                               deco0])
for i in range(-6,0):
    deco = autoencoder.layers[i](deco)
decoder_argmax = ArgMax()(deco)
decoder = Model(encoded_input,
                decoder_argmax)

#save trained encoder and decoder
if not os.path.exists(dir + '/export'):
    os.makedirs(dir + '/export')
encoder.save(dir + '/export/keras_weights_encoder.h5')
decoder.save(dir + '/export/keras_weights_decoder.h5')