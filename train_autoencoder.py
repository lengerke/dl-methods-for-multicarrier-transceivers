#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, TimeDistributed, MaxPooling1D, Conv1D, Reshape, Flatten, Concatenate, Multiply, Embedding
from keras.models import Model

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from custom_layers import GaussianNoiseCustomComplex, Complex2Real, Real2Complex, ComplexRotate, Normalize, Serialize, MergeRandomTimeShift, ArgMax
from custom_functions import generate_data_sparse_fair, make_window_data_sparse, generate_phase, generate_shift, generate_attenuation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #prevent warning about not using all available CPU instructions 
dir = os.path.dirname(os.path.realpath(__file__))

# log2(M-1) bits to encode
M = 2**7
# Training dataset size
N = 6000000
#number of complex samples per symbol
symbol_length = 8
# number of messages simultaneously looked at by the decoder
no_data_encoders=1 #number of data symbols per preamble symbol
window_size = 3*symbol_length-1#(no_data_encoders+1)*2-1
no_encoder = 5#(no_data_encoders+1)*3#window_size*2-1
paramter_detector_output_width = 128
decoder_width = 32
# phase 
random_phase = True
# timeshift
random_shift = True
random_attenuation = True
# SNR
EsN0_dB = 9
EsN0 = np.power(10,EsN0_dB/10)
EbN0 = EsN0 * symbol_length / int(np.log2(M))
# training parameters
epochs = 8
batch_size = 300

#np.random.seed(1234)
train_input_symbol = generate_data_sparse_fair(N,M)
train_phase = generate_phase(N, random_phase = random_phase)
train_shift = generate_shift(N, window_size,symbol_length,no_encoder,random_shift = random_shift)
window_data = make_window_data_sparse(train_input_symbol, no_encoder)
train_attenuation = generate_attenuation(N, window_size,random_attenuation = random_attenuation)
    
tbCallBack = TensorBoard(log_dir=dir+'/tf_graphs/train_autoencoder',
                         histogram_freq=0, write_graph=True, write_images=True)
#==============================================================================
#Encoder
#==============================================================================
input_symbol = Input(shape=(1,),name='input', dtype = 'int32')
with tf.name_scope('encoder'):
    encoder0 = Embedding(M,M,input_length=1,name='Embedding')(input_symbol)
    encoder1 = Dense(M, activation='relu',name='Encoder_Dense_1')(encoder0)
    encoder2 = Dense(2*symbol_length, activation='relu',name='Encoder_Dense_2')(encoder1)
    to_normalize = Dense(2*symbol_length, activation='linear',name='Encoder_Dense_3')(encoder2)
    to_complex = Normalize(norm_max_value =1, sparse=True,name='Encoder_Normalization') (to_normalize)
    to_channel = Real2Complex()(to_complex)

encoder = Model(input_symbol, to_channel)

input_sequences = Input(shape=(no_encoder, 1),name = 'window_inputs')
processed_sequences = TimeDistributed(encoder, name = 'parallel_encoders')(input_sequences)                         
#==============================================================================
#Channel
#==============================================================================
input_shift = Input(shape=(1,),name='Shift_Input',dtype='int32') 
input_phase = Input(shape=(1,),name='Phase_Input')   
input_attenuation = Input(shape=(window_size,),name='Attenuation_Input',dtype='complex64')  
with tf.name_scope('channel'):
    channel = Serialize()(processed_sequences)    
    channel = MergeRandomTimeShift(no_encoder=no_encoder,window_size=window_size)([channel,input_shift])
    channel = ComplexRotate(name='Phase_Noise') ([channel,input_phase])
    channel = GaussianNoiseCustomComplex(1.0/(np.sqrt(2*EsN0)),name='Channel_Noise')(channel)
    channel = Multiply(name='Attenuation')([channel,input_attenuation])
#==============================================================================
#Decoder
#==============================================================================
deco_comp2real = Complex2Real()(channel)
with tf.name_scope('parameterdetector'):
    detector = Reshape((window_size ,2,))(deco_comp2real)
    detector = Conv1D(M, kernel_size=3, strides=1, activation='relu')(detector)
    detector = MaxPooling1D(pool_size=1, strides=1)(detector)
    detector = Conv1D(128, kernel_size=2, activation='relu')(detector)
    detector = MaxPooling1D(pool_size=2)(detector)
    detector = Flatten()(detector)
    detector = Dense(M, activation='relu' )(detector)
    detector = Dense(paramter_detector_output_width, activation='relu')(detector)

with tf.name_scope('decoder'):
    decoder1 = Concatenate(axis=-1)([detector,deco_comp2real])
    decoder1 = Dense(512, activation='relu')(decoder1)
    decoder1 = Dense(512, activation='relu')(decoder1)   
    decoder1 = Dense(M*2, activation='relu')(decoder1)
    decoder1 = Dense(M, activation='relu')(decoder1)
    decoder1 = Dense(M, activation='relu')(decoder1)
    decoder1 = Dense(M, activation='softmax',name='Decoder_Softmax')(decoder1)

autoencoder = Model(inputs = [input_sequences,input_shift,input_phase,input_attenuation], 
                    outputs = decoder1)
autoencoder.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy')

print (autoencoder.summary())
#==============================================================================
#End of Model Definition, Start of Training
#==============================================================================
history = autoencoder.fit([window_data,train_shift,train_phase,train_attenuation], train_input_symbol,
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
#Extraction of Decoder
#==============================================================================
encoded_input = Input(shape=(window_size,),dtype='complex64')
deco0 = autoencoder.layers[-16](encoded_input)
deco = autoencoder.layers[-15](deco0)
for i in range(-14,-7):
    deco = autoencoder.layers[i](deco)
deco = autoencoder.layers[-7]([deco,deco0])
for i in range(-6,0):
    deco = autoencoder.layers[i](deco)
decoder_argmax = ArgMax()(deco)
decoder = Model(encoded_input, decoder_argmax)
if not os.path.exists(dir + '/export'):
    os.makedirs(dir + '/export')
encoder.save(dir + '/export/keras_weights_encoder.h5')
decoder.save(dir + '/export/keras_weights_decoder.h5')
