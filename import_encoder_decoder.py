#!/usr/bin/env python3
import tensorflow as tf
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, Reshape, Flatten, Concatenate
from keras.models import Model
from custom_layers import  Normalize, Real2Complex, Complex2Real,ArgMax

from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #prevent warning about not using all available CPU instructions 

K.clear_session()
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

input_symbol = Input(shape=(1,),name='input', dtype = 'int32')
with tf.name_scope('encoder'):
    encoder0 = Embedding(M,M,input_length=1,name='Embedding')(input_symbol)
    encoder1 = Dense(M, activation='relu',name='Encoder_Dense_1')(encoder0)
    encoder2 = Dense(2*symbol_length, activation='relu',name='Encoder_Dense_2')(encoder1)
    to_normalize = Dense(2*symbol_length, activation='linear',name='Encoder_Dense_3')(encoder2)
    to_complex = Normalize(norm_max_value =1, sparse=True,name='Encoder_Normalization') (to_normalize)
    to_channel = Real2Complex()(to_complex)

encoder = Model(input_symbol, to_channel)
dir = os.path.dirname(os.path.realpath(__file__))
encoder.load_weights(dir + '/export/keras_weights_encoder.h5')

encoded_input = Input(shape=(window_size,),name='input',dtype='complex64')
deco_comp2real = Complex2Real()(encoded_input)
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

decoder_argmax = ArgMax()(decoder1)
decoder = Model(encoded_input, decoder_argmax)
dir = os.path.dirname(os.path.realpath(__file__))
decoder.load_weights(dir + '/export/keras_weights_decoder.h5')