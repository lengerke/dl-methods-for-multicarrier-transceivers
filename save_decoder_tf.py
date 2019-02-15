#!/usr/bin/env python3
import tensorflow as tf
from keras.layers import Input, Dense, MaxPooling1D, Conv1D, Reshape, Flatten, Concatenate
from keras.models import Model
from custom_layers import Complex2Real, ArgMax

from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #prevent warning about not using all available CPU instructions 
K.clear_session()

#number of source symbols, correpsonds to k=log2(M) bits
M = 256 #8 bits
#number of complex samples n per symbol
symbol_length = 8

#number of data symbols per preamble symbol
no_data_encoders=1
#number of complex samples simultaneously looked at by the decoder
window_size = 3*symbol_length-1

encoded_input = Input(shape=(window_size,),name='input',dtype='complex64')
deco_comp2real = Complex2Real()(encoded_input)

with tf.name_scope('decoder'):
    decoder1 = Dense(512, activation='relu')(deco_comp2real)
    decoder1 = Dense(512, activation='relu')(decoder1)   
    decoder1 = Dense(256, activation='relu')(decoder1)
    decoder1 = Dense(256, activation='relu')(decoder1)
    decoder1 = Dense(M, activation='relu')(decoder1)
    decoder1 = Dense(M, activation='softmax',name='Decoder_Softmax')(decoder1)

decoder_argmax = ArgMax()(decoder1)
decoder = Model(encoded_input, decoder_argmax)
tf.identity(decoder_argmax, name='output')

print (decoder.summary())

dir = os.path.dirname(os.path.realpath(__file__))

decoder.load_weights(dir + '/export/keras_weights_decoder.h5')
sess=K.get_session()
saver = tf.train.Saver()
saver.save(sess, dir + '/export/decoder/tf_model')

# write graph
tf.summary.FileWriter(dir + '/tf_graphs/decoder',sess.graph)

K.clear_session()