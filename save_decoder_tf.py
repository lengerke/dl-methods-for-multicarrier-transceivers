#!/usr/bin/env python3

#Imports weights from an .h5 file to save them in the tensorflow format.
#The generated format can then be imported into GNURadio Tensorflow blocks.
#Note that the layer structure here has to match the layer structure defined in 
#the train_autoencoder.py, if they as the script will only import the weights 
#into the structure defined here. If the layout does not match, the import will 
#fail.
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
#output width of synchronization feature detector_layer sub-network
paramter_detector_layer_output_width = 10

encoded_input = Input(shape=(window_size,),name='input',dtype='complex64')
deco_comp2real = Complex2Real()(encoded_input)

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

decoder_argmax = ArgMax()(decoder_layer)
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