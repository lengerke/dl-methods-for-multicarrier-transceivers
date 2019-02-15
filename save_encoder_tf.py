#!/usr/bin/env python3
import tensorflow as tf
from keras.layers import Input, Dense, Embedding
from keras.models import Model
from custom_layers import  Normalize, Real2Complex

from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #prevent warning about not using all available CPU instructions 
K.clear_session()
# log2(M-1) bits to encode
M = 128
# Training dataset size
N = 2400000
#number of complex samples per symbol
symbol_length = 16
# number of messages simultaneously looked at by the decoder
no_data_encoders=1 #number of data symbols per preamble symbol
window_size = 3*symbol_length-1#(no_data_encoders+1)*2-1
no_encoder = 5#(no_data_encoders+1)*3#window_size*2-1
paramter_detector_output_width = 10
decoder_width = 32

input_symbol = Input(shape=(1,),name='input', dtype = 'int32')
with tf.name_scope('encoder'):
    encoder0 = Embedding(M,M,input_length=1,name='Embedding')(input_symbol)
    encoder1 = Dense(M, activation='relu',name='Encoder_Dense_1')(encoder0)
    encoder2 = Dense(2*symbol_length, activation='relu',name='Encoder_Dense_2')(encoder1)
    to_normalize = Dense(2*symbol_length, activation='linear',name='Encoder_Dense_3')(encoder2)
    to_complex = Normalize(norm_max_value =1, sparse=True,name='Encoder_Normalization') (to_normalize)
    to_channel = Real2Complex()(to_complex)

encoder = Model(input_symbol, to_channel)
tf.identity(to_channel, name='output')
print(encoder.summary())
dir = os.path.dirname(os.path.realpath(__file__))

encoder.load_weights(dir + '/export/keras_weights_encoder.h5')
sess=K.get_session()
saver = tf.train.Saver()
saver.save(sess, dir + '/export/encoder/tf_model')
# write graph
tf.summary.FileWriter(dir + '/tf_graphs/encoder',sess.graph)

K.clear_session()