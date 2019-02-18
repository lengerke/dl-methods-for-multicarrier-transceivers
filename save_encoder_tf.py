#!/usr/bin/env python3

#Imports weights from an .h5 file to save them in the tensorflow format.
#The generated format can then be imported into GNURadio Tensorflow blocks.
#Note that the layer structure here has to match the layer structure defined in 
#the train_autoencoder.py, if they as the script will only import the weights 
#into the structure defined here. If the layout does not match, the import will 
#fail.
import tensorflow as tf
from keras.layers import Input, Dense, Embedding
from keras.models import Model
from custom_layers import  Normalize, Real2Complex

from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #prevent warning about not using all available CPU instructions 
K.clear_session()

#number of source symbols, correpsonds to k=log2(M) bits
M = 256 #8 bits
#number of complex samples n per symbol
symbol_length = 8

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