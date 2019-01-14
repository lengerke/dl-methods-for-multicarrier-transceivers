#!/usr/bin/env python3
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec
import tensorflow as tf

class ArgMax(Layer):
    def __init__(self, **kwargs):
        super(ArgMax, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.argmax(inputs,axis=1,output_type=tf.int32)

class Rotate(Layer):
    
    def __init__(self, output_dim, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Rotate, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs):
        
        a=K.cos(-inputs[:,-1])
        b=K.sin(-inputs[:,-1])
        rotation1=K.stack([a,-b],axis=1)
        rotation2=K.stack([b,a],axis=1)
        rotation=K.stack([rotation1,rotation2],axis=2)
        output=K.batch_dot(rotation,inputs[:,:-1])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class GaussianNoiseCustom(Layer):

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseCustom, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoiseCustom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape 
 
class GaussianNoiseCustomComplex(Layer):

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseCustomComplex, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        a = tf.real(inputs) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        b = tf.imag(inputs) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return tf.complex(a,b)        

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoiseCustom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape 
     
class Serialize(Layer):
    
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Serialize, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs):
        inputshape=inputs.shape.as_list()
        outputsize=int(inputshape[1]*inputshape[2])
        output=tf.reshape(inputs,[-1,outputsize])#-1 will be the batch size
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*input_shape[2])
      
class Real2Complex(Layer):
    
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Real2Complex, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs):
        inputshape=inputs.shape.as_list()
        outputsize=int(inputshape[-1]/2) #number of complex outputs per input
        a=tf.reshape(inputs,[-1,outputsize,2])#-1 will be the batch size
        output = tf.complex(a[:,:,0],a[:,:,1])

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[-1]/2))


class Complex2Real(Layer):
    
    def __init__(self,  **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Complex2Real, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs):
        inputshape=inputs.shape.as_list()
        outputsize=int(inputshape[-1]*2) #number of real outputs per input
        a=tf.stack([tf.real(inputs), tf.imag(inputs )], axis=2)
        output=tf.reshape(a,[-1,outputsize],)#-1 will be the batch size
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[-1]*2)) 
      
class RandomTimeShift(Layer):
    
    def __init__(self, window_size,no_encoder, random_flag,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RandomTimeShift, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.window_size = window_size
        self.no_encoder = no_encoder
        self.random_flag = random_flag

    def call(self, inputs):
        inputshape=inputs.shape.as_list()
        outputsize=int(inputshape[-1]/self.no_encoder*self.window_size) 
        self.outputsize = outputsize
        random_shift = tf.random_uniform((1,1),minval=0, maxval=inputshape[-1]-outputsize, dtype='int32')        
        fixed_shift = tf.constant(1, shape = (1,1))
        if self.random_flag == True:
            shift = random_shift
        else:
            shift = fixed_shift        
        a = tf.stack([tf.zeros((1,1),dtype='int32'),shift  ])
        b = tf.squeeze(a)
        output = tf.slice(inputs, b, [-1,outputsize])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.outputsize) 

class NormalizeExpMirror(Layer):
    def __init__(self, norm_max_value, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NormalizeExpMirror, self).__init__(**kwargs)        
        self.input_spec = InputSpec(min_ndim=2)
        self.norm_max_value = norm_max_value
    def call(self, inputs, training=None):
        def output():
            inputshape=inputs.shape.as_list()
            a=tf.reshape(inputs,[-1,int(inputshape[1]/2),2])
            b=tf.norm(a)
            c=tf.where (tf.greater_equal(b,tf.ones_like(b)), tf.multiply(a,tf.exp(tf.ones_like(a)-tf.pow(a,2))), a) 
            return tf.reshape(c,[-1,inputshape[1]])
         
        return K.in_train_phase(output, inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape  

class NormalizeRandom(Layer):
    def __init__(self, norm_max_value, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NormalizeRandom, self).__init__(**kwargs)        
        self.input_spec = InputSpec(min_ndim=2)
        self.norm_max_value = norm_max_value
    def call(self, inputs, training=None):
        def output():
            inputshape=inputs.shape.as_list()
            z=int(inputshape[1]/2)
            a=tf.reshape(inputs,[-1,z,2])
            b=tf.norm(a)
            x=tf.random_uniform(K.shape(inputs),minval=-1,maxval=1)    
            y=tf.reshape(x,[-1,z,2])
            d=tf.where (tf.greater_equal(b,tf.ones_like(b)), y, a) 
            return tf.reshape(d,[-1,inputshape[1]])
        return K.in_train_phase(output, inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape  
     
class Normalize(Layer):
    def __init__(self, norm_max_value, sparse=False, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Normalize, self).__init__(**kwargs)        
        self.input_spec = InputSpec(min_ndim=2)
        self.norm_max_value = norm_max_value
        self.sparse = sparse
    def call(self, inputs):
        inputshape=inputs.shape.as_list()
        if self.sparse == True:
            inputshape[1]=inputshape[2]
        a=tf.reshape(inputs,[-1,int(inputshape[1]/2),2])        
        b=tf.clip_by_norm(a, self.norm_max_value, axes = 2)
        output=tf.reshape(b,[-1,inputshape[1]]) 
        return output

    def compute_output_shape(self, input_shape):
        return input_shape    
      
class _Merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        # Arguments
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        # Returns
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        # Raises
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Can not merge tensors with different '
                             'batch sizes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if self._reshape_required:            
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = K.expand_dims(x, 1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = K.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate([x_shape[1:], K.expand_dims(batch_size)])
                        x_transposed = K.reshape(x, K.stack([batch_size, K.prod(x_shape[1:])]))
                        x_transposed = K.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = K.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(K.permute_dimensions(x, dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed, we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = K.shape(y)
                        y_ndim = K.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([K.expand_dims(batch_size), y_shape[:y_ndim - 1]])
                        y = K.reshape(y, (-1, batch_size))
                        y = K.permute_dimensions(y, (1, 0))
                        y = K.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = K.permute_dimensions(y, dims)
                return y
        else:            
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)

class ComplexRotate(_Merge):
    
    def __init__(self, **kwargs):
        super(ComplexRotate, self).__init__(**kwargs)

        self.supports_masking = False
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on exactly 2 inputs')
        x1 = inputs[0]
        x2 = inputs[1] 
        x2_len = inputs[1].shape
        x3 = tf.exp(tf.complex(tf.zeros(x2_len[1],1),x2))       
        output = tf.multiply(x1,x3)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on a list of 2 inputs.')
        #as the first input is only rotated by an amount determined by the second
        #input, the output shape equals the shape of the first input
        output_shape = list(input_shape[0])
        return tuple(output_shape)

    def get_config(self):
        config = {
        }
        base_config = super(ComplexRotate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MergeRandomTimeShift(_Merge):
    
    def __init__(self, window_size,no_encoder, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MergeRandomTimeShift, self).__init__(**kwargs)
       # self.input_spec = InputSpec(min_ndim=2)
        self.window_size = window_size
        self.no_encoder = no_encoder
        self.supports_masking = False
        self._reshape_required = False
        
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on exactly 2 inputs')
        batch_size = 300
        x = inputs[0]
        shift = inputs[1]    
        inputshape=x.shape.as_list() 
        outputsize=int(self.window_size) 
        self.outputsize = outputsize 
        
        b = []
        for i in range(0,batch_size):
            b.append(tf.slice(x[i,:],shift[i,:],[outputsize]))
        
        output = tf.convert_to_tensor(b)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ComplexRotate` layer should be called '
                             'on a list of 2 inputs.')
        a = input_shape[0]
        return (a[0], self.outputsize)
#        output_shape = list(input_shape[0])
#        return tuple(output_shape)

    def get_config(self):
        config = {
        }
        base_config = super(ComplexRotate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   