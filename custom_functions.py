#!/usr/bin/env python3
import numpy as np
def generate_data(N,M):
    label = np.random.randint(M,size=N).astype(np.int32)
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data).astype(np.int32)
    return data, label
def generate_data_sparse(N,M):
    return np.random.randint(M,size=N).astype(np.int32)
def generate_data_sparse_fair(N,M):
    a=int(N/M)
    b=[]
    for i in range (M):
        b.append(i*np.ones((a,1)).astype(np.int32))
    c=np.array(b).flatten()
    b=[]
    for j in range (N-c.size):
        b.append(j)
    c=np.append(c,np.array(b))
    np.random.shuffle(c)
    return c
def make_window_data(data, no_encoder):
    data_length = data.shape[0]
    data_width = data.shape[1]
    padding_size = 1
    padding = np.zeros((data_length,data_width,padding_size))
    padding[:,0,:] = np.squeeze(np.ones((data_length,padding_size)))
    data = np.expand_dims(data,axis=2)
    a = np.append(data,padding,axis = 2)
    output = np.roll(a,int(padding_size/2),axis = 2)
    output = np.transpose(output, (0,2,1))
    return output
def make_window_data_sparse(data,no_encoder):
    shift=((no_encoder+1)/2-1)/2
    data_needed=int(np.ceil(no_encoder/2))
    
    a1=np.roll(np.expand_dims(data,axis=1), shift=int(shift))
    output=a1
    for i in range(1,data_needed):   
        output=np.append(output, np.zeros_like(a1),axis=1)
        output=np.append(output, np.roll(a1, shift=-i),axis=1)
    return output[:,:,None]
def make_window_data_sparse_PID(data,no_encoder):
#    shift=((no_encoder+1)/2-1)/2
#    data_needed=int(np.ceil(no_encoder/2))
#    
#    a1=np.roll(np.expand_dims(data,axis=1), shift=int(shift))
#    output=a1
#    for i in range(1,data_needed):   
#        output=np.append(output, np.zeros_like(a1),axis=1)
#        output=np.append(output, np.roll(a1, shift=-i),axis=1)
    return 0
def generate_phase(N, random_phase = True):
    if random_phase == True:
        phase = np.random.uniform(size=N,low=0.0, high=2*np.pi)
        #phase.append(random_phase[0])#5+counter*0.0001)
    else:
        phase = np.zeros((N,1))
    return phase

def generate_shift(N,window_size,symbol_length, no_encoder, random_shift = True):
    if random_shift == True:
        shift = np.random.randint(size=(N,1),low=1, high=(no_encoder*symbol_length-window_size))
    else:
        shift = symbol_length*np.ones((N,1)).astype(np.int32)
    return shift

def generate_attenuation(N, window_size,random_attenuation = True,a_min=0.001):
    if random_attenuation == True:
        a = np.random.uniform(low=a_min,high=1.0, size=(N,1)) +1j*np.zeros((N,1))
        return np.repeat(a, window_size, axis=1)
    else:
        return np.ones((N,window_size)) +1j*np.zeros((N,window_size))
