#!/usr/bin/env python3
import scipy
import matplotlib as mpl
import numpy as np
from custom_functions import generate_data_sparse_fair, make_window_data_sparse, generate_phase, generate_shift, generate_attenuation
N_test = 10000
EbN0_dB_range = np.arange(0,8,1)
EbN0_range = np.power(10,EbN0_dB_range/10)
EsN0_range  = EbN0_range *  int(np.log2(M)) / symbol_length
EsN0_dB_range = 10*np.log10(EsN0_range)

test_input_symbol = generate_data_sparse_fair(N_test,M)
test_input_symbol = test_input_symbol.astype(np.int)
test_phase = generate_phase(N_test, random_phase)
test_shift = generate_shift(N_test, window_size,symbol_length,no_encoder,random_shift=random_shift)
test_attenuation = generate_attenuation(N_test, window_size,random_attenuation)
test_window_data = make_window_data_sparse(test_input_symbol, no_encoder)

channel_symbols = encoder.predict(np.arange(M))

def get_timeshift_channeloutput(test_window_data,channel_symbols,N_test,shift,window_size):
    test_window_data = np.squeeze(test_window_data).astype(np.int)
    a = channel_symbols[test_window_data,:]
    a1=np.reshape(a,(N_test,-1))
    shift_indices = shift
    for i in range(1,window_size):
        shift_indices=np.append(shift_indices, shift+i,axis=1)
    b=[]
    for i in range(N_test):
        b.append(np.take(a1[i],shift_indices[i]))
    return np.array(b)
def add_channel_noise(channelinput, EsN0_dB):
    a = np.shape(channelinput)
    EsN0 = np.power(10,EsN0_dB/10)
    stddev = 1.0/(np.sqrt(2*EsN0))
    return channelinput + np.random.normal(size = a, scale=stddev) + 1j*np.random.normal(size = a,scale=stddev) 
def add_phase_noise(channelinput,phase,N_test):
    a = np.exp(np.zeros((N_test,1))+1j*np.expand_dims(phase,axis=1))    
    #return np.multiply(channelinput,a.transpose())
    return np.multiply(channelinput,a)
def attenuate(channelinput,attenuation):
    return channelinput*attenuation


#M1=128
#M2=64
#M3=32
#qamM1 = 1-(1-2*(1-1/np.sqrt(M1))*1/2*scipy.special.erfc(np.sqrt(3/2*np.log2(M1)/(M1-1)*EbN0_range)))**2
#qamM2 = 1-(1-2*(1-1/np.sqrt(M2))*1/2*scipy.special.erfc(np.sqrt(3/2*np.log2(M2)/(M2-1)*EbN0_range)))**2
#qamM3 = 1-(1-2*(1-1/np.sqrt(M3))*1/2*scipy.special.erfc(np.sqrt(3/2*np.log2(M3)/(M3-1)*EbN0_range)))**2
bpsk = scipy.special.erfc(np.sqrt(EbN0_range))/2

conf_matrix = np.zeros((EbN0_dB_range.size,M,M))
no_errors = np.zeros((EbN0_dB_range.size,1))
signal = get_timeshift_channeloutput(test_window_data,channel_symbols,N_test,test_shift,window_size)
signal = add_phase_noise(signal,test_phase,N_test)
counter=0
for i in EsN0_dB_range:
    print('testing EsN0 ',i, 'dB')
    channel_output = add_channel_noise(signal, i)
    channel_output = attenuate(channel_output,test_attenuation)    
    channel_output = decoder.predict(channel_output)    
    for j in range(channel_output.size):
        conf_matrix[counter,test_input_symbol[j],channel_output[j]] += 1
    no_errors[counter] += N_test-np.sum(np.diag(conf_matrix[counter])).astype(np.int)
    no_errors_byshift=(np.array(np.where(channel_output != test_input_symbol)).squeeze())
    counter += 1
ser = np.divide(no_errors,(N_test))
print ('Symbol Error Rate:',ser)
#=============plotting===============================
mpl.use("pgf")
import matplotlib.pyplot as plt
pgf_with_rc_fonts = {
    "font.family": "Latin Modern Roman",
    "font.serif": [],                   # use latex default serif font    
    "font.sans-serif": [], # use a specific sans-serif font
    'mathtext.bf': 'sans:bold',
    'mathtext.cal': 'cursive',
    'mathtext.default': 'it',
    'mathtext.fallback_to_cm': True,
    'mathtext.fontset': 'custom',
    'mathtext.it': "Latin Modern Math",
    'mathtext.rm': "Latin Modern Math",
    'mathtext.sf': "Latin Modern Math",
    'mathtext.tt': 'monospace',
}
mpl.rcParams.update(pgf_with_rc_fonts)
fig = plt.figure()
ax = fig.gca()
plt.grid(True, which="both")
cmap = [(0, 0.32941176470588235, 0.6235294117647059, 1),           #blue
 (0.8, 0.027450980392156862, 0.11764705882352941, 1),              #red
 (0.3411764705882353, 0.6705882352941176, 0.15294117647058825, 1), #green
 (0.9647058823529412, 0.6588235294117647, 0.0, 1),                 #orange
 (0.0, 0.596078431372549, 0.6313725490196078, 1),                  #petrol
 (0.6313725490196078, 0.06274509803921569, 0.20784313725490197, 1),#burgund
 (0.5568627450980392, 0.7294117647058823, 0.8980392156862745, 1),  #light blue
 (0.8156862745098039, 0.8509803921568627, 0.3607843137254902, 1)]  #light green

line1, = ax.plot(EbN0_dB_range, ser, color=cmap[0], label='Autoencoder')
#line2, = ax.plot(EbN0_dB_range, ser_improved, color=cmap[1], label='first and last shitft removed')
                                                                
plt.legend(handles=[line1])
#ax.set_xticks(np.arange(-4, 13, 2))
ax.set_xlabel('$E_b/N_0$ in dB')
ax.set_ylabel('$P_s$')
ax.set_yscale('log') 
ax.autoscale(enable=True, axis='x', tight=True)
plt.savefig("ser.pdf", bbox_inches='tight')
plt.savefig("ser.pgf", bbox_inches='tight')

#===================confusion matrix=====================
import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    if len(classes)<=10:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j]>=0.25:
                plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
plt.figure()
plot_confusion_matrix(conf_matrix[0], classes=np.arange(M), 
                      title='Normalized confusion matrix')
plt.savefig("conf.pdf", bbox_inches='tight')
plt.savefig("conf.pgf", bbox_inches='tight')

#=================fft plot=======================================
a=np.expand_dims(np.arange(M),1)
np.random.shuffle(a)
#a=np.append(a,np.zeros((M,1)),axis=1)
#a=np.reshape(a,(2*M,-1))
#a=np.roll(a,-1)
b=channel_symbols[a.astype(np.int32),:]
b=np.reshape(b,(-1))
z=np.expand_dims(np.arange(M*symbol_length),axis=1)
z=np.append(z,z,axis=1)
zval=np.linspace(0,M*symbol_length,num=2*M*symbol_length)
#real_interpolated = np.interp(zval, z[:,0], b.real)
#imag_interpolated = np.interp(zval, z[:,1], b.imag)
tck = scipy.interpolate.splrep(z[:,0], np.real(b), s=0)
real_interpolated = scipy.interpolate.splev(zval, tck, der=0)
tck = scipy.interpolate.splrep(z[:,0], np.imag(b), s=0)
imag_interpolated = scipy.interpolate.splev(zval, tck, der=0)
b_fft=np.fft.fft(real_interpolated+1j*imag_interpolated,n=512)
fig = plt.figure()
ax = fig.gca()
plt.grid(True, which="both")
axis_intervals = np.linspace(-2,2,num=512)
line1, = ax.plot(axis_intervals,10*np.log10(np.abs(np.fft.fftshift(b_fft))), color=cmap[0], label='Magnitude')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlabel('$f_g$')
ax.set_ylabel('Magnitude (dB)')
#ax.set_yscale('log')
plt.savefig("FFT.pdf", bbox_inches='tight')
plt.savefig("FFT.pgf", bbox_inches='tight')

#=======================================signal======
mpl.use("pgf")
import matplotlib.pyplot as plt
pgf_with_rc_fonts = {
    "font.family": "Latin Modern Roman",
    "font.serif": [],                   # use latex default serif font    
    "font.sans-serif": [], # use a specific sans-serif font
    'mathtext.bf': 'sans:bold',
    'mathtext.cal': 'cursive',
    'mathtext.default': 'it',
    'mathtext.fallback_to_cm': True,
    'mathtext.fontset': 'custom',
    'mathtext.it': "Latin Modern Math",
    'mathtext.rm': "Latin Modern Math",
    'mathtext.sf': "Latin Modern Math",
    'mathtext.tt': 'monospace',
}
mpl.rcParams.update(pgf_with_rc_fonts)
cmap = [(0, 0.32941176470588235, 0.6235294117647059, 1),           #blue
 (0.8, 0.027450980392156862, 0.11764705882352941, 1),              #red
 (0.3411764705882353, 0.6705882352941176, 0.15294117647058825, 1), #green
 (0.9647058823529412, 0.6588235294117647, 0.0, 1),                 #orange
 (0.0, 0.596078431372549, 0.6313725490196078, 1),                  #petrol
 (0.6313725490196078, 0.06274509803921569, 0.20784313725490197, 1),#burgund
 (0.5568627450980392, 0.7294117647058823, 0.8980392156862745, 1),  #light blue
 (0.8156862745098039, 0.8509803921568627, 0.3607843137254902, 1)]  #light green
a=np.array([0,5,0,16,0,150,0])
oversampling = 2
b=channel_symbols[a.astype(np.int32),:]
b=np.reshape(b,(-1))
z=np.expand_dims(np.arange(a.size*symbol_length),axis=1)
z=np.append(z,z,axis=1)
zval=np.linspace(0,a.size*symbol_length,num=oversampling*a.size*symbol_length)
#real_interpolated = np.interp(zval, z[:,0], b.real)
#imag_interpolated = np.interp(zval, z[:,1], b.imag)
tck = scipy.interpolate.splrep(z[:,0], np.real(b), s=0)
real_interpolated = scipy.interpolate.splev(zval, tck, der=0)
tck = scipy.interpolate.splrep(z[:,0], np.imag(b), s=0)
imag_interpolated = scipy.interpolate.splev(zval, tck, der=0)

fig = plt.figure()
ax = fig.gca()
plt.grid(False, which="both")
line1, = ax.plot( real_interpolated[oversampling*a.size:-oversampling*a.size], color=cmap[0], label='Real')
line3, = ax.plot(np.arange(oversampling-1,(a.size-2)*symbol_length*oversampling+oversampling,oversampling), b[symbol_length:1-symbol_length].real, '.', color=cmap[0])
line2, = ax.plot( imag_interpolated[oversampling*a.size:-oversampling*a.size], color=cmap[1], label='Imag')

line4, = ax.plot(np.arange(oversampling-1,(a.size-2)*symbol_length*oversampling+oversampling,oversampling), b[symbol_length:1-symbol_length].imag, '.', color=cmap[1])
plt.legend(handles=[line1, line2])
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_ylim([-5,5])
ax.set_xlabel('time')
plt.savefig("signal.svg", bbox_inches='tight')