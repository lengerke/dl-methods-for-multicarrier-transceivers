# DL Methods for Multicarrier Transceivers

The scripts presented here are part of my Master thesis, the results of which
were published in the paper [A Deep Learning Wireless Transceiver with Fully
Learned Modulation and 
Synchronization](https://ieeexplore.ieee.org/abstract/document/8757051). 
They allow training and evaluation of autoencoders, that are able to transmit
data over a wireless channel, using TensorFlow and Keras frameworks. The trained
autoencoder can be imported into GNURadio for over-the-air transmission using 
[Tensorflow GNURadio Blocks](https://github.com/lengerke/gr-tensorflow_cc).

The export files of four examplary trained autoencoder used to be found 
[here](https://cloud.ti.rwth-aachen.de/index.php/s/P4bADCELmJKba6N). Their names
indicate the ratio of transmitted bits per complex baseband samples, for example
AE-7/8 transmits 7 bits using 8 samples. Our experiments showed that AE-7/16 and
AE-8/8 perform best, both over the air and when evaluated over the channel model 
they were trained on.
