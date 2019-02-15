# DL Methods for Multicarrier Transceivers

The scripts presented here are part of my 
[Master thesis](https://cloud.ti.rwth-aachen.de/index.php/s/8QyW6NsPCj2PnQ9) and 
allow training and evaluation of autoencoders, that are able to transmit data over 
a wireless channel, using TensorFlow and Keras frameworks. The trained autoencoder 
can be imported into GNURadio for over-the-air transmission using 
[Tensorflow GNURadio Blocks](https://github.com/johschmitz/gr-tensorflow_cc).

The export files of four examplary trained autoencoder can be found 
[here](https://cloud.ti.rwth-aachen.de/index.php/s/P4bADCELmJKba6N). Their names
indicate the ratio of transmitted bits per complex baseband samples, for example
AE-7/16 transmits 7 bits using 16 samples.