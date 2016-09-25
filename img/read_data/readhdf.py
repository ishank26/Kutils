import h5py
import matplotlib.pyplot as plt
import numpy as np


f = h5py.File('weights_vgg_train.hdf5','r')

print f
print f.keys().index('dense_12')

print f.values()[13]

g=h5py.File('vgg16_weights.h5','r')

#for i in g.values()[13]:
	#print dim