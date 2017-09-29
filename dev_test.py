# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 08:52:04 2017

@author: Richard
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from neuralnet import neural_net as nn
if 0:
    data = sio.loadmat('test_data/ex4data1.mat')
    data['y'] = data['y']-1 # convert to index starting at 0.
else:
    dd = np.loadtxt('test_data/train.csv', skiprows = 1, delimiter = ',')
    data = {'X':dd[:,1:],
            'y':dd[:,0].astype(int)}
    
layers = (data['X'].shape[1],25,10)
nnet = nn(layers)
#nnet.train(data['X'], data['y'], 
#           lmbda = 0.01, 
#           epochs = 1500, 
#           mini_batch_size = 100, 
#           eta = 3.0, 
#           test_data = None)

init_weights = nnet.weights

nnet.train_optimize(data['X'], data['y'], lmbda = 0.01,
                    method = 'TNC',
                    init_weights = init_weights,
                    opts = {'gtol':1e-6,
                            'disp':True})

pred = nnet.predict(data['X'])
pred = pred.reshape((pred.size,1))
print "Training accuracy:", np.mean(pred == data['y'])*100



'''

print np.mean(pred == data['y'])*100

plt.imshow(data['X'][0].reshape((20,20)))

label = (pred + 1).flatten()
label[np.where(label==10)] = 0

true = (data['y']+1).flatten()
true[np.where(true == 10)] = 0

idx = random.sample(range(len(label)),50)

for i in range(len(idx)):
    plt.imshow(1-data['X'][idx[i]].reshape((20,20)).T, cmap = 'gray')
    plt.title("Pred: %d, Label: %d"%(label[idx[i]], true[idx[i]]))
    plt.pause(1)


# plot network


layers = (3, 5, 4, 3)
nnet = nn(layers)
weights = nnet.weights
unrolled = nnet.unroll(nnet.weights)
rolled = nnet.roll(unrolled)
'''