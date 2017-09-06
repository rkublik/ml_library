# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:50:11 2017

@author: Richard
"""

import numpy as np

class neural_net:
    """ Implements a neural network, following Andrew Ng ML course
    
    Attributes:
        layers: a tuple containin the size of each layer
    """
    layers = None
    weights = None
    def __init__(self, layers, epsilon = 0.12):
        """Return a neural_network object"""
        self.layers = layers
        for l in range(len(layers)-1):
            w = np.random.uniform(0, epsilon,(layers[l+1], layers[l] + 1))
            if self.weights is None:
                self.weights = w
            else:
                self.weights = (self.weights, w)
                
    def sigmoid(self, z):
        """ computes sigmoid of input z """
        
        return 1.0 / (1.0 + np.exp(-z))
        
        
        
    def cost_function(self, X, y, lmbda):
        """ Calculate the cost function given X, y, and 
        regularization parameter lmbda
        
        Also compute cost function gradient. Doing this in the same function to 
        re-use computed values.
        """
        grad = None
        a = [None]*len(self.layers)
        z = [None]*len(self.layers)
        
        h = X
        m = float(np.shape(X)[0])
        
        yvec = np.zeros((np.shape(X)[0],self.layers[-1]))
        for i in range(int(m)):
            yvec[i,y[i]] = 1
            
        for w in range(len(self.weights)):
            a = np.hstack((np.ones((np.shape(h)[0],1)),h))
            h = self.sigmoid(np.dot(a,self.weights[w].T))
            
        # compute cost function
        J = 0
        for k in range(self.layers[-1]):
            J = J + 1/m * (np.dot(-yvec[:,k].T, np.log(h[:,k])) - 
                            np.dot((1 - yvec[:,k]).T, np.log(1-h[:,k])))
        # add regularization:
        for w in range(len(self.weights)):
            J = J + lmbda/(2*m) * np.sum(np.square(self.weights[w][:,1:]).flatten())
            
            
        # Compute cost function gradient
            
        return J, grad
    
    def sigmoidGradinet(self,z):
        h = self.sigmoid(z)
        return np.multiply(h, 1-h)
    
    