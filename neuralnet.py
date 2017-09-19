# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:50:11 2017

@author: Richard
"""

import numpy as np
import scipy.optimize as optimize

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
                self.weights = [w]
            else:
                self.weights.append(w)
                
        
    def sigmoid(self, z):
        """ computes sigmoid of input z """
        
        return 1.0 / (1.0 + np.exp(-z))
        
    def sigmoidGradient(self,z):
        h = self.sigmoid(z)
        return np.multiply(h, 1-h)
        
        
    def roll(self, params):
        weights = [None] * len(self.weights)
        start = 0
        for w in range(len(self.weights)):
            end = start + self.weights[w].size
            weights[w] = params[start:end].reshape(self.weights[w].shape)
            start = end
            
        return weights
        
    def unroll(self, w):
        weights = w[0].flatten().reshape(w[0].size,1)
        for i in range(1, len(w)):
            weights = np.vstack((weights, w[i].reshape(w[i].size,1)))
        return weights

    
    def cost_function(self, weights, X, y, lmbda):
        """ Calculate the cost function given X, y, and 
        regularization parameter lmbda
        
        Also compute cost function gradient. Doing this in the same function to 
        re-use computed values.
        """
        # forward propagation
        if len(weights) != len(self.weights):
            weights = self.roll(weights)
            
        grad = None
        grads = [None]*len(weights)
        for g in range(len(grads)):
            grads[g] = np.zeros(weights[g].shape)
            
        a = [None]*len(self.layers)
        z = [None]*len(weights)
        
        h = X
        m = float(np.shape(X)[0])
        
        yvec = np.zeros((np.shape(X)[0],self.layers[-1]))

        for i in range(int(m)):
            yvec[i,y[i]] = 1
            
        a[0] = X
        for w in range(len(weights)):
            a[w] =  np.hstack((np.ones((np.shape(a[w])[0],1)),a[w]))
            z[w] = np.dot(a[w],weights[w].T)
            a[w+1] = self.sigmoid(z[w])
            
        h = a[-1]
        # compute cost function
        J = 0
        for k in range(self.layers[-1]):
            J = J + 1/m * (np.dot(-yvec[:,k].T, np.log(h[:,k])) - 
                            np.dot((1 - yvec[:,k]).T, np.log(1-h[:,k])))
        # add regularization:
        for w in range(len(weights)):
            J = J + lmbda/(2*m) * np.sum(np.square(weights[w][:,1:]).flatten())
            
        
        # Back propagation:
        d = [None] * len(weights)
        for t in range(int(m)):
            d[-1] = h[t,:] - yvec[t,:]
            for k in xrange(len(d)-2, -1, -1):
                d[k]  = np.multiply(np.dot(weights[k+1].T, d[k+1])[1:],
                 self.sigmoidGradient(z[k][t,:]))
            for k in xrange(len(d)-1, -1, -1):
                grads[k] = grads[k] + np.dot(d[k].reshape((d[k].size,1)), a[k][t,:].reshape((1,a[k].shape[1])))
                
        for g in range(len(grads)):
            grads[g] = 1/m * grads[g]
            grads[g][:, 1:] = grads[g][:,1:] + lmbda/m * weights[g][:,1:]
            
        # unroll gradients
        grad = self.unroll(grads)
        
        #grad = grads[0].reshape((grads[0].size,1))
        #for i in range(1, len(grads)):
        #    grad = np.vstack((grad, grads[i].reshape(grads[i].size,1)))
            
        return J, grad
    
    def gradient_descent(self, X, y, lmbda = 0.01, alpha = 0.01, max_iter = 1500, tol = 1e-6):
        cost = []
        Jold = 1e1000
        weights = self.unroll(self.weights)
        for i in range(max_iter):
            J, g = self.cost_function(weights, X, y, lmbda)
            weights = weights - alpha * g
            cost.append(J)
            if np.abs(J - Jold) < tol:
                break
        self.weights = weights
        self.cost_training = cost
        
    def train(self, X, y, lmbda = 0.001):
        init_weights = self.unroll(self.weights)
        
        params, feval, rc = optimize.fmin_tnc(lambda W: self.cost_function(W, X, y, lmbda), x0 = init_weights)
        self.weights = self.roll(params)
        '''
        res = optimize.minimize(lambda W: self.cost_function(W, X, y, lmbda), 
                                x0 = init_weights, 
                                method = 'CG',
                                jac = True)
        # store the weights
        print res.x
        self.weights = self.roll(res.x)
        '''
        
    def predict(self, X):
        """ return the index of the output layer with the largest value """
        h = X
        for i in range(len(self.weights)):
            z = np.hstack((np.ones((np.shape(h)[0],1)),h))
            h = self.sigmoid(np.dot(z,self.weights[i].T))
        return np.argmax(h, axis = 1)
    
    