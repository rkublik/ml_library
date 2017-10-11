# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:50:11 2017

@author: Richard
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
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
        self.weights = [np.random.randn(y, x+1) for x,y in zip(layers[:-1], layers[1:])] # list comprehension!
                
        
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
        return weights.flatten() # so scipy optimizers will work.

    
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

    def stochastic_gradient_descent(self, X, y, lmbda = 0.1, epochs = 1500, 
                                    mini_batch_size = 10, eta = 0.50, tol = 1e-6,
                                    adaptive_eta = True,
                                    test_data = None, verbose = False):
        ''' train neural network using mini-batch stochastic gradient descent.
        Adapted from : http://neuralnetworksanddeeplearning.com/chap1.html
        X = training features
        y = training labels
        lmbda = regularization parameter
        epochs = number of training epochs
        mini_batch_size = number of samples in each mini batch
        eta = learning rate
        test_data = if supplied as a dictionary {X:, y:}, the network will be tested using this data at the end of each epoch
                    default = None
        '''
        if test_data:
            n_test = test_data['y'].size
        n = X.shape[0]
        self.cost_training = []
        if mini_batch_size < 0:
            mini_batch_size = X.shape[0]
        for j in xrange(epochs):
            idx = range(n)
            np.random.shuffle(idx)
            mini_batch_idx = [
                    idx[k:k+mini_batch_size] 
                    for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batch_idx:
                ecost,g = self.update_mini_batch(X[mini_batch,:], y[mini_batch], lmbda, eta)
            self.cost_training.append(ecost)
            if adaptive_eta and len(self.cost_training) > 2 and self.cost_training[-1] > self.cost_training[-2] + tol:
                eta = eta*0.95
            if verbose:
                if test_data:
                    print "Epoch {0}. cv accuracy: {1}/{2}".format(j, self.evaluate(test_data['X'], test_data['y']), n_test)
                else:
                    print "Epoch {0}. Cost Function Value: {1}".format(j, self.cost_training[-1])
            if np.max(np.abs(g))<tol:
                break
                
    def update_mini_batch(self, X, y, lmbda, eta):
        weights = self.unroll(self.weights)
        J,g = self.cost_function(weights, X, y, lmbda)
        weights = weights - eta * g
        self.weights = self.roll(weights)
        return J,g

    def train_optimize(self, X,y, lmbda, init_weights=None, method = "CG",
                       opts = None):
        if init_weights is None:
            init_weights = self.weights
            
        init_weights = self.unroll(init_weights)
            
        res = optimize.minimize(lambda W: self.cost_function(W,X,y,lmbda), 
                                jac = True, x0 = init_weights,
                                method = method,
                                options = opts)
        self.weights = self.roll(res.x)
        
    def train(self, X, y, lmbda = 0.1, epochs = 1500, 
              mini_batch_size = 10, eta = 0.50, adaptive_eta = True, tol = 1e-6,
              test_data = None, verbose = False):
        # init_weights = self.unroll(self.weights)
        # 
        # params, feval, rc = optimize.fmin_tnc(lambda W: self.cost_function(W, X, y, lmbda), x0 = init_weights)
        # self.weights = self.roll(params)
       
        self.stochastic_gradient_descent(X, y, lmbda, epochs, mini_batch_size, 
                                         eta, tol, adaptive_eta, test_data, 
                                         verbose)
        
    def predict(self, X):
        """ return the index of the output layer with the largest value """
        h = X
        for i in range(len(self.weights)):
            z = np.hstack((np.ones((np.shape(h)[0],1)),h))
            h = self.sigmoid(np.dot(z,self.weights[i].T))
        return np.argmax(h, axis = 1)
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)
    
    
if __name__=="__main__":
    import dev_test
    