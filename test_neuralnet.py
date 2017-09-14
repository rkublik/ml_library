# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:31:52 2017
    Test suite for neuralnet.py
@author: Richard
"""

import unittest
#from mock import MagicMock

import numpy as np
import scipy.io as sio

from neuralnet import neural_net as nn

class initTestCases(unittest.TestCase):
    """ Tests for initialization """
    
    def test_shape_of_weight_matrices(self):
        layers = (5,4,3)
        nnet = nn(layers)        
        for i in range(len(layers)-1):
            self.assertEqual(np.shape(nnet.weights[i]),(layers[i+1],layers[i]+1))
            
        layers = (3,4,5)
        nnet = nn(layers)        
        for i in range(len(layers)-1):
            self.assertEqual(np.shape(nnet.weights[i]),(layers[i+1],layers[i]+1))
        
        
class SigmoidTestCases(unittest.TestCase):
    """ Tests for sigmoid function"""
    
    def test_sigmoid_0(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        # Check that a single value gives the correct result
        self.assertTrue(nnet.sigmoid(0) == 0.5)

    def test_sigmoid_float_0(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        self.assertTrue(nnet.sigmoid(0.0) == 0.5)
        
    def test_sigmoid_array(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        # check that sigmoid works for arrays
        z = np.array([0, 0, 0])
        res = np.array([0.5, 0.5, 0.5])
        self.assertTrue((nnet.sigmoid(z) == 0.5).all())
        self.assertTrue((nnet.sigmoid(z.reshape(3,1)) == res.reshape(3,1)).all())


    def test_sigmoid_large_input(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        
        # check that large input values return something close to 1
        self.assertAlmostEqual(nnet.sigmoid(100),1)
        
    def test_sigmoid_large_negative_input(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        
        # check that large input values return something close to 1
        self.assertAlmostEqual(nnet.sigmoid(-100),0)
        
    def test_sigmoid_range(self):
        layers = (5,4,3)
        nnet = nn(layers)  
        
        z = np.array([-100, 0, 100])
        res = np.array([0, 0.5, 1])
        h = nnet.sigmoid(z)
        for i in range(len(h)):
            self.assertAlmostEqual(h[i], res[i])

    def test_sigmoid_gradient(self):
        layers = (1,1)
        nnet = nn(layers)
        g = nnet.sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
        a = np.array([0.196612, 0.235004, 0.250000, 0.235004, 0.196612])
        for i in range(len(g)):
            self.assertAlmostEqual(g[i], a[i], places = 6)


class CostFunctionTestCases(unittest.TestCase):
    """ Test the cost function, using the data from Ng's ML course """
    
    def test_cost_function_ML_no_regularization(self):
        data = sio.loadmat('test_data/ex4data1.mat')
        #yy = data['y']
        #yy[np.where(yy==10)] = 0
        #data['y'] = yy
        data['y'] = data['y']-1 # convert to index starting at 0.
        
        weights = sio.loadmat('test_data/ex4weights.mat')
        layers = (400,25,10)
        nnet = nn(layers)
        nnet.weights = (weights['Theta1'], weights['Theta2'])
        
        J,_ = nnet.cost_function(nnet.weights, data['X'], data['y'], lmbda = 0)
       
        self.assertAlmostEqual(J, 0.287629, places = 6)
        
        
    def test_cost_function_ML_regularization(self):
        data = sio.loadmat('test_data/ex4data1.mat')
        #yy = data['y']
        #yy[np.where(yy==10)] = 0
        #data['y'] = yy
        data['y'] = data['y']-1 # convert to index starting at 0.
        
        weights = sio.loadmat('test_data/ex4weights.mat')
        layers = (400,25,10)
        nnet = nn(layers)
        nnet.weights = (weights['Theta1'], weights['Theta2'])
        
        J,_ = nnet.cost_function(nnet.weights, data['X'], data['y'], lmbda = 1)
       
        self.assertAlmostEqual(J, 0.383770, places = 6)
         
    def test_roll_unroll(self):
        layers = (3, 5, 3)
        nnet = nn(layers)
        weights = nnet.weights
        unrolled = nnet.unroll(nnet.weights)
        rolled = nnet.roll(unrolled)
        
        for r in range(len(rolled)):
            self.assertEqual(rolled[r].shape, weights[r].shape)
            for i in range(rolled[r].size):
                self.assertAlmostEqual(rolled[r].flatten()[i], weights[r].flatten()[i])
        
    def test_cost_function_gradients(self):
        """ Check implementation of cost function gradients, bu comparing to 
        numerically calculated gradients
        
        This is so long, it should have its own test cases....
        """
        layers = (3, 5, 3)
        weights = [None] * 2
        m = 5
        nnet = nn(layers)
        
        for w in range(len(weights)):
            weights[w] = debugInitializeWeights(layers[w+1], layers[w])
        X = debugInitializeWeights(m, layers[0]-1)
        y = np.array(range(m)).reshape((m,1))
        y = np.mod(y,layers[-1])  

        nnparams = nnet.unroll(weights)

        J, g = nnet.cost_function(nnparams, X, y, lmbda = 0)
        
        numgrad = computeNumericalGradient(lambda theta: nnet.cost_function(theta, X, y, lmbda = 0), nnparams)
        
        diff = np.linalg.norm(numgrad - g)/np.linalg.norm(numgrad + g)
        #print np.linalg.norm(numgrad-g), np.linalg.norm(numgrad + g)
        #print np.hstack((numgrad, g))
        self.assertAlmostEqual(diff, 0, places = 9)
        
        
        
class PredictionTestCases(unittest.TestCase):
    
    def test_predict_ML(self):
        data = sio.loadmat('test_data/ex4data1.mat')
        data['y'] = data['y']-1 # convert to index starting at 0.
        layers = (400,25,10)
        nnet = nn(layers)
        
        nnet.train(data['X'], data['y'], lmbda = 0.001)
        pred = nnet.predict(data['X'])
        
        #print np.mean(pred == data['y'].flatten())*100
        
        self.assertTrue(np.mean(pred == data['y'].flatten())*100>97)
        
        
        
''' utility functions '''
def debugInitializeWeights(output_size, input_size):
    w = np.zeros((output_size, 1 + input_size))
    w = np.sin(xrange(1,w.size+1)).reshape(w.shape)/10.0
    return w

def computeNumericalGradient(func, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        loss1,_ = func(theta - perturb)
        loss2,_ = func(theta + perturb)
        
        
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad
        
        

if __name__=="__main__":
    unittest.main()