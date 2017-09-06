# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:31:52 2017
    Test suite for neuralnet.py
@author: Richard
"""

import unittest
from mock import MagicMock

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
        
        J,_ = nnet.cost_function(data['X'], data['y'], lmbda = 0)
       
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
        
        J,_ = nnet.cost_function(data['X'], data['y'], lmbda = 1)
       
        self.assertAlmostEqual(J, 0.383770, places = 6)
         
    def test_sigmoid_gradient(self):
        layers = (1,1)
        nnet = nn(layers)
        g = nnet.sigmoidGradinet(np.array([-1, -0.5, 0, 0.5, 1]))
        a = np.array([0.196612, 0.235004, 0.250000, 0.235004, 0.196612])
        for i in range(len(g)):
            self.assertAlmostEqual(g[i], a[i], places = 6)


if __name__=="__main__":
    unittest.main()