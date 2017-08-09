# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:31:52 2017
    Test suite for logistic_regression.py
@author: Richard
"""

import unittest
from mock import MagicMock

import numpy as np


from logistic_regression import logistic_regression

class SigmoidTestCases(unittest.TestCase):
    """ Tests for sigmoid function"""
    
    def test_sigmoid_0(self):
        lr = logistic_regression()
        # Check that a single value gives the correct result
        self.assertTrue(lr.sigmoid(0) == 0.5)

    def test_sigmoid_float_0(self):
        lr = logistic_regression()
        self.assertTrue(lr.sigmoid(0.0) == 0.5)
        
    def test_sigmoid_array(self):
        lr = logistic_regression()
        # check that sigmoid works for arrays
        z = np.array([0, 0, 0])
        res = np.array([0.5, 0.5, 0.5])
        self.assertTrue((lr.sigmoid(z) == 0.5).all())
        self.assertTrue((lr.sigmoid(z.reshape(3,1)) == res.reshape(3,1)).all())


    def test_sigmoid_large_input(self):
        lr = logistic_regression()
        
        # check that large input values return something close to 1
        self.assertAlmostEqual(lr.sigmoid(100),1)
        
    def test_sigmoid_large_negative_input(self):
        lr = logistic_regression()
        
        # check that large input values return something close to 1
        self.assertAlmostEqual(lr.sigmoid(-100),0)
        
    def test_sigmoid_range(self):
        lr = logistic_regression()
        
        z = np.array([-100, 0, 100])
        res = np.array([0, 0.5, 1])
        h = lr.sigmoid(z)
        for i in range(len(h)):
            self.assertAlmostEqual(h[i], res[i])


class CostFunctionTestCases(unittest.TestCase):
    """Tests for Cost Function calculation"""
    
    def test_cost_sigmoid_half_no_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((4,1)))
        theta = np.ones((6,1))
        x = np.ones((4,6))
        y = np.ones((4,1)) # y's all cancel out in this case
        lmbda = 0
        self.assertAlmostEqual(lr.cost(theta, x, y, lmbda),-np.log(0.5))

    def test_cost_sigmoid_half_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((4,1)))
        theta = np.ones((6,1))
        x = np.ones((4,6))
        y = np.ones((4,1)) # y's all cancel out in this case
        lmbda = 1
        self.assertAlmostEqual(lr.cost(theta, x, y, lmbda),-np.log(0.5) + 5/8.0)
        
    def test_cost_sigmoid_half_scalar_no_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((1,1)))
        theta = np.array([1,1])
        x = np.array([1,1]) # not used
        y = 1 # y's all cancel out in this case
        lmbda = 0
        self.assertAlmostEqual(lr.cost(theta, x, y, lmbda),-np.log(0.5))
        
    def test_cost_sigmoid_half_scalar_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((1,1)))
        theta = np.array([1,1])
        x = np.array([1,1]) # not used
        y = 1 # y's all cancel out in this case
        lmbda = 1
        self.assertAlmostEqual(lr.cost(theta, x, y, lmbda),-np.log(0.5) + 0.5)
        

class CostFunctionGradientTestCases(unittest.TestCase):
    """Tests for cost_gradient function"""
    def test_cost_gradient_shape(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = np.ones((4,1)))
        theta = np.ones((5,1))
        x = np.ones((4,5))
        y = np.ones(4)
        lmbda = 0
        grad = lr.cost_gradient(theta,x,y,lmbda)
        self.assertFalse(grad.shape==(5,1))
        y = np.ones((4,1))
        grad = lr.cost_gradient(theta,x,y,lmbda)
        self.assertTrue(grad.shape==(5,1))
        
        
    def test_cost_gradient_h_equals_y_no_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = np.ones((4,1)))
        theta = np.ones((5,1))
        x = np.ones((4,5))
        y = np.ones((4,1))
        lmbda = 0
        grad = lr.cost_gradient(theta,x,y,lmbda)
        self.assertTrue(grad.shape==(5,1))
        for i in range(len(grad)):
            self.assertAlmostEqual(grad[i],0)

    def test_cost_gradient_h_equals_y_regularization(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = np.ones((4,1)))
        theta = np.ones((5,1))
        x = np.ones((4,5))
        y = np.ones((4,1))
        lmbda = 1
        grad = lr.cost_gradient(theta,x,y,lmbda)
        self.assertAlmostEqual(grad[0],0)
        for i in range(1,len(grad)):
            self.assertAlmostEqual(grad[i],0.25)

    def test_cost_gradient_integer_m(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = np.ones((4,1)))
        theta = np.ones((5,1))
        x = np.ones((4,5))
        y = np.ones((4,1))
        lmbda = 1
        grad = lr.cost_gradient(theta,x,y,lmbda)
        self.assertAlmostEqual(grad[0],0)
        for i in range(1,len(grad)):
            self.assertAlmostEqual(grad[i],0.25)
        

class ErrorCalcTestCases(unittest.TestCase):
    """Tests for calc_error function"""
    
    def test_error_calc_size_x(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 2*np.ones((400,1)))
        theta = np.ones((5,1)) # not used
        x = np.ones((400,5))
        y = np.ones((400,1))
        self.assertAlmostEqual(lr.calc_error(theta,x,y),x.size)
    
        
class TrainTestCases(unittest.TestCase):
    """Tests for the main gradient descent loop"""

    def test_theta_evolves_after_one_step(self):
        lr = logistic_regression()
        a = 0.5
        lr.sigmoid = MagicMock(return_value = np.ones((4,1)))
        features = np.ones((4,5))
        target = np.ones((4,1))
        lr.cost_gradient = MagicMock(return_value = 0.5)

        lr.train(features, target, calc_cost = False, alpha = a, max_iter = 1)
        for i in range(0, len(lr.theta)):
            self.assertNotEqual(lr.theta[i], 0)    
            
            
    def test_gradient_sigmoid_0_5(self):
        lr = logistic_regression()
        a = 0.5
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((4,1)))
        features = np.ones((4,5))
        target = np.ones((4,1))
        target[0]=0
        target[3]=0
        lr.cost_gradient = MagicMock(return_value = 0.5)

        lr.train(features, target, calc_cost = False, alpha = a, max_iter = 1)
        for i in range(0, len(lr.theta)):
            self.assertNotEqual(lr.theta[i], 0)    
        
    def test_train_single_feature(self):
        lr = logistic_regression()
        lr.sigmoid = MagicMock(return_value = 0.5*np.ones((4,1)))
        features = np.ones((4,1))
        target = np.ones((4,1))
        lr.cost_gradient = MagicMock(return_value = 0.5)
        lr.train(features,target, calc_cost = False, max_iter = 1)
        
        for i in range(0, len(lr.theta)):
            self.assertNotEqual(lr.theta[i], 0)    
        
    def test_realistic_data(self):
        lr = logistic_regression()
        data = np.array([[ 34.62365962,  78.02469282,   0.        ],
                        [ 30.28671077,  43.89499752,   0.        ],
                        [ 35.84740877,  72.90219803,   0.        ],
                        [ 60.18259939,  86.3085521 ,   1.        ],
                        [ 79.03273605,  75.34437644,   1.        ],
                        [ 45.08327748,  56.31637178,   0.        ],
                        [ 61.10666454,  96.51142588,   1.        ],
                        [ 75.02474557,  46.55401354,   1.        ],
                        [ 76.0987867 ,  87.42056972,   1.        ],
                        [ 84.43281996,  43.53339331,   1.        ]])
        features = data[:,0].reshape(data.shape[0],1)
        target = data[:,2].reshape(data.shape[0],1)
        lr.train(features,target,calc_cost = False)
        
if __name__=="__main__":
    unittest.main()