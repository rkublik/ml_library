# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:59:47 2017

@author: Richard
"""

import numpy as np

class logistic_regression:
    """ Implements logistic regression, using gradient descent
    
    Attributes:
        theta : parameters for logistic regression
        alpha : gradient descent learning rate
        lbda: regularization constant
    """
    theta = None
    cost_training = None
    alpha = None
    lbda = None
    tol = None
    max_iter = None
    
    def __init__(self):
        """Return a logistic_regression object"""
        pass
    
    def sigmoid(self, z):
        """Computes the sigmoid of z"""
        return 1/(1+np.exp(-z))
        
    
    def cost(self, theta, X, y, lbda):
        """Compute the cost function for logistic regression"""
        try:
            m = float(len(y))
        except TypeError:
            y = y*np.ones((1))
            m = 1.0

        h = self.sigmoid(np.dot(X,theta))
        J = 1/m * (np.dot(-y.T,np.log(h)) - np.dot((1-y).T,np.log(1-h))) + lbda/(2*m) * np.dot(theta[1:].T,theta[1:])
        return float(J)
        
    def cost_gradient(self, theta, X, y, lbda):
        """Compute the cost function gradient for logistic regression"""
        try: 
            m = float(len(y))
        except TypeError:
            m = float(1)
            
        h = self.sigmoid(np.dot(X,theta))

        grad = 1/m * np.dot(X.T,h-y)
        grad[1:] = grad[1:] + lbda/m * theta[1:]

        return grad
            
    def calc_error(self, theta, X, y):
        """Calculate one update of gradient descent"""
        return np.sum(np.multiply((self.sigmoid(np.dot(X,theta)) - y),X))
    
    def train(self, features, target, theta = None, alpha = 0.01, lbda = 1, max_iter = 1500, tol = 1e-6, calc_cost = False):
        """Use provided training set to get optimal parameters for logistic 
        regression model."""
        
        # Set learning parameters:
        self.alpha = alpha
        self.lbda = lbda
        self.tol = tol
        self.max_iter = max_iter
        
        # Add column of ones to features
        X = np.hstack((np.ones((features.shape[0],1)), features))

        if theta is None:
            theta = np.zeros((X.shape[1],1))
        cost_iteration = list()
        err_old = 1e100
        for i in range(max_iter):
            err = self.calc_error(theta, X, target)
            theta = theta - alpha * self.cost_gradient(theta, X, target, lbda)

            if calc_cost:
                cost_iteration.append(self.cost(theta,X,target,lbda))
            if abs(err - err_old)<tol:
                break
            err_old = err
        self.theta = theta
        
        if calc_cost:
            iterations = np.array(range(i+1)).reshape((i+1,1))
            cost_iteration = np.array(cost_iteration).reshape((i+1,1))
            self.cost_training=np.hstack((iterations,cost_iteration))
        else:
            self.cost_training = None
            