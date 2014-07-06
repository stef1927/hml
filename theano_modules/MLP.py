import os 
import sys
import random
import math
import string

import numpy as np
import scipy as sci

import theano
import theano.tensor as T

from theano_modules.HiddenLayer import HiddenLayer
from theano_modules.LogisticRegression import LogisticRegression

"""Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
"""
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.sigmoid)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm 
        self.L1 = abs(self.hiddenLayer.W).sum() 
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm 
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        
        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
        self.score = self.logRegressionLayer.score
        self.classify = self.logRegressionLayer.classify
