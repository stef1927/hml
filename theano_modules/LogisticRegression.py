import os 
import sys
import random
import math
import string

import numpy as np
import scipy as sci

import theano
import theano.tensor as T

""" Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
                  architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
                 which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
                  which the labels lie

"""
class LogisticRegression(object):
    
    def __init__(self, input, n_in, n_out):
        # The weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                            dtype=theano.config.floatX),
                            name='W', borrow=True)
        # The baises b as a vector of n_out elements
        self.b = theano.shared(value=np.zeros((n_out,),
                            dtype=theano.config.floatX),
                            name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        self.score = theano.function(inputs=[x], outputs=self.p_y_given_x)
        self.classify = theano.function(inputs=[x], outputs=self.y_pred)
    
    def negative_log_likelihood(self, y):
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()