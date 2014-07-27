import os 
import sys
import random
import math
import string
import time

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from theano_modules.LogisticRegression import LogisticRegression
from theano_modules.HiddenLayer import HiddenLayer
from theano_modules.MLP import MLP
from theano_modules.AutoEncoders import dA
from theano_modules.AutoEncoders import SdA

""" Functions that load the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""

def shared_dataset(data_x, data_y, borrow=True):
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')

def shared_dataset_x(data_x, borrow=True):
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	return shared_x

class TheanoNet(object):

	def __init__(self, xs):
		xs.split()
		
		self.train_set_x, self.train_set_y = shared_dataset(xs.train, xs.labelsTrain)
		self.valid_set_x, self.valid_set_y = shared_dataset(xs.validation, xs.labelsValidation)
		self.test_set_x,  self.test_set_y = shared_dataset(xs.test, xs.labelsTest)

		self.datasets = [ (self.train_set_x, self.train_set_y), 
						  (self.valid_set_x, self.valid_set_y), 
						  (self.test_set_x, self.test_set_y) ]
		self.xs = xs
	
	""" 
		Train a stochastic denoising autoencoder 
		TODO: get rid of cross validation and test data, get rid of AMS part
			  just fit the model with some other convergence
			  and move parameters to constructor
			  and generally fit the interface of scikit learn
	"""
	def fit(self, finetune_lr=0.1, pretraining_epochs=15,
			 pretrain_lr=0.001, training_epochs=75, batch_size=10):
		
		n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size

		numpy_rng = np.random.RandomState(89677)
		print '... building the model'

		self.classifier = SdA(numpy_rng=numpy_rng, n_ins=self.xs.numFeatures,
				  hidden_layers_sizes=[100, 100],
				  n_outs=2)

		print '... getting the pretraining functions'
		pretraining_fns = \
			self.classifier.pretraining_functions(train_set_x=self.train_set_x, 
												   batch_size=batch_size)

		print '... pre-training the model'
		start_time = time.clock()

		## Pre-train layer-wise
		corruption_level = .1
		for i in xrange(self.classifier.n_layers):
			for epoch in xrange(pretraining_epochs):
				c = []
				for batch_index in xrange(n_train_batches):
					c.append(pretraining_fns[i](index=batch_index,
							 corruption=corruption_level,
							 lr=pretrain_lr))
				print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
				print np.mean(c)

		end_time = time.clock()

		print '... getting the finetuning functions'
		train_fn, train_score_fn, validate_score_fn, test_score_fn = \
			self.classifier.build_finetune_functions(
			datasets=self.datasets, batch_size=batch_size, learning_rate=finetune_lr)

		print '... finetunning the model'
		patience = 10 * n_train_batches  
		patience_increase = 1.5
		improvement_threshold = 0.005  

		validation_frequency = min(n_train_batches, patience / 2)

		best_params = None
		best_validation_loss = np.inf
		best_train_loss = np.inf
		best_ams = 0.
		self.best_threshold = 0.
		start_time = time.clock()

		self.errorsTrain = np.zeros((training_epochs+1))
		self.errorsValidation = np.zeros((training_epochs+1))    

		done_looping = False
		epoch = 0

		while (epoch < training_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in xrange(n_train_batches):
				minibatch_avg_cost = train_fn(minibatch_index)
				iter = (epoch - 1) * n_train_batches + minibatch_index

				if (iter + 1) % validation_frequency == 0:
					validation_losses = validate_score_fn()
					this_validation_loss = np.mean(validation_losses)

					if this_validation_loss < best_validation_loss:
						best_validation_loss = this_validation_loss

						train_losses = train_score_fn()
						best_train_loss = np.mean(train_losses)
						
						test_losses = test_score_fn()
						best_test_loss = np.mean(test_losses)
						
						(amss, amss_max, threshold) = self.calculateAMS(self.getTestScores())

						if ((amss_max - best_ams) > improvement_threshold):
							best_ams = amss_max
							self.best_threshold = threshold
							best_iter = iter
							best_params = self.classifier.params
							patience = max(patience, iter * patience_increase)

					print('epoch %i, patience %i, iter %i, test %f %%, valid %f %%, AMS %f Threshold %f' %
						(epoch, patience, iter, best_test_loss * 100., best_validation_loss * 100., amss_max, threshold))

				if patience <= iter:
					done_looping = True
					break

			self.errorsTrain[epoch] = best_train_loss
			self.errorsValidation[epoch] = best_validation_loss
			self.epoch = epoch

		end_time = time.clock()
		print(('Optimization complete with best validation %f %%, test %f %%, ams %f, threshold %f') %
				(best_validation_loss * 100., best_test_loss * 100., best_ams, self.best_threshold))
		
		print >> sys.stderr, ('The training code ran for %.2fm' % 
			((end_time - start_time) / 60.))

	def plotErrors(self):	
		fig = plt.figure()
		fig.suptitle('Learning curves', fontsize=14, fontweight='bold')
		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		ax.set_xlabel('number of epochs')
		ax.set_ylabel('errors')

		ax.plot(range(self.epoch), self.errorsTrain[:self.epoch] * 100, 'b-')
		ax.plot(range(self.epoch), self.errorsValidation[:self.epoch] * 100, 'r-')

		ax.axis([0, self.epoch, 0, 100])

		plt.show()

	def scores(self):
		#TODO

	def predict(self, xs):
		return self.classifier.scores(xs)[:,1]

	
