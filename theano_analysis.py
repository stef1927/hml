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

""" Calculate the AMS"""
def AMS(s,b):
	assert s >= 0
	assert b >= 0
	bReg = 10.
	return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))

class Analysis(object):

	def __init__(self, xs):
		xs.split()
		
		self.train_set_x, self.train_set_y = shared_dataset(xs.train, xs.labelsTrain)
		self.valid_set_x, self.valid_set_y = shared_dataset(xs.validation, xs.labelsValidation)
		self.test_set_x,  self.test_set_y = shared_dataset(xs.test, xs.labelsTest)

		self.datasets = [ (self.train_set_x, self.train_set_y), 
						  (self.valid_set_x, self.valid_set_y), 
						  (self.test_set_x, self.test_set_y) ]
		self.xs = xs

	""" Train a MLP via stochastic gradient descent """
	def train(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20, n_hidden=500):
		
		# compute number of minibatches for training, validation and testing
		n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / batch_size

		print('Building the model, num train batches %i, num valid batches %i' % (n_train_batches, n_valid_batches));

		index = T.lscalar()  
		x = T.matrix('x')  
		y = T.ivector('y')  

		rng = np.random.RandomState(1234)

		self.classifier = MLP(rng=rng, input=x, n_in=self.xs.numFeatures, 
							  n_hidden=n_hidden, n_out=2)

		cost = self.classifier.negative_log_likelihood(y) \
			 + L1_reg * self.classifier.L1 \
			 + L2_reg * self.classifier.L2_sqr

		# classification errors
		test_error_model = theano.function(inputs=[index],
				outputs=self.classifier.errors(y),
				givens={
					x: self.test_set_x[index * batch_size:(index + 1) * batch_size],
					y: self.test_set_y[index * batch_size:(index + 1) * batch_size]})

		validate_error_model = theano.function(inputs=[index],
				outputs=self.classifier.errors(y),
				givens={
					x: self.valid_set_x[index * batch_size:(index + 1) * batch_size],
					y: self.valid_set_y[index * batch_size:(index + 1) * batch_size]})
		
		train_error_model = theano.function(inputs=[index],
				outputs=self.classifier.errors(y),
				givens={
					x: self.train_set_x[index * batch_size:(index + 1) * batch_size],
					y: self.train_set_y[index * batch_size:(index + 1) * batch_size]})	
			
		# compute the gradient of cost with respect to theta (sotred in params)
		# the resulting gradients will be stored in a list gparams
		gparams = []
		for param in self.classifier.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)

		# specify how to update the parameters of the model as a list of
		# (variable, update expression) pairs
		updates = []
		# given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
		# same length, zip generates a list C of same size, where each element
		# is a pair formed from the two lists :
		#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
		for param, gparam in zip(self.classifier.params, gparams):
			updates.append((param, param - learning_rate * gparam))

		# compiling a Theano function `train_model` that returns the cost, but
		# in the same time updates the parameter of the model based on the rules
		# defined in `updates`
		train_model = theano.function(inputs=[index], outputs=cost,
				updates=updates,
				givens={
					x: self.train_set_x[index * batch_size:(index + 1) * batch_size],
					y: self.train_set_y[index * batch_size:(index + 1) * batch_size]})

		print '... training'

		# early-stopping parameters
		patience = 20000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is found
		improvement_threshold = 0.995  # a relative improvement of this much is considered significant
		validation_frequency = min(n_train_batches, patience / 2)
									  # go through this many
									  # minibatche before checking the network
									  # on the validation set; in this case we
									  # check every epoch

		best_params = None
		best_test_loss = np.inf
		best_validation_loss = np.inf
		best_train_loss = np.inf
		best_iter = 0
		start_time = time.clock()

		self.epoch = 0
		done_looping = False
		
		self.errorsTrain = np.zeros((n_epochs+1))
		self.errorsValidation = np.zeros((n_epochs+1))    

		while (self.epoch < n_epochs) and (not done_looping):
			self.epoch = self.epoch + 1
			for minibatch_index in xrange(n_train_batches):

				minibatch_avg_cost = train_model(minibatch_index)
				# iteration number
				iter = (self.epoch - 1) * n_train_batches + minibatch_index

				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					validation_losses = [validate_error_model(i) for i in xrange(n_valid_batches)]
					this_validation_loss = np.mean(validation_losses)

					# if we got the best validation score until now
					if this_validation_loss < best_validation_loss:
						#improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss * improvement_threshold:
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss
						
						train_losses = [train_error_model(i) for i in xrange(n_valid_batches)]
						best_train_loss = np.mean(train_losses)
						
						test_losses = [test_error_model(i) for i in xrange(n_test_batches)]
						
						best_test_loss = np.mean(test_losses)
						best_iter = iter

					print('epoch %i, minibatch %i/%i, patience %i, iter %i, test error %f %%, valid error %f %%' %
						 (self.epoch, minibatch_index + 1, n_train_batches, patience, iter, 
						  best_test_loss * 100., best_validation_loss * 100.))     

				if patience <= iter:
					done_looping = True
					break
						
			self.errorsTrain[self.epoch] = best_train_loss
			self.errorsValidation[self.epoch] = best_validation_loss
			
		end_time = time.clock()
		print(('Optimization complete. Best validation score of %f %% '
			   'obtained at iteration %i, with test performance %f %%') %
			  (best_validation_loss * 100., best_iter + 1, best_test_loss * 100.))
		print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

	
	""" Train a stochastic denoising autoencoder """
	def train_SdA(self, finetune_lr=0.1, pretraining_epochs=15,
			 pretrain_lr=0.001, training_epochs=1000, batch_size=1):
		
		n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size

		numpy_rng = np.random.RandomState(89677)
		print '... building the model'

		sda = SdA(numpy_rng=numpy_rng, n_ins=self.xs.numFeatures,
				  hidden_layers_sizes=[100],
				  n_outs=2)

		print '... getting the pretraining functions'
		pretraining_fns = sda.pretraining_functions(train_set_x=self.train_set_x,
													batch_size=batch_size)

		print '... pre-training the model'
		start_time = time.clock()

		## Pre-train layer-wise
		corruption_levels = [.1, .2, .3]
		for i in xrange(sda.n_layers):
			# go through pretraining epochs
			for epoch in xrange(pretraining_epochs):
				# go through the training set
				c = []
				for batch_index in xrange(n_train_batches):
					c.append(pretraining_fns[i](index=batch_index,
							 corruption=corruption_levels[i],
							 lr=pretrain_lr))
				print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
				print np.mean(c)

		end_time = time.clock()

		print '... getting the finetuning functions'
		train_fn, train_score_fn, validate_score_fn, test_score_fn = \
			sda.build_finetune_functions(
			datasets=self.datasets, batch_size=batch_size, learning_rate=finetune_lr)

		print '... finetunning the model'
		patience = 10 * n_train_batches 
		patience_increase = 2. 
		improvement_threshold = 0.995  

		validation_frequency = min(n_train_batches, patience / 2)

		best_params = None
		best_validation_loss = np.inf
		best_train_loss = np.inf
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
					print('epoch %i, minibatch %i/%i, validation error %f %%' %
						  (epoch, minibatch_index + 1, n_train_batches,
						   this_validation_loss * 100.))

					if this_validation_loss < best_validation_loss:

						if (this_validation_loss < best_validation_loss *
							improvement_threshold):
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss

						train_losses = train_score_fn()
						best_train_loss = np.mean(train_losses)
						
						test_losses = test_score_fn()
						best_test_loss = np.mean(test_losses)
						
						best_iter = iter

						print(('     epoch %i, minibatch %i/%i, test error of '
							   'best model %f %%') %
							  (epoch, minibatch_index + 1, n_train_batches,
							   best_test_loss * 100.))

				if patience <= iter:
					done_looping = True
					break

			self.errorsTrain[epoch] = best_train_loss
			self.errorsValidation[epoch] = best_validation_loss
			self.epoch = epoch
			
		end_time = time.clock()
		print(('Optimization complete with best validation score of %f %%,'
			   'with test performance %f %%') %
				(best_validation_loss * 100., best_test_loss * 100.))
		
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

	def getTestScores(self):
		return self.getScores(self.xs.test)

	def getScores(self, xs):
		predict = theano.function(
			inputs=[self.classifier.input], 
			outputs=self.classifier.p_y_given_x)
		return predict(np.asarray(xs, dtype=theano.config.floatX))[:,1]	

	def calculateAMS(self, scores):
		tIIs = scores.argsort()

		s = np.sum(self.xs.weightsTest[self.xs.sSelectorTest])
		b = np.sum(self.xs.weightsTest[self.xs.bSelectorTest])
		amss = np.empty([len(tIIs)])
		amsMax = 0
		self.threshold = 0.0
		
		for tI in range(len(tIIs)):
			# don't forget to renormalize the weights to the same sum 
			# as in the complete training set
			amss[tI] = AMS(max(0,s * self.xs.wFactor),max(0,b * self.xs.wFactor))
			# careful with small regions, they fluctuate a lot
			if tI < 0.9 * len(tIIs) and amss[tI] > amsMax:
				amsMax = amss[tI]
				self.threshold = scores[tIIs[tI]]
				#print tI,self.threshold
			if self.xs.sSelectorTest[tIIs[tI]]:
				s -= self.xs.weightsTest[tIIs[tI]]
			else:
				b -= self.xs.weightsTest[tIIs[tI]]

		return amss

	def plotAMSvsRank(self, amss):
		fig = plt.figure()
		fig.suptitle('AMS curves', fontsize=14, fontweight='bold')
		vsRank = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		vsRank.set_xlabel('rank')
		vsRank.set_ylabel('AMS')

		vsRank.plot(amss,'b-')
		vsRank.axis([0,len(amss), 0, 4])

		plt.show()

	def plotAMSvsScore(self, scores, amss):
		tIIs = scores.argsort()

		fig = plt.figure()
		fig.suptitle('AMS curves', fontsize=14, fontweight='bold')
		vsScore = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		vsScore.set_xlabel('score')
		vsScore.set_ylabel('AMS')

		vsScore.plot(scores[tIIs],amss,'b-')
		vsScore.axis([scores[tIIs[0]], scores[tIIs[-1]] , 0, 4])

		plt.show()    

	def computeSubmission(self, xsTest, output_file):	

		## We divide in batches to avoid out of memory errors, you can
		## simply do scores = self.getScores(xsTest.xs) if you have sufficient
		## memory in your GPU
		num_batches = 20
		num_points = xsTest.numPoints / (num_batches * 1.0)
		scores = np.empty([])
		for i in range(num_batches):
			start = i * num_points
			end = ((i+1) * num_points)
			print(('Calculating test scores for batch %i, from %i to %i') % 
				(i, start, end))

			if i == 0:
				scores = self.getScores(xsTest.xs[start : end,])
			else:	
				scores = np.append(scores, self.getScores(xsTest.xs[start : end,]))

		print "Num test scores:", scores.shape
		print scores

		testInversePermutation = scores.argsort()
		testPermutation = list(testInversePermutation)
		for tI,tII in zip(range(len(testInversePermutation)), testInversePermutation):
			testPermutation[tII] = tI

		submission = np.array([[str(xsTest.testIds[tI]),str(testPermutation[tI]+1),
			's' if scores[tI] >= self.threshold else 'b'] for tI in range(len(xsTest.testIds))])

		submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
		np.savetxt(output_file, submission, fmt='%s', delimiter=',')
