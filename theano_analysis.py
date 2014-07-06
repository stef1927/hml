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

def shared_dataset(data_x, data_y, borrow=True):
	""" Function that loads the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""

	if data_y is None:
        data_y = np.zeros((data_x.shape[0]), dtype=theano.config.floatX)

	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
	
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
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

		self.xs = xs

	""" Train a neural net via stochastic gradient descent """
	def train(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, 
					n_epochs=1000, batch_size=20, n_hidden=500):
		
		# compute number of minibatches for training, validation and testing
		n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / batch_size

		print('Building the model, num train batches %i, num valid batches %i' % (n_train_batches, n_valid_batches));

		index = T.lscalar()  
		x = T.matrix('x')  
		y = T.ivector('y')  

		rng = np.random.RandomState(1234)

		self.classifier = MLP(rng=rng, input=x, n_in=self.xs.numFeatures, n_hidden=n_hidden, n_out=2)

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
		test_score = 0.
		start_time = time.clock()

		epoch = 0
		done_looping = False
		
		errorsTrain = np.zeros((n_epochs+1))
		self.errorsValidation = np.zeros((n_epochs+1))    

		while (epoch < n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in xrange(n_train_batches):

				minibatch_avg_cost = train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * n_train_batches + minibatch_index

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
						 (epoch, minibatch_index + 1, n_train_batches, patience, iter, 
						  best_test_loss * 100., best_validation_loss * 100.))     

				if patience <= iter:
					done_looping = True
					break
						
			self.errorsTrain[epoch] = best_train_loss
			self.errorsValidation[epoch] = best_validation_loss
			
		end_time = time.clock()
		print(('Optimization complete. Best validation score of %f %% '
			   'obtained at iteration %i, with test performance %f %%') %
			  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
		print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

	def plotErrors(self):	
		fig = plt.figure()
		fig.suptitle('Learning curves', fontsize=14, fontweight='bold')
		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		ax.set_xlabel('number of epochs')
		ax.set_ylabel('errors')

		ax.plot(range(epoch), self.errorsTrain[:epoch] * 100, 'b-')
		ax.plot(range(epoch), self.errorsValidation[:epoch] * 100, 'r-')

		ax.axis([0, epoch, 0, 100])

		plt.show()

	def getTestScores(self):
		return getScores(self, self.xs.test)

	def getScores(self, xs):
		return self.classifier.score(inputs=self.xs)	

	def calculateAMS(self, scores):
		tIIs = scores.argsort()

		s = np.sum(self.xs.weightsTest[self.xs.sSelectorTest])
		b = np.sum(self.xs.weightsTest[self.xs.bSelectorTest])
		amss = np.empty([len(tIIs)])
		amsMax = 0
		threshold = 0.0
		
		for tI in range(len(tIIs)):
		    # don't forget to renormalize the weights to the same sum 
		    # as in the complete training set
		    amss[tI] = AMS(max(0,s * self.xs.wFactor),max(0,b * self.xs.wFactor))
		    # careful with small regions, they fluctuate a lot
		    if tI < 0.9 * len(tIIs) and amss[tI] > amsMax:
		        amsMax = amss[tI]
		        threshold = scores[tIIs[tI]]
		        #print tI,threshold
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
		vsScore.axis(scores[tIIs[0]], scores[tIIs[-1]] , 0, 4])

		plt.show()    

	def computeSubmission(self, xsTest, output_file):	
		testIds = np.array([int(row[0]) for row in xsTest.all[1:]])

		scores = getScores(xsTest.xs)

    	testInversePermutation = scores.argsort()
		testPermutation = list(testInversePermutation)
		for tI,tII in zip(range(len(testInversePermutation)), testInversePermutation):
    		testPermutation[tII] = tI

    	submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
            's' if scores[tI] >= threshold else 'b'] for tI in range(len(testIds))])

    	submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
    	np.savetxt(output_file, submission, fmt='%s', delimiter=',')
