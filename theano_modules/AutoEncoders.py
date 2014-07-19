import os 
import sys
import gzip
import time
import string

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theano_modules.LogisticRegression import LogisticRegression
from theano_modules.HiddenLayer import HiddenLayer

"""
 Denoising autoencoders are the building blocks for SdA (stacked denoising 
 autoencoders). They are based on auto-encoders as the ones used in Bengio 
 et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
	  - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
class dA(object):
   
	def __init__(self, numpy_rng, theano_rng=None, input=None,
				 n_visible=784, n_hidden=500,
				 W=None, bhid=None, bvis=None):
		
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# note : W' was written as `W_prime` and b' as `b_prime`
		if not W:
			# W is initialized with `initial_W` which is uniformely sampled
			# from -4*sqrt(6./(n_visible+n_hidden)) and
			# 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
			# converted using asarray to dtype
			# theano.config.floatX so that the code is runable on GPU
			initial_W = np.asarray(numpy_rng.uniform(
					  low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
					  high=4 * np.sqrt(6. / (n_hidden + n_visible)),
					  size=(n_visible, n_hidden)), dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if not bvis:
			bvis = theano.shared(value=np.zeros(n_visible,
								 dtype=theano.config.floatX),
								 borrow=True)

		if not bhid:
			bhid = theano.shared(value=np.zeros(n_hidden,
								 dtype=theano.config.floatX),
								 name='b',
								 borrow=True)

		self.W = W

		self.b = bhid

		self.b_prime = bvis

		self.W_prime = self.W.T

		self.theano_rng = theano_rng

		if input == None:
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

		self.params = [self.W, self.b, self.b_prime]

	def get_corrupted_input(self, input, corruption_level):
		"""This function keeps ``1-corruption_level`` entries of the inputs the
		same and zero-out randomly selected subset of size ``coruption_level``
		Note : first argument of theano.rng.binomial is the shape(size) of
			   random numbers that it should produce
			   second argument is the number of trials
			   third argument is the probability of success of any trial

				this will produce an array of 0s and 1s where 1 has a
				probability of 1 - ``corruption_level`` and 0 with
				``corruption_level``

				The binomial function return int64 data type by
				default.  int64 multiplicated by the input
				type(floatX) always return float64.  To keep all data
				in floatX when floatX is float32, we set the dtype of
				the binomial to floatX. As in our case the value of
				the binomial is always 0 or 1, this don't change the
				result. This is needed to allow the gpu to work
				correctly as it only support float32 for now.

		"""
		return  self.theano_rng.binomial(size=input.shape, n=1,
										 p=1 - corruption_level,
										 dtype=theano.config.floatX) * input

	def get_hidden_values(self, input):
		return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

	def get_reconstructed_input(self, hidden):
		return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

	def get_cost_updates(self, corruption_level, learning_rate):
		tilde_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)
		# note : we sum over the size of a datapoint; if we are using
		#        minibatches, L will be a vector, with one entry per
		#        example in minibatch
		L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
		# note : L is now a vector, where each element is the
		#        cross-entropy cost of the reconstruction of the
		#        corresponding example of the minibatch. We need to
		#        compute the average of all these to get the cost of
		#        the minibatch
		cost = T.mean(L)

		# compute the gradients of the cost of the `dA` with respect
		# to its parameters
		gparams = T.grad(cost, self.params)
		# generate the list of updates
		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append((param, param - learning_rate * gparam))

		return (cost, updates)

	
"""Stacked denoising auto-encoder class (SdA)

A stacked denoising autoencoder model is obtained by stacking several
dAs. The hidden layer of the dA at layer `i` becomes the input of
the dA at layer `i+1`. The first layer dA gets as input the input of
the SdA, and the hidden layer of the last dA represents the output.
Note that after pretraining, the SdA is dealt with as a normal MLP,
the dAs are only used to initialize the weights.
"""
class SdA(object):
	
	def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
				 hidden_layers_sizes=[500, 500], n_outs=10,
				 corruption_levels=[0.1, 0.1]):

		self.sigmoid_layers = []
		self.dA_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)

		assert self.n_layers > 0

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.matrix('x')  
		self.y = T.ivector('y')  

		# The SdA is an MLP, for which all weights of intermediate layers
		# are shared with a different denoising autoencoders
		# We will first construct the SdA as a deep multilayer perceptron,
		# and when constructing each sigmoidal layer we also construct a
		# denoising autoencoder that shares weights with that layer
		# During pretraining we will train these autoencoders (which will
		# lead to chainging the weights of the MLP as well)
		# During finetunining we will finish training the SdA by doing
		# stochastich gradient descent on the MLP

		for i in xrange(self.n_layers):
			# construct the sigmoidal layer

			if i == 0:
				input_size = n_ins
			else:
				input_size = hidden_layers_sizes[i - 1]

			if i == 0:
				layer_input = self.x
			else:
				layer_input = self.sigmoid_layers[-1].output

			sigmoid_layer = HiddenLayer(rng=numpy_rng,
										input=layer_input,
										n_in=input_size,
										n_out=hidden_layers_sizes[i],
										activation=T.nnet.sigmoid)

			self.sigmoid_layers.append(sigmoid_layer)

			# its arguably a philosophical question...
			# but we are going to only declare that the parameters of the
			# sigmoid_layers are parameters of the StackedDaA
			# the visible biases in the dA are parameters of those
			# dA, but not the SdA
			self.params.extend(sigmoid_layer.params)

			dA_layer = dA(numpy_rng=numpy_rng,
						  theano_rng=theano_rng,
						  input=layer_input,
						  n_visible=input_size,
						  n_hidden=hidden_layers_sizes[i],
						  W=sigmoid_layer.W,
						  bhid=sigmoid_layer.b)

			self.dA_layers.append(dA_layer)

		# The logistic layer on top of the MLP
		self.logLayer = LogisticRegression(
						 input=self.sigmoid_layers[-1].output,
						 n_in=hidden_layers_sizes[-1], n_out=n_outs)

		self.params.extend(self.logLayer.params)

		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

		self.errors = self.logLayer.errors(self.y)

		self.p_y_given_x = self.logLayer.p_y_given_x

		self.input = self.x

	''' Generates a list of functions, each of them implementing one
		step in trainnig the dA corresponding to the layer with same index.
		The function will require as input the minibatch index, and to train
		a dA you just need to iterate, calling the corresponding function on
		all minibatch indexes.

		:type train_set_x: theano.tensor.TensorType
		:param train_set_x: Shared variable that contains all datapoints used
							for training the dA

		:type batch_size: int
		:param batch_size: size of a [mini]batch

		:type learning_rate: float
		:param learning_rate: learning rate used during training for any of
							  the dA layers
	'''	
	def pretraining_functions(self, train_set_x, batch_size):
		index = T.lscalar('index')  
		corruption_level = T.scalar('corruption')  
		learning_rate = T.scalar('lr')  
		
		n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

		batch_begin = index * batch_size
		batch_end = batch_begin + batch_size

		pretrain_fns = []
		for dA in self.dA_layers:
			cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
			
			fn = theano.function(inputs=[index,
							  theano.Param(corruption_level, default=0.2),
							  theano.Param(learning_rate, default=0.1)],
								 outputs=cost,
								 updates=updates,
								 givens={self.x: train_set_x[batch_begin:
															 batch_end]})
			pretrain_fns.append(fn)

		return pretrain_fns

	''' Generates a function `train` that implements one step of
		finetuning, a function `validate` that computes the error on
		a batch from the validation set, and a function `test` that
		computes the error on a batch from the testing set

		:type datasets: list of pairs of theano.tensor.TensorType
		:param datasets: It is a list that contain all the datasets;
						 the has to contain three pairs, `train`,
						 `valid`, `test` in this order, where each pair
						 is formed of two Theano variables, one for the
						 datapoints, the other for the labels

		:type batch_size: int
		:param batch_size: size of a minibatch

		:type learning_rate: float
		:param learning_rate: learning rate used during finetune stage
	'''
	def build_finetune_functions(self, datasets, batch_size, learning_rate):
		(train_set_x, train_set_y) = datasets[0]
		(valid_set_x, valid_set_y) = datasets[1]
		(test_set_x, test_set_y) = datasets[2]

		n_train_batches = train_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size

		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_valid_batches /= batch_size

		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_test_batches /= batch_size

		index = T.lscalar('index')  

		gparams = T.grad(self.finetune_cost, self.params) #gradient

		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append((param, param - gparam * learning_rate))

		train_fn = theano.function(inputs=[index],
			  outputs=self.finetune_cost,
			  updates=updates,
			  givens={
				self.x: train_set_x[index * batch_size:
									(index + 1) * batch_size],
				self.y: train_set_y[index * batch_size:
									(index + 1) * batch_size]},
			  name='train')

		train_score_i = theano.function([index], self.errors,
				 givens={
				   self.x: train_set_x[index * batch_size:
									  (index + 1) * batch_size],
				   self.y: train_set_y[index * batch_size:
									  (index + 1) * batch_size]},
					  name='train')

		valid_score_i = theano.function([index], self.errors,
			  givens={
				 self.x: valid_set_x[index * batch_size:
									 (index + 1) * batch_size],
				 self.y: valid_set_y[index * batch_size:
									 (index + 1) * batch_size]},
					  name='valid')

		test_score_i = theano.function([index], self.errors,
				 givens={
				   self.x: test_set_x[index * batch_size:
									  (index + 1) * batch_size],
				   self.y: test_set_y[index * batch_size:
									  (index + 1) * batch_size]},
					  name='test')

		def train_score():
			return [train_score_i(i) for i in xrange(n_train_batches)]

		def valid_score():
			return [valid_score_i(i) for i in xrange(n_valid_batches)]

		def test_score():
			return [test_score_i(i) for i in xrange(n_test_batches)]

		return train_fn, train_score, valid_score, test_score        
