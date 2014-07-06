import os 
import sys
import random
import math
import string
import csv

import numpy as np
import scipy as sci

"""Perform feature scaling ((x - min) / (max - min)) * 1000 
	so that all features are between 0 and 1000, 
	missing values become -1.
"""
def normalize(inData, c1, c2):
	ret = np.array([map(float, row[c1 : c2]) for row in inData])
	ret[ret == -999.0] = np.nan

	ret_min = np.nanmin(ret,0)
	ret_max = np.nanmax(ret,0)

	ret = ((ret - ret_min) / (ret_max - ret_min)) * 1000 
	ret[np.isnan(ret)] = -1
	return ret

class DataSet(object):
	
	def __init__(self, file_name):
		self.all = list(csv.reader(open(file_name,"rb"), delimiter=','))

		self.weights = np.array([float(row[-2]) for row in self.all[1:]])
		self.labels = np.array([map(lambda l: 1.0 if l == 's' else 0.0, row[-1]) for row in self.all[1:]]).flatten()

		self.xs = normalize(self.all[1:], 1, -2)
		(self.numPoints, self.numFeatures) = self.xs.shape

	""" Split data set into training, validation and test"""	
	def split(self): 
		sIndexes = self.labels == 1.0
		bIndexes = self.labels == 0.0

		self.sumWeights = np.sum(self.weights)
		self.sumSWeights = np.sum(self.weights[sIndexes])
		self.sumBWeights = np.sum(self.weights[bIndexes])

		randomPermutation = random.sample(range(len(self.xs)), len(self.xs))
		#np.savetxt("randomPermutation.csv",randomPermutation,fmt='%d',delimiter=',')
		#randomPermutation = np.array(map(int,np.array(list(csv.reader(open("randomPermutation.csv","rb"), delimiter=','))).flatten()))

		numPointsTrain = int(self.numPoints*0.8)
		numPointsValidation = (self.numPoints - numPointsTrain) / 2
		numPointsTest = numPointsValidation
		self.wFactor = 1.* self.numPoints / numPointsTest

		print(('Num points training %d, validation %d, testing %d') %
			  (numPointsTrain, numPointsValidation, numPointsTest))

		self.train = self.xs[randomPermutation[:numPointsTrain]]
		self.validation = self.xs[randomPermutation[numPointsTrain:numPointsTrain+numPointsValidation]]
		self.test = self.xs[randomPermutation[numPointsTrain+numPointsValidation:]]

		self.sSelectorTrain = sIndexes[randomPermutation[:numPointsTrain]]
		self.bSelectorTrain = bIndexes[randomPermutation[:numPointsTrain]]
		self.sSelectorValidation = sIndexes[randomPermutation[numPointsTrain:numPointsTrain+numPointsValidation]]
		self.bSelectorValidation = bIndexes[randomPermutation[numPointsTrain:numPointsTrain+numPointsValidation]]
		self.sSelectorTest = sIndexes[randomPermutation[numPointsTrain+numPointsValidation:]]
		self.bSelectorTest = bIndexes[randomPermutation[numPointsTrain+numPointsValidation:]]

		self.weightsTrain = self.weights[randomPermutation[:numPointsTrain]]
		self.weightsValidation = self.weights[randomPermutation[numPointsTrain:numPointsTrain+numPointsValidation]]
		self.weightsTest = self.weights[randomPermutation[numPointsTrain+numPointsValidation:]]

		self.labelsTrain = self.labels[randomPermutation[:numPointsTrain]]
		self.labelsValidation = self.labels[randomPermutation[numPointsTrain:numPointsTrain+numPointsValidation]]
		self.labelsTest = self.labels[randomPermutation[numPointsTrain+numPointsValidation:]]

		self.sumWeightsTrain = np.sum(self.weightsTrain)
		self.sumSWeightsTrain = np.sum(self.weightsTrain[self.sSelectorTrain])
		self.sumBWeightsTrain = np.sum(self.weightsTrain[self.bSelectorTrain])	
