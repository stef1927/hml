import os 
import sys
import random
import math
import string
import csv

import numpy as np
import scipy as sci

class TrainingSet(object):
	
	def __init__(self, file_name):
		self.all = list(csv.reader(open(file_name,"rb"), delimiter=','))

		self.nLabel = -1.0
		self.pLabel = 1.0

		self.weights = np.array([float(row[-2]) for row in self.all[1:]])
		self.labels = np.array([map(lambda l: self.pLabel if l == 's' else self.nLabel, row[-1]) 
			for row in self.all[1:]]).flatten()

		self.header = np.array(self.all[0])[1:31]
		self.data = np.array([map(float, row[1 : 31]) for row in self.all[1:]])
		
		(self.numPoints, self.numFeatures) = self.data.shape
		print "Finished reading training set :", self.data.shape

	def getWeights(self):
		weights = self.weights
		pIndexes = self.labels == self.pLabel
		nIndexes = self.labels == self.nLabel

		w = np.sum(weights)
		p = np.sum(weights[pIndexes])
		n = np.sum(weights[nIndexes])
	
		return { self.nLabel : n/w, self.pLabel: p/w }	


class TestSet(object):
	
	def __init__(self, file_name):
		self.all = list(csv.reader(open(file_name,"rb"), delimiter=','))
		self.testIds = np.array([int(row[0]) for row in self.all[1:]])

		self.data = np.array([map(float, row[1 : 31]) for row in self.all[1:]])

		(self.numPoints, self.numFeatures) = self.data.shape
		print "Finished reading test set :", self.data.shape