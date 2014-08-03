import os 
import sys
import random
import math
import string
import csv

import numpy as np
import scipy as sci

from sklearn import preprocessing

"""Perform feature scaling ((x - min) / (max - min)) 
	so that all features are between 0 and 1
	missing values become 0.

	Labels are converted to 1 and -1.
"""
def normalize(inData, c1, c2):
	ret = np.array([map(float, row[c1 : c2]) for row in inData])
	
	#ret[ret == -999.0] = np.nan

	#ret_min = np.nanmin(ret,0)
	#ret_max = np.nanmax(ret,0)

	#ret = ((ret - ret_min) / (ret_max - ret_min)) * 100 + 0.0000001
	#ret[np.isnan(ret)] = 0
	
	return preprocessing.scale(ret)
	

class TrainingSet(object):
	
	def __init__(self, file_name):
		self.all = list(csv.reader(open(file_name,"rb"), delimiter=','))

		self.nLabel = -1.0
		self.pLabel = 1.0

		self.weights = np.array([float(row[-2]) for row in self.all[1:]])
		self.labels = np.array([map(lambda l: self.pLabel if l == 's' else self.nLabel, row[-1]) 
			for row in self.all[1:]]).flatten()

		self.header = np.array(self.all[0])[1:31]
		self.data = normalize(self.all[1:], 1, 31)
		
		(self.numPoints, self.numFeatures) = self.data.shape
		print "Finished reading training set :", self.data.shape


class TestSet(object):
	
	def __init__(self, file_name):
		self.all = list(csv.reader(open(file_name,"rb"), delimiter=','))
		self.testIds = np.array([int(row[0]) for row in self.all[1:]])

		self.data = normalize(self.all[1:], 1, 31)
		#self.data = normalize(self.all[1:], 14, 31) # Primary only

		(self.numPoints, self.numFeatures) = self.data.shape
		print "Finished reading test set :", self.data.shape