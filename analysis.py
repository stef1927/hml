import os 
import sys
import random
import math
import string
import time


import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.externals import joblib

#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def AMS(s,b):
	assert s >= 0
	assert b >= 0
	bReg = 10.
	return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))

class Analysis:

	def __init__(self, xs):
		self.xs = xs

		self.classifiers = [
			#LogisticRegression(),
			#LinearSVC(C=1, dual=False, verbose=0),
			#AdaBoostClassifier(n_estimators=50),
			#GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0),
			#RandomForestClassifier(n_estimators=10),
			#RandomForestClassifier(n_estimators=50)

			Pipeline(steps=[('pca', PCA()), ('classifier', LogisticRegression())]),
			Pipeline(steps=[('pca', PCA()), ('classifier', AdaBoostClassifier(n_estimators=50))]),
			Pipeline(steps=[('pca', PCA()), ('classifier', GradientBoostingClassifier(n_estimators=50))]),
			Pipeline(steps=[('pca', PCA()), ('classifier', RandomForestClassifier(n_estimators=50))])
		]

	def evaluate(self):
		#rs = KFold(self.xs.numPoints, n_folds=10, shuffle=true)
		rs = ShuffleSplit(self.xs.numPoints, n_iter=5, test_size=.10, random_state=0)

		best_ams = 0

		for clf in self.classifiers:
			print "===>", clf
			for train, test in rs:
				x_train, y_train = self.xs.data[train], self.xs.labels[train]
				x_test, y_test = self.xs.data[test], self.xs.labels[test]

				clf.fit(x_train, y_train)
				
				score = clf.score(x_test, y_test) 
				amss, amsMax, t, scores = self.calculateAMS(test, clf)

				print(('Score %f, AMS max %f, threshold %f') % (score, amsMax, t));

				if (amsMax > best_ams):
					best_ams = amsMax
					best_amss = amss
					best_scores = scores

					self.threshold = t
					self.classifier = clf

		print "Best AMS ", best_ams, "with ", self.classifier, "and threshold ", self.threshold	
		return best_ams, best_amss, best_scores		

	def train(self):
		self.classifier.fit(self.xs.data, self.xs.labels)

	def save(self):
		joblib.dump(self.classifier, 'model.pkl') 

	def load(self):
		self.classifier = joblib.load('model.pkl') 

	def get_scores(self, data, clf):
		#print clf.classes_
		return clf.predict_proba(data)[:,1]
		#return clf.decision_function(data)

	def calculateAMS(self, indexes, clf):
		scores = self.get_scores(self.xs.data[indexes], clf)
		#print scores

		sortedIndexes = scores.argsort()

		labels = self.xs.labels[indexes]
		weights = self.xs.weights[indexes]

		sIndexes = labels == self.xs.pLabel
		bIndexes = labels == self.xs.nLabel

		s = np.sum(weights[sIndexes])
		b = np.sum(weights[bIndexes])

		amss = np.empty([len(sortedIndexes)])
		amsMax = 0
		threshold = 0.0
		
		numPoints = len(sortedIndexes)
		wFactor = 1. * self.xs.numPoints / numPoints
		#print('Num points %f, Factor: %f' % (numPoints, wFactor))

		for tI in range(numPoints):
			# don't forget to renormalize the weights to the same sum 
			# as in the complete training set
			amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
			# careful with small regions, they fluctuate a lot
			if tI < 0.9 * numPoints and amss[tI] > amsMax:
				amsMax = amss[tI]
				threshold = scores[sortedIndexes[tI]]
		
			if sIndexes[sortedIndexes[tI]]:
				s -= weights[sortedIndexes[tI]]
			else:
				b -= weights[sortedIndexes[tI]]

		#print('Max AMS is %f, threshold %f' % (amsMax, threshold))
		return (amss, amsMax, threshold, scores)

	def computeSubmission(self, xsTest, output_file):	
		scores = self.get_scores(xsTest.data, self.classifier)
		sortedIndexes = scores.argsort()

		rankOrder = list(sortedIndexes)
		for tI,tII in zip(range(len(sortedIndexes)), sortedIndexes):
			rankOrder[tII] = tI

		submission = np.array([[str(xsTest.testIds[tI]),str(rankOrder[tI]+1),
			's' if scores[tI] >= self.threshold else 'b'] for tI in range(len(xsTest.testIds))])

		submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
		np.savetxt(output_file, submission, fmt='%s', delimiter=',')

		print "FInished generating submission file"	

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
		sortedIndexes = scores.argsort()

		fig = plt.figure()
		fig.suptitle('AMS curves', fontsize=14, fontweight='bold')
		vsScore = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		vsScore.set_xlabel('score')
		vsScore.set_ylabel('AMS')

		vsScore.plot(scores[sortedIndexes],amss,'b-')
		vsScore.axis([scores[sortedIndexes[0]], scores[sortedIndexes[-1]] , 0, 4])

		plt.show()    
	
