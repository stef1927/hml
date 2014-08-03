import os 
import sys
import random
import math
import string
import time


import numpy as np
import scipy as sci
import matplotlib.pyplot as plt


from sklearn.cross_validation import ShuffleSplit
from sklearn.externals import joblib

from sklearn.cluster import FeatureAgglomeration

#Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence


from xgboost_classifier import XgBoost

def getWeights(xs):
	weights = xs.weights

	pIndexes = xs.labels == xs.pLabel
	nIndexes = xs.labels == xs.nLabel

	w = np.sum(weights)
	p = np.sum(weights[pIndexes])
	n = np.sum(weights[nIndexes])
	
	return { xs.nLabel : n/w, xs.pLabel: p/w }

class Analysis:

	def __init__(self, xs):
		self.xs = xs
		
		weights = getWeights(xs) 

		num_features = 15

		#self.transformer = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
		#self.transformer = RBFSampler(gamma=1, random_state=1, n_components=10)
		self.transformer = FeatureAgglomeration(n_clusters=num_features)

		self.classifier = GradientBoostingClassifier(n_estimators=25, max_depth=10, 
			min_samples_leaf=500, max_features=num_features, verbose=1)

		self.classifiers = [
			LogisticRegression(),
			LinearSVC(C=1, dual=False, class_weight=weights), 

			SGDClassifier(loss="hinge", penalty="l2"),
			SGDClassifier(loss="log", penalty="l2"),
			SGDClassifier(loss="modified_huber", penalty="l2"),

			GradientBoostingClassifier(n_estimators=25, max_depth=10, 
				min_samples_leaf=500, max_features=num_features, verbose=1),

			RandomForestClassifier(n_estimators=25, n_jobs=2),
			#XgBoost(num_round=25),	
		]


	def evaluate(self, n_iter=5):

		self.labels = self.xs.labels
		
		if hasattr(self, 'transformer'):
			print "Transforming data..."
			self.data = self.transformer.fit_transform(self.xs.data, self.labels)

			if hasattr(self.transformer, 'feature_importances_'):
				print "Feature importances :", self.transformer.feature_importances_
				print "Feature labels by importance :", \
					self.xs.header[self.transformer.feature_importances_.argsort()[::-1]]
			print "New shape :", self.data.shape
		else:
			self.data = self.xs.data
		
		if hasattr(self, 'classifier'):
			print "Skipping evaluation, classifier alreday pre-selected"
			return

		best_ams = 0
		best_threshold = 0
		amss = np.empty([n_iter])
		thresholds = np.empty([n_iter])
		scores = np.empty([n_iter])

		rs = ShuffleSplit(self.xs.numPoints, n_iter=n_iter, test_size=.10, random_state=0)
		for clf in self.classifiers:
			print "===>", clf
			i = 0
			for train, test in rs:
				x_train, y_train = self.data[train], self.labels[train]
				x_test, y_test = self.data[test], self.labels[test]

				start_time = time.clock()
				clf.fit(x_train, y_train)
				#print clf.classes_

				scores[i] = clf.score(x_test, y_test) 
				amss[i], thresholds[i] = self.calculateAMS(test, clf)

				end_time = time.clock()

				print(('Score %f, AMS max %f, threshold %f ran for %.2fs') % 
					(scores[i], amss[i], thresholds[i], (end_time - start_time)));
				
				i = i + 1

			ams = amss.mean() 
			if (ams > best_ams):
				best_ams = ams
				best_threshold = thresholds.mean()
				self.classifier = clf
			
			#fig, axs = plot_partial_dependence(clf, x_train, np.arange(self.data.shape[1]), n_cols=6) 
			#plt.show()

			print(('Score %0.4f (+/- %0.4f), AMS %0.4f (+/- %0.4f), threshold %0.4f (+/- %0.4f)') % 
				(scores.mean(), scores.std(), amss.mean(), amss.std(), thresholds.mean(), thresholds.std()));

		print "Best AMS ", best_ams, "with ", self.classifier, "and threshold ", best_threshold	

	def train(self):
		print "Training best classifier..."
		print "===>", self.classifier
		self.classifier.fit(self.data, self.labels)
		ams, self.threshold = self.calculateAMS(np.arange(self.xs.numPoints), self.classifier, True) 

	def save(self):
		if hasattr(self.classifier, 'save'):
			self.classifier.save('saved/model.dmp')
		else:	
			joblib.dump(self.classifier, 'saved/model.pkl') 

	def load(self):
		if hasattr(self.classifier, 'load'):
			self.classifier.load('saved/model.dmp')
		else:	
			self.classifier = joblib.load('saved/model.pkl') 

	def get_scores(self, data, clf):
		if hasattr(clf, 'predict_proba'):
			return clf.predict_proba(data)[:,1]
		return clf.decision_function(data)

	def calculateAMS(self, indexes, clf, plot=False):
		def AMS(s,b):
			bReg = 10.
			return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))

		scores = self.get_scores(self.data[indexes], clf)
		threshold = np.percentile(scores, 85)
		pred = scores >= threshold 

		numPoints = len(scores)

		labels = self.xs.labels[indexes]
		weights = self.xs.weights[indexes]

		sIndexes = labels == self.xs.pLabel # true positive
		bIndexes = labels == self.xs.nLabel # true negative

		s = 0
		b = 0
		wFactor = 1. * self.xs.numPoints / numPoints
		for i in range(numPoints):
			if pred[i]:
				if sIndexes[i]:
					s += weights[i]	* wFactor
				else:
					b += weights[i] * wFactor

		ams = AMS(max(0, s), max(0, b))

		if plot:
			s = b = 0
			amss = np.empty([numPoints])
			sortedIndexes = scores.argsort()
			for tI in range(numPoints):
				amss[tI] = AMS(max(0, s), max(0, b))
		
				if pred[sortedIndexes[tI]]:
					if sIndexes[sortedIndexes[tI]]:
						s += weights[sortedIndexes[tI]]
					else:
						b += weights[sortedIndexes[tI]]

			print('AMS %f, threshold %f' % (ams, threshold))
			self.plotAMSvsRank(amss)
			self.plotAMSvsScore(scores, amss)
		
		return (ams, threshold)

	def computeSubmission(self, xsTest, output_file):	
		if hasattr(self, 'transformer'):
			data = self.transformer.transform(xsTest.data)
		else:
			data = xsTest.data

		scores = self.get_scores(data, self.classifier)
		sortedIndexes = scores.argsort()

		rankOrder = list(sortedIndexes)
		for tI,tII in zip(range(len(sortedIndexes)), sortedIndexes):
			rankOrder[tII] = tI

		submission = np.array([[str(xsTest.testIds[tI]),str(rankOrder[tI]+1),
			's' if scores[tI] >= self.threshold else 'b'] for tI in range(len(xsTest.testIds))])

		submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
		np.savetxt(output_file, submission, fmt='%s', delimiter=',')

		print "Finished generating submission file"	

	def plotAMSvsRank(self, amss):
		fig = plt.figure()
		fig.suptitle('AMS curves', fontsize=14, fontweight='bold')
		vsRank = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)

		vsRank.set_xlabel('rank')
		vsRank.set_ylabel('AMS')

		vsRank.plot(amss,'b-')
		vsRank.axis([0,len(amss), 0, 4.0])

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
		vsScore.axis([scores[sortedIndexes[0]], scores[sortedIndexes[-1]] , 0, 4.0])

		plt.show()    
	
