import os 
import sys
import random
import math
import string
import time

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold

from sklearn.neural_network import BernoulliRBM

from sklearn.cross_validation import ShuffleSplit
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from xgboost_classifier import XgBoost

class Analysis:

	def __init__(self, xs):
		np.random.seed(0)
		self.xs = xs
		
		self.transformers = [
			StandardScaler(),

			#BernoulliRBM(n_components=2),

			#VarianceThreshold(threshold=(.5 * (1 - .5))),

			#LinearSVC(C=0.01, penalty="l1", dual=False),

			SelectKBest(f_classif, k=14),
			
			#GradientBoostingClassifier(max_depth=25, max_features=15, min_samples_leaf=100, 
			#	n_estimators=5, verbose=1),

			#PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
		]

		self.labels = self.xs.labels
		self.weights = self.xs.weights
		self.data = self.transform(self.xs.data, self.labels)		

		#self.classifier = GradientBoostingClassifier(n_estimators=10)
		#self.param_grid={'max_depth' : [5, 25, 50], 
		#				 'min_samples_leaf' : [10, 100, 1000]}

		#self.n_estimators = 5
		#self.classifier = GradientBoostingClassifier(max_depth=25, min_samples_leaf=100, 
		#	n_estimators=self.n_estimators, verbose=1)

		#self.classifier = SVC(kernel='rbf', C=1.0, probability=True)

		#hinge for linear svm, log for logistic regression, modified_huber			
		self.classifier = SGDClassifier(loss="log", n_iter=100, 
			penalty="l2", fit_intercept=True, verbose=0)

		#base_estimator=DecisionTreeClassifier,
		#self.classifier = AdaBoostClassifier(n_estimators=100, learning_rate=1.0),
		
		#self.classifier = RandomForestClassifier(n_estimators=30, n_jobs=2),
		#self.classifier = XgBoost(num_round=25),	


	def transform(self, data, labels=None):
		for trf in self.transformers:
			if labels is not None: 
				print "Fit transform ===>", trf
				data = trf.fit_transform(data, labels)
				print "New shape :", data.shape
				#print "Mean : ", data.mean(axis=0), "Std : ", data.std(axis=0)
				#print data
				if hasattr(trf, 'feature_importances_'):
					print "Feature importances :", trf.feature_importances_
					sortedIndexes = trf.feature_importances_.argsort()[::-1]
					print sortedIndexes
					print trf.feature_importances_[sortedIndexes]
					print self.xs.header[sortedIndexes]
			else:
				print "Simple transform ===>", trf
				data = trf.transform(data)

		return data	

	""" Purpose: to find the best hyper parameters """	
	def grid_search(self, n_iter=5):
		rs = ShuffleSplit(self.data.shape[0], n_iter=1, test_size=.1, random_state=0)
		train, test = rs.__iter__().next()

		x_train, y_train, w_train = self.data[train], self.labels[train], self.weights[train]
		x_test, y_test = self.data[test], self.labels[test]

		scores = ['precision', 'recall']
		for score in scores:
			print("# Tuning hyper-parameters for %s" % score)

			clf = GridSearchCV(self.classifier, self.param_grid, cv=2, 
					scoring=score, verbose=3, n_jobs=2)
			clf.fit(x_train, y_train)

			print("Best parameters set found on development set:")
			print(clf.best_estimator_)
			print("Grid scores on development set:")
			for params, mean_score, scores in clf.grid_scores_:
				print("%0.3f (+/-%0.03f) for %r"
					  % (mean_score, scores.std() / 2, params))

			print("Detailed classification report:")
			print("The model is trained on the full development set.")
			print("The scores are computed on the full evaluation set.")
			y_true, y_pred = y_test, clf.predict(x_test)
			print(classification_report(y_true, y_pred))

			test_ams, test_threshold = self.calculateAMS(test, clf.best_estimator_)
			train_ams, train_threshold = self.calculateAMS(train, clf.best_estimator_)

			print(('Test AMS %f, Train AMS %f') % (test_ams, train_ams))

	""" Purpose: To determine when we start over-fitting """
	def evaluate(self, n_iter=10):
		test_amss = np.empty([n_iter])
		train_amss = np.empty([n_iter])

		i = 0
		rs = ShuffleSplit(self.data.shape[0], n_iter=n_iter, test_size=.1, random_state=0)
		for train, test in rs:
			x_train, y_train, w_train = self.data[train], self.labels[train], self.weights[train]
			x_test, y_test = self.data[test], self.labels[test]

			start_time = time.clock()
			self.classifier.fit(x_train, y_train)

			test_score = self.classifier.score(x_test, y_test) 
			train_score = self.classifier.score(x_train, y_train) 

			test_amss[i], test_threshold = self.calculateAMS(test, self.classifier)
			train_amss[i], train_threshold = self.calculateAMS(train, self.classifier)

			end_time = time.clock()

			print(('Test score %f / AMS %f, Train score %f / AMS %f ran for %.2fs') % 
				(test_score, test_amss[i], train_score, train_amss[i], (end_time - start_time)));
			
			i = i + 1	

		print(('AMS %0.4f (+/- %0.4f)') %  (test_amss.mean(), test_amss.std()))


	""" Purpose: to train the best classifier with full data set """
	def train(self):
		self.classifier.fit(self.data, self.labels)
		ams, self.threshold = self.calculateAMS(np.arange(self.data.shape[0]), self.classifier) 
		
		print(('AMS %f, threshold %f') % (ams, self.threshold));

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
		# Appendix B of http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf	
		return (np.clip(clf.decision_function(data), -1, 1) + 1) / 2  

	def calculateAMS(self, indexes, clf):
		def AMS(s,b):
			bReg = 10.
			return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))

		scores = self.get_scores(self.data[indexes], clf)
		threshold = np.percentile(scores, 85)
		
		pred = scores >= threshold 
		#pred = clf.predict(self.data[indexes])

		numPoints = len(scores)

		labels = self.labels[indexes]
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
	
		return (ams, threshold)

	def computeSubmission(self, xsTest, output_file):	
		data = self.transform(xsTest.data)
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
	
