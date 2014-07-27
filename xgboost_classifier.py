import inspect
import os
import sys
import numpy as np

sys.path.append("../xgboost/python")
import xgboost as xgb

class XgBoost(object):
	def __init__(self, num_round=120):
		self.num_round = num_round
		self.param = {}

		self.params['objective'] = 'binary:logitraw'
		self.params['bst:eta'] = 0.1 
		self.params['bst:max_depth'] = 6
		self.params['eval_metric'] = 'auc'
		self.params['silent'] = 1
		self.params['nthread'] = 16

	def fit(self, x, y):
		xgmat = xgb.DMatrix(x, label=y)

		watchlist = [ (xgmat, 'train') ]
		self.bst = xgb.train(list(self.params.items()), xgmat, self.num_round, watchlist);
		
		#self.bst.save_model('model')
	
	def score(self, x, y):
		return 0.0 #TODO
		#xgmat = xgb.DMatrix(x)
		#ygmat = xgb.DMatrix(y)
		#ret = self.bst.eval(xgmat, ygmat)
		#print ret
		#return ret

	def decision_function(self, x):	
		xgmat = xgb.DMatrix(x)
		return self.bst.predict(xgmat)

	def save(self, fileName):
		self.bst.save_model(fileName)

	def load(self, fileName):
		self.bst = xgb.Booster(self.params)
		self.bst.load_model(fileName)

