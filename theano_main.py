from data_set import DataSet
from theano_analysis import Analysis

print "Reading dataset...."
#xs = DataSet("../data/training_sub_set.csv")
xs = DataSet("../data/training.csv")

print "Building model...."
analysis = Analysis(xs)
analysis.train(learning_rate=0.001, L1_reg=0.00, L2_reg=0.0005, 
			   n_epochs=1, batch_size=50, n_hidden=500)
	
scores = analysis.getTestScores()  
print ("Training completed, test scores: " , scores.shape)
print scores

analysis.plotErrors()

amss = analysis.calculateAMS(scores)
analysis.plotAMSvsRank(amss)
analysis.plotAMSvsScore(scores, amss)

xsTest = DataSet("../data/test.csv")
analysis.computeSubmission(xsTest, "submission.csv")